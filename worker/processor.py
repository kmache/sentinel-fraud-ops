"""
ROLE: The Brain (Worker)
RESPONSIBILITIES:
1. Connects to Kafka (Consumer)
2. Loads ML Models (SentinelInference)
3. Consumes Raw Transactions in BATCHES
4. Predicts Fraud (Vectorized) - Fast Path
5. Calculates SHAP (Threaded) - Async Path
6. Saves Full JSON to 'sentinel_stream' (Capped at 1000)
7. Saves Minimal Arrays to 'stats:hist_...' (Uncapped for ROI/AUC)
"""
import os
import sys
import json
import logging
import time
import redis
import threading
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable, CommitFailedError
from circuitbreaker import circuit, CircuitBreakerError

# Add src to path for shared library
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from sentinel.inference import SentinelInference
except ImportError:
    print("❌ Critical: Could not import SentinelInference. Check PYTHONPATH.")
    sys.exit(1)

# ==============================================================================
# 1. STRUCTURED LOGGING
# ==============================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": "sentinel-worker"
        }
        if hasattr(record, 'extra_data'):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("Worker")

def log_event(msg, **kwargs):
    logger.info(msg, extra={'extra_data': kwargs})

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
KAFKA_BROKER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
INPUT_TOPIC = os.getenv('INPUT_TOPIC', 'transactions')    
OUTPUT_TOPIC = os.getenv('OUTPUT_TOPIC', 'predictions')    
DLQ_TOPIC = os.getenv('DLQ_TOPIC', 'sentinel_dlq')       
CONSUMER_GROUP = os.getenv('CONSUMER_GROUP_ID', 'sentinel_worker_group')

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))
BATCH_TIMEOUT = float(os.getenv('BATCH_TIMEOUT', 5.0))

REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

MODEL_DIR = os.getenv('MODEL_DIR', '/app/models/prod_v1')

# Cache for drift monitoring
_CACHED_MONITORED_FEATURES = []
_LAST_IMPORTANCE_REFRESH = 0
IMPORTANCE_REFRESH_INTERVAL = 300  # Refresh Top 50 list every 5 minutes (300s)

# ==============================================================================
# 3. INITIALIZE CONNECTIONS
# ==============================================================================
try:
    redis_client = redis.Redis(                                                
        host=REDIS_HOST,                                                     
        port=REDIS_PORT,                                                      
        password=REDIS_PASSWORD,                                         
        db=REDIS_DB,                                               
        decode_responses=True,                                       
        socket_timeout=2,                                                      
        socket_connect_timeout=2,                                                    
    )
    redis_client.ping()
    log_event("✅ Redis connected", host=REDIS_HOST)
except Exception as e:
    logger.critical(json.dumps({"event": "❌ redis_connection_failed", "error": str(e)}))
    sys.exit(1)

kafka_producer = None
try:
    kafka_producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all', retries=3, compression_type='lz4'
    )
    log_event("✅ Kafka Producer ready")
except Exception as e:
    logger.warning(json.dumps({"event": "⚠️ kafka_producer_failed", "error": str(e)}))



log_event("⏳ Loading Inference Engine ...")
def load_training_distribution_to_redis():  
    """
    Reads feature_baseline.json from the model directory and 
    caches it in Redis for the drift detection worker.
    """
    baseline_path = Path(MODEL_DIR) / "feature_baseline.json"
    
    try:
        if baseline_path.exists():
            with open(baseline_path, 'r') as f: baseline_data = json.load(f)
            
            redis_client.set("stats:training_distribution", json.dumps(baseline_data))
            
            log_event(
                "✅ Training Baseline Loaded", 
                path=str(baseline_path), 
                features_mapped=len(baseline_data)
            )
        else:
            logger.error(f"❌ CRITICAL: feature_baseline.json not found at {baseline_path}. Drift analysis will fail.")
            
    except Exception as e:
        logger.error(f"❌ Failed to load training distribution to Redis: {e}")

try:
    model_engine = SentinelInference(model_dir=MODEL_DIR)
    redis_client.set('config:threshold', model_engine.config.get('threshold', 0.5))
    load_training_distribution_to_redis() 
    log_event("Model Weights Loaded", version=model_engine.config.get('weights', 'Unknown'))
except Exception as e:
    logger.critical(json.dumps({"event": "❌ model_load_failed", "error": str(e)}))
    sys.exit(1)

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def get_consumer():
    while True:
        try:
            return KafkaConsumer(
                INPUT_TOPIC, 
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id=CONSUMER_GROUP, 
                auto_offset_reset='latest', 
                enable_auto_commit=False
            )
        except NoBrokersAvailable:
            logger.warning("⏳ Waiting for Kafka...")
            time.sleep(3)
        except Exception as e:
            logger.error(f"⚠️ Kafka Error: {e}")
            time.sleep(3)

def send_to_dlq(raw_data_list, error_reason, exception=None):
    if not kafka_producer: return
    if not isinstance(raw_data_list, list): raw_data_list = [raw_data_list]
    
    for raw_data in raw_data_list:
        dlq_payload = {
            "original_message": raw_data,
            "error_reason": error_reason,
            "exception": str(exception) if exception else None,
            "timestamp": datetime.now().replace(microsecond=0).isoformat() 
        }
        try:
            kafka_producer.send(DLQ_TOPIC, value=dlq_payload)
        except Exception as e:
            logger.error(f"❌ Failed to write to DLQ: {e}")


# ==============================================================================
# 5. ASYNC SHAP ENRICHMENT
# ==============================================================================
def update_global_feature_importance(batch_explanations):
    """
    Updates the Global Feature Importance in Redis using Exponential Moving Average.
    This runs fast and doesn't require storing raw data.
    """
    try:
        if not batch_explanations:
            return
        
        batch_totals = {}
        count = 0
        
        for tx_expl in batch_explanations:
            count += 1
            for item in tx_expl:
                feat = item['feature']
                impact = abs(item['impact']) 
                batch_totals[feat] = batch_totals.get(feat, 0.0) + impact
        batch_avg = {k: v / count for k, v in batch_totals.items()}
        redis_key = "stats:global_feature_importance"
        old_data_json = redis_client.get(redis_key)
        
        # Apply Moving Average
        if old_data_json:
            old_data = json.loads(old_data_json)
            alpha = 0.1 
            merged_data = {}
            all_keys = set(batch_avg.keys()) | set(old_data.keys())
            
            for k in all_keys:
                old_val = old_data.get(k, 0.0)
                new_val = batch_avg.get(k, old_val) 
                merged_data[k] = (old_val * (1 - alpha)) + (new_val * alpha)
        else:
            merged_data = batch_avg

        sorted_items = sorted(merged_data.items(), key=lambda x: x[1], reverse=True)[:50]
        final_payload = dict(sorted_items)

        redis_client.set(redis_key, json.dumps(final_payload))

    except Exception as e:
        logger.error(f"⚠️ Global SHAP Update Failed: {e}")

def log_raw_features_for_drift(pipe, raw_data_processed         ):
    """
    Pushes raw feature values to Redis lists for drift analysis.
    Uses a cache to avoid hitting Redis for the 'Top 50' list every batch.
    """
    global _CACHED_MONITORED_FEATURES, _LAST_IMPORTANCE_REFRESH
    
    now = time.time()
    
    # 1. Refresh the list of features to monitor every 5 minutes
    if (now - _LAST_IMPORTANCE_REFRESH) > IMPORTANCE_REFRESH_INTERVAL or not _CACHED_MONITORED_FEATURES:
        try:
            importance_json = redis_client.get("stats:global_feature_importance")
            
            if importance_json:
                importance_data = json.loads(importance_json)
                # Cache the Top 50 features
                _CACHED_MONITORED_FEATURES = sorted(
                    importance_data, key=importance_data.get, reverse=True
                )[:50]
                _LAST_IMPORTANCE_REFRESH = now
                log_event("Refresh Monitored Features", count=len(_CACHED_MONITORED_FEATURES))
        except Exception as e:
            logger.error(f"⚠️ Failed to refresh monitored features: {e}")

    # 2. If we have features to monitor, push their values to Redis
    pushed_count = 0
    if _CACHED_MONITORED_FEATURES:
        for tx_features in raw_data_processed:
            for feat in _CACHED_MONITORED_FEATURES:
                if feat in tx_features:
                    val = tx_features[feat]
                    if val is None:
                        val = np.nan
                    try:
                        float_val = float(val) 
                        
                        list_key = f"stats:hist_feat:{feat}"
                        pipe.rpush(list_key, float_val)
                        pipe.ltrim(list_key, -1000, -1)
                        pushed_count += 1
                        
                    except (ValueError, TypeError):
                        continue
    return pushed_count


def async_enrich_shap(ids, raw_data):
    """
    Background Thread: Calculates SHAP and updates Redis keys.
    This runs parallel to the main Kafka loop.
    """
    try:
        batch_explanations = model_engine.explain(raw_data, n=None)

        update_global_feature_importance(batch_explanations)
        
        pipe = redis_client.pipeline()
        
        for i, tx_id in enumerate(ids):
            key = f"prediction:{tx_id}"
            
            current_json = redis_client.get(key)
            if current_json:
                data = json.loads(current_json)
                data['explanations'] = batch_explanations[i][:10]
                data['explanation_status'] = "ready"
                pipe.setex(key, 3600, json.dumps(data))
        
        pipe.execute()
            
    except Exception as e:
        logger.error(f"❌ Async SHAP Failed: {str(e)}")

# ==============================================================================
# 5. BATCH LOGIC WITH CIRCUIT BREAKER
# ==============================================================================
@circuit(failure_threshold=5, recovery_timeout=30)
def execute_redis_pipeline(pipe):
    """
    Executes the batch write to Redis. 
    Wrapped in Circuit Breaker to fail fast if Redis is down.
    """
    pipe.execute()

def flush_batch(messages):
    if not messages: return

    raw_data = [m.value for m in messages]
    
    try:
        results = model_engine.predict(raw_data)
        raw_processed = results.get("processed_data_records", {}).get('xgb', {}) 
    except Exception as e:
        log_event("Inference Batch Failed", error=str(e))
        logger.error("❌ CRITICAL INFERENCE ERROR:")
        logger.error(traceback.format_exc())
        send_to_dlq(raw_data, "inference_batch_error", e)
        return

    TS_MIN_INTERVAL = 10 
    TS_VALUE_DELTA = 500.0 
    # --- 2. PREPARE PIPELINE ---
    ids = results['transaction_id']
    probs = results['probabilities']
    is_frauds = results['is_frauds'] # Predict as Fraud by the model 
    actions = results['actions']
    y_trues = results['y_true']
    dashboard_records = results['dashboard_data']
    weights = results.get("meta", {}).get("weights", {})

    latency_ms = results.get("meta", {}).get("latency_ms", -1)
    pipe = redis_client.pipeline()
    pipe.set("stats:live_latency_ms", latency_ms) 
    pipe.execute()  

    batch_savings_delta = 0.0
    batch_loss_delta = 0.0

    for i in range(len(ids)):
        amt = float(dashboard_records[i].get('TransactionAmt', 0.0))
        truth = int(y_trues[i])
        decision = int(is_frauds[i]) # 1 = Blocked, 0 = Allowed
        if truth == 1:
            if decision == 1:
                batch_savings_delta += amt
            else:
                batch_loss_delta += amt

    pipe = redis_client.pipeline()
    pipe.incrbyfloat('stats:global_savings', batch_savings_delta)
    pipe.incrbyfloat('stats:global_loss', batch_loss_delta)

    new_totals = pipe.execute() 
    current_total_saved = float(new_totals[0])
    current_total_loss = float(new_totals[1])

    global _LAST_TS_WRITE_TIME, _LAST_SAVED_VALUE
    _LAST_TS_WRITE_TIME = 0
    _LAST_SAVED_VALUE = -1.0
    now = time.time()
    should_write = False

    if (now - _LAST_TS_WRITE_TIME) >= TS_MIN_INTERVAL:
        should_write = True

    elif abs(current_total_saved - _LAST_SAVED_VALUE) >= TS_VALUE_DELTA:
        should_write = True

    if should_write:
        ts_point = {
            "timestamp": datetime.now().isoformat(),
            "cumulative_savings": current_total_saved,
            "cumulative_loss": current_total_loss
        }
        pipe = redis_client.pipeline()
        pipe.rpush('sentinel_timeseries', json.dumps(ts_point))

        _LAST_TS_WRITE_TIME = now
        _LAST_SAVED_VALUE = current_total_saved
        pipe.ltrim('sentinel_timeseries', -100000, -1) 

        log_raw_features_for_drift(pipe, raw_processed)

        execute_redis_pipeline(pipe)

        _LAST_TS_WRITE_TIME = now
        _LAST_SAVED_VALUE = current_total_saved 

    count_fraud = 0
    count_legit = 0
    for i in range(len(ids)):
        score = float(probs[i])
        amt = float(dashboard_records[i].get('TransactionAmt', 0.0)) 
        truth = int(y_trues[i])
        payload = dashboard_records[i]
        payload.update({
            "transaction_id": ids[i],
            "score": score,
            "is_fraud": int(is_frauds[i]),
            "ground_truth": truth,
            "y_true": truth,
            "action": actions[i],
            "model_weights": weights,
            "timestamp": datetime.now().isoformat(),
            "explanation_status": "processing",
            "explanations": []
        })
        
        payload_json = json.dumps(payload)
        
        # Redis Commands
        pipe.lpush('sentinel_stream', payload_json)
        pipe.setex(f"prediction:{ids[i]}", 3600, payload_json)

        pipe.rpush('stats:hist_y_prob', score)
        pipe.rpush('stats:hist_y_true', truth)
        pipe.rpush('stats:hist_amounts', amt)
        
        if payload['is_fraud'] == 1:
            count_fraud += 1
            pipe.lpush('sentinel_alerts', payload_json)
            pipe.ltrim('sentinel_alerts', 0, 99)
        else:
            count_legit += 1
        if kafka_producer:
            kafka_producer.send(OUTPUT_TOPIC, value=payload)

    # Redis Cleanup
    pipe.ltrim('sentinel_stream', 0, 999)
    pipe.incrby('stats:fraud_count', count_fraud)
    pipe.incrby('stats:legit_count', count_legit)

    # --- 3. EXECUTE PIPELINE (Circuit Protected) ---
    try:
        execute_redis_pipeline(pipe)
        if len(ids) > 0:
            t = threading.Thread(target=async_enrich_shap, args=(ids, raw_data))
            t.daemon = True
            t.start()

        if count_fraud > 0:
            log_event("Batch Processed", size=len(ids), frauds=count_fraud)
    
    except CircuitBreakerError as e:
        log_event("Circuit Breaker Open", error=str(e))
        send_to_dlq(raw_data, "redis_circuit_breaker", e)
    
    except Exception as e:
        log_event("Redis Pipeline Failed", error=str(e))
        send_to_dlq(raw_data, "redis_pipeline_error", e)

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def run():
    consumer = get_consumer()
    log_event("System Started", role="Worker/Brain", topic=INPUT_TOPIC, mode="BATCH")
    batch = []
    last_flush_time = time.time()

    try:
        for message in consumer:
            batch.append(message)

            if len(batch) >= BATCH_SIZE or (time.time() - last_flush_time) >= BATCH_TIMEOUT:
                flush_batch(batch)
                
                try:
                    consumer.commit()
                except CommitFailedError:
                    logger.warning("⚠️ Commit Failed: Rebalance in progress")
                except Exception as e:
                    log_event("Offset Commit Failed: Rebalancing in progress", error=str(e))
                
                batch = []
                last_flush_time = time.time()

    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested...")
    except Exception as e:
        log_event("💥 Fatal Loop Error", error=str(e))
        traceback.print_exc()
    finally:
        if consumer: consumer.close()
        if kafka_producer: kafka_producer.close()
        logger.info("👋 Worker Closed")

if __name__ == "__main__":
    run()

