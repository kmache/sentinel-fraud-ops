"""
ROLE: The Brain (Worker)
RESPONSIBILITIES:
1. Connects to Kafka (Consumer)
2. Loads ML Models (SentinelInference)
3. Consumes Raw Transactions in BATCHES
4. Predicts Fraud (Vectorized)
5. Saves Enriched Results to Redis (Pipelined & Circuit Broken)
"""
import os
import sys
import json
import logging
import time
import redis
import traceback
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable
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
try:
    model_engine = SentinelInference(model_dir=MODEL_DIR)
    log_event("Model Loaded", version=model_engine.config.get('selected_model', 'Unknown'))
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
        
    except Exception as e:
        log_event("Inference Batch Failed", error=str(e))
        logger.error("❌ CRITICAL INFERENCE ERROR:")
        logger.error(traceback.format_exc())
        send_to_dlq(raw_data, "inference_batch_error", e)
        return

    # --- 2. PREPARE PIPELINE ---
    ids = results['transaction_id']
    probs = results.get('probabilities', [])
    is_frauds = results.get('is_frauds', [])
    actions = results.get('actions', [])
    y_trues = results.get('y_true', [])

    amounts = [d.get('TransactionAmt', 0.0) for d in raw_data]
    timestamps = [d.get('timestamp', datetime.now().replace(microsecond=0).isoformat()) for d in raw_data]

    pipe = redis_client.pipeline()
    count_fraud = 0
    count_legit = 0

    for i in range(len(ids)):
        payload = {
            "transaction_id": ids[i],
            "timestamp": timestamps[i],
            "score": probs[i],
            "is_fraud": int(is_frauds[i]),
            "action": actions[i],
            "ground_truth": int(y_trues[i]),
            "amount": amounts[i],
            "ProductCD": raw_data[i].get("ProductCD", "U"),
            "dist1": raw_data[i].get("dist1", 0),
            "addr1": raw_data[i].get("addr1", 0),
            "C1": raw_data[i].get("C1", 0),
            "C13": raw_data[i].get("C13", 0),
            "C14": raw_data[i].get("C14", 0),
            "UID_velocity_24h": raw_data[i].get("UID_velocity_24h", 0),
            "card_email_combo_fraud_rate": raw_data[i].get("card_email_combo_fraud_rate", 0),
            "P_emaildomain": raw_data[i].get("P_emaildomain", ""),
            "device_vendor": raw_data[i].get("device_vendor", ""),
            "D15": raw_data[i].get("D15", 0)
        }
        payload_json = json.dumps(payload)
        
        # Redis Commands
        pipe.lpush('sentinel_stream', payload_json)
        pipe.setex(f"prediction:{ids[i]}", 3600, payload_json)
        
        if payload['is_fraud'] == 1:
            count_fraud += 1
            pipe.lpush('sentinel_alerts', payload_json)
            pipe.ltrim('sentinel_alerts', 0, 99)
        else:
            count_legit += 1

        # Kafka Send (Output)
        if kafka_producer:
            kafka_producer.send(OUTPUT_TOPIC, value=payload)

    # Redis Cleanup
    pipe.ltrim('sentinel_stream', 0, 999)
    pipe.incrby('stats:fraud_count', count_fraud)
    pipe.incrby('stats:legit_count', count_legit)

    # --- 3. EXECUTE PIPELINE (Circuit Protected) ---
    try:
        execute_redis_pipeline(pipe)
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
                except Exception as e:
                    log_event("Offset Commit Failed", error=str(e))
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

