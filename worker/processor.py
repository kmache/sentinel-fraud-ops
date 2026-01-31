"""
ROLE: The Brain (Worker)
RESPONSIBILITIES:
1. Connects to Kafka (Consumer)
2. Loads ML Models (SentinelInference)
3. Consumes Raw Transactions
4. Predicts Fraud & Generates Features
5. Saves Enriched Results to Redis (for Backend/Dashboard)
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
from pydantic import BaseModel, ValidationError, Field
from circuitbreaker import circuit, CircuitBreakerError

# Add src to path for shared library
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from sentinel.inference import SentinelInference
except ImportError:
    print("❌ Critical: Could not import SentinelInference. Check PYTHONPATH.")
    sys.exit(1)

# ==============================================================================
# 1. STRUCTURED LOGGING (For Observability)
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
# Kafka (Source & Sink)
KAFKA_BROKER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
INPUT_TOPIC = os.getenv('INPUT_TOPIC', 'transactions')    
OUTPUT_TOPIC = os.getenv('OUTPUT_TOPIC', 'predictions')    
DLQ_TOPIC = os.getenv('DLQ_TOPIC', 'sentinel_dlq')       
CONSUMER_GROUP = os.getenv('CONSUMER_GROUP_ID', 'sentinel_worker_group')

# Redis (State Store for Backend)
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# Model Artifacts
MODEL_DIR = os.getenv('MODEL_DIR', '/app/models/prod_v1')

# ==============================================================================
# 3. SCHEMA VALIDATION (Raw Input from Simulator)
# ==============================================================================
class TransactionInput(BaseModel):
    """
    Validates data coming from the Simulator/Producer.
    Does NOT expect features like 'dist1' or 'ProductCD' yet.
    """
    transaction_id: str
    TransactionAmt: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    class Config:
        extra = "allow" 

# ==============================================================================
# 4. INITIALIZE CONNECTIONS
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
        retry_on_timeout=True,   
        health_check_interval=30 
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
        acks='all',          
        retries=3,           
        compression_type='lz4'
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
# 5. HELPER FUNCTIONS
# ==============================================================================
def get_consumer():
    """Connect to Kafka Consumer with retries and Manual Commit"""
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
            logger.warning("⏳ Waiting for Kafka Broker (Broker not available)...")
            time.sleep(3)
        except Exception as e:
            logger.error(f"⚠️ Kafka Connection Error: {e}")
            time.sleep(3)

def send_to_dlq(raw_data, error_reason, exception=None):
    """Must Implement: Dead Letter Queue for failed messages"""
    if not kafka_producer: return
    
    dlq_payload = {
        "original_message": raw_data,
        "error_reason": error_reason,
        "exception": str(exception) if exception else None,
        "timestamp": datetime.now().isoformat()
    }
    try:
        kafka_producer.send(DLQ_TOPIC, value=dlq_payload)
        log_event("Message sent to DLQ", reason=error_reason)
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to write to DLQ: {e}")

def log_statistics(start_time: float, processed_count: int):
    """Logs system health as JSON (Compatible with Datadog/Splunk)"""
    elapsed = time.time() - start_time
    rate = processed_count / elapsed if elapsed > 0 else 0
    try:
        fraud = int(redis_client.get('stats:fraud_count') or 0)
        legit = int(redis_client.get('stats:legit_count') or 0)
        total = fraud + legit
        fraud_rate = (fraud / total * 100) if total > 0 else 0.0
        log_event("system_stats", processed=processed_count, rate_tps=round(rate, 2), fraud_rate=round(fraud_rate, 2))
    except Exception:
        pass

@circuit(failure_threshold=5, recovery_timeout=60)
def write_to_redis(key, data, is_fraud):
    """
    Responsibility: Save Enriched Data to Redis.
    This allows backend/main.py to serve stats without loading the model.
    """
    payload_json = json.dumps(data)
    pipe = redis_client.pipeline()
    
    pipe.lpush('sentinel_stream', payload_json)
    pipe.ltrim('sentinel_stream', 0, 999)

    pipe.setex(key, 3600, payload_json)
    
    if is_fraud:
        pipe.incr('stats:fraud_count')
        pipe.lpush('sentinel_alerts', payload_json)
        pipe.ltrim('sentinel_alerts', 0, 99)
    else:
        pipe.incr('stats:legit_count')
        
    pipe.execute()

def process_transaction(raw_data):
    """
    Orchestrates the 'Brain' Logic:
    Raw Data -> Validation -> Feature Eng (Model) -> Prediction -> Storage
    """
    try:
        validated_input = TransactionInput(**raw_data)
    except ValidationError as e:
        log_event("Schema Validation Failed", error=str(e))
        send_to_dlq(raw_data, "schema_validation_failed", e)
        return
    try:
        prediction = model_engine.predict(validated_input.dict())
    except Exception as e:
        log_event("Inference Failed", tx_id=validated_input.transaction_id)
        send_to_dlq(raw_data, "inference_error", e)
        return

    try:
        full_payload = {
            "transaction_id": validated_input.transaction_id,
            "timestamp": validated_input.timestamp,
            
            # Model Output
            "is_fraud": prediction.get("is_fraud", 0),
            "score": prediction.get("probability", 0.0),
            "action": prediction.get("action", "unknown"),
            
            # Raw Feature
            "amount": validated_input.TransactionAmt,
            
            # Features for Dashboard
            "ProductCD": prediction.get("ProductCD", "U"),
            "dist1": prediction.get("dist1", 0),
            "addr1": prediction.get("addr1", 0),
            "C1": prediction.get("C1", 0),
            "C13": prediction.get("C13", 0),
            "C14": prediction.get("C14", 0),
            "UID_velocity_24h": prediction.get("UID_velocity_24h", 0),
            "card_email_combo_fraud_rate": prediction.get("card_email_combo_fraud_rate", 0),
            "P_emaildomain": prediction.get("P_emaildomain", ""),
            "device_vendor": prediction.get("device_vendor", ""),
            "D15": prediction.get("D15", 0)
        }

        write_to_redis(f"prediction:{validated_input.transaction_id}", full_payload, full_payload['is_fraud'])
        
        if kafka_producer:
            kafka_producer.send(OUTPUT_TOPIC, value=full_payload)
            
        if full_payload['is_fraud'] == 1:
            log_event("FRAUD DETECTED", id=full_payload['transaction_id'], score=full_payload['score'])

    except CircuitBreakerError as e:
        log_event("Circuit Breaker Open", error=str(e))
        send_to_dlq(raw_data, "redis_circuit_breaker", e)
    except Exception as e:
        log_event("Processing Error", error=str(e))
        send_to_dlq(raw_data, "processing_error", e)

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def run():
    consumer = get_consumer()
    log_event("System Started", role="Worker/Brain", topic=INPUT_TOPIC)
    
    processed_count = 0
    start_time = time.time()
    last_stat_time = time.time()

    try:
        for message in consumer:
            process_transaction(message.value)
            try:
                consumer.commit()
            except Exception as e:
                log_event("Offset Commit Failed", error=str(e))
            processed_count += 1
            if time.time() - last_stat_time > 10:
                log_statistics(start_time, processed_count)
                last_stat_time = time.time()
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
