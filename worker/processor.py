"""
Sentinel Stream Processor (Production Grade)
1. Consumes Transactions (Kafka)
2. Predicts Fraud (SentinelInference Class)
3. Publishes Results (Redis + Output Kafka)
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

print("=" * 60)
print("🚀 Starting Fraud Detection Stream Processor")
print("=" * 60)

# Add src to path so we can import the shared library
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from sentinel.inference import SentinelInference
except ImportError:
    print("❌ Critical: Could not import SentinelInference. Check PYTHONPATH.")
    sys.exit(1)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - WORKER - %(message)s')
logger = logging.getLogger("Worker")

# ==============================================================================
# 2. CONFIGURATION (Robust)
# ==============================================================================
# Kafka
KAFKA_BROKER = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
INPUT_TOPIC = os.getenv('INPUT_TOPIC', 'transactions')
OUTPUT_TOPIC = os.getenv('OUTPUT_TOPIC', 'predictions')
CONSUMER_GROUP = os.getenv('CONSUMER_GROUP_ID', 'sentinel_worker_group')

# Redis
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_DB = int(os.getenv('REDIS_DB', '0'))

# Model
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
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
    )
    redis_client.ping()
    logger.info(f"✅ Redis connected at {REDIS_HOST}")
except Exception as e:
    logger.critical(f"❌ Redis connection failed: {e}")
    sys.exit(1)

kafka_producer = None
try:
    kafka_producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3
    )
    logger.info("✅ Kafka Producer ready")
except Exception as e:
    logger.warning(f"⚠️ Kafka Producer failed (Output stream disabled): {e}")

logger.info("⏳ Loading Sentinel Inference Engine...")
try:
    model_engine = SentinelInference(model_dir=MODEL_DIR)
    logger.info(f"✅ Model Loaded. Version: {model_engine.config.get('selected_model', 'Unknown')}")
except Exception as e:
    logger.critical(f"❌ Failed to load model artifacts: {e}")
    sys.exit(1)

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def get_consumer():
    """Connect to Kafka Consumer with retries"""
    while True:
        try:
            return KafkaConsumer(
                INPUT_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id=CONSUMER_GROUP,
                auto_offset_reset='latest'
            )
        except NoBrokersAvailable:
            logger.warning("⏳ Waiting for Kafka Broker...")
            time.sleep(3)
        except Exception as e:
            logger.error(f"Kafka Error: {e}")
            time.sleep(3)

def store_prediction(prediction: dict, raw_data: dict):
    """
    Robust storage:
    1. Augments prediction with Dashboard Features.
    2. Uses Redis Pipeline for atomicity.
    3. Sends to Output Kafka Topic.
    """
    try:
        full_payload = {
            "transaction_id": prediction["transaction_id"],
            "timestamp": raw_data.get("timestamp", datetime.now().replace(microsecond=0).isoformat()), 
            
            "is_fraud": prediction["is_fraud"],
            "action": prediction["action"],
            "final_prob": prediction["probability"],
            "score": prediction["probability"],
            
            "amount": raw_data.get("TransactionAmt", 0),
            "ProductCD": raw_data.get("ProductCD", "U"),
            "dist1": raw_data.get("dist1", 0),
            "addr1": raw_data.get("addr1", 0),
            "C1": raw_data.get("C1", 0),
            "C13": raw_data.get("C13", 0),
            "C14": raw_data.get("C14", 0),
            "UID_velocity_24h": raw_data.get("UID_velocity_24h", 0),
            "card_email_combo_fraud_rate": raw_data.get("card_email_combo_fraud_rate", 0),
            "P_emaildomain": raw_data.get("P_emaildomain", ""),
            "D15": raw_data.get("D15", 0),
            "device_vendor": raw_data.get("device_vendor", "")
        }
        
        payload_json = json.dumps(full_payload)
        
        pipe = redis_client.pipeline()
        
        pipe.lpush('sentinel_stream', payload_json)
        pipe.ltrim('sentinel_stream', 0, 999)

        pipe.setex(f"prediction:{prediction['transaction_id']}", 3600, payload_json)
        
        if full_payload['is_fraud'] == 1:
            pipe.incr('stats:fraud_count')
            pipe.lpush('sentinel_alerts', payload_json)
            pipe.ltrim('sentinel_alerts', 0, 99)
        else:
            pipe.incr('stats:legit_count')
            
        pipe.execute()
        
        if kafka_producer:
            kafka_producer.send(OUTPUT_TOPIC, value=full_payload)
            
    except Exception as e:
        logger.error(f"Storage Error: {e}")

def print_statistics(start_time: float, processed_count: int):
    """Logs system health to console"""
    elapsed = time.time() - start_time
    rate = processed_count / elapsed if elapsed > 0 else 0
    
    try:
        fraud = int(redis_client.get('stats:fraud_count') or 0)
        legit = int(redis_client.get('stats:legit_count') or 0)
        total = fraud + legit
        fraud_rate = (fraud / total * 100) if total > 0 else 0
        
        logger.info(f"📊 Stats: {processed_count} processed | {rate:.1f} tx/s | Fraud Rate: {fraud_rate:.1f}%")
    except Exception:
        pass

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def run():
    consumer = get_consumer()
    logger.info(f"🎧 Listening on topic: {INPUT_TOPIC}")
    
    processed_count = 0
    start_time = time.time()
    last_stat_time = time.time()

    try:
        for message in consumer:
            raw_data = message.value
            prediction = model_engine.predict(raw_data)
            store_prediction(prediction, raw_data)
            processed_count += 1
            if time.time() - last_stat_time > 10: # Print every 10s
                print_statistics(start_time, processed_count)
                last_stat_time = time.time()
            if prediction['is_fraud'] == 1:
                logger.info(f"🚨 FRAUD DETECTED: {prediction['transaction_id']} (Score: {prediction['probability']:.4f})")
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested...")
    except Exception as e:
        logger.error(f"💥 Fatal Loop Error: {e}")
        traceback.print_exc()
    finally:
        if consumer: consumer.close()
        if kafka_producer: kafka_producer.close()
        logger.info("👋 Worker Closed")

if __name__ == "__main__":
    run()
