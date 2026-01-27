import os
import json
import time
import redis
import traceback
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer

# --- THE CLEAN IMPORT ---
from sentinel.inference import SentinelInference

print("=" * 60)
print("🚀 Sentinel Stream Worker Starting")
print("=" * 60)

# 1. CONFIG
KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
INPUT_TOPIC = os.getenv('INPUT_TOPIC', 'transactions')
OUTPUT_TOPIC = os.getenv('OUTPUT_TOPIC', 'predictions')
GROUP_ID = os.getenv('CONSUMER_GROUP_ID', 'sentinel-worker-group')
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/prod_v1')

# 2. INIT
try:
    print(f"🤖 Loading Sentinel Engine: {MODEL_PATH}")
    # This single line replaces 100 lines of old code
    sentinel = SentinelInference(model_dir=MODEL_PATH)
    print("✅ Model Loaded")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load model: {e}")
    exit(1)

try:
    redis_client = redis.Redis(host=REDIS_HOST, decode_responses=True)
    redis_client.ping()
    print("✅ Redis Connected")
except Exception as e:
    print(f"❌ Redis Failed: {e}")
    exit(1)

producer = None
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print("✅ Kafka Producer Ready")
except:
    print("⚠️ Kafka Producer unavailable")

# 3. PROCESSING
def process_message(raw_data: dict):
    start_time = time.time()
    
    # A. Validate
    if 'TransactionID' not in raw_data:
        raw_data['TransactionID'] = f"tx_{int(time.time()*1000)}"
    
    # B. Inference (The Library does the heavy lifting)
    try:
        # Sentinel Pipeline: Preprocess -> Feature Eng -> Predict
        result = sentinel.predict(raw_data)
        # result = {'probability': 0.95, 'is_fraud': True, ...}
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return None

    # C. Format Output for Dashboard
    # We combine raw input + model results for the UI
    processing_ms = (time.time() - start_time) * 1000
    
    output = {
        'transaction_id': raw_data['TransactionID'],
        'amount': raw_data.get('TransactionAmt', 0),
        'final_prob': result['probability'],
        'is_fraud': int(result['is_fraud']),
        'composite_risk_score': result['probability'], # Mapping for UI compatibility
        'timestamp': datetime.now().isoformat(),
        'processing_time_ms': round(processing_ms, 2),
        
        # Pass through some raw fields needed for Dashboard UI
        'device_vendor': raw_data.get('DeviceInfo', 'Unknown'),
        'card_email_combo': 'N/A', # Simplified for V2
        'model_version': result.get('model_version', 'prod_v1')
    }
    
    # D. Save to Redis (For Dashboard)
    save_to_redis(output)
    
    # E. Output to Kafka (For downstream systems)
    if producer:
        producer.send(OUTPUT_TOPIC, output)
        
    return output

def save_to_redis(data):
    try:
        pipe = redis_client.pipeline()
        j = json.dumps(data)
        
        # Live Feed
        pipe.lpush('sentinel_stream', j)
        pipe.ltrim('sentinel_stream', 0, 1999)
        
        # Stats
        if data['is_fraud']:
            pipe.incr('stats:fraud_count')
            pipe.lpush('sentinel_alerts', j)
            pipe.ltrim('sentinel_alerts', 0, 99)
        else:
            pipe.incr('stats:legit_count')
            
        pipe.incr('total_processed')
        pipe.execute()
    except Exception as e:
        print(f"⚠️ Redis Error: {e}")

# 4. LOOP
def main():
    print(f"🎧 Listening to {INPUT_TOPIC}...")
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest'
    )
    
    for msg in consumer:
        res = process_message(msg.value)
        if res:
            icon = "🚨" if res['is_fraud'] else "✅"
            print(f"{icon} Tx: {res['transaction_id']} | Score: {res['final_prob']:.4f}")

if __name__ == "__main__":
    main()