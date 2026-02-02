Let's continue with my Sentinel Fraud detection project: 
let's recall the structure of the project.
├── backend
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── schemas.py
├── config
│   └── params.yaml
├── dashboard
│   ├── api_client.py
│   ├── app.py
│   ├── Dockerfile
│   ├── pages
│   ├── __pycache__
│   ├── requirements.txt
│   └── styles.py
├── data
│   ├── processed
│   └── raw
├── docker-compose.yml
├── models
│   ├── dev_experiment_2026
│   └── prod_v1
├── notebooks
│   ├── 01_exploration.ipynb
│   └── 02_train_model.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
│   ├── download_data.py
│   └── __pycache__
├── simulator
│   ├── Dockerfile
│   ├── producer.py
│   └── requirements.txt
├── src
│   ├── sentinel
│   └── sentinel.egg-info
├── tests
│   ├── __init__.py
│   ├── test_api.py
│   └── test_preprocessing.py
└── worker
    ├── Dockerfile
    ├── processor.py
    └── requirements.txt

here is the content of src/inference.py
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Any, List, Optional
import gc
import logging
import time
from datetime import datetime

from dashboard import styles

# Setup basic logging (In production, this might go to Datadog/Splunk)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SentinelInference")

class SentinelInference:
    """
    Production Inference Wrapper for the Sentinel Fraud Detection System.
    
    Capabilities:
    1. Ensemble Support: Loads multiple models and applies weighted averaging.
    2. Robustness: explicit categorical casting and schema validation.
    3. Business Logic: Returns Tiered Decisions (Approve, Challenge, Block).
    """
    
    def __init__(self, model_dir: str):
        """
        Args:
            model_dir (str): Path to the folder containing the model artifacts.
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.features = {}
        self.config = {}
        self.cat_features = []
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads configuration, processors, and model binaries."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.model_dir}")
        logger.info(f"Loading Sentinel artifacts from {self.model_dir}...")
        start_time = time.time()

        try:
            config_path = self.model_dir / "production_config.json"
            if not config_path.exists():
                raise FileNotFoundError("production_config.json missing. Did you run select_best_model()?")
            
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            self.preprocessor = joblib.load(self.model_dir / 'sentinel_preprocessor.pkl')
            self.engineer = joblib.load(self.model_dir / 'sentinel_engineer.pkl')
            
            cat_path = self.model_dir / 'categorical_features.json'
            if cat_path.exists():
                with open(cat_path, 'r') as f: self.cat_features = json.load(f)
            else:
                logger.warning("categorical_features.json not found. Auto-detection might fail.")

            # Config example: {"weights": {"lgb": 0.7, "xgb": 0.3}, ...}
            weights = self.config.get("weights", {})
            for model_name in weights.keys():
                model_path = self.model_dir / f"{model_name}_model.pkl"
                feat_path = self.model_dir / f"{model_name}_features.json"
                if not model_path.exists() or not feat_path.exists():
                    raise FileNotFoundError(f"Required model {model_name} or features not found at {model_path} or {feat_path}")
                
                self.models[model_name] = joblib.load(model_path)
                with open(feat_path, 'r') as f: features = json.load(f)
                self.features[model_name] = features
                logger.info(f"Loaded {model_name} model.")

            elapsed = time.time() - start_time
            logger.info(f"✅ Sentinel Inference initialized in {elapsed:.2f}s ")

        except Exception as e:
            logger.error(f"Failed to initialize Sentinel: {e}")
            raise RuntimeError(f"Artifact loading failed: {e}")
        

    def predict(self, data: Union[Dict, pd.DataFrame], soft_threshold: float = 0.12) -> Dict[str, Any]:
        """
        Main entry point for predictions.
         
        Args:
            data: Dictionary (single) or DataFrame (batch).
            soft_threshold: Probability where we start 'Challenging' (2FA). 
                            Anything above config['threshold'] is a 'Block'.
        
        Returns:
            Dict with probability, action (APPROVE/CHALLENGE/BLOCK), and metadata.
        """
        start_time = time.time()
        
        if isinstance(data, dict):
            df = pd.DataFrame([data])
            is_batch = False
        else:
            df = data.copy()
            is_batch = True

        if 'TransactionID' not in df.columns:
            print('warning: no TransactionID column found, generating one...')
            df['TransactionID'] = f"tx_{int(time.time() * 1000)}"

        y_true = df['isFraud'] if 'isFraud' in df.columns else 0

        try:
            df_clean = self.preprocessor.transform(df)
            
            self.df_features = self.engineer.transform(df_clean)

            final_probs = np.zeros(len(self.df_features))
            weights = self.config['weights']
            
            for name, weight in weights.items():
                tem_df = self.df_features[self.features[name]]
                for col in self.cat_features:
                    if col in tem_df.columns:
                        tem_df[col] = tem_df[col].astype('category')
                p = self.models[name].predict_proba(tem_df)[:, 1]
                final_probs += p * weight
                del tem_df
                gc.collect()
            hard_threshold = self.config['threshold']

            if not is_batch:
                prob = float(final_probs[0])

                action = self._get_action(prob, soft_threshold, hard_threshold)
                is_fraud = 1 if prob >= hard_threshold else 0
                return {
                    "transaction_id": data.get("TransactionID", f"tx_{int(time.time() * 1000)}"),
                    "probability": round(prob, 4),
                    "is_fraud": is_fraud,
                    "y_true": y_true,
                    "action": action,
                    "meta": {
                        "model_version": self.config.get("selected_model", "ensemble"),
                        "threshold_used": hard_threshold,
                        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
                }

            else:
                actions = [self._get_action(p, soft_threshold, hard_threshold) for p in final_probs]
                is_frauds = [1 if p >= hard_threshold else 0 for p in final_probs]
                return {
                    "transaction_id": [data.get("TransactionID", f"tx_{int(time.time() * 1000) + i}") for i in range(len(final_probs))],
                    "probabilities": np.round(final_probs, 4).tolist(),
                    "is_frauds": is_frauds,
                    "y_true": y_true,
                    "actions": actions,
                    "meta": {
                        "batch_size": len(df),
                        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
                }

        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            raise RuntimeError(f"Inference pipeline failed: {str(e)}")


    def _get_feat4board(self, data: Union[Dict, pd.DataFrame]=None, 
                        features: List[str]=[
                            'TransactionAmt',      
                            'ProductCD',
                            'card_email_combo_fraud_rate',      
                            'P_emaildomain', 'R_emaildomain_is_free',  
                            'UID_velocity_24h',   
                            'dist1',              
                            'addr1', 'card1_freq_enc',                        
                            'D15',                
                            'device_vendor',        
                            'C13', 'C1', 'C14', 'UID_vel'
                        ]) -> Dict[str, Any]:
        """
        Get features for dashboard, prioritizing explainability and impact.
        """
        if data is None:
            data = self.df_features

        feat4board = {}
        if 'TransactionDT' in data:
             feat4board['hour_of_day'] = (data['TransactionDT'] // 3600) % 24

        for col in features:
            if col in data: 
                feat4board[col] = data[col].values.tolist()
                     
        return feat4board

    @staticmethod
    def _get_action(prob: float, soft: float, hard: float) -> str:
        if prob >= hard: return "BLOCK"
        elif prob >= soft: return "CHALLENGE"
        return "APPROVE" 


if __name__ == "__main__":
    print("Sentinel Inference Module Loaded")

here is the content of simulator/producer.py
"""
Transaction Data Simulator for Fraud Detection System
Reads from CSV and streams to Kafka continuously with mock data fallback.
Robust, Docker-friendly, and Schema-aligned.
"""
import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ============================================================================
# CONFIGURATION
# ============================================================================
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('INPUT_TOPIC', 'transactions')
CSV_FILE_PATH = os.getenv('DATA_FILE', '/app/data/raw/test_raw.csv')
ROWS_PER_SECOND = float(os.getenv('SIMULATION_RATE', 2.0))
MAX_KAFKA_RETRIES = int(os.getenv('MAX_KAFKA_RETRIES', 30))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Simulator")

# ============================================================================
# KAFKA CONNECTION
# ============================================================================
def get_kafka_producer(max_retries: int = 30) -> Optional[KafkaProducer]:
    """Establish connection to Kafka with exponential backoff retry logic."""
    retry_count = 0
    base_delay = 2
    
    while retry_count < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info(f"✅ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer

        except NoBrokersAvailable as e:
            retry_count += 1
            delay = min(base_delay * (2 ** retry_count), 30)
            logger.warning(f"⏳ Kafka not available. Retry {retry_count}/{max_retries} in {delay}s...")
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"❌ Kafka Error: {e}")
            time.sleep(2)
            retry_count += 1
    return None

# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_clean_csv(file_path: str) -> Optional[pd.DataFrame]:
    """Load and clean CSV data for JSON serialization."""
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ CSV file not found at {file_path}")
        return None
    try:
        logger.info(f"📖 Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        df = df.replace(['nan', 'NaN', 'Nan', np.nan], None)
        df = df.where(pd.notnull(df), None)
        
        if 'TransactionID' not in df.columns:
            if 'transaction_id' in df.columns:
                df['TransactionID'] = df['transaction_id']
            else:
                df['TransactionID'] = [f"tx_{int(time.time())}_{i}" for i in range(len(df))]
        logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.error(f"❌ CSV Load Error: {e}")
        return None

# ============================================================================
# MOCK DATA
# ============================================================================
def generate_mock_transaction() -> Dict[str, Any]:
    """Generates data compatible with Sentinel Dashboard visualization."""
    is_fraud = random.random() < 0.10 
    amt = round(random.uniform(10, 150), 2)
    if is_fraud:
        amt = round(random.uniform(300, 2000), 2)
    c_counts = random.randint(1, 4) if not is_fraud else random.randint(15, 80)
    velocity = random.randint(1, 5) if not is_fraud else random.randint(20, 60)
    risk_rate = random.uniform(0.0, 0.05) if not is_fraud else random.uniform(0.75, 0.99)
    device = random.choice(['iOS Device', 'Windows', 'MacOS', 'Samsung'])
    email_domain = random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'icloud.com'])
    if is_fraud:
        email_domain = random.choice(['tempmail.net', 'yopmail.com', 'protonmail.com'])
    
    return {
        'TransactionID': f"mock_{int(time.time()*1000)}",
        'timestamp': datetime.now().replace(microsecond=0).isoformat(),
        'isFraud': 1 if is_fraud else 0, 
        'TransactionAmt': amt,
        'ProductCD': random.choice(['W', 'H', 'C', 'R']), 
        'card1': random.randint(1000, 9999),
        'addr1': random.choice([325, 330, 440, 120]), 
        'P_emaildomain': email_domain,
        'dist1': random.randint(0, 20) if not is_fraud else random.randint(200, 3000),        
        'card_email_combo_fraud_rate': round(risk_rate, 4), 
        'R_emaildomain_is_free': random.choice([0, 1]),
        'UID_velocity_24h': velocity,
        'UID_vel': velocity, 
        'C1': c_counts,
        'C13': c_counts,
        'C14': c_counts,        
        'card1_freq_enc': random.randint(50, 5000), 
        'D15': random.randint(100, 800) if not is_fraud else random.randint(0, 5), 
        'device_vendor': device,
        'DeviceInfo': device 
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    logger.info("=" * 60)
    logger.info("🚀 SENTINEL SIMULATOR STARTING")
    logger.info(f"📡 Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"📄 Topic: {KAFKA_TOPIC}")
    logger.info(f"📂 CSV:   {CSV_FILE_PATH}")
    logger.info(f"⚡ Rate:  {ROWS_PER_SECOND} tx/sec")
    logger.info("=" * 60)
    
    producer = get_kafka_producer(MAX_KAFKA_RETRIES)
    if not producer:
        logger.critical("❌ Failed to connect to Kafka. Exiting.")
        sys.exit(1)
    
    try:
        while True:
            df = load_and_clean_csv(CSV_FILE_PATH)

            if df is not None and not df.empty:
                records = df.to_dict('records')
                logger.info(f"▶️ Streaming {len(records)} records from CSV...")
                
                for i, record in enumerate(records):
                    if ROWS_PER_SECOND > 0:
                        time.sleep(1.0 / ROWS_PER_SECOND)
                    
                    producer.send(KAFKA_TOPIC, value=record)
                    
                    if i % 10 == 0:
                        logger.info(f"📤 CSV Tx: {record.get('TransactionID')} | ${record.get('TransactionAmt')}")
                
                logger.info("🔄 CSV finished. Restarting in 5 seconds...")
                time.sleep(5)
                
            else:
                logger.warning("⚠️ CSV not found. Switching to MOCK DATA generator.")
                
                while True:
                    if ROWS_PER_SECOND > 0:
                        time.sleep(1.0 / ROWS_PER_SECOND)

                    record = generate_mock_transaction()
                    producer.send(KAFKA_TOPIC, value=record)
                    logger.info(f"📤 Mock Tx: {record['TransactionID']} | ${record['TransactionAmt']}")
                    
                    if os.path.exists(CSV_FILE_PATH) and random.random() < 0.05:
                        logger.info("📁 CSV file detected! Switching modes...")
                        break

    except KeyboardInterrupt:
        logger.info("\n🛑 Simulation stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal Error: {e}")
    finally:
        if producer:
            producer.close()
            logger.info("👋 Kafka producer closed")

if __name__ == "__main__":
    main()
here is the content of worker/processor.py 
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

here is the content of backend/main.py
"""
ROLE: The Gateway (API)
INTEGRATED: Resilience, Validation, and Monitoring
RESPONSIBILITIES:
1. Connects to Redis (Read-only for Dashboard).
2. Serves Statistics (/stats).
3. Serves Recent Transactions (/recent).
4. Serves Fraud Alerts (/alerts).
"""
import os
import json
import logging
import time
import asyncio
import psutil
from datetime import datetime
from typing import List, Optional
import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================================================================
# 1. RESPONSE SCHEMAS (Must Implement: Review Point 3)
# ==============================================================================
class Transaction(BaseModel):
    transaction_id: str
    timestamp: str
    amount: float
    is_fraud: int
    score: float
    action: str
    # Enriched fields from Worker
    ProductCD: Optional[str] = "U"
    device_vendor: Optional[str] = ""
    dist1: Optional[float] = 0

    class Config:
        extra = "allow"

class StatsResponse(BaseModel):
    total_processed: int
    fraud_detected: int
    legit_transactions: int
    fraud_rate: float
    queue_depth: int
    updated_at: str

# ==============================================================================
# 2. STRUCTURED LOGGING
# ==============================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": "sentinel-gateway"
        })

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("Gateway")

# ==============================================================================
# 3. CONFIGURATION & APP INIT
# ==============================================================================
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')

app = FastAPI(title="Sentinel Gateway", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Connection Pool for better performance (Review Point 4)
redis_pool = redis.ConnectionPool(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    password=REDIS_PASSWORD, 
    decode_responses=True,
    socket_timeout=2
)
redis_client = redis.Redis(connection_pool=redis_pool)

# ==============================================================================
# 4. LIFECYCLE & RESILIENCE (Must Implement: Review Point 1)
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    app.state.startup_time = time.time()
    retries = 5
    for i in range(retries):
        try:
            redis_client.ping()
            logger.info("✅ Gateway connected to Redis")
            return
        except Exception as e:
            logger.warning(f"⏳ Redis not ready (Attempt {i+1}/{retries}). Waiting...")
            await asyncio.sleep(2)
    
    logger.critical("❌ Could not connect to Redis. Gateway starting in degraded mode.")

# ==============================================================================
# 5. ENDPOINTS
# ==============================================================================
@app.get("/", tags=["System"])
def root():
    return {"service": "Sentinel Gateway", "status": "active"}

@app.get("/health", tags=["System"])
def health():
    try:
        redis_client.ping()
        return {"status": "healthy", "uptime": f"{int(time.time() - app.state.startup_time)}s"}
    except:
        raise HTTPException(status_code=503, detail="Redis unreachable")

@app.get("/metrics", tags=["System"])
def get_metrics():
    """System Monitoring (Review Point 6)"""
    return {
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
        "cpu_usage_percent": psutil.Process().cpu_percent(),
        "redis_connected": True if redis_client.ping() else False
    }

@app.get("/stats", response_model=StatsResponse, tags=["Dashboard"])
def get_stats():
    """Aggregated Dashboard Metrics"""
    try:
        fraud = int(redis_client.get('stats:fraud_count') or 0)
        legit = int(redis_client.get('stats:legit_count') or 0)
        total = fraud + legit
        
        return {
            "total_processed": total,
            "fraud_detected": fraud,
            "legit_transactions": legit,
            "fraud_rate": round((fraud / total * 100), 2) if total > 0 else 0,
            "queue_depth": redis_client.llen('sentinel_stream'),
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Stats Error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/recent", response_model=List[Transaction], tags=["Dashboard"])
def get_recent(limit: int = Query(15, ge=1, le=100)):
    """Live Transaction Feed (Review Point 2: Validation added)"""
    try:
        data = redis_client.lrange('sentinel_stream', 0, limit - 1)
        return [json.loads(item) for item in data]
    except Exception as e:
        logger.error(f"Recent Feed Error: {e}")
        return []

@app.get("/alerts", response_model=List[Transaction], tags=["Dashboard"])
def get_alerts(limit: int = Query(10, ge=1, le=50)):
    """High-Risk Alerts Only"""
    try:
        data = redis_client.lrange('sentinel_alerts', 0, limit - 1)
        return [json.loads(item) for item in data]
    except Exception as e:
        logger.error(f"Alerts Feed Error: {e}")
        return []

now I want to build the dashboard, I subdivide in different pages:
I have dashboard/styles.py
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# 1. COLOR PALETTE
# ==============================================================================
COLORS = {
    "background": "#0E1117",      # Main App Background
    "card_bg": "#181b21",         # Card Background
    "text": "#FAFAFA",
    "safe": "#00CC96",            # Green
    "danger": "#EF553B",          # Red
    "warning": "#FFA15A",         # Amber
    "neutral": "#8b92a1",         # Subtext Gray
    "border": "#2b3b4f",          # Card Border
    "highlight": "#00CC96"        # Title Color
}

# ==============================================================================
# 2. CSS INJECTION (Optimized)
# ==============================================================================
# In dashboard/styles.py, update the apply_custom_css function:
# def apply_custom_css():
#     st.markdown(f"""
#     <style>
#         /* MAIN BACKGROUND */
#         .stApp {{
#             background-color: {COLORS['background']};
#         }}
        
#         /* HIDE DEFAULT MENU */
#         #MainMenu {{ visibility: hidden; }}
#         footer {{ visibility: hidden; }}
#         header {{ visibility: hidden; }}

#         /* CARD STYLING (Glassmorphism Lite) */
#         .kpi-card {{
#             background-color: rgba(24, 27, 33, 0.7); /* Slight transparency */
#             border: 1px solid {COLORS['border']};
#             border-radius: 8px;
#             padding: 20px;
#             text-align: center;
#             backdrop-filter: blur(5px); /* The Blur Effect */
#             box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#             transition: transform 0.2s;
#         }}
#         .kpi-card:hover {{
#             transform: translateY(-5px);
#             border-color: {COLORS['highlight']};
#         }}
        
#         /* HEADERS */
#         h1, h2, h3 {{
#             font-family: 'Inter', sans-serif;
#             font-weight: 700;
#             color: {COLORS['text']};
#         }}
        
#         /* TABS (If you use them later) */
#         .stTabs [data-baseweb="tab-list"] {{
#             gap: 10px;
#         }}
#         .stTabs [data-baseweb="tab"] {{
#             background-color: {COLORS['card_bg']};
#             border-radius: 4px;
#             color: {COLORS['neutral']};
#         }}
#         .stTabs [data-baseweb="tab"][aria-selected="true"] {{
#             background-color: {COLORS['highlight']};
#             color: #000;
#         }}
#     </style>
#     """, unsafe_allow_html=True)


def apply_custom_css():
    """Injects global CSS styles into the Streamlit app."""
    st.markdown(f"""
    <style>
        /* ---------------------------------------------------------------------
           RESET & MOBILE
           --------------------------------------------------------------------- */
        .stDeployButton {{ display: none; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        
        @media (max-width: 768px) {{
            .kpi-card {{ min-height: 120px; padding: 10px; }}
            .kpi-value {{ font-size: 22px; }}
            .global-title {{ font-size: 2rem; }}
        }}

        /* ---------------------------------------------------------------------
           GLOBAL THEME
           --------------------------------------------------------------------- */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        .block-container {{
            padding-top: 2rem; 
            padding-bottom: 2rem;
        }}

        /* ---------------------------------------------------------------------
           TYPOGRAPHY
           --------------------------------------------------------------------- */
        .global-title {{
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 5px;
            color: {COLORS['highlight']};
            line-height: 1.2;
            text-shadow: 0px 0px 10px rgba(0, 204, 150, 0.3);
        }}
        
        .page-header {{
            text-align: left;
            margin-top: 10px; margin-bottom: 15px;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 5px;
        }}
        .page-header h2 {{
            font-size: 22px; font-weight: 700;
            color: {COLORS['text']}; margin: 0;
        }}
        
        /* ---------------------------------------------------------------------
           KPI CARDS
           --------------------------------------------------------------------- */
        .kpi-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 10px;
            height: 100%; 
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 140px; 
        }}
        .kpi-title {{ font-size: 14px; font-weight: 600; color: {COLORS['text']}; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; }}
        .kpi-subtext {{ font-size: 11px; color: {COLORS['neutral']}; font-style: italic; }}

        /* ---------------------------------------------------------------------
           STATUS INDICATORS & UTILS
           --------------------------------------------------------------------- */
        .status-indicator {{
            height: 12px; width: 12px; border-radius: 50%;
            display: inline-block; margin-right: 8px;
        }}
        .status-green {{ background-color: {COLORS['safe']}; box-shadow: 0 0 6px {COLORS['safe']}; }}
        .status-orange {{ background-color: {COLORS['warning']}; box-shadow: 0 0 6px {COLORS['warning']}; }}
        .status-red {{ background-color: {COLORS['danger']}; box-shadow: 0 0 6px {COLORS['danger']}; }}
        
        div.stButton > button {{
            width: 100%;
            background-color: {COLORS['card_bg']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            height: 45px; font-weight: 600;
        }}
        div.stButton > button:hover {{
            border-color: {COLORS['safe']}; color: {COLORS['safe']};
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. UI HELPERS
# ==============================================================================
def render_header(title, subtitle=""):
    """Renders a consistent section header."""
    sub_html = f"<div style='font-size:12px; color:{COLORS['neutral']}; margin-top:2px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <h2>{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def kpi_card(title, value, subtext, value_color=COLORS['safe']):
    """Returns HTML for a styled KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {value_color}">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """

def apply_plot_style(fig, title="", height=350):
    """Applies the Dashboard Dark Theme to any Plotly figure."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 14, 'color': COLORS['text'], 'family': "Helvetica Neue, sans-serif"}
        },
        height=height,
        font=dict(color=COLORS['neutral'], family="Helvetica Neue, sans-serif"),
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=COLORS['text'])
        )
    )
    # Subtle Grid
    grid_style = dict(showgrid=True, gridcolor="rgba(43, 59, 79, 0.5)", linecolor=COLORS['border'], zeroline=False)
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(**grid_style)
    return fig
 
dashboard/api_client.py
"""
ROLE: API Client (Adapter)
RESPONSIBILITIES:
1.  Abstracts HTTP requests to the Backend Service.
2.  Handles connection errors and timeouts gracefully.
3.  Converts raw JSON responses into Pandas DataFrames for Streamlit.
4.  Provides fallback data structures to prevent UI crashes.
"""
import os
import requests
import logging
import pandas as pd
from typing import Dict, Any, Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Default to 'backend' service name in Docker Compose network
API_BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000")
REQUEST_TIMEOUT = 3  # Seconds

logger = logging.getLogger("ApiClient")

class SentinelApiClient:
    """
    Client for interacting with the Sentinel Fraud Ops Backend.
    """

    def __init__(self):
        self.base_url = API_BASE_URL
        self.session = requests.Session()
        logger.info(f"🔌 API Client initialized pointing to: {self.base_url}")

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Internal helper to perform GET requests with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Connection Error: Could not reach {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"⏳ Timeout: Backend did not respond in {REQUEST_TIMEOUT}s")
            return None
        except Exception as e:
            logger.error(f"⚠️ API Error ({endpoint}): {e}")
            return None

    # ==========================================================================
    # SYSTEM HEALTH
    # ==========================================================================
    def is_backend_alive(self) -> bool:
        """Checks if the backend is reachable and Redis is connected."""
        data = self._get("/health")
        return data is not None and data.get("status") == "healthy"

    def get_system_metrics(self) -> Dict[str, Any]:
        """Fetches CPU and Memory usage of the backend service."""
        default = {"memory_usage_mb": 0, "cpu_usage_percent": 0, "redis_connected": False}
        data = self._get("/metrics")
        return data if data else default

    # ==========================================================================
    # DATA & STATISTICS
    # ==========================================================================
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Fetches aggregated statistics (Fraud Rate, Total Count, etc.).
        Returns zeroed-out dict on failure.
        """
        default = {
            "total_processed": 0,
            "fraud_detected": 0,
            "legit_transactions": 0,
            "fraud_rate": 0.0,
            "queue_depth": 0,
            "updated_at": "N/A"
        }
        data = self._get("/stats")
        return data if data else default

    def get_recent_transactions(self, limit: int = 20) -> pd.DataFrame:
        """
        Fetches the latest transactions stream.
        Returns: Pandas DataFrame suitable for Streamlit display.
        """
        data = self._get("/recent", params={"limit": limit})
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Data Formatting for UI
        if not df.empty:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            cols = ['timestamp', 'transaction_id', 'amount', 'score', 'is_fraud', 'action']
            existing_cols = [c for c in cols if c in df.columns]
            extra_cols = [c for c in df.columns if c not in cols]
            df = df[existing_cols + extra_cols]

        return df

    def get_fraud_alerts(self, limit: int = 10) -> pd.DataFrame:
        """
        Fetches only high-risk transactions (Fraud Alerts).
        Returns: Pandas DataFrame.
        """
        data = self._get("/alerts", params={"limit": limit})
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    
    dashboard/pages/executive.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from styles import COLORS, kpi_card, apply_plot_style, render_header

def render_page(df: pd.DataFrame, metrics: dict, threshold: float):
    render_header("Executive Overview", "Financial Impact & ROI Analysis")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # --- KPI ROW ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        st.markdown(kpi_card("Net Business Benefit", f"${metrics['net_benefit']/1000:,.1f}K", "Saved - Costs - Missed", COLORS['safe']), unsafe_allow_html=True)
    with c2: 
        st.markdown(kpi_card("Total Fraud Prevented", f"${metrics['fraud_prevented']/1000:,.1f}K", "Gross Value Protected", COLORS['safe']), unsafe_allow_html=True)
    with c3:
        ratio_color = COLORS['safe'] if metrics['fp_ratio'] < 3 else COLORS['danger']
        st.markdown(kpi_card("False Positive Ratio", f"1 : {metrics['fp_ratio']}", "Target < 1:3", ratio_color), unsafe_allow_html=True)
    with c4: 
        st.markdown(kpi_card("Global Recall", f"{metrics['recall']:.1%}", "Of known patterns", COLORS['warning']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- FINANCIAL CHART ---
    daily = df.copy().sort_values('timestamp')
    # Cumulative savings
    daily['cum_saved'] = daily[daily['ground_truth']==1]['TransactionAmt'].cumsum().ffill().fillna(0)
    
    # Realized Loss (Missed Fraud)
    daily['missed'] = (daily['ground_truth']==1) & (daily['composite_risk_score'] <= threshold)
    daily['cum_loss'] = daily[daily['missed']]['TransactionAmt'].cumsum().ffill().fillna(0)

    from plotly.subplots import make_subplots
    fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
    fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_saved'], name="Savings", fill='tozeroy', line=dict(color=COLORS['safe'])), secondary_y=False)
    fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_loss'], name="Loss", line=dict(color=COLORS['danger'], dash='dot')), secondary_y=True)

    fig_fin = apply_plot_style(fig_fin, title="Cumulative Value vs. Realized Loss")
    st.plotly_chart(fig_fin, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- COMPOSITION & SENSITIVITY ---
    c1, c2 = st.columns(2)
    with c1:
        # Fraud Composition
        fraud_only = df[df['ground_truth']==1]
        available_cols = [c for c in ['ProductCD', 'device_vendor', 'card_type'] if c in fraud_only.columns]
        
        if not fraud_only.empty and available_cols:
            # Use the first available column for simple sunburst or specific ones
            path = available_cols[:2] 
            fig_sun = px.sunburst(fraud_only, path=path, values='TransactionAmt', color_discrete_sequence=px.colors.sequential.RdBu)
            fig_sun = apply_plot_style(fig_sun, title=f"Fraud Composition ({' > '.join(path)})")
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.markdown(kpi_card("Fraud Composition", "No Data", "Waiting for fraud patterns", COLORS['neutral']), unsafe_allow_html=True)
            
    with c2:
        # Cost Curve Simulation
        x = np.linspace(0, 1, 100)
        # Fake cost curve equation for demo visualization
        y = 1000 * ((x - 0.6)**2 * 20 + 2) 
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=x, y=y, name="Cost Function", line=dict(color="white")))
        
        current_y = 1000 * ((threshold - 0.6)**2 * 20 + 2)
        fig_curve.add_trace(go.Scatter(x=[threshold], y=[current_y], mode='markers', 
                                     marker=dict(color=COLORS['warning'], size=15), name="Current Threshold"))
        
        fig_curve = apply_plot_style(fig_curve, title="Cost-Benefit Sensitivity Analysis")
        fig_curve.update_layout(xaxis_title="Threshold", yaxis_title="Est. Operational Cost ($)")
        st.plotly_chart(fig_curve, use_container_width=True)

dashboard/pages/forensics.py
import streamlit as st
import pandas as pd
import plotly.express as px
from styles import COLORS, apply_plot_style, render_header

def render_page(df: pd.DataFrame):
    render_header("Forensics & Search", "Deep Dive into Transaction Details")

    if df.empty:
        st.info("Waiting for data...")
        return

    # --- 1. SEARCH BAR ---
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            search_term = st.text_input("Search Transaction ID, Product, or Amount", placeholder="e.g., tx_1234 or 150.00")
        with c2:
            st.markdown("<br>", unsafe_allow_html=True) 
            filter_fraud_only = st.checkbox("Show Fraud Only", value=False)

    # --- 2. FILTER LOGIC ---
    filtered_df = df.copy()
    
    if filter_fraud_only:
        filtered_df = filtered_df[filtered_df['is_fraud'] == 1]

    if search_term:
        # Simple string matching across columns
        mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
        filtered_df = filtered_df[mask]

    # --- 3. RESULTS AREA ---
    st.markdown(f"**Found {len(filtered_df)} transactions**")

    if not filtered_df.empty:
        # Split screen: Table on Left, Details on Right
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.dataframe(
                filtered_df[['transaction_id', 'timestamp', 'amount', 'score', 'is_fraud', 'ProductCD']],
                use_container_width=True,
                height=500,
                hide_index=True
            )

        with c2:
            st.markdown("### 🔍 Risk Analysis")
            if len(filtered_df) == 1:
                # Detail View for Single Record
                record = filtered_df.iloc[0]
                
                score_color = COLORS['danger'] if record['score'] > 0.5 else COLORS['safe']
                st.markdown(f"""
                <div style="background-color: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                    <h1 style="color: {score_color}; margin:0;">{record['score']:.2f}</h1>
                    <div style="color: {COLORS['neutral']}; margin-bottom: 20px;">Risk Score</div>
                    
                    <p><b>ID:</b> {record['transaction_id']}</p>
                    <p><b>Amount:</b> ${record['amount']}</p>
                    <p><b>Product:</b> {record['ProductCD']}</p>
                    <p><b>Timestamp:</b> {record['timestamp']}</p>
                    <hr style="border-color: {COLORS['border']}">
                    <p><b>Action Taken:</b> {record.get('action', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Aggregate View for Multiple Records
                avg_score = filtered_df['score'].mean()
                total_amt = filtered_df['amount'].sum()
                
                fig = px.pie(filtered_df, names='is_fraud', title="Fraud vs Legit in Search",
                             color='is_fraud', color_discrete_map={0: COLORS['safe'], 1: COLORS['danger']})
                fig = apply_plot_style(fig, height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Total Amount in View", f"${total_amt:,.2f}")
                st.metric("Average Risk Score", f"{avg_score:.2f}")

    else:
        st.warning("No transactions match your search criteria.")

dashboard/pages/ml.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve
from styles import COLORS, kpi_card, apply_plot_style, render_header

def plot_pr_vs_threshold(df, current_threshold):
    """Calculates and plots Precision-Recall curve"""
    try:
        y_true = df['ground_truth']
        y_scores = df['composite_risk_score']
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    except Exception:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions[:-1], name="Precision", line=dict(color=COLORS['safe'], width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls[:-1], name="Recall", line=dict(color=COLORS['warning'], width=3)))
    fig.add_vline(x=current_threshold, line_width=2, line_dash="dash", line_color="white")

    fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1]), hovermode="x unified")
    return apply_plot_style(fig, title="Precision & Recall Trade-off")

def render_page(df, threshold):
    render_header("ML Integrity", "Model Drift & Performance Monitoring")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # --- DRIFT CALCULATION ---
    drift_score = 0.0
    min_samples = 50
    if len(df) > min_samples:
        try:
            # Simple Drift: KS Test between first half and second half of buffer
            mid = len(df) // 2
            ks_stat, _ = ks_2samp(df.iloc[:mid]['composite_risk_score'], df.iloc[mid:]['composite_risk_score'])
            drift_score = ks_stat
        except: pass

    # --- METRICS ROW ---
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(kpi_card("PR-AUC", "0.89", "Precision-Recall AUC", COLORS['safe']), unsafe_allow_html=True)
    with c2: 
        status = 'Stable' if drift_score < 0.1 else 'Drift Detected'
        color = COLORS['safe'] if drift_score < 0.1 else COLORS['warning']
        st.markdown(kpi_card("PSI (Data Drift)", f"{drift_score:.2f}", status, color), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Concept Drift", "0.02", "Label Consistency", COLORS['safe']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PLOTS ---
    c1, c2 = st.columns(2)
    with c1:
        # Score Separation (KDE approximation via Hist)
        fig_kde = go.Figure()
        fig_kde.add_trace(go.Histogram(x=df[df['ground_truth']==0]['composite_risk_score'], name='Legit', marker_color=COLORS['safe'], opacity=0.6))
        fig_kde.add_trace(go.Histogram(x=df[df['ground_truth']==1]['composite_risk_score'], name='Fraud', marker_color=COLORS['danger'], opacity=0.6))
        fig_kde.add_vline(x=threshold, line_width=3, line_color="white")
        fig_kde = apply_plot_style(fig_kde, title="Score Separation (Legit vs Fraud)")
        fig_kde.update_layout(barmode='overlay')
        st.plotly_chart(fig_kde, use_container_width=True)
    
    with c2:
        # Precision Recall Curve
        if df['ground_truth'].sum() > 0:
            fig_pr = plot_pr_vs_threshold(df, threshold)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            # Placeholder if no fraud in buffer
            st.info("Insufficient fraud labels in buffer to generate PR Curve.")

    # --- STABILITY ---
    if len(df) > min_samples:
         roll = df['composite_risk_score'].rolling(window=20).mean()
         fig_drift = go.Figure(go.Scatter(y=roll, name="Mean Score", line=dict(color=COLORS['neutral'])))
         fig_drift = apply_plot_style(fig_drift, title=f"Score Stability (Rolling Mean)")
         st.plotly_chart(fig_drift, use_container_width=True)

dashboard/pages/ops.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from styles import COLORS, kpi_card, apply_plot_style, render_header

def render_page(df, threshold):
    render_header("Real-Time Operations", "SOC Monitoring & Case Management")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # --- OPERATIONAL METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # Calculate Risk Index
        risk_mean = df.tail(50)['composite_risk_score'].mean() if 'composite_risk_score' in df.columns else 0
        risk_index = int(risk_mean * 100)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = risk_index,
            number = {'font': {'color': COLORS['text'], 'size': 24}, 'suffix': "%"}, 
            gauge = {
                'axis': {'range': [None, 100], 'visible': False}, 
                'bar': {'color': "rgba(0,0,0,0)"}, 
                'steps': [
                    {'range': [0, 40], 'color': COLORS['safe']},
                    {'range': [40, 75], 'color': COLORS['warning']},
                    {'range': [75, 100], 'color': COLORS['danger']}
                ],
                'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': risk_index}
            }
        ))
        fig_gauge = apply_plot_style(fig_gauge, title="Live Risk Index", height=155)
        fig_gauge.update_layout(margin=dict(l=25, r=25, t=35, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    
    # Latency simulation if column missing
    proc_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 45
    
    with c2: st.markdown(kpi_card("Mean Latency", f"{proc_time:.0f}ms", "SLA: < 100ms", COLORS['safe']), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Throughput", f"{len(df)}", "Transactions Buffered", COLORS['text']), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Analyst Overturn", "1.2%", "Label Correction Rate", COLORS['warning']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TRAFFIC & BOT HUNTER ---
    c1, c2 = st.columns([1.5, 1])
    with c1:
        stream_df = df.tail(100).copy()
        stream_df['legit_vol'] = 1 
        stream_df['blocked_vol'] = stream_df['composite_risk_score'].apply(lambda x: 1 if x > threshold else 0)
        
        fig_pulse = go.Figure()
        fig_pulse.add_trace(go.Scatter(x=stream_df['timestamp'], y=stream_df['legit_vol'], stackgroup='one', name='Legit', line=dict(color=COLORS['safe'])))
        fig_pulse.add_trace(go.Scatter(x=stream_df['timestamp'], y=stream_df['blocked_vol'], stackgroup='one', name='Blocked', line=dict(color=COLORS['danger'])))
        fig_pulse = apply_plot_style(fig_pulse, title="Traffic Pulse (Rolling Window)")
        st.plotly_chart(fig_pulse, use_container_width=True)
        
    with c2:
        # Scatter Plot: Velocity vs Amount
        plot_df = df.tail(200).copy()
        if 'UID_velocity_24h' not in plot_df.columns: plot_df['UID_velocity_24h'] = 0
            
        fig_scatter = px.scatter(
            plot_df, 
            x='UID_velocity_24h', 
            y='TransactionAmt', 
            color='composite_risk_score',
            color_continuous_scale='Reds',
            size='TransactionAmt',
            size_max=15,
            labels={'UID_velocity_24h': 'Velocity', 'TransactionAmt': 'Amt ($)'}
        )
        fig_scatter.add_vline(x=40, line_dash="dash", line_color=COLORS['warning'])
        fig_scatter.update_layout(coloraxis_colorbar=dict(title="Risk", orientation="v", title_side="right"))
        fig_scatter = apply_plot_style(fig_scatter, title="Bot Hunter (Vel vs Amt)")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # --- ALERT QUEUE ---
    st.markdown("<h3 style='text-align:left'>🔥 High-Priority Investigation Queue</h3>", unsafe_allow_html=True)
    queue = df[df['composite_risk_score'] > threshold].copy()
    queue = queue.sort_values('composite_risk_score', ascending=False).head(10)

    if not queue.empty:
        queue['time_formatted'] = queue['timestamp'].dt.strftime('%H:%M:%S')
        queue['Action'] = 'REVIEW'
        
        # Select relevant columns for the analyst
        cols = ['transaction_id', 'time_formatted', 'composite_risk_score', 'TransactionAmt', 'Action']
        display_queue = queue[[c for c in cols if c in queue.columns]]
        
        st.dataframe(display_queue.style.background_gradient(subset=['composite_risk_score'], cmap="Reds"), use_container_width=True)
    else:
        st.success("✅ Queue is empty. System healthy.")

dashboard/pages/strategy.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from styles import COLORS, apply_plot_style, render_header

def render_page(df: pd.DataFrame):
    render_header("Strategy & Product", "Risk Profiling & Link Analysis")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    c1, c2 = st.columns(2)

    # --- DEVICE RISK ---
    with c1:
        if 'device_vendor' in df.columns:
            # Simple aggregation
            dev_risk = df.groupby('device_vendor')['ground_truth'].mean().reset_index()
            # Filter for visualisation
            dev_risk = dev_risk.sort_values('ground_truth', ascending=True).tail(10)
            
            fig_dev = px.bar(dev_risk, y='device_vendor', x='ground_truth', orientation='h', 
                             color='ground_truth', color_continuous_scale='Reds', 
                             labels={'ground_truth': 'Fraud Rate'})
            fig_dev.update_layout(coloraxis_colorbar=dict(title="Rate", orientation="v", title_side="right"))
            fig_dev = apply_plot_style(fig_dev, title="Risk by Device Vendor")
            st.plotly_chart(fig_dev, use_container_width=True)
        else:
            st.warning("Device data not available in stream.")

    # --- EMAIL DOMAIN RISK (Mock if column missing) ---
    with c2:
        # Mocking this specific chart if data is missing, as per original dashboard intent
        email_data = pd.DataFrame({
            'Domain': ['Proton', 'TempMail', 'Gmail', 'Yahoo', 'Corporate'], 
            'Risk_Rate': [0.85, 0.95, 0.02, 0.03, 0.01]
        }).sort_values('Risk_Rate')
        
        fig_email = px.bar(email_data, y='Domain', x='Risk_Rate', orientation='h', 
                           color='Risk_Rate', color_continuous_scale='Reds')
        fig_email = apply_plot_style(fig_email, title="Risk by Email Domain (Global Stats)")
        st.plotly_chart(fig_email, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- NETWORK GRAPH (FRAUD RINGS) ---
    try:
        # Simulated Network Graph
        G = nx.Graph()
        center = "Bad_Actor_X"
        G.add_node(center, type='User')
        for i in range(5):
            ip = f"Device_{i}" 
            G.add_node(ip, type='Device')
            G.add_edge(center, ip)
            linked = f"User_{i}" 
            G.add_node(linked, type='User')
            G.add_edge(ip, linked)
            
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        node_x, node_y, node_color, node_text = [], [], [], []
        
        for node in G.nodes():
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            node_text.append(node)
            if node == center: node_color.append(COLORS['danger'])
            elif "Device" in node: node_color.append(COLORS['warning'])
            else: node_color.append(COLORS['neutral'])
            
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, 
                                marker=dict(showscale=False, color=node_color, size=20))
                                
        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net = apply_plot_style(fig_net, title="Fraud Ring Analysis (Linked via C13/Device)")
        fig_net.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig_net, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render Network Graph: {e}")

and dashboard/app.py
"""
Sentinel Fraud Ops - Main Dashboard Controller
"""
import sys
import os
import time
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from styles import apply_custom_css, COLORS
from api_client import SentinelApiClient
from pages import executive, ops, ml, strategy, forensics

# ==============================================================================
# 1. SETUP
# ==============================================================================
st.set_page_config(
    page_title="Sentinel Ops",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

if 'api_client' not in st.session_state:
    st.session_state.api_client = SentinelApiClient()
    # Initialize Pause State
    st.session_state.is_paused = False

client = st.session_state.api_client

# ==============================================================================
# 2. SIDEBAR (NAVIGATION & CONTROLS)
# ==============================================================================
with st.sidebar:
    st.markdown(f"<h2 style='text-align: center; color: {COLORS['highlight']}; letter-spacing: 2px;'>SENTINEL</h2>", unsafe_allow_html=True)
    st.caption("Enterprise Fraud Detection System")
    st.markdown("---")
    
    # 2.1 Navigation
    page = st.radio(
        "MODULES", 
        ["Executive View", "Ops Center", "ML Monitor", "Strategy", "Forensics"],
        index=0
    )
    
    st.markdown("---")
    
    # 2.2 Live Control (The "Lightness" Fix)
    st.markdown("**LIVE FEED CONTROL**")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏸ PAUSE" if not st.session_state.is_paused else "▶ RESUME"):
            st.session_state.is_paused = not st.session_state.is_paused
            st.rerun()
            
    with c2:
        if st.button("🔄 REFRESH"):
            st.session_state.is_paused = False # Unpause on manual refresh
            st.rerun()

    if st.session_state.is_paused:
        st.warning("⚠️ Live Feed Paused")
    else:
        st.success("🟢 Live Feed Active")

    # 2.3 System Health
    st.markdown("---")
    alive = client.is_backend_alive()
    st.caption(f"System Status: {'Online ✅' if alive else 'Offline ❌'}")

# ==============================================================================
# 3. DATA LOADING
# ==============================================================================
def load_data():
    if st.session_state.is_paused and 'last_df' in st.session_state:
        # Return cached data if paused
        return st.session_state.last_stats, st.session_state.last_df

    stats = client.get_dashboard_stats()
    df = client.get_recent_transactions(limit=500)
    
    # Cache for pause functionality
    st.session_state.last_stats = stats
    st.session_state.last_df = df
    
    return stats, df

# ==============================================================================
# 4. MAIN RENDER
# ==============================================================================
def main():
    stats, df = load_data()
    
    # Routing
    if page == "Executive View":
        executive.render_page(stats, df)
    elif page == "Ops Center":
        ops.render_page(stats, df)
    elif page == "ML Monitor":
        ml.render_page(stats, df)
    elif page == "Strategy":
        strategy.render_page(df)
    elif page == "Forensics":
        forensics.render_page(df)

    # Auto-Refresh Logic (Only if not paused)
    if not st.session_state.is_paused:
        time.sleep(5) # Slower refresh for smoother UX
        st.rerun()

if __name__ == "__main__":
    main() 




styles.py
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# 1. COLOR PALETTE
# ==============================================================================
COLORS = {
    "background": "#0E1117",      # Main App Background
    "card_bg": "#181b21",         # Card Background
    "text": "#FAFAFA",
    "safe": "#00CC96",            # Green
    "danger": "#EF553B",          # Red
    "warning": "#FFA15A",         # Amber
    "neutral": "#8b92a1",         # Subtext Gray
    "border": "#2b3b4f",          # Card Border
    "highlight": "#00CC96"        # Title Color
}

# ==============================================================================
# 2. CSS INJECTION (Optimized)
# ==============================================================================
# In dashboard/styles.py, update the apply_custom_css function:
# def apply_custom_css():
#     st.markdown(f"""
#     <style>
#         /* MAIN BACKGROUND */
#         .stApp {{
#             background-color: {COLORS['background']};
#         }}
        
#         /* HIDE DEFAULT MENU */
#         #MainMenu {{ visibility: hidden; }}
#         footer {{ visibility: hidden; }}
#         header {{ visibility: hidden; }}

#         /* CARD STYLING (Glassmorphism Lite) */
#         .kpi-card {{
#             background-color: rgba(24, 27, 33, 0.7); /* Slight transparency */
#             border: 1px solid {COLORS['border']};
#             border-radius: 8px;
#             padding: 20px;
#             text-align: center;
#             backdrop-filter: blur(5px); /* The Blur Effect */
#             box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#             transition: transform 0.2s;
#         }}
#         .kpi-card:hover {{
#             transform: translateY(-5px);
#             border-color: {COLORS['highlight']};
#         }}
        
#         /* HEADERS */
#         h1, h2, h3 {{
#             font-family: 'Inter', sans-serif;
#             font-weight: 700;
#             color: {COLORS['text']};
#         }}
        
#         /* TABS (If you use them later) */
#         .stTabs [data-baseweb="tab-list"] {{
#             gap: 10px;
#         }}
#         .stTabs [data-baseweb="tab"] {{
#             background-color: {COLORS['card_bg']};
#             border-radius: 4px;
#             color: {COLORS['neutral']};
#         }}
#         .stTabs [data-baseweb="tab"][aria-selected="true"] {{
#             background-color: {COLORS['highlight']};
#             color: #000;
#         }}
#     </style>
#     """, unsafe_allow_html=True)


def apply_custom_css():
    """Injects global CSS styles into the Streamlit app."""
    st.markdown(f"""
    <style>
        /* ---------------------------------------------------------------------
           RESET & MOBILE
           --------------------------------------------------------------------- */
        .stDeployButton {{ display: none; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        
        @media (max-width: 768px) {{
            .kpi-card {{ min-height: 120px; padding: 10px; }}
            .kpi-value {{ font-size: 22px; }}
            .global-title {{ font-size: 2rem; }}
        }}

        /* ---------------------------------------------------------------------
           GLOBAL THEME
           --------------------------------------------------------------------- */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        .block-container {{
            padding-top: 2rem; 
            padding-bottom: 2rem;
        }}

        /* ---------------------------------------------------------------------
           TYPOGRAPHY
           --------------------------------------------------------------------- */
        .global-title {{
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 5px;
            color: {COLORS['highlight']};
            line-height: 1.2;
            text-shadow: 0px 0px 10px rgba(0, 204, 150, 0.3);
        }}
        
        .page-header {{
            text-align: left;
            margin-top: 10px; margin-bottom: 15px;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 5px;
        }}
        .page-header h2 {{
            font-size: 22px; font-weight: 700;
            color: {COLORS['text']}; margin: 0;
        }}
        
        /* ---------------------------------------------------------------------
           KPI CARDS
           --------------------------------------------------------------------- */
        .kpi-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 10px;
            height: 100%; 
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 140px; 
        }}
        .kpi-title {{ font-size: 14px; font-weight: 600; color: {COLORS['text']}; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; }}
        .kpi-subtext {{ font-size: 11px; color: {COLORS['neutral']}; font-style: italic; }}

        /* ---------------------------------------------------------------------
           STATUS INDICATORS & UTILS
           --------------------------------------------------------------------- */
        .status-indicator {{
            height: 12px; width: 12px; border-radius: 50%;
            display: inline-block; margin-right: 8px;
        }}
        .status-green {{ background-color: {COLORS['safe']}; box-shadow: 0 0 6px {COLORS['safe']}; }}
        .status-orange {{ background-color: {COLORS['warning']}; box-shadow: 0 0 6px {COLORS['warning']}; }}
        .status-red {{ background-color: {COLORS['danger']}; box-shadow: 0 0 6px {COLORS['danger']}; }}
        
        div.stButton > button {{
            width: 100%;
            background-color: {COLORS['card_bg']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            height: 45px; font-weight: 600;
        }}
        div.stButton > button:hover {{
            border-color: {COLORS['safe']}; color: {COLORS['safe']};
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. UI HELPERS
# ==============================================================================
def render_header(title, subtitle=""):
    """Renders a consistent section header."""
    sub_html = f"<div style='font-size:12px; color:{COLORS['neutral']}; margin-top:2px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <h2>{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def kpi_card(title, value, subtext, value_color=COLORS['safe']):
    """Returns HTML for a styled KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {value_color}">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """

def apply_plot_style(fig, title="", height=350):
    """Applies the Dashboard Dark Theme to any Plotly figure."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 14, 'color': COLORS['text'], 'family': "Helvetica Neue, sans-serif"}
        },
        height=height,
        font=dict(color=COLORS['neutral'], family="Helvetica Neue, sans-serif"),
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=COLORS['text'])
        )
    )
    # Subtle Grid
    grid_style = dict(showgrid=True, gridcolor="rgba(43, 59, 79, 0.5)", linecolor=COLORS['border'], zeroline=False)
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(**grid_style)
    return fig