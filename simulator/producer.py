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

