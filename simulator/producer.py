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

# Environment variables with defaults
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('INPUT_TOPIC', 'transactions') # Matched to Processor Env Var
CSV_FILE_PATH = os.getenv('DATA_FILE', '/app/data/test_raw.csv') 
ROWS_PER_SECOND = float(os.getenv('SIMULATION_RATE', 2.0))
MAX_KAFKA_RETRIES = int(os.getenv('MAX_KAFKA_RETRIES', 30))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# KAFKA CONNECTION
# ============================================================================

def get_kafka_producer(max_retries: int = 30) -> Optional[KafkaProducer]:
    """Establish connection to Kafka with exponential backoff retry logic."""
    retry_count = 0
    base_delay = 2  # seconds
    
    while retry_count < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            
            # Test connection
            logger.info(f"✅ Successfully connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer
            
        except NoBrokersAvailable as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"❌ Failed to connect to Kafka after {max_retries} attempts: {e}")
                return None
            
            # Exponential backoff
            delay = min(base_delay * (2 ** retry_count), 30)
            logger.warning(f"⏳ Kafka not available. Retry {retry_count}/{max_retries} in {delay:.1f}s...")
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"❌ Unexpected Kafka connection error: {e}")
            time.sleep(2)
            retry_count += 1
    
    return None

# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================

def load_and_clean_csv(file_path: str) -> Optional[pd.DataFrame]:
    """Load and clean CSV data for JSON serialization."""
    if not os.path.exists(file_path):
        logger.error(f"❌ CSV file not found: {file_path}")
        return None
    
    try:
        logger.info(f"📖 Loading CSV file: {file_path}")
        
        # Try loading
        df = pd.read_csv(file_path)
        
        # CRITICAL: Kafka/JSON cannot handle NaN/Infinity. 
        # We replace NaNs with None (which becomes null in JSON)
        df = df.replace(['nan', 'NaN', 'Nan', np.nan], None)
        df = df.where(pd.notnull(df), None)
        
        # Ensure TransactionID exists
        if 'TransactionID' not in df.columns:
            if 'transaction_id' in df.columns:
                df['TransactionID'] = df['transaction_id']
            else:
                print("No TransactionID or transaction_id column found")
                df['TransactionID'] = [f"tx_{i}" for i in range(len(df))]

        logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading CSV: {e}")
        return None

# ============================================================================
# MOCK DATA GENERATOR (IEEE-CIS Schema Compatible)
# ============================================================================

def generate_mock_transaction() -> Dict[str, Any]:
    """
    Generate mock data that matches the IEEE-CIS Fraud Detection schema.
    Aligned with Dashboard requirements (dist1, C13, DeviceInfo).
    """
    is_fraud = random.random() < 0.05
    
    # Simulate fraud patterns (High amount, high velocity logic simulated downstream)
    amt = round(random.uniform(10, 1000), 2)
    if is_fraud:
        amt = round(random.uniform(200, 2000), 2)

    return {
        'TransactionID': f"mock_{int(time.time()*1000)}",
        'isFraud': 1 if is_fraud else 0, # Ground truth for testing
        'TransactionDT': int(time.time()),
        'TransactionAmt': amt,
        'ProductCD': random.choice(['C', 'W', 'H', 'R', 'S']),
        'card1': random.randint(1000, 9999),
        'card2': random.randint(100, 600),
        'card3': 150.0,
        'card4': random.choice(['mastercard', 'visa', 'american express']),
        'card5': random.randint(100, 200),
        'card6': random.choice(['credit', 'debit']),
        'addr1': random.choice([300, 315, 325, 400]),
        'addr2': 87.0,
        'dist1': random.randint(0, 500) if not is_fraud else random.randint(400, 2000), # Dashboard Feature
        'P_emaildomain': random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'protonmail.com']),
        'R_emaildomain': random.choice(['gmail.com', 'yahoo.com', None]),
        'C1': random.randint(1, 10),
        'C13': random.randint(1, 5) if not is_fraud else random.randint(10, 50), # Dashboard Feature (Graph)
        'D1': random.randint(0, 500), # Dashboard Feature (Account Age)
        'D2': random.randint(0, 500),
        'D4': random.randint(0, 500),
        'D10': random.randint(0, 500),
        'D15': random.randint(0, 500),
        'DeviceInfo': random.choice(['SM-G9600', 'iOS Device', 'Windows', 'MacOS', 'Trident/7.0']), # Dashboard Feature
        'DeviceType': random.choice(['mobile', 'desktop']),
        'id_31': random.choice(['chrome 63.0', 'mobile safari 11.0', 'safari generic']),
        'timestamp': datetime.now().replace(microsecond=0).isoformat()
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("🚀 TRANSACTION DATA SIMULATOR")
    logger.info("=" * 60)
    logger.info(f"📡 Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"📄 Topic: {KAFKA_TOPIC}")
    logger.info(f"📂 CSV File: {CSV_FILE_PATH}")
    logger.info(f"⚡ Rate: {ROWS_PER_SECOND} transactions/sec")
    logger.info("=" * 60)
    
    # Step 1: Connect to Kafka
    producer = get_kafka_producer(MAX_KAFKA_RETRIES)
    if not producer:
        logger.error("❌ Failed to connect to Kafka. Exiting.")
        sys.exit(1)
    
    # Main simulation loop
    try:
        while True:
            # Step 2: Try to load CSV data
            df = load_and_clean_csv(CSV_FILE_PATH)
            
            if df is not None and not df.empty:
                # --- MODE A: STREAM CSV ---
                records = df.to_dict('records')
                logger.info(f"▶️ Streaming {len(records)} records from CSV...")
                
                for i, record in enumerate(records):
                    # Rate Limiting
                    if ROWS_PER_SECOND > 0:
                        time.sleep(1.0 / ROWS_PER_SECOND)
                    
                    # Send
                    producer.send(KAFKA_TOPIC, value=record)
                    
                    # Log occasionally
                    if i % 10 == 0:
                        tx_id = record.get('TransactionID', 'unknown')
                        amt = record.get('TransactionAmt', 0)
                        logger.info(f"📤 Sent CSV: {tx_id} | ${amt}")
                
                logger.info("🔄 CSV finished. Restarting in 5 seconds...")
                time.sleep(5)
                
            else:
                # --- MODE B: MOCK DATA FALLBACK ---
                logger.warning("⚠️ CSV unavailable. Streaming MOCK DATA.")
                while True:
                    if ROWS_PER_SECOND > 0:
                        time.sleep(1.0 / ROWS_PER_SECOND)
                        
                    record = generate_mock_transaction()
                    producer.send(KAFKA_TOPIC, value=record)
                    logger.info(f"📤 Sent MOCK: {record['TransactionID']} | ${record['TransactionAmt']}")
                    
                    # If file appears, break mock loop and reload
                    if os.path.exists(CSV_FILE_PATH) and random.random() < 0.05:
                        break

    except KeyboardInterrupt:
        logger.info("\n🛑 Simulation stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error in main loop: {e}")
    finally:
        if producer:
            producer.close()
            logger.info("👋 Kafka producer closed")

if __name__ == "__main__":
    main()



