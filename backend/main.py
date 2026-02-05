"""
ROLE: The Gateway (API)
INTEGRATED: Thin Client Architecture (Reads Pre-computed Data)
RESPONSIBILITIES:
1. Connects to Redis.
2. Serves Real-time Feed (/recent, /alerts).
3. Serves Pre-computed Metrics (/performance, /stats).
4. Serves Forensics Data (/transactions/{id}).
"""
import os
import sys
import json
import logging
import time
import psutil
import redis
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Add src to path to import the Sentinel Evaluator 
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas import Transaction, StatsResponse 

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')

# ==============================================================================
# 3. STRUCTURED LOGGING
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
# 4. REDIS & UTILS
# ==============================================================================
redis_pool = redis.ConnectionPool(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    password=REDIS_PASSWORD, 
    decode_responses=True,
    socket_timeout=2
)
redis_client = redis.Redis(connection_pool=redis_pool)

# ==============================================================================
# 5. LIFECYCLE MANAGEMENT
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----Startup----------
    app.state.startup_time = time.time()
    logger.info("🚀 Gateway starting...")

    retries = 5
    redis_is_ready = False

    for i in range(retries):
        try:
            if redis_client.ping():
                logger.info("✅ Gateway connected to Redis")
                redis_is_ready = True
                break
        except Exception:
            logger.warning(f"⏳ Redis not ready (Attempt {i+1}/{retries}). Waiting...")
            await asyncio.sleep(2)
    
    if not redis_is_ready:
        logger.critical("❌ Could not connect to Redis. Gateway starting in degraded mode.")

    yield
    
    # ----Shutdown----------
    logger.info("🛑 Gateway shutting down...")
    try:
        redis_client.close()
    except Exception as e:
        logger.error(f"⚠️ Error closing Redis: {e}")

# ==============================================================================
# 6. APP & ENDPOINTS
# ==============================================================================
app = FastAPI(title="Sentinel Gateway", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

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
    
    try:
        redis_status = True if redis_client.ping() else False
    except:
        redis_status = False
    return {
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
        "cpu_usage_percent": psutil.Process().cpu_percent(),
        "redis_connected": redis_status,
    }

@app.get("/stats", response_model=StatsResponse, tags=["Dashboard"])
def get_stat_performance_report():
    try: 
        raw_data = redis_client.get('stats:stat_bussiness_report')
        if not raw_data:
            raise HTTPException(status_code=404, detail="Stats not yet computed")
        
        # CRITICAL FIX: Load the JSON string into a dict
        data = json.loads(raw_data)
        
        # Mapping with safe .get() defaults
        counts = data.get('counts', {})
        finance = data.get('financials', {})
        perf = data.get('performance', {})

        return {
            "precision": perf.get('precision', 0),
            "recall": perf.get('recall', 0),
            "fpr_insult_rate": perf.get('fpr_insult_rate', 0),
            "auc": perf.get('auc', 0),
            "fraud_rate": perf.get('fraud_rate', 0),
            "fraud_stopped_val": finance.get('fraud_stopped_val', 0),
            "fraud_missed_val": finance.get('fraud_missed_val', 0),
            "false_positive_loss": finance.get('false_positive_loss', 0),
            "net_savings": finance.get('net_savings', 0),
            "total_eval": counts.get('total_eval', 0),
            "treshold": data.get('meta', {}).get('threshold', 0.5)
        }
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Stats Error: {e}")
        raise HTTPException(status_code=500, detail="Internal data error")


@app.get("/exec/series", tags=["Dashboard"])
def get_financial_timeseries():
    try:
        raw_data = redis_client.lrange('sentinel_timeseries', 0, -1)
        if not raw_data:
            return []

        # Downsampling logic
        TARGET = 1000
        total_points = len(raw_data)
        step = max(1, total_points // TARGET) 
        sampled_raw = raw_data[::step]
        
        series = [json.loads(x) for x in sampled_raw]
        
        # Ensure the most recent point is always included
        last_point = json.loads(raw_data[-1])
        if series and series[-1].get('timestamp') != last_point.get('timestamp'):
            series.append(last_point)
            
        return series
    except Exception as e:
        logger.error(f"Timeseries error: {e}")
        return []       

@app.get("/recent", response_model=List[Transaction], tags=["Dashboard"])
def get_recent_stream(limit: int = Query(100, ge=1, le=1000)):
    try:
        raw_list = redis_client.lrange('sentinel_stream', 0, limit - 1)
        return [json.loads(x) for x in raw_list]
    except Exception as e:
        logger.error(f"Stream error: {e}")
        return [] 

@app.get("/alerts", response_model=List[Transaction], tags=["Dashboard"])
def get_alerts(limit: int = Query(50, ge=1, le=200)):
    try:
        data = redis_client.lrange('sentinel_alerts', 0, limit - 1)
        return [json.loads(item) for item in data]
    except Exception as e:
        logger.error(f"Alerts Feed Error: {e}")
        return []

@app.get("/transactions/{transaction_id}", tags=["Forensics"])
def get_transaction_detail(transaction_id: str):
    """
    Fetches the individual 'Case File'. 
    Used when an analyst clicks a row to see SHAP explanations.
    """
    try:
        key = f"prediction:{transaction_id}"
        data_json = redis_client.get(key)
        
        if not data_json:
            raise HTTPException(status_code=404, detail="Transaction Expired or Not Found")

        return json.loads(data_json)

    except HTTPException:  
        raise
    except Exception as e:
        logger.error(f"Forensics Detail Error for {transaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Lookup Error")
