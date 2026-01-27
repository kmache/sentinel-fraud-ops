import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis

# IMPORT YOUR LIBRARY
from sentinel.inference import SentinelInference

# ============================================================================
# CONFIGURATION
# ============================================================================
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/prod_v1')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentinelBackend")

# ============================================================================
# INITIALIZATION
# ============================================================================
app = FastAPI(title="Sentinel Fraud API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
redis_pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
sentinel_engine = None

@app.on_event("startup")
async def startup_event():
    global sentinel_engine
    logger.info(f"🚀 Sentinel Backend Starting... Env: {ENVIRONMENT}")
    
    # Load Model Logic via Library
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"🤖 Loading Model Bundle: {MODEL_PATH}")
            sentinel_engine = SentinelInference(model_dir=MODEL_PATH)
        else:
            logger.warning(f"⚠️ Model path {MODEL_PATH} not found. /predict will fail.")
    except Exception as e:
        logger.error(f"❌ Failed to load Sentinel Engine: {e}")

    app.state.startup_time = time.time()

def get_redis():
    try:
        return redis.Redis(connection_pool=redis_pool)
    except Exception:
        return None

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "Sentinel Fraud API", 
        "status": "running", 
        "model_loaded": sentinel_engine is not None
    }

@app.get("/health")
def health_check():
    r = get_redis()
    redis_status = "connected" if r and r.ping() else "disconnected"
    return {
        "status": "healthy",
        "redis": redis_status,
        "uptime": round(time.time() - app.state.startup_time, 2)
    }

@app.post("/predict")
def predict_transaction(txn: Dict[str, Any]):
    """REST Endpoint for Synchronous Prediction"""
    if not sentinel_engine:
        raise HTTPException(503, "Model not loaded")
    
    try:
        # The library handles everything
        result = sentinel_engine.predict(txn)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/stats")
def get_stats():
    """Fetches stats for the Dashboard"""
    r = get_redis()
    if not r: raise HTTPException(503, "Redis unavailable")
    
    try:
        return {
            "summary": {
                "total_processed": int(r.get('total_processed') or 0),
                "fraud_detected": int(r.get('stats:fraud_count') or 0),
                "legit_transactions": int(r.get('stats:legit_count') or 0),
                "dashboard_queue_depth": r.llen('sentinel_stream')
            }
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {}

@app.get("/recent")
def get_recent(limit: int = 10):
    r = get_redis()
    if not r: raise HTTPException(503, "Redis unavailable")
    
    items = r.lrange('sentinel_stream', 0, limit - 1)
    parsed = [json.loads(i) for i in items if i]
    return {"transactions": parsed}