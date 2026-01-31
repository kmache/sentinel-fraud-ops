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

