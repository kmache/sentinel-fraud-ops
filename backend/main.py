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
import secrets
import psutil
import redis
import asyncio
from datetime import datetime
from typing import Any, List, cast
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.status import HTTP_403_FORBIDDEN
from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore[import-untyped]
from slowapi.util import get_remote_address  # type: ignore[import-untyped]
from slowapi.errors import RateLimitExceeded  # type: ignore[import-untyped]

# Add src to path to import the Sentinel Evaluator 
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from schemas import Transaction, StatsResponse

try:
    from config.config import get_api_config  # type: ignore[import-not-found]
    _api_cfg = get_api_config()
except Exception:
    _api_cfg = {}

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
API_KEY = os.getenv('SENTINEL_API_KEY', '')
ALLOWED_ORIGINS = os.getenv(
    'ALLOWED_ORIGINS', 
    'http://localhost:8501,http://dashboard:8501'
).split(',')

# ==============================================================================
# 2. API KEY AUTHENTICATION
# ==============================================================================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate API key if SENTINEL_API_KEY is configured."""
    if not API_KEY:
        return None  # Auth disabled when no key is set
    if not api_key or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API key")
    return api_key

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
    socket_timeout=2,
    retry_on_timeout=True
)
redis_client: Any = redis.Redis(connection_pool=redis_pool)


def redis_safe(fn: Any, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Execute a Redis call with error handling. Returns default on failure."""
    try:
        return fn(*args, **kwargs)
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(f"Redis call failed: {e}")
        return default

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
    
    logger.info("🛑 Gateway shutting down...")
    try:
        redis_client.close()
    except Exception as e:
        logger.error(f"⚠️ Error closing Redis: {e}")

# ==============================================================================
# 6. APP & ENDPOINTS
# ==============================================================================
app = FastAPI(title="Sentinel Gateway", version="2.5.0", lifespan=lifespan)

app.add_middleware(GZipMiddleware, minimum_size=1000)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_methods=["GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)

@app.get("/", tags=["System"])
def root():
    return {"service": "Sentinel Gateway", "status": "active"}

@app.get("/health", tags=["System"])
def health():
    try:
        redis_client.ping()
        return {"status": "healthy", "uptime": f"{int(time.time() - app.state.startup_time)}s"}
    except Exception:
        raise HTTPException(status_code=503, detail="Redis unreachable")

@app.get("/metrics", tags=["System"])
@limiter.limit("30/minute")
def get_metrics(request: Request, _key: str = Depends(verify_api_key)):
    
    try:
        redis_status = True if redis_client.ping() else False
    except Exception:
        redis_status = False

    uptime_s = int(time.time() - app.state.startup_time)
    proc = psutil.Process()

    return {
        "memory_usage_mb": round(proc.memory_info().rss / 1024 / 1024, 2),
        "cpu_usage_percent": proc.cpu_percent(),
        "redis_connected": redis_status,
        "uptime_seconds": uptime_s,
        "open_connections": len(proc.net_connections(kind="tcp")),
    }

@app.get("/metrics/prometheus", tags=["System"])
@limiter.limit("30/minute")
def get_metrics_prometheus(request: Request, _key: str = Depends(verify_api_key)):
    """Prometheus-compatible text exposition endpoint."""
    try:
        redis_status = 1 if redis_client.ping() else 0
    except Exception:
        redis_status = 0

    uptime_s = int(time.time() - app.state.startup_time)
    proc = psutil.Process()
    mem_bytes = proc.memory_info().rss
    cpu_pct = proc.cpu_percent()
    conns = len(proc.net_connections(kind="tcp"))

    total_processed = int(redis_safe(redis_client.get, "count:total_processed", default=0) or 0)
    total_frauds = int(redis_safe(redis_client.get, "count:total_flagged", default=0) or 0)

    lines = [
        "# HELP sentinel_up Whether the gateway is up (1=up).",
        "# TYPE sentinel_up gauge",
        "sentinel_up 1",
        "# HELP sentinel_redis_connected Whether Redis is reachable.",
        "# TYPE sentinel_redis_connected gauge",
        f"sentinel_redis_connected {redis_status}",
        "# HELP sentinel_uptime_seconds Gateway uptime in seconds.",
        "# TYPE sentinel_uptime_seconds gauge",
        f"sentinel_uptime_seconds {uptime_s}",
        "# HELP sentinel_memory_bytes Resident memory in bytes.",
        "# TYPE sentinel_memory_bytes gauge",
        f"sentinel_memory_bytes {mem_bytes}",
        "# HELP sentinel_cpu_percent CPU usage percent.",
        "# TYPE sentinel_cpu_percent gauge",
        f"sentinel_cpu_percent {cpu_pct}",
        "# HELP sentinel_open_connections Number of open TCP connections.",
        "# TYPE sentinel_open_connections gauge",
        f"sentinel_open_connections {conns}",
        "# HELP sentinel_transactions_total Total transactions processed.",
        "# TYPE sentinel_transactions_total counter",
        f"sentinel_transactions_total {total_processed}",
        "# HELP sentinel_frauds_total Total fraud alerts.",
        "# TYPE sentinel_frauds_total counter",
        f"sentinel_frauds_total {total_frauds}",
    ]
    from starlette.responses import Response
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

@app.get("/stats", response_model=StatsResponse, tags=["Dashboard"])
def get_stat_performance_report(_key: str = Depends(verify_api_key)):
    try: 
        raw_data = redis_client.get('stats:stat_business_report')
        if not raw_data:
            raise HTTPException(status_code=404, detail="Stats not available")
        
        data = json.loads(raw_data)
        
        perf = data.get('performance', {})
        finance = data.get('financials', {})
        counts = data.get('counts', {})
        meta = data.get('meta', {})

        return {
            "precision": perf.get('precision', 0),
            "recall": perf.get('recall', 0),
            "fpr_insult_rate": perf.get('fpr_insult_rate', 0),
            "auc": perf.get('auc', 0),
            "f1_score": perf.get('f1_score', 0),
            "fraud_rate": perf.get('fraud_rate', 0),

            "fraud_stopped_val": finance.get('fraud_stopped_val', 0),
            "fraud_missed_val": finance.get('fraud_missed_val', 0),
            "false_positive_loss": finance.get('false_positive_loss', 0),
            "net_savings": finance.get('net_savings', 0),

            "total_processed": counts.get('total_processed', 0),
            "live_latency_ms": counts.get('live_latency_ms', 0),
            
            "threshold": meta.get('threshold', 0.5),
            "total_lifetime_count": meta.get('total_lifetime_count', 0),
            "queue_depth": counts.get('queue_depth', 0), 
            "updated_at": meta.get('updated_at', ""),
        }

    except Exception as e:
        logger.error(f"Stats Report Mapping Error: {e}")
        raise HTTPException(status_code=500, detail=f"Data mapping error: {str(e)}")

@app.get("/exec/threshold-optimization", tags=["Dashboard"])
def get_threshold_curve(_key: str = Depends(verify_api_key)):
    data = redis_client.get("stats:threshold_cost_curve")
    if not data:
        return []
    return json.loads(data)
      
@app.get("/exec/series", tags=["Dashboard"])
def get_financial_timeseries(_key: str = Depends(verify_api_key)):
    try:
        raw_data = redis_client.lrange('sentinel_timeseries', 0, -1)
        if not raw_data:
            return []

        cleaned_data = []
        last_point = None
        
        for item in raw_data:
            current_point = json.loads(item)
            if last_point is None:
                cleaned_data.append(current_point)
            else:
                if (current_point['cumulative_savings'] != last_point['cumulative_savings'] or
                    current_point['cumulative_loss'] != last_point['cumulative_loss']):
                    cleaned_data.append(current_point)
            last_point = current_point
        
        TARGET = int(_api_cfg.get('timeseries_max_points', 1000))
        total_points = len(cleaned_data)
        
        if total_points <= TARGET:
            series = cleaned_data
        else:
            step = (total_points + TARGET - 1) // TARGET 
            series = cleaned_data[::step]
            
            if cleaned_data and series[-1]['timestamp'] != cleaned_data[-1]['timestamp']:
                series.append(cleaned_data[-1])
                
        return series
    except Exception as e:
        logger.error(f"Timeseries error: {e}")
        return []

@app.get("/recent", response_model=List[Transaction], tags=["Dashboard"])
def get_recent_stream(limit: int = Query(100, ge=1, le=1000), _key: str = Depends(verify_api_key)):
    try:
        raw_list = redis_client.lrange('sentinel_stream', 0, limit - 1)
        return [json.loads(x) for x in raw_list]
    except Exception as e:
        logger.error(f"Stream error: {e}")
        return [] 

@app.get("/alerts", response_model=List[Transaction], tags=["Dashboard"])
def get_alerts(limit: int = Query(50, ge=1, le=200), _key: str = Depends(verify_api_key)):
    try:
        data = redis_client.lrange('sentinel_alerts', 0, limit - 1)
        return [json.loads(item) for item in data]
    except Exception as e:
        logger.error(f"Alerts Feed Error: {e}")
        return []

@app.get("/transactions/{transaction_id}", tags=["Forensics"])
def get_transaction_detail(transaction_id: str, _key: str = Depends(verify_api_key)):
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
    
@app.get("/exec/global-feature-importance", tags=["ML Monitor"])
@limiter.limit("30/minute")
def get_global_feature_importance(request: Request, _key: str = Depends(verify_api_key)):
    """
    Returns the aggregated feature importance (Average Absolute SHAP).
    """
    try:
        data = redis_client.get("stats:global_feature_importance")
        if not data:
            return []
        return json.loads(data)
    except Exception as e:
        logger.error(f"Global SHAP API Error: {e}")  
        return []

@app.get("/exec/feature-drift", tags=["ML Monitor"])
@limiter.limit("30/minute")
def get_feature_drift_report(request: Request, _key: str = Depends(verify_api_key)):
    try:
        data = redis_client.get("stats:feature_drift_report")

        if not data:
            return {}
        return json.loads(data)
    except Exception as e:
        logger.error(f"Feature Drift Report API Error: {e}")  
        return {} 
    
@app.get("/exec/performance-lookup", tags=["ML Monitor"])
@limiter.limit("30/minute")
def get_performance_lookup(request: Request, _key: str = Depends(verify_api_key)):
    data = redis_client.get("stats:simulation_table")
    return json.loads(data) if data else {}

@app.get("/exec/calibration", tags=["ML Monitor"])
@limiter.limit("30/minute")
def get_calibration_report(request: Request, _key: str = Depends(verify_api_key)):
    data = redis_client.get("stats:calibration_data")
    return json.loads(data) if data else {}

