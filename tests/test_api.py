"""
Basic API smoke tests.
Uses fakeredis to avoid requiring a running Redis instance.
"""
import pytest
import json
import sys
import os
import time
from unittest.mock import patch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import fakeredis


@pytest.fixture
def fake_redis():
    """Create a fakeredis instance with minimal test data."""
    r = fakeredis.FakeRedis(decode_responses=True)
    # Seed stats for /stats endpoint
    stats = {
        "performance": {"precision": 0.85, "recall": 0.92, "fpr_insult_rate": 0.01,
                        "auc": 0.97, "f1_score": 0.88, "fraud_rate": 3.5},
        "financials": {"fraud_stopped_val": 50000.0, "fraud_missed_val": 2000.0,
                       "false_positive_loss": 500.0, "net_savings": 47500.0},
        "counts": {"total_processed": 10000, "live_latency_ms": 12.5, "queue_depth": 42},
        "meta": {"threshold": 0.5, "total_lifetime_count": 10000,
                 "updated_at": datetime.now().isoformat()},
    }
    r.set("stats:stat_business_report", json.dumps(stats))
    # Seed stream
    for i in range(5):
        tx = {"transaction_id": f"TX_{i}", "timestamp": datetime.now().isoformat(),
              "TransactionAmt": 100.0 + i * 50, "is_fraud": 0, "score": 0.05 + i * 0.1,
              "action": "APPROVE", "ground_truth": 0}
        r.lpush("sentinel_stream", json.dumps(tx))
    return r


@pytest.fixture
def client(fake_redis):
    """Create FastAPI TestClient with mocked Redis."""
    with patch("main.redis_client", fake_redis), \
         patch("main.redis_pool", None), \
         patch("main.API_KEY", ""):  # Disable auth for unit tests
        from main import app
        from fastapi.testclient import TestClient
        app.state.startup_time = time.time()
        yield TestClient(app)


def test_health_endpoint(client):
    """Test basic health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client):
    """Test root endpoint returns service info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Sentinel Gateway"
    assert data["status"] == "active"


def test_metrics_endpoint(client):
    """Test system metrics return expected structure"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "memory_usage_mb" in data
    assert "cpu_usage_percent" in data
    assert isinstance(data["redis_connected"], bool)


def test_stats_schema_validation(client):
    """Test /stats returns valid StatsResponse structure"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    required_fields = [
        "precision", "recall", "fpr_insult_rate", "auc",
        "net_savings", "threshold", "total_processed"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
        assert isinstance(data[field], (int, float))


def test_exec_series_empty(client):
    """Test timeseries endpoint handles empty data gracefully"""
    response = client.get("/exec/series")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_transaction_detail_404(client):
    """Test forensics endpoint returns 404 for missing transaction"""
    response = client.get("/transactions/nonexistent_id")
    assert response.status_code == 404