"""
Integration tests for the FastAPI Gateway.
Uses fakeredis to mock Redis, testing the full request/response cycle.
"""
import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import fakeredis
from unittest.mock import patch
from datetime import datetime


@pytest.fixture
def fake_redis():
    """Create a fakeredis instance with test data pre-loaded."""
    r = fakeredis.FakeRedis(decode_responses=True)

    # Seed stats
    stats = {
        "performance": {
            "precision": 0.85, "recall": 0.92, "fpr_insult_rate": 0.01,
            "auc": 0.97, "f1_score": 0.88, "fraud_rate": 3.5,
        },
        "financials": {
            "fraud_stopped_val": 50000.0, "fraud_missed_val": 2000.0,
            "false_positive_loss": 500.0, "net_savings": 47500.0,
        },
        "counts": {"total_processed": 10000, "live_latency_ms": 12.5, "queue_depth": 42},
        "meta": {
            "threshold": 0.5, "total_lifetime_count": 10000,
            "updated_at": datetime.now().isoformat(),
        },
    }
    r.set("stats:stat_business_report", json.dumps(stats))

    # Seed stream
    for i in range(5):
        tx = {
            "transaction_id": f"TX_{i}",
            "timestamp": datetime.now().isoformat(),
            "TransactionAmt": 100.0 + i * 50,
            "is_fraud": 0,
            "score": 0.05 + i * 0.1,
            "action": "APPROVE",
            "ground_truth": 0,
        }
        r.lpush("sentinel_stream", json.dumps(tx))

    # Seed one alert
    alert = {
        "transaction_id": "TX_FRAUD_1",
        "timestamp": datetime.now().isoformat(),
        "TransactionAmt": 999.99,
        "is_fraud": 1,
        "score": 0.95,
        "action": "BLOCK",
        "ground_truth": 1,
    }
    r.lpush("sentinel_alerts", json.dumps(alert))

    # Seed one transaction detail
    detail = {
        "transaction_id": "TX_DETAIL_1",
        "score": 0.87,
        "action": "BLOCK",
        "explanations": [{"feature": "TransactionAmt", "value": "9999", "impact": 0.42}],
    }
    r.setex("prediction:TX_DETAIL_1", 3600, json.dumps(detail))

    # Seed cost curve
    curve = [{"threshold": round(t, 3), "total_loss": 1000 - t * 500} for t in [0.1, 0.3, 0.5, 0.7, 0.9]]
    r.set("stats:threshold_cost_curve", json.dumps(curve))

    # Seed timeseries
    for i in range(3):
        point = {
            "timestamp": datetime.now().isoformat(),
            "cumulative_savings": 1000.0 * (i + 1),
            "cumulative_loss": 100.0 * (i + 1),
        }
        r.rpush("sentinel_timeseries", json.dumps(point))

    return r


@pytest.fixture
def client(fake_redis):
    """Create FastAPI TestClient with mocked Redis."""
    import time as _time
    with patch("main.redis_client", fake_redis), \
         patch("main.redis_pool", None), \
         patch("main.API_KEY", ""):  # Disable auth for integration tests
        from main import app
        from fastapi.testclient import TestClient
        app.state.startup_time = _time.time()
        yield TestClient(app)


class TestHealthEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "active"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_metrics(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "memory_usage_mb" in data
        assert data["redis_connected"] is True


class TestDashboardEndpoints:
    def test_stats(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["precision"] == 0.85
        assert data["recall"] == 0.92
        assert data["net_savings"] == 47500.0
        assert data["threshold"] == 0.5

    def test_recent_stream(self, client):
        r = client.get("/recent?limit=3")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3
        assert all("transaction_id" in tx for tx in data)

    def test_recent_default_limit(self, client):
        r = client.get("/recent")
        assert r.status_code == 200
        assert len(r.json()) == 5  # we seeded 5

    def test_alerts(self, client):
        r = client.get("/alerts")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["action"] == "BLOCK"

    def test_threshold_curve(self, client):
        r = client.get("/exec/threshold-optimization")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5

    def test_timeseries(self, client):
        r = client.get("/exec/series")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        assert "cumulative_savings" in data[0]


class TestForensicsEndpoints:
    def test_transaction_detail(self, client):
        r = client.get("/transactions/TX_DETAIL_1")
        assert r.status_code == 200
        data = r.json()
        assert data["score"] == 0.87
        assert len(data["explanations"]) == 1

    def test_transaction_not_found(self, client):
        r = client.get("/transactions/NONEXISTENT")
        assert r.status_code == 404


class TestAPIKeySecurity:
    """Test that API key auth works when enabled."""

    def test_protected_endpoint_rejects_without_key(self, fake_redis):
        with patch("main.redis_client", fake_redis), \
             patch("main.redis_pool", None), \
             patch("main.API_KEY", "test-secret-key"):
            from main import app
            from fastapi.testclient import TestClient
            c = TestClient(app)
            r = c.get("/stats")
            assert r.status_code == 403

    def test_protected_endpoint_accepts_valid_key(self, fake_redis):
        with patch("main.redis_client", fake_redis), \
             patch("main.redis_pool", None), \
             patch("main.API_KEY", "test-secret-key"):
            from main import app
            from fastapi.testclient import TestClient
            c = TestClient(app)
            r = c.get("/stats", headers={"X-API-Key": "test-secret-key"})
            assert r.status_code == 200

    def test_health_accessible_without_key(self, fake_redis):
        with patch("main.redis_client", fake_redis), \
             patch("main.redis_pool", None), \
             patch("main.API_KEY", "test-secret-key"):
            from main import app
            from fastapi.testclient import TestClient
            c = TestClient(app)
            r = c.get("/health")
            assert r.status_code == 200
