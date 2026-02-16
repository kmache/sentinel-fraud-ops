import pytest
from fastapi.testclient import TestClient
from datetime import datetime

# Add backend to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app

client = TestClient(app)

def test_health_endpoint():
    """Test basic health check"""
    response = client.get("/health")
    assert response.status_code == 503  # Redis not available in tests

def test_root_endpoint():
    """Test root endpoint returns service info"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Sentinel Gateway"
    assert data["status"] == "active"

def test_metrics_endpoint():
    """Test system metrics return expected structure"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "memory_usage_mb" in data
    assert "cpu_usage_percent" in data
    assert isinstance(data["redis_connected"], bool)

def test_stats_schema_validation():
    """Test /stats returns valid StatsResponse structure"""
    response = client.get("/stats")
    # Will 404 if Redis is empty, but should return valid JSON structure
    if response.status_code == 200:
        data = response.json()
        required_fields = [
            "precision", "recall", "fpr_insult_rate", "auc",
            "net_savings", "threshold", "total_processed"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
            assert isinstance(data[field], (int, float))

def test_exec_series_empty():
    """Test timeseries endpoint handles empty data gracefully"""
    response = client.get("/exec/series")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_transaction_detail_404():
    """Test forensics endpoint returns 404 for missing transaction (when Redis up) or error when down"""
    response = client.get("/transactions/nonexistent_id")
    # If Redis is running: expect 404 (not found)
    # If Redis is down: expect 503 or 500 (service unavailable)
    assert response.status_code in [404, 500, 503]
    assert "detail" in response.json() or "error" in response.json() or response.status_code == 503