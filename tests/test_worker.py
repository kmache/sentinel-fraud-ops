"""
Tests for the worker/processor.py module.
Uses mocks for Redis, Kafka, and the ML model to test batch processing logic.

Since processor.py has module-level side effects (Redis/Kafka connections, model loading),
we must mock those before importing. We use a module-level fixture approach.
"""
import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fakeredis


@pytest.fixture(scope="module")
def processor_module():
    """Import processor.py with all external connections mocked."""
    fake_r = fakeredis.FakeRedis(decode_responses=True)

    mock_sentinel = MagicMock()
    mock_sentinel.config = {"threshold": 0.5, "weights": {}}
    mock_sentinel.predict = MagicMock()
    mock_sentinel.explain = MagicMock(return_value=[])

    # Remove processor from cache to allow fresh import with mocks
    if "processor" in sys.modules:
        del sys.modules["processor"]

    with patch("redis.Redis", return_value=fake_r), \
         patch("redis.ConnectionPool", return_value=MagicMock()):
        with patch("sentinel.inference.SentinelInference", return_value=mock_sentinel):
            import processor
            processor.redis_client = fake_r
            processor.model_engine = mock_sentinel
            processor.kafka_producer = None

    return processor


@pytest.fixture
def mock_redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def sample_predict_results():
    """Sample output from model_engine.predict()."""
    return {
        "transaction_id": ["TX_001", "TX_002", "TX_003"],
        "probabilities": [0.95, 0.12, 0.87],
        "is_frauds": [1, 0, 1],
        "actions": ["BLOCK", "APPROVE", "BLOCK"],
        "y_true": [1, 0, 1],
        "dashboard_data": [
            {"TransactionAmt": 500.0, "ProductCD": "W", "card4": "visa"},
            {"TransactionAmt": 25.0, "ProductCD": "C", "card4": "mastercard"},
            {"TransactionAmt": 1200.0, "ProductCD": "H", "card4": "visa"},
        ],
        "processed_data_records": {"xgb": [{}, {}, {}]},
        "meta": {
            "latency_ms": 15.3,
            "weights": {"xgb": 0.4, "lgb": 0.35, "cb": 0.25},
        },
    }


@pytest.fixture
def sample_messages():
    """Create mock Kafka messages."""
    messages = []
    for tx in [
        {"TransactionAmt": 500.0, "ProductCD": "W", "card4": "visa"},
        {"TransactionAmt": 25.0, "ProductCD": "C", "card4": "mastercard"},
        {"TransactionAmt": 1200.0, "ProductCD": "H", "card4": "visa"},
    ]:
        msg = MagicMock()
        msg.value = tx
        messages.append(msg)
    return messages


# ==========================================================================
# TESTS: flush_batch pipeline separation
# ==========================================================================
class TestFlushBatch:

    def test_flush_batch_writes_correct_keys(
        self, processor_module, mock_redis, sample_messages, sample_predict_results
    ):
        """Verify flush_batch writes expected Redis keys."""
        mock_engine = MagicMock()
        mock_engine.predict.return_value = sample_predict_results
        mock_engine.config = {"threshold": 0.5, "weights": {}}

        processor_module.redis_client = mock_redis
        processor_module.model_engine = mock_engine
        processor_module.kafka_producer = None
        processor_module.SENTINEL_MODE = "demo"

        processor_module.flush_batch(sample_messages)

        assert mock_redis.llen("sentinel_stream") == 3
        assert mock_redis.llen("sentinel_alerts") == 2
        for tx_id in ["TX_001", "TX_002", "TX_003"]:
            assert mock_redis.get(f"prediction:{tx_id}") is not None
        assert mock_redis.llen("stats:hist_y_prob") == 3
        assert mock_redis.llen("stats:hist_y_true") == 3
        assert mock_redis.llen("stats:hist_amounts") == 3

    def test_flush_batch_empty(self, processor_module, mock_redis):
        """flush_batch with empty list should be a no-op."""
        processor_module.redis_client = mock_redis
        processor_module.flush_batch([])
        assert mock_redis.llen("sentinel_stream") == 0

    def test_flush_batch_sends_to_dlq_on_inference_error(
        self, processor_module, mock_redis, sample_messages
    ):
        """Verify DLQ handling when inference fails."""
        mock_engine = MagicMock()
        mock_engine.predict.side_effect = RuntimeError("Model crashed")
        mock_producer = MagicMock()

        processor_module.redis_client = mock_redis
        processor_module.model_engine = mock_engine
        processor_module.kafka_producer = mock_producer

        processor_module.flush_batch(sample_messages)

        assert mock_producer.send.call_count == 3

    def test_flush_batch_financial_metrics_skipped_in_production(
        self, processor_module, mock_redis, sample_messages, sample_predict_results
    ):
        """In production mode with no ground truth, financial metrics should be zero."""
        results_no_gt = sample_predict_results.copy()
        results_no_gt["y_true"] = None

        mock_engine = MagicMock()
        mock_engine.predict.return_value = results_no_gt
        mock_engine.config = {"threshold": 0.5}

        processor_module.redis_client = mock_redis
        processor_module.model_engine = mock_engine
        processor_module.kafka_producer = None
        processor_module.SENTINEL_MODE = "production"

        processor_module.flush_batch(sample_messages)

        savings = float(mock_redis.get("stats:global_savings") or 0)
        loss = float(mock_redis.get("stats:global_loss") or 0)
        assert savings == 0.0
        assert loss == 0.0


# ==========================================================================
# TESTS: DLQ
# ==========================================================================
class TestDLQ:

    def test_send_to_dlq_single_message(self, processor_module):
        mock_producer = MagicMock()
        processor_module.kafka_producer = mock_producer
        processor_module.DLQ_TOPIC = "test_dlq"
        processor_module.send_to_dlq({"key": "value"}, "test_error", ValueError("test"))

        mock_producer.send.assert_called_once()
        call_args = mock_producer.send.call_args
        assert call_args[0][0] == "test_dlq"
        payload = call_args[1]["value"]
        assert payload["error_reason"] == "test_error"

    def test_send_to_dlq_batch(self, processor_module):
        mock_producer = MagicMock()
        processor_module.kafka_producer = mock_producer
        processor_module.DLQ_TOPIC = "test_dlq"
        processor_module.send_to_dlq([{"a": 1}, {"b": 2}], "batch_error")
        assert mock_producer.send.call_count == 2

    def test_send_to_dlq_no_producer(self, processor_module):
        processor_module.kafka_producer = None
        # Should not raise
        processor_module.send_to_dlq({"key": "value"}, "test_error")


# ==========================================================================
# TESTS: SHAP enrichment
# ==========================================================================
class TestSHAPEnrichment:

    def test_update_global_feature_importance_new(self, processor_module, mock_redis):
        batch_explanations = [
            [{"feature": "amt", "impact": 0.5}, {"feature": "card", "impact": 0.3}],
            [{"feature": "amt", "impact": 0.7}, {"feature": "card", "impact": 0.1}],
        ]
        processor_module.redis_client = mock_redis
        processor_module.update_global_feature_importance(batch_explanations)

        result = json.loads(mock_redis.get("stats:global_feature_importance"))
        assert "amt" in result
        assert "card" in result
        assert result["amt"] > result["card"]

    def test_update_global_feature_importance_ema(self, processor_module, mock_redis):
        existing = {"amt": 1.0, "card": 0.5}
        mock_redis.set("stats:global_feature_importance", json.dumps(existing))

        batch_explanations = [
            [{"feature": "amt", "impact": 0.0}, {"feature": "card", "impact": 0.0}],
        ]
        processor_module.redis_client = mock_redis
        processor_module.update_global_feature_importance(batch_explanations)

        result = json.loads(mock_redis.get("stats:global_feature_importance"))
        assert 0.85 < result["amt"] < 0.95

    def test_update_global_feature_importance_empty(self, processor_module, mock_redis):
        processor_module.redis_client = mock_redis
        processor_module.update_global_feature_importance([])
        assert mock_redis.get("stats:global_feature_importance") is None


# ==========================================================================
# TESTS: Graceful Shutdown
# ==========================================================================
class TestGracefulShutdown:

    def test_run_flushes_on_keyboard_interrupt(self, processor_module, mock_redis):
        mock_consumer = MagicMock()
        mock_msg = MagicMock()
        mock_msg.value = {"TransactionAmt": 100.0}

        call_count = [0]
        def message_iter():
            yield mock_msg
            raise KeyboardInterrupt()

        mock_consumer.__iter__ = MagicMock(side_effect=message_iter)
        mock_consumer.close = MagicMock()
        mock_consumer.commit = MagicMock()

        orig_get_consumer = processor_module.get_consumer
        orig_flush = processor_module.flush_batch
        orig_batch_size = processor_module.BATCH_SIZE
        orig_redis = processor_module.redis_client
        orig_producer = processor_module.kafka_producer

        flush_calls = []
        def track_flush(msgs):
            flush_calls.append(len(msgs))

        processor_module.get_consumer = lambda: mock_consumer
        processor_module.flush_batch = track_flush
        processor_module.BATCH_SIZE = 100
        processor_module.redis_client = mock_redis
        processor_module.kafka_producer = None

        try:
            processor_module.run()
        finally:
            processor_module.get_consumer = orig_get_consumer
            processor_module.flush_batch = orig_flush
            processor_module.BATCH_SIZE = orig_batch_size
            processor_module.redis_client = orig_redis
            processor_module.kafka_producer = orig_producer

        assert len(flush_calls) >= 1
        mock_consumer.close.assert_called_once()


# ==========================================================================
# TESTS: Circuit Breaker → DLQ Integration
# ==========================================================================
class TestCircuitBreakerDLQ:

    def test_flush_batch_routes_to_dlq_on_circuit_breaker(
        self, processor_module, mock_redis, sample_messages, sample_predict_results
    ):
        """When Redis circuit breaker opens, batch should be routed to DLQ."""
        mock_engine = MagicMock()
        mock_engine.predict.return_value = sample_predict_results
        mock_engine.config = {"threshold": 0.5, "weights": {}}
        mock_producer = MagicMock()

        processor_module.redis_client = mock_redis
        processor_module.model_engine = mock_engine
        processor_module.kafka_producer = mock_producer
        processor_module.SENTINEL_MODE = "demo"

        # The second call to execute_redis_pipeline is the txn batch write.
        # Let the timeseries pipeline succeed but the txn pipeline fail.
        from circuitbreaker import CircuitBreakerError

        call_count = [0]
        original_execute = processor_module.execute_redis_pipeline

        def selective_fail(pipe):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise CircuitBreakerError(MagicMock())
            pipe.execute()

        processor_module.execute_redis_pipeline = selective_fail

        try:
            processor_module.flush_batch(sample_messages)
        finally:
            processor_module.execute_redis_pipeline = original_execute

        # DLQ should have received messages
        assert mock_producer.send.call_count >= 1

    def test_dlq_payload_contains_error_metadata(self, processor_module):
        """DLQ messages must contain error_reason, exception, and timestamp."""
        mock_producer = MagicMock()
        processor_module.kafka_producer = mock_producer
        processor_module.DLQ_TOPIC = "test_dlq"

        processor_module.send_to_dlq(
            {"TransactionAmt": 100.0},
            "redis_circuit_breaker",
            RuntimeError("Connection refused"),
        )

        payload = mock_producer.send.call_args[1]["value"]
        assert payload["error_reason"] == "redis_circuit_breaker"
        assert "Connection refused" in payload["exception"]
        assert "timestamp" in payload


# ==========================================================================
# TESTS: SHAP Failure Status Marking
# ==========================================================================
class TestSHAPFailureMarking:

    def test_shap_failure_marks_transactions_as_failed(
        self, processor_module, mock_redis
    ):
        """When SHAP computation fails, transactions should have explanation_status='failed'."""
        processor_module.redis_client = mock_redis

        # Pre-populate predictions in Redis (simulating flush_batch output)
        for tx_id in ["TX_A", "TX_B"]:
            payload = {
                "transaction_id": tx_id,
                "score": 0.9,
                "explanation_status": "processing",
                "explanations": [],
            }
            mock_redis.setex(f"prediction:{tx_id}", 3600, json.dumps(payload))

        # Make model_engine.explain raise an error
        original_engine = processor_module.model_engine
        mock_engine = MagicMock()
        mock_engine.explain.side_effect = RuntimeError("SHAP OOM")
        processor_module.model_engine = mock_engine

        try:
            processor_module.async_enrich_shap(
                ["TX_A", "TX_B"], [{"amt": 1}, {"amt": 2}]
            )
        finally:
            processor_module.model_engine = original_engine

        # Both transactions should now have failed status
        for tx_id in ["TX_A", "TX_B"]:
            data = json.loads(mock_redis.get(f"prediction:{tx_id}"))
            assert data["explanation_status"] == "failed"
            assert "SHAP OOM" in data["explanation_error"]

    def test_shap_success_marks_transactions_as_ready(
        self, processor_module, mock_redis
    ):
        """Successful SHAP should set explanation_status='ready'."""
        processor_module.redis_client = mock_redis

        for tx_id in ["TX_C", "TX_D"]:
            payload = {
                "transaction_id": tx_id,
                "score": 0.8,
                "explanation_status": "processing",
                "explanations": [],
            }
            mock_redis.setex(f"prediction:{tx_id}", 3600, json.dumps(payload))

        original_engine = processor_module.model_engine
        mock_engine = MagicMock()
        mock_engine.explain.return_value = [
            [{"feature": "amt", "impact": 0.5}] * 10,
            [{"feature": "card", "impact": 0.3}] * 10,
        ]
        processor_module.model_engine = mock_engine

        try:
            processor_module.async_enrich_shap(
                ["TX_C", "TX_D"], [{"amt": 1}, {"amt": 2}]
            )
        finally:
            processor_module.model_engine = original_engine

        for tx_id in ["TX_C", "TX_D"]:
            data = json.loads(mock_redis.get(f"prediction:{tx_id}"))
            assert data["explanation_status"] == "ready"
            assert len(data["explanations"]) > 0


# ==========================================================================
# TESTS: SIGTERM Graceful Shutdown
# ==========================================================================
class TestSIGTERMShutdown:

    def test_sigterm_sets_shutdown_flag(self, processor_module):
        """SIGTERM should set the _shutdown_requested event."""
        processor_module._shutdown_requested.clear()

        mock_consumer = MagicMock()
        msgs_yielded = []

        def message_iter():
            # Simulate SIGTERM after first message
            msg = MagicMock()
            msg.value = {"TransactionAmt": 50.0}
            yield msg
            msgs_yielded.append(1)
            processor_module._shutdown_requested.set()
            # Yield one more — run() should break before processing it
            msg2 = MagicMock()
            msg2.value = {"TransactionAmt": 75.0}
            yield msg2

        mock_consumer.__iter__ = MagicMock(side_effect=message_iter)
        mock_consumer.close = MagicMock()
        mock_consumer.commit = MagicMock()

        orig_get_consumer = processor_module.get_consumer
        orig_flush = processor_module.flush_batch
        orig_batch_size = processor_module.BATCH_SIZE
        orig_producer = processor_module.kafka_producer

        flush_calls = []

        def track_flush(msgs):
            flush_calls.append(len(msgs))

        processor_module.get_consumer = lambda: mock_consumer
        processor_module.flush_batch = track_flush
        processor_module.BATCH_SIZE = 1000  # Large so auto-flush doesn't trigger
        processor_module.kafka_producer = None

        try:
            processor_module.run()
        finally:
            processor_module.get_consumer = orig_get_consumer
            processor_module.flush_batch = orig_flush
            processor_module.BATCH_SIZE = orig_batch_size
            processor_module.kafka_producer = orig_producer
            processor_module._shutdown_requested.clear()

        mock_consumer.close.assert_called_once()


# ==========================================================================
# TESTS: Custom Exception Hierarchy
# ==========================================================================
class TestExceptionHierarchy:

    def test_sentinel_error_is_base(self):
        from sentinel.exceptions import (
            SentinelError,
            PreprocessingError,
            FeatureEngineeringError,
            InferenceError,
            ArtifactLoadError,
            CalibrationError,
        )
        for exc_cls in [
            PreprocessingError,
            FeatureEngineeringError,
            InferenceError,
            ArtifactLoadError,
            CalibrationError,
        ]:
            assert issubclass(exc_cls, SentinelError)
            assert issubclass(exc_cls, Exception)

    def test_inference_error_raised_by_predict(self):
        """SentinelInference.predict should raise InferenceError, not RuntimeError."""
        from sentinel.exceptions import InferenceError

        # Build a minimal SentinelInference with broken preprocessor
        from sentinel.inference import SentinelInference
        from unittest.mock import PropertyMock

        engine = object.__new__(SentinelInference)
        engine.verbose = False
        engine.config = {"threshold": 0.5, "weights": {"xgb": 1.0}}
        engine.models = {"xgb": MagicMock()}
        engine.features = {"xgb": ["f1", "f2"]}
        engine.calibrator = MagicMock()
        engine.preprocessor = MagicMock()
        engine.preprocessor.transform.side_effect = ValueError("bad data")
        engine.engineer = MagicMock()

        with pytest.raises(InferenceError, match="Inference pipeline failed"):
            engine.predict({"TransactionID": 1, "TransactionAmt": 100.0})

    def test_catch_all_sentinel_errors(self):
        """A single except SentinelError should catch all domain exceptions."""
        from sentinel.exceptions import SentinelError, InferenceError, PreprocessingError

        for exc in [InferenceError("test"), PreprocessingError("test")]:
            try:
                raise exc
            except SentinelError:
                pass  # Expected
            except Exception:
                pytest.fail(f"{type(exc).__name__} was not caught by SentinelError")
