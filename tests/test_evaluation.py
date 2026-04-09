"""
Unit tests for SentinelEvaluator and SentinelCalibrator.
"""
import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sentinel.evaluation import SentinelEvaluator
from sentinel.calibration import SentinelCalibrator


# ==============================================================================
# EVALUATOR TESTS
# ==============================================================================
class TestEvaluatorMetrics:
    """Test metric computation correctness."""

    def test_auc_perfect_model(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        ev = SentinelEvaluator(y_true, y_prob)
        assert ev.get_auc() == 1.0

    def test_auc_random_model(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 1000)
        y_prob = rng.random(1000)
        ev = SentinelEvaluator(y_true, y_prob)
        auc = ev.get_auc()
        assert 0.4 < auc < 0.6  # should be ~0.5

    def test_auc_single_class(self):
        y_true = np.zeros(10)
        y_prob = np.random.random(10)
        ev = SentinelEvaluator(y_true, y_prob)
        assert ev.get_auc() == 0.5

    def test_core_metrics_keys(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        metrics = ev.get_core_metrics()
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics


class TestThresholdOptimization:
    """Test threshold finding methods."""

    def test_cost_threshold_in_range(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        t = ev.find_best_threshold(method="cost", cb_fee=30.0, support_cost=12.5, churn_factor=0.15)
        assert 0.0 <= t <= 1.0

    def test_fbeta_threshold_in_range(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        t = ev.find_best_threshold(method="fbeta", beta=2.0)
        assert 0.0 <= t <= 1.0

    def test_friction_threshold_in_range(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        t = ev.find_best_threshold(method="friction", max_fpr=0.02)
        assert 0.0 <= t <= 1.0

    def test_empty_data(self):
        ev = SentinelEvaluator([], [], [])
        t = ev.find_best_threshold(method="cost")
        assert t == 0.5


class TestBusinessImpactReport:
    """Test P&L report structure and logic."""

    def test_report_structure(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        report = ev.report_business_impact(threshold=0.5)

        assert "performance" in report
        assert "financials" in report
        assert "counts" in report

        perf = report["performance"]
        assert 0 <= perf["precision"] <= 1
        assert 0 <= perf["recall"] <= 1
        assert 0 <= perf["auc"] <= 1

    def test_high_threshold_catches_less(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        r_low = ev.report_business_impact(threshold=0.1)
        r_high = ev.report_business_impact(threshold=0.9)
        # Higher threshold → lower recall (catches fewer frauds)
        assert r_low["performance"]["recall"] >= r_high["performance"]["recall"]

    def test_cost_curve_length(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        curve = ev.get_cost_curve()
        assert len(curve) == 50

    def test_cost_curve_sorted(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        curve = ev.get_cost_curve()
        thresholds = [p["threshold"] for p in curve]
        assert thresholds == sorted(thresholds)

    def test_tiered_strategy(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        actions = ev.get_tiered_strategy(soft_threshold=0.15, hard_threshold=0.75)
        valid_actions = {"APPROVE", "REVIEW", "BLOCK"}
        assert all(a in valid_actions for a in actions)

    def test_simulation_table(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        table = ev.get_simulation_table()
        assert isinstance(table, dict)
        assert len(table) == 101  # 0.00 to 1.00

    def test_calibration_report(self, evaluator_data):
        y_true, y_prob, amounts = evaluator_data
        ev = SentinelEvaluator(y_true, y_prob, amounts)
        report = ev.get_calibration_report()
        assert "prob_true" in report
        assert "prob_pred" in report


# ==============================================================================
# CALIBRATOR TESTS
# ==============================================================================
class TestCalibrator:
    """Test SentinelCalibrator fit/transform/save/load."""

    def test_isotonic_fit_transform(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.1, 500)
        y_prob = np.clip(y_true + rng.normal(0, 0.3, 500), 0, 1)

        cal = SentinelCalibrator(method="isotonic")
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)

        assert calibrated.shape == y_prob.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_sigmoid_fit_transform(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.1, 500)
        y_prob = np.clip(y_true + rng.normal(0, 0.3, 500), 0, 1)

        cal = SentinelCalibrator(method="sigmoid")
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)

        assert calibrated.shape == y_prob.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_unfitted_returns_input(self):
        cal = SentinelCalibrator()
        raw = np.array([0.1, 0.5, 0.9])
        np.testing.assert_array_equal(cal.transform(raw), raw)

    def test_save_and_load(self):
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.1, 200)
        y_prob = np.clip(y_true + rng.normal(0, 0.3, 200), 0, 1)

        cal = SentinelCalibrator(method="isotonic")
        cal.fit(y_true, y_prob)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            cal.save(path)
            loaded = SentinelCalibrator.load(path)
            test_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            np.testing.assert_array_almost_equal(
                cal.transform(test_probs),
                loaded.transform(test_probs),
            )
        finally:
            os.unlink(path)

    def test_load_missing_returns_uncalibrated(self):
        cal = SentinelCalibrator.load("/nonexistent/calibrator.pkl")
        raw = np.array([0.2, 0.8])
        np.testing.assert_array_equal(cal.transform(raw), raw)

    def test_too_few_samples_skips(self):
        cal = SentinelCalibrator()
        cal.fit(np.array([0, 1]), np.array([0.1, 0.9]))
        assert not cal._fitted
