"""
Unit tests for monitoring (PSI, drift detection).
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sentinel.monitoring import calculate_psi, check_feature_drift, generate_feature_baseline


class TestPSI:
    """Population Stability Index calculation tests."""

    def test_identical_distributions(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 1000)
        _, bin_edges = np.histogram(data, bins=10)
        counts, _ = np.histogram(data, bins=bin_edges)
        expected_pct = counts / len(data)

        psi = calculate_psi(expected_pct, data, bin_edges)
        assert psi < 0.01  # essentially no drift

    def test_shifted_distribution(self):
        rng = np.random.RandomState(42)
        train = rng.normal(0, 1, 1000)
        _, bin_edges = np.histogram(train, bins=10)
        counts, _ = np.histogram(train, bins=bin_edges)
        expected_pct = counts / len(train)

        # Significant shift
        live = rng.normal(3, 1, 1000)
        psi = calculate_psi(expected_pct, live, bin_edges)
        assert psi > 0.2  # should detect drift

    def test_empty_actual_returns_zero(self):
        psi = calculate_psi([0.1, 0.9], np.array([]), [0, 0.5, 1])
        assert psi == 0.0

    def test_inf_values_filtered(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 100)
        _, bin_edges = np.histogram(data, bins=10)
        counts, _ = np.histogram(data, bins=bin_edges)
        expected_pct = counts / len(data)

        # Add infs
        live = np.concatenate([data, [np.inf, -np.inf, np.nan]])
        psi = calculate_psi(expected_pct, live, bin_edges)
        assert np.isfinite(psi)


class TestFeatureBaseline:
    """Test baseline generation."""

    def test_baseline_structure(self):
        rng = np.random.RandomState(42)
        df_data = {"f1": rng.normal(0, 1, 1000), "f2": rng.exponential(1, 1000)}
        import pandas as pd
        df = pd.DataFrame(df_data)
        baseline = generate_feature_baseline(df, ["f1", "f2"])

        for feat in ["f1", "f2"]:
            assert feat in baseline
            assert "expected_pct" in baseline[feat]
            assert "bin_edges" in baseline[feat]
            assert "expected_null_rate" in baseline[feat]
            assert len(baseline[feat]["bin_edges"]) == 11  # 10 bins + 1 edge

    def test_sparse_feature_flagged(self):
        import pandas as pd
        data = np.full(100, np.nan)
        data[:5] = [1, 2, 3, 4, 5]
        df = pd.DataFrame({"sparse_feat": data})
        baseline = generate_feature_baseline(df, ["sparse_feat"])
        assert baseline["sparse_feat"]["is_sparse"] is True


class TestDriftCheck:
    """Test the full drift check function."""

    def test_no_drift(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 500)
        _, bin_edges = np.histogram(data, bins=10)
        counts, _ = np.histogram(data, bins=bin_edges)
        expected_pct = (counts / len(data)).tolist()

        baseline_item = {
            "expected_null_rate": 0.0,
            "expected_pct": expected_pct,
            "bin_edges": bin_edges.tolist(),
            "is_sparse": False,
        }

        result = check_feature_drift(data, baseline_item)
        assert result["status"] == "GREEN"
        assert result["psi"] < 0.1

    def test_drift_detected(self):
        rng = np.random.RandomState(42)
        train = rng.normal(0, 1, 500)
        _, bin_edges = np.histogram(train, bins=10)
        counts, _ = np.histogram(train, bins=bin_edges)
        expected_pct = (counts / len(train)).tolist()

        baseline_item = {
            "expected_null_rate": 0.0,
            "expected_pct": expected_pct,
            "bin_edges": bin_edges.tolist(),
            "is_sparse": False,
        }

        live = rng.normal(5, 2, 500)
        result = check_feature_drift(live, baseline_item)
        assert result["status"] == "RED"
        assert result["psi"] > 0.2
