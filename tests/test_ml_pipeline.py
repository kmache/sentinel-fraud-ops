"""
Tests for the full ML pipeline: Preprocessing → Feature Engineering → Serialization.
Covers round-trip save/load, feature engineering fit/transform consistency,
and the complete pipeline contract.
"""
import pytest
import joblib
import tempfile
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sentinel.preprocessing import SentinelPreprocessing
from sentinel.features import SentinelFeatureEngineering
from sentinel.calibration import SentinelCalibrator


class TestPreprocessorSerialization:
    """Verify preprocessor survives save → load → predict round-trip."""

    def test_save_load_produces_same_output(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        out_original = pp.transform(batch_df, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            joblib.dump(pp, path)
            pp_loaded = joblib.load(path)
            out_loaded = pp_loaded.transform(batch_df, verbose=False)
            pd.testing.assert_frame_equal(out_original, out_loaded)
        finally:
            os.unlink(path)

    def test_fitted_state_persists(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            joblib.dump(pp, path)
            pp_loaded = joblib.load(path)
            assert pp_loaded._fitted is True
            assert pp_loaded.cols_to_drop == pp.cols_to_drop
        finally:
            os.unlink(path)


class TestFeatureEngineerSerialization:
    """Verify feature engineer survives save → load → predict round-trip."""

    def test_save_load_produces_same_output(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out_original = fe.transform(clean, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            joblib.dump(fe, path)
            fe_loaded = joblib.load(path)
            out_loaded = fe_loaded.transform(clean, verbose=False)
            pd.testing.assert_frame_equal(
                out_original.reset_index(drop=True),
                out_loaded.reset_index(drop=True),
            )
        finally:
            os.unlink(path)

    def test_learned_state_persists(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            joblib.dump(fe, path)
            fe_loaded = joblib.load(path)
            assert fe_loaded._fitted is True
            assert len(fe_loaded.user_start_dates) > 0
            assert len(fe_loaded.frequency_maps) > 0
            assert len(fe_loaded.risk_maps) > 0
        finally:
            os.unlink(path)


class TestFullPipelineContract:
    """End-to-end: Preprocessing → Feature Engineering → Calibration."""

    def test_full_pipeline_produces_valid_output(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        features = fe.transform(clean, verbose=False)

        assert len(features) == len(batch_df)
        assert len(features.columns) > len(batch_df.columns)

        # No inf values should remain
        numeric = features.select_dtypes(include=[np.number])
        for col in numeric.columns:
            vals = numeric[col].dropna().values
            assert not np.any(np.isinf(vals)), f"inf in {col}"

    def test_pipeline_with_calibration(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        features = fe.transform(clean, verbose=False)

        # Simulate model probabilities
        rng = np.random.RandomState(42)
        y_true = batch_df["isFraud"].values
        y_prob = np.clip(y_true * rng.uniform(0.5, 0.9, len(y_true)) +
                         (1 - y_true) * rng.uniform(0.0, 0.3, len(y_true)), 0, 1)

        cal = SentinelCalibrator(method="isotonic")
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)

        assert calibrated.shape == y_prob.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_pipeline_serialization_roundtrip(self, batch_df):
        """Full round-trip: fit → save all → load all → transform → compare."""
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        features_original = fe.transform(clean, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            pp_path = os.path.join(tmpdir, "preprocessor.pkl")
            fe_path = os.path.join(tmpdir, "engineer.pkl")
            joblib.dump(pp, pp_path)
            joblib.dump(fe, fe_path)

            pp2 = joblib.load(pp_path)
            fe2 = joblib.load(fe_path)

            clean2 = pp2.transform(batch_df, verbose=False)
            features2 = fe2.transform(clean2, verbose=False)

            pd.testing.assert_frame_equal(
                features_original.reset_index(drop=True),
                features2.reset_index(drop=True),
            )

    def test_unseen_data_after_fit(self, batch_df):
        """Pipeline handles data not seen during fit."""
        train = batch_df.iloc[:80]
        test = batch_df.iloc[80:]

        pp = SentinelPreprocessing(verbose=False)
        pp.fit(train)
        clean_test = pp.transform(test, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        clean_train = pp.transform(train, verbose=False)
        fe.fit(clean_train, y="isFraud")
        features_test = fe.transform(clean_test, verbose=False)

        assert len(features_test) == len(test)
