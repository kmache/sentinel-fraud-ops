"""
Unit tests for SentinelPreprocessing and SentinelFeatureEngineering.
Covers: normal flow, NaN handling, inf handling, empty inputs, fit/transform consistency.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from sentinel.preprocessing import SentinelPreprocessing
from sentinel.features import SentinelFeatureEngineering


def _make_transaction(**overrides):
    base = {
        "TransactionID": 100001,
        "TransactionDT": 86400,
        "TransactionAmt": 150.00,
        "ProductCD": "C",
        "card1": 1234,
        "card2": 300.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 142.0,
        "card6": "debit",
        "addr1": 315.0,
        "addr2": 87.0,
        "dist1": 19.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "M1": "T", "M2": "T", "M3": "T", "M4": "M0", "M5": "F", "M6": "T",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows",
        "isFraud": 0,
    }
    base.update(overrides)
    return base


def _make_batch(n=50, fraud_ratio=0.05):
    rng = np.random.RandomState(42)
    n_fraud = max(1, int(n * fraud_ratio))
    rows = []
    for i in range(n):
        rows.append(_make_transaction(
            TransactionID=100001 + i,
            TransactionDT=86400 + i * 100,
            TransactionAmt=float(rng.choice([49.99, 150.0, 500.0, 999.99, 50.0])),
            ProductCD=rng.choice(["C", "H", "R", "S", "W"]),
            card1=int(rng.randint(1000, 9999)),
            card4=rng.choice(["visa", "mastercard", "discover"]),
            card6=rng.choice(["debit", "credit"]),
            P_emaildomain=rng.choice(["gmail.com", "yahoo.com", "protonmail.com", "hotmail.com"]),
            isFraud=1 if i < n_fraud else 0,
        ))
    return pd.DataFrame(rows)


# ==============================================================================
# PREPROCESSING TESTS
# ==============================================================================
class TestPreprocessingFitTransform:
    """Core fit/transform contract."""

    def test_fit_returns_self(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        result = pp.fit(batch_df)
        assert result is pp
        assert pp._fitted is True

    def test_transform_preserves_row_count(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        out = pp.transform(batch_df, verbose=False)
        assert len(out) == len(batch_df)

    def test_fit_then_transform_is_deterministic(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        out1 = pp.transform(batch_df, verbose=False)
        out2 = pp.transform(batch_df, verbose=False)
        pd.testing.assert_frame_equal(out1, out2)

    def test_transform_adds_time_features(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        out = pp.transform(batch_df, verbose=False)
        assert "hour_of_day" in out.columns
        assert "day_of_week" in out.columns

    def test_product_cd_mapped_to_int(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        out = pp.transform(batch_df, verbose=False)
        assert pd.api.types.is_numeric_dtype(out["ProductCD"])

    def test_email_features_extracted(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        out = pp.transform(batch_df, verbose=False)
        assert "P_emaildomain_vendor_id" in out.columns
        assert "P_emaildomain_is_free" in out.columns
        assert "P_emaildomain_risk_score" in out.columns

    def test_m_columns_encoded(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        out = pp.transform(batch_df, verbose=False)
        for col in ["M1", "M2", "M3"]:
            if col in out.columns:
                assert pd.api.types.is_numeric_dtype(out[col])

    def test_card_columns_encoded(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        out = pp.transform(batch_df, verbose=False)
        assert pd.api.types.is_numeric_dtype(out["card4"])
        assert pd.api.types.is_numeric_dtype(out["card6"])


class TestPreprocessingEdgeCases:
    """Handles NaN, inf, empty inputs."""

    def test_nan_transaction_survives(self, nan_transaction):
        df = pd.DataFrame([nan_transaction])
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        out = pp.transform(df, verbose=False)
        assert len(out) == 1

    def test_inf_values_replaced(self, inf_transaction):
        df = pd.DataFrame([inf_transaction])
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        out = pp.transform(df, verbose=False)
        # inf should be replaced with NaN during cleanup
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.any(np.isinf(out[col].dropna().values)), f"inf found in {col}"

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"])
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        out = pp.transform(df, verbose=False)
        assert len(out) == 0

    def test_single_row(self, single_transaction):
        df = pd.DataFrame([single_transaction])
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        out = pp.transform(df, verbose=False)
        assert len(out) == 1

    def test_missing_email_domain(self):
        tx = _make_transaction(P_emaildomain=None, R_emaildomain=None)
        df = pd.DataFrame([tx])
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        out = pp.transform(df, verbose=False)
        assert "country" in out.columns

    def test_unknown_product_cd(self):
        tx = _make_transaction(ProductCD="Z")
        df = pd.DataFrame([tx])
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        out = pp.transform(df, verbose=False)
        assert out["ProductCD"].iloc[0] == 0  # fallback to 0


# ==============================================================================
# FEATURE ENGINEERING TESTS
# ==============================================================================
class TestFeatureEngineering:
    """Core feature engineering tests."""

    def test_fit_returns_self(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        result = fe.fit(clean, y="isFraud")
        assert result is fe
        assert fe._fitted is True

    def test_transform_adds_features(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)

        # Should have more columns after engineering
        assert len(out.columns) > len(clean.columns)

    def test_uid_created(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)

        # UID may be hashed, but UID_hash should exist
        assert "UID_hash" in out.columns or "UID" in out.columns

    def test_amount_features_created(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)

        assert "TransactionAmt_log" in out.columns
        assert "TransactionAmt_suspicious" in out.columns

    def test_velocity_features_created(self):
        # Need duplicate UIDs (same card1+addr1) so velocity is computed
        rows = [_make_transaction(
            TransactionID=100001 + i,
            TransactionDT=86400 + i * 100,
            card1=1234,  # same card1 for all
            addr1=315.0, # same addr1 for all
        ) for i in range(20)]
        df = pd.DataFrame(rows)
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        clean = pp.transform(df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)

        # At least one velocity column should exist
        velocity_cols = [c for c in out.columns if "velocity" in c.lower()]
        assert len(velocity_cols) > 0

    def test_risk_scores_created(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)

        fraud_rate_cols = [c for c in out.columns if "fraud_rate" in c]
        assert len(fraud_rate_cols) > 0

    def test_row_count_preserved(self, batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(batch_df)
        clean = pp.transform(batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)
        assert len(out) == len(batch_df)

    def test_deterministic_transform(self, small_batch_df):
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(small_batch_df)
        clean = pp.transform(small_batch_df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out1 = fe.transform(clean, verbose=False)
        out2 = fe.transform(clean, verbose=False)
        pd.testing.assert_frame_equal(out1.reset_index(drop=True), out2.reset_index(drop=True))


class TestFeatureEngineeringEdgeCases:
    """Edge cases for feature engineering."""

    def test_nan_amount_transaction(self):
        # addr1 must be present for UID creation in feature engineering
        tx = _make_transaction(TransactionAmt=np.nan)
        df = pd.DataFrame([tx])
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        clean = pp.transform(df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)
        assert len(out) == 1

    def test_zero_amount(self):
        tx = _make_transaction(TransactionAmt=0.0)
        df = pd.DataFrame([tx])
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        clean = pp.transform(df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)
        assert len(out) == 1

    def test_suspicious_amounts(self):
        """Test detection of round numbers and .99 cents."""
        amounts = [100.00, 99.99, 49.95, 123.45]
        rows = [_make_transaction(TransactionID=i, TransactionAmt=a) for i, a in enumerate(amounts)]
        df = pd.DataFrame(rows)
        pp = SentinelPreprocessing(verbose=False)
        pp.fit(df)
        clean = pp.transform(df, verbose=False)

        fe = SentinelFeatureEngineering(verbose=False)
        fe.fit(clean, y="isFraud")
        out = fe.transform(clean, verbose=False)

        assert "TransactionAmt_suspicious" in out.columns
        # $100.00 (exact dollar) should have higher suspicious score than $123.45
        sorted_out = out.sort_values("TransactionID").reset_index(drop=True)
        assert sorted_out.loc[0, "TransactionAmt_suspicious"] >= sorted_out.loc[3, "TransactionAmt_suspicious"]
