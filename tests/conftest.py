"""
Shared test fixtures and factories for Sentinel Fraud Ops.
"""
import pytest
import numpy as np
import pandas as pd
from typing import Dict, List


# ==============================================================================
# TRANSACTION FACTORIES
# ==============================================================================
def make_transaction(**overrides) -> Dict:
    """Factory for a single raw transaction dict."""
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
        "M1": "T",
        "M2": "T",
        "M3": "T",
        "M4": "M0",
        "M5": "F",
        "M6": "T",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows",
        "isFraud": 0,
    }
    base.update(overrides)
    return base


def make_batch(n: int = 50, fraud_ratio: float = 0.05) -> pd.DataFrame:
    """Factory for batch transaction DataFrame with realistic distribution."""
    rng = np.random.RandomState(42)
    n_fraud = max(1, int(n * fraud_ratio))
    n_legit = n - n_fraud

    rows = []
    for i in range(n):
        is_fraud = 1 if i < n_fraud else 0
        rows.append(make_transaction(
            TransactionID=100001 + i,
            TransactionDT=86400 + i * 100,
            TransactionAmt=float(rng.choice([49.99, 150.0, 500.0, 999.99, 50.0])),
            ProductCD=rng.choice(["C", "H", "R", "S", "W"]),
            card1=int(rng.randint(1000, 9999)),
            card4=rng.choice(["visa", "mastercard", "discover"]),
            card6=rng.choice(["debit", "credit"]),
            P_emaildomain=rng.choice(["gmail.com", "yahoo.com", "protonmail.com", "hotmail.com"]),
            isFraud=is_fraud,
        ))
    return pd.DataFrame(rows)


# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture
def single_transaction() -> Dict:
    return make_transaction()


@pytest.fixture
def batch_df() -> pd.DataFrame:
    return make_batch(n=100, fraud_ratio=0.05)


@pytest.fixture
def small_batch_df() -> pd.DataFrame:
    return make_batch(n=10, fraud_ratio=0.2)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


@pytest.fixture
def nan_transaction() -> Dict:
    """Transaction with many NaN / None fields."""
    return make_transaction(
        TransactionAmt=np.nan,
        card4=None,
        P_emaildomain=None,
        R_emaildomain=None,
        DeviceInfo=None,
        DeviceType=None,
        dist1=np.nan,
        addr1=np.nan,
    )


@pytest.fixture
def inf_transaction() -> Dict:
    """Transaction with inf values."""
    return make_transaction(
        TransactionAmt=np.inf,
        dist1=-np.inf,
    )


@pytest.fixture
def evaluator_data():
    """Realistic evaluator inputs."""
    rng = np.random.RandomState(42)
    n = 500
    y_true = rng.binomial(1, 0.035, n)
    # Make probabilities somewhat correlated with labels
    y_prob = np.clip(y_true * rng.uniform(0.4, 0.95, n) + (1 - y_true) * rng.uniform(0.0, 0.3, n), 0, 1)
    amounts = rng.exponential(200, n)
    return y_true, y_prob, amounts
