import os
from typing import Dict, Final
from dataclasses import dataclass

# ==============================================================================
# 1. SYSTEM & CONNECTION
# ==============================================================================
ENVIRONMENT: Final = os.getenv("ENVIRONMENT", "development").lower()
BACKEND_URL: Final = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip('/')
REQUEST_TIMEOUT: Final = float(os.getenv("REQUEST_TIMEOUT", "5.0"))

# ==============================================================================
# 2. STANDARDIZED COLUMN NAMES
# ==============================================================================
class StandardColumns:
    """Centralized column names for type-safe access."""
    # Transaction Data
    AMOUNT = "TransactionAmt"
    SCORE = "score"
    IS_FRAUD = "is_fraud"
    TRANSACTION_ID = "transaction_id"
    TIMESTAMP = "timestamp"
    
    # Context Data
    DEVICE = "device"
    NETWORK = "card4"
    EMAIL = "P_emaildomain"
    IP = "ip_address"
    PRODUCT = "ProductCD"  
    
    # Business Metrics
    NET_SAVINGS = "net_savings"
    FRAUD_PREVENTED = "fraud_stopped_val"
    FALSE_POSITIVE_LOSS = "false_positive_loss"
    PRECISION = "precision"
    RECALL = "recall"

# Mapping: { Raw_Backend_Field : StandardColumns_Value }
COLUMN_MAPPING: Dict[str, str] = {
    # IEEE-CIS / Raw Dataset columns
    'TransactionAmt':       StandardColumns.AMOUNT,
    'TransactionID':        StandardColumns.TRANSACTION_ID,
    'TransactionDT':        StandardColumns.TIMESTAMP,
    'DeviceType':           StandardColumns.DEVICE,
    'card4':                StandardColumns.NETWORK,
    'P_emaildomain':        StandardColumns.EMAIL,
    'addr1':                StandardColumns.IP,
    'ProductCD':            StandardColumns.PRODUCT,
    
    # Cleaned / Pydantic Schema
    'amount':               StandardColumns.AMOUNT,
    'transaction_id':       StandardColumns.TRANSACTION_ID,
    'timestamp':            StandardColumns.TIMESTAMP,
    'risk_score':           StandardColumns.SCORE,
    'prediction':           StandardColumns.SCORE, 
    'is_fraud':             StandardColumns.IS_FRAUD,
    'product':              StandardColumns.PRODUCT,
    
    # Stats
    'net_savings':          StandardColumns.NET_SAVINGS,
    'fraud_stopped_val':    StandardColumns.FRAUD_PREVENTED,
    'false_positive_loss':  StandardColumns.FALSE_POSITIVE_LOSS
}

# ==============================================================================
# 3. UI CONFIGURATION
# ==============================================================================
@dataclass(frozen=True)
class UIConfig:
    APP_TITLE = "Sentinel Fraud Detection"
    APP_ICON = "🛡️"
    HIGH_RISK_LIMIT: float = 0.8
    MEDIUM_RISK_LIMIT: float = 0.4
    COLOR_RISK_HIGH: str = "#EF553B"
    COLOR_RISK_MEDIUM: str = "#FFA15A"
    COLOR_RISK_LOW: str = "#00CC96"

    @classmethod
    def get_risk_color(cls, score: float) -> str:
        if score is None: return cls.COLOR_RISK_LOW
        if score >= cls.HIGH_RISK_LIMIT: return cls.COLOR_RISK_HIGH
        elif score >= cls.MEDIUM_RISK_LIMIT: return cls.COLOR_RISK_MEDIUM
        return cls.COLOR_RISK_LOW

# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================
class Endpoints:
    HEALTH = "/health"
    SYSTEM_METRICS = "/metrics"
    STATS = "/stats"
    TIMESERIES = "/exec/series"
    RECENT_TRANSACTIONS = "/recent"
    ALERTS = "/alerts"
    TRANSACTION_DETAIL = "/transactions/{id}"
    THRESHOLD_OPTIMIZATION = "/exec/threshold-optimization"
    GLOBAL_FEATURE_IMPORTANCE = "/exec/global-feature-importance"
    FEATURE_DRIFT = "/exec/feature-drift"
    PERFORMANCE_LOOKUP = "/exec/performance-lookup"
    CALIBRATION_DATA = "/exec/calibration"
    
    @classmethod
    def build_url(cls, endpoint: str, **params) -> str:
        if params:
            try:
                endpoint = endpoint.format(**params)
            except KeyError:
                pass
        return f"{BACKEND_URL}{endpoint}" 

