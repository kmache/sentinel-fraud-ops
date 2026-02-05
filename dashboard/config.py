import os
from typing import Dict, Final
from dataclasses import dataclass
from urllib.parse import urljoin

# ==============================================================================
# 1. SYSTEM & CONNECTION
# ==============================================================================
ENVIRONMENT: Final = os.getenv("ENVIRONMENT", "development").lower()
BACKEND_URL: Final = os.getenv("BACKEND_URL", "http://backend:8000").rstrip('/')
REQUEST_TIMEOUT: Final = float(os.getenv("REQUEST_TIMEOUT", "10.0"))
MAX_RETRIES: Final = int(os.getenv("MAX_RETRIES", "3"))

# Quick validation
if ENVIRONMENT not in {"development", "staging", "production"}:
    raise ValueError(f"Invalid ENVIRONMENT: {ENVIRONMENT}")

# ==============================================================================
# 2. STANDARDIZED COLUMN NAMES (Type Safety)
# ==============================================================================
class StandardColumns:
    """Centralized column names for type-safe access across the dashboard."""
    # Core transaction
    AMOUNT = "amount"
    SCORE = "score"
    IS_FRAUD = "is_fraud"
    TRANSACTION_ID = "transaction_id"
    TIMESTAMP = "timestamp"
    
    # Device & network
    DEVICE = "device"
    NETWORK = "network"
    EMAIL = "email"
    
    # System metrics
    LATENCY = "latency_ms"
    PRODUCT = "product"
    
    # Derived/calculated (optional)
    RISK_LEVEL = "risk_level"  # "high", "medium", "low"
    AMOUNT_BUCKET = "amount_bucket"

# Mapping: {Raw_Backend_Field : StandardColumns_Value}
COLUMN_MAPPING: Dict[str, str] = {
    'TransactionAmt':       StandardColumns.AMOUNT,
    'composite_risk_score': StandardColumns.SCORE,
    'risk_score':           StandardColumns.SCORE,  # Alternative name
    'ground_truth':         StandardColumns.IS_FRAUD,
    'TransactionID':        StandardColumns.TRANSACTION_ID,
    'TransactionDT':        StandardColumns.TIMESTAMP,  # If available
    'DeviceType':           StandardColumns.DEVICE,
    'card4':                StandardColumns.NETWORK,
    'P_emaildomain':        StandardColumns.EMAIL,
    'ProductCD':            StandardColumns.PRODUCT,
    'processing_time_ms':   StandardColumns.LATENCY
}

# Reverse mapping for when we need to send data back
REVERSE_MAPPING: Dict[str, str] = {v: k for k, v in COLUMN_MAPPING.items()}

# Column display types for rendering
COLUMN_TYPES: Dict[str, str] = {
    StandardColumns.AMOUNT: 'currency',
    StandardColumns.SCORE: 'percentage',
    StandardColumns.LATENCY: 'duration_ms',
    StandardColumns.IS_FRAUD: 'boolean',
    StandardColumns.TIMESTAMP: 'datetime',
}

# ==============================================================================
# 3. UI & PERFORMANCE SETTINGS
# ==============================================================================
@dataclass(frozen=True)
class UIConfig:
    """Dashboard UI and performance configuration."""
    # Refresh logic
    DEFAULT_REFRESH_RATE: int = int(os.getenv("UI_REFRESH_RATE", "10"))
    GLOBAL_TTL: int = int(os.getenv("GLOBAL_TTL", "7200"))  # 2 hours
    
    # Table settings
    TABLE_PAGE_SIZE: int = int(os.getenv("TABLE_PAGE_SIZE", "20"))
    MAX_ROWS_PER_PAGE: int = 100  # Hard limit for performance
    
    # Risk thresholds for UI coloring
    HIGH_RISK_LIMIT: float = 0.8
    MEDIUM_RISK_LIMIT: float = 0.5
    
    # Chart defaults
    CHART_HEIGHT: int = 350
    COLOR_RISK_HIGH: str = "#EF4444"    # Red-500
    COLOR_RISK_MEDIUM: str = "#F59E0B"  # Amber-500
    COLOR_RISK_LOW: str = "#10B981"     # Emerald-500
    
    # Refresh rate options for dropdown
    REFRESH_OPTIONS = [5, 10, 30, 60, 300]  # 5s, 10s, 30s, 1m, 5m
    
    @classmethod
    def get_risk_color(cls, score: float) -> str:
        """Get color based on risk score."""
        if score >= cls.HIGH_RISK_LIMIT:
            return cls.COLOR_RISK_HIGH
        elif score >= cls.MEDIUM_RISK_LIMIT:
            return cls.COLOR_RISK_MEDIUM
        return cls.COLOR_RISK_LOW

# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================
class Endpoints:
    """API endpoint registry with URL building."""
    HEALTH = "/health"
    STATS = "/stats"
    TIMESERIES = "/exec/series"
    PERFORMANCE = "/ml/report"
    STREAM = "/recent"
    
    # Optional additional endpoints
    TRANSACTION_DETAIL = "/transactions/{id}"
    BATCH_STATS = "/stats/batch"
    
    @classmethod
    def build_url(cls, endpoint_name: str, **params) -> str:
        """Build full URL for an endpoint."""
        endpoint = getattr(cls, endpoint_name.upper(), None)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        
        # Replace path parameters
        if params:
            endpoint = endpoint.format(**params)
        
        return urljoin(BACKEND_URL, endpoint)
    
    @classmethod
    def get_all(cls) -> Dict[str, str]:
        """Get all endpoints as dict."""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and isinstance(v, str)}

# ==============================================================================
# 5. DASHBOARD MODULES CONFIG
# ==============================================================================
class DashboardModules:
    """Configuration for individual dashboard modules."""
    ENABLED_MODULES = {
        "kpi_summary": True,
        "risk_trend": True,
        "performance_metrics": True,
        "forensics_table": True,
        "alerts_feed": bool(os.getenv("ENABLE_ALERTS", "true")),
    }
    
    # Module-specific settings
    FORENSICS_TABLE_LIMIT = int(os.getenv("FORENSICS_LIMIT", "100"))
    ALERTS_FEED_LIMIT = 20
    KPI_REFRESH_INTERVAL = 30  # More frequent for KPIs

# ==============================================================================
# 6. QUICK VALIDATION & DEBUG
# ==============================================================================
def debug_config():
    """Print config for debugging (development only)."""
    if ENVIRONMENT == "development":
        print(f"🚀 Dashboard Config:")
        print(f"   Environment: {ENVIRONMENT}")
        print(f"   Backend URL: {BACKEND_URL}")
        print(f"   Timeout: {REQUEST_TIMEOUT}s")
        print(f"   UI Refresh: {UIConfig.DEFAULT_REFRESH_RATE}s")
        print(f"   Mapped Columns: {len(COLUMN_MAPPING)}")
        print(f"   Enabled Modules: {sum(DashboardModules.ENABLED_MODULES.values())}")

# Auto-debug in development
if __name__ == "__main__":
    debug_config()