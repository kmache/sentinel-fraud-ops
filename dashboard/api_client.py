"""
ROLE: API Client (Adapter)
RESPONSIBILITIES:
1.  Abstracts HTTP requests to the Backend Service.
2.  Handles connection errors and timeouts gracefully.
3.  Converts raw JSON responses into Pandas DataFrames for Streamlit.
4.  Provides fallback data structures to prevent UI crashes.
"""
import os
import requests
import logging
import pandas as pd
from typing import Dict, Any, Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Default to 'backend' service name in Docker Compose network
API_BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000")
REQUEST_TIMEOUT = 3  # Seconds

logger = logging.getLogger("ApiClient")

class SentinelApiClient:
    """
    Client for interacting with the Sentinel Fraud Ops Backend.
    """

    def __init__(self):
        self.base_url = API_BASE_URL
        self.session = requests.Session()
        logger.info(f"🔌 API Client initialized pointing to: {self.base_url}")

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Internal helper to perform GET requests with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Connection Error: Could not reach {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"⏳ Timeout: Backend did not respond in {REQUEST_TIMEOUT}s")
            return None
        except Exception as e:
            logger.error(f"⚠️ API Error ({endpoint}): {e}")
            return None

    # ==========================================================================
    # SYSTEM HEALTH
    # ==========================================================================
    def is_backend_alive(self) -> bool:
        """Checks if the backend is reachable and Redis is connected."""
        data = self._get("/health")
        return data is not None and data.get("status") == "healthy"

    def get_system_metrics(self) -> Dict[str, Any]:
        """Fetches CPU and Memory usage of the backend service."""
        default = {"memory_usage_mb": 0, "cpu_usage_percent": 0, "redis_connected": False}
        data = self._get("/metrics")
        return data if data else default

    # ==========================================================================
    # DATA & STATISTICS
    # ==========================================================================
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Fetches aggregated statistics (Fraud Rate, Total Count, etc.).
        Returns zeroed-out dict on failure.
        """
        default = {
            "total_processed": 0,
            "fraud_detected": 0,
            "legit_transactions": 0,
            "fraud_rate": 0.0,
            "queue_depth": 0,
            "updated_at": "N/A"
        }
        data = self._get("/stats")
        return data if data else default

    def get_recent_transactions(self, limit: int = 20) -> pd.DataFrame:
        """
        Fetches the latest transactions stream.
        Returns: Pandas DataFrame suitable for Streamlit display.
        """
        data = self._get("/recent", params={"limit": limit})
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Data Formatting for UI
        if not df.empty:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            cols = ['timestamp', 'transaction_id', 'amount', 'score', 'is_fraud', 'action']
            existing_cols = [c for c in cols if c in df.columns]
            extra_cols = [c for c in df.columns if c not in cols]
            df = df[existing_cols + extra_cols]

        return df

    def get_fraud_alerts(self, limit: int = 10) -> pd.DataFrame:
        """
        Fetches only high-risk transactions (Fraud Alerts).
        Returns: Pandas DataFrame.
        """
        data = self._get("/alerts", params={"limit": limit})
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    
    