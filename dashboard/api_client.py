import requests
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from config import Endpoints, REQUEST_TIMEOUT

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ApiClient")

class SentinelClient:
    """
    The Bridge: Fetches data from the Sentinel Backend (FastAPI).
    converts JSON responses into Pandas DataFrames where appropriate.
    """

    def __init__(self):
        self.session = requests.Session()
        logger.info(f"🔌 Sentinel Client Initialized")

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Internal helper for robust GET requests."""
        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Connection Refused: {url}")
            return None
        except Exception as e:
            logger.error(f"❌ API Error ({url}): {e}")
            return None

    # ==========================================================================
    # 1. SYSTEM HEALTH
    # ==========================================================================
    def get_system_health(self) -> bool:
        """Checks if Backend + Redis are alive."""
        url = Endpoints.build_url(Endpoints.HEALTH)
        data = self._get(url)
        return data is not None and data.get("status") == "healthy"

    def get_system_metrics(self) -> Dict[str, Any]:
        """Fetches CPU/RAM usage."""
        url = Endpoints.build_url(Endpoints.SYSTEM_METRICS)
        return self._get(url) or {}

    # ==========================================================================
    # 2. EXECUTIVE DASHBOARD (KPIs & Charts)
    # ==========================================================================
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Fetches the Business Report (KPI Cards).
        Endpoint: /stats
        """
        url = Endpoints.build_url(Endpoints.STATS)
        return self._get(url) or {}

    def get_financial_timeseries(self) -> pd.DataFrame:
        """
        Fetches data for the 'Savings over Time' chart.
        Endpoint: /exec/series
        """
        url = Endpoints.build_url(Endpoints.TIMESERIES)
        data = self._get(url)
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        try:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.error(f"Timestamp parsing error: {e}")
            return pd.DataFrame()

    # ==========================================================================
    # 3. LIVE STREAMS & TABLES
    # ==========================================================================
    def get_recent_transactions(self, limit: int = 20) -> pd.DataFrame:
        """
        Fetches the live stream of recent transactions.
        Endpoint: /recent
        """
        url = Endpoints.build_url(Endpoints.RECENT_TRANSACTIONS)
        data = self._get(url, params={"limit": limit})
        
        if not data:
            return pd.DataFrame()
            
        return pd.DataFrame(data)

    def get_alerts(self, limit: int = 20) -> pd.DataFrame:
        """
        Fetches only high-risk alerts.
        Endpoint: /alerts
        """
        url = Endpoints.build_url(Endpoints.ALERTS)
        data = self._get(url, params={"limit": limit})
        
        if not data:
            return pd.DataFrame()
            
        return pd.DataFrame(data)

    def get_threshold_optimization_curve(self) -> pd.DataFrame:
        """
        Fetches the threshold vs. total loss data for the optimization plot.
        Endpoint: /exec/threshold-optimization
        """
        # Ensure 'THRESHOLD_OPTIMIZATION' is defined in your Endpoints config
        url = Endpoints.build_url(Endpoints.THRESHOLD_OPTIMIZATION)
        data = self._get(url)
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)

    # ==========================================================================
    # 4. FORENSICS (Drill Down)
    # ==========================================================================
    def get_transaction_detail(self, transaction_id: str) -> Dict[str, Any]:
        """
        Fetches deep-dive data (SHAP values) for a specific ID.
        Endpoint: /transactions/{id}
        """
        url = Endpoints.build_url(Endpoints.TRANSACTION_DETAIL, id=transaction_id)
        return self._get(url) or {}
    
