import requests
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from config import BACKEND_URL, REQUEST_TIMEOUT, ENDPOINTS

logger = logging.getLogger("ApiClient")

class SentinelApiClient:
    """
    Thin Client: Fetches PRE-COMPUTED data from the Sentinel Fraud Backend.
    No heavy calculations are performed here.
    """

    def __init__(self):
        self.base_url = BACKEND_URL
        self.session = requests.Session()
        logger.info(f"🔌 API Client linked to: {self.base_url}")

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Internal helper for robust GET requests."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # We don't log health check failures as 'Errors' to keep logs clean
            if endpoint != ENDPOINTS["health"]:
                logger.error(f"⚠️ API Fetch Failed ({endpoint}): {e}")
            return None

    # ==========================================================================
    # 1. SYSTEM HEALTH
    # ==========================================================================
    def is_backend_alive(self) -> bool:
        """Returns True if the backend is reachable and healthy."""
        data = self._get(ENDPOINTS["health"])
        return data is not None and data.get("status") == "healthy"

    def get_system_metrics(self) -> Dict[str, Any]:
        """Fetches pre-computed CPU/RAM load for the sidebar status."""
        return self._get(ENDPOINTS["metrics"]) or {}

    # ==========================================================================
    # 2. EXECUTIVE DATA (Pre-Computed)
    # ==========================================================================
    def get_executive_stats(self) -> Dict[str, Any]:
        """
        Fetches backend-calculated totals: Net Benefit, Fraud Prevented, etc.
        Backend calculates these over the WHOLE database.
        """
        return self._get(ENDPOINTS["stats"]) or {}

    def get_financial_timeseries(self) -> pd.DataFrame:
        """
        Fetches an array of points for the ROI line chart.
        The backend has already calculated 'cumulative_savings' per hour/minute.
        """
        data = self._get("/executive/timeseries") # Custom endpoint for plot-ready data
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    # ==========================================================================
    # 3. OPERATION & ML DATA (Pre-Computed)
    # ==========================================================================
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Fetches pre-calculated PR-AUC, Drift scores, and Threshold data.
        """
        return self._get(ENDPOINTS["performance"]) or {}

    def get_model_curves(self) -> Dict[str, List[float]]:
        """
        Fetches pre-calculated X and Y coordinates for Precision-Recall curves.
        Dashboard just draws the lines; no math involved.
        """
        return self._get("/ml/curves") or {"precision": [], "recall": [], "thresholds": []}

    # ==========================================================================
    # 4. RAW DATA (For Tables/Forensics Only)
    # ==========================================================================
    def get_transaction_stream(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetches raw transactions for the Forensics table.
        This is the only 'heavy' data, used exclusively for the search/table view.
        """
        data = self._get(ENDPOINTS["recent"], params={"limit": limit})
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

        