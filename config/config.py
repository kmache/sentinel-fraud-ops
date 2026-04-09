"""
Centralized Configuration Loader.
Single source of truth for all parameters — loaded from config/params.yaml.
Used by: worker/metrics.py, worker/processor.py, backend/main.py, src/sentinel/evaluation.py
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any

_CONFIG_CACHE: Dict[str, Any] = {}

def _find_config_path() -> Path:
    """Locate params.yaml relative to this file or via env var."""
    env_path = os.getenv("SENTINEL_CONFIG_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Relative to this file: config/params.yaml
    here = Path(__file__).resolve().parent
    candidate = here / "params.yaml"
    if candidate.exists():
        return candidate

    # Fallback: workspace root / config / params.yaml
    root = here.parent
    candidate = root / "config" / "params.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "params.yaml not found. Set SENTINEL_CONFIG_PATH env var or ensure config/params.yaml exists."
    )


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load and cache the full config from params.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE and not force_reload:
        return _CONFIG_CACHE

    path = _find_config_path()
    with open(path, "r") as f:
        _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def get_business_costs() -> Dict[str, float]:
    """Return business cost parameters used for threshold optimization."""
    cfg = load_config()
    costs = cfg.get("evaluation", {}).get("business_costs", {})
    return {
        "cb_fee": float(costs.get("cb_fee", 30.0)),
        "support_cost": float(costs.get("support_cost", 12.50)),
        "churn_factor": float(costs.get("churn_factor", 0.15)),
    }


def get_evaluation_config() -> Dict[str, Any]:
    """Return the full evaluation section."""
    cfg = load_config()
    return cfg.get("evaluation", {})


def get_model_paths() -> Dict[str, str]:
    """Return configured paths for data and models."""
    cfg = load_config()
    return cfg.get("paths", {})


def get_monitoring_config() -> Dict[str, Any]:
    """Return monitoring parameters (PSI thresholds, drift intervals)."""
    cfg = load_config()
    return cfg.get("monitoring", {})


def get_worker_config() -> Dict[str, Any]:
    """Return worker tuning parameters (EMA alpha, refresh intervals)."""
    cfg = load_config()
    return cfg.get("worker", {})


def get_api_config() -> Dict[str, Any]:
    """Return API-specific parameters (timeseries limits)."""
    cfg = load_config()
    return cfg.get("api", {})
