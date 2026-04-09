"""
Sentinel exception hierarchy.

All domain-specific errors inherit from ``SentinelError`` so callers
can catch the family with a single ``except SentinelError``.
"""


class SentinelError(Exception):
    """Base exception for the Sentinel fraud-detection library."""


class PreprocessingError(SentinelError):
    """Raised when data cleaning or transformation fails."""


class FeatureEngineeringError(SentinelError):
    """Raised when feature construction fails."""


class InferenceError(SentinelError):
    """Raised when model prediction or calibration fails."""


class ArtifactLoadError(SentinelError):
    """Raised when a required model artifact is missing or corrupt."""


class CalibrationError(SentinelError):
    """Raised when probability calibration fails."""
