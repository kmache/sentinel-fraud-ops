"""
Model Probability Calibration.
Fits Isotonic Regression or Platt (Sigmoid) scaling on held-out validation predictions.
Ensures output probabilities reflect true fraud likelihood.
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Literal
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import logging
from sentinel.exceptions import CalibrationError

logger = logging.getLogger("SentinelCalibration")


class SentinelCalibrator:
    """
    Standalone calibrator that maps raw model scores to calibrated probabilities.

    Usage:
        # After training, on validation fold:
        calibrator = SentinelCalibrator(method='isotonic')
        calibrator.fit(y_val, y_prob_val)
        calibrator.save(model_dir / 'calibrator.pkl')

        # During inference:
        calibrator = SentinelCalibrator.load(model_dir / 'calibrator.pkl')
        calibrated = calibrator.transform(raw_probs)
    """

    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic"):
        self.method = method
        self._calibrator = None
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "SentinelCalibrator":
        """
        Fit calibrator on validation data.

        Args:
            y_true: Ground truth binary labels.
            y_prob: Raw predicted probabilities from the ensemble.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if len(y_true) < 50:
            logger.warning("Too few samples for calibration (%d). Skipping.", len(y_true))
            self._fitted = False
            return self

        if self.method == "isotonic":
            self._calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._calibrator.fit(y_prob, y_true)
        elif self.method == "sigmoid":
            self._calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self._calibrator.fit(y_prob.reshape(-1, 1), y_true)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True
        logger.info("Calibrator fitted (%s) on %d samples.", self.method, len(y_true))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if not self._fitted or self._calibrator is None:
            return np.asarray(y_prob)

        y_prob = np.asarray(y_prob)

        if self.method == "isotonic" and isinstance(self._calibrator, IsotonicRegression):
            return self._calibrator.transform(y_prob)
        elif self.method == "sigmoid" and isinstance(self._calibrator, LogisticRegression):
            return self._calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return y_prob

    def save(self, path: str) -> None:
        """Persist calibrator to disk."""
        joblib.dump(
            {"method": self.method, "calibrator": self._calibrator, "fitted": self._fitted},
            path,
        )
        logger.info("Calibrator saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "SentinelCalibrator":
        """Load a persisted calibrator."""
        p = Path(path)
        if not p.exists():
            logger.warning("Calibrator not found at %s. Using uncalibrated scores.", path)
            return cls()

        data = joblib.load(p)
        obj = cls(method=data["method"])
        obj._calibrator = data["calibrator"]
        obj._fitted = data["fitted"]
        logger.info("Calibrator loaded from %s (method=%s, fitted=%s)", path, obj.method, obj._fitted)
        return obj

