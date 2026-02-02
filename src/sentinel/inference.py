import joblib
import json
from matplotlib.widgets import EllipseSelector
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Any, List, Optional
import logging
import time
import sklearn
from datetime import datetime
import xgboost as xgb
import sklearn
import pandas
print(f"XGBoost version: {xgb.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pandas.__version__}")
# Setup basic logging (In production, this might go to Datadog/Splunk)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SentinelInference")

class SentinelInference:
    """
    Production Inference Wrapper for the Sentinel Fraud Detection System.
    
    Capabilities:
    1. Ensemble Support: Loads multiple models and applies weighted averaging.
    2. Robustness: explicit categorical casting and schema validation.
    3. Business Logic: Returns Tiered Decisions (Approve, Challenge, Block).
    """

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir (str): Path to the folder containing the model artifacts.
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.features = {}
        self.config = {}
        self.cat_features = []
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads configuration, processors, and model binaries."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.model_dir}")
        logger.info(f"Loading Sentinel artifacts from {self.model_dir}...")
        start_time = time.time()
        
        try:
            config_path = self.model_dir / "production_config.json"
            if not config_path.exists():
                raise FileNotFoundError("production_config.json missing. Did you run select_best_model()?")
            
            with open(config_path, "r") as f: self.config = json.load(f)
            
            self.preprocessor = joblib.load(self.model_dir / 'sentinel_preprocessor.pkl')
            self.engineer = joblib.load(self.model_dir / 'sentinel_engineer.pkl')
            
            cat_path = self.model_dir / 'categorical_features.json'
            if cat_path.exists():
                with open(cat_path, 'r') as f: self.cat_features = json.load(f)
            else:
                logger.warning("categorical_features.json not found. Auto-detection might fail.")

            # Config example: {"weights": {"lgb": 0.7, "xgb": 0.3}, ...}
            weights = self.config.get("weights", {})
            for model_name in weights.keys():
                model_path = self.model_dir / f"{model_name}_model.pkl" 
                if not model_path.exists():
                    raise FileNotFoundError(f"Required model {model_name} not found at {model_path}")
                
                self.models[model_name] = joblib.load(model_path)
                self.features[model_name] = self._get_features(self.models[model_name], model_name)
                logger.info(f"Loaded {model_name} model.")

            elapsed = time.time() - start_time
            logger.info(f"✅ Sentinel Inference initialized in {elapsed:.2f}s ")

        except Exception as e:
            logger.error(f"Failed to initialize Sentinel: {e}")
            raise RuntimeError(f"Artifact loading failed: {e}")


    def predict(self, data: Union[Dict, List, pd.DataFrame], soft_threshold: float = 0.12) -> Dict[str, Any]:
        """
        Main entry point for predictions.
         
        Args:
            data: Dictionary (single) or DataFrame (batch).
            soft_threshold: Probability where we start 'Challenging' (2FA). 
                            Anything above config['threshold'] is a 'Block'.
        
        Returns:
            Dict with probability, action (APPROVE/CHALLENGE/BLOCK), and metadata.
        """
        start_time = time.time()
        
        if isinstance(data, dict): 
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame): 
            df = data.copy()
        else: 
            raise TypeError("data must be a dict, list of dicts, or pandas DataFrame")
        df = df.replace({None: np.nan})
        
        df = self._type_consistency(df)

        if 'TransactionID' not in df.columns:
            print('warning: no TransactionID column found, generating one...') 
            df['TransactionID'] = f"tx_{int(time.time() * 1000)}"

        try:
            df_clean = self.preprocessor.transform(df)
            df_features = self.engineer.transform(df_clean)
            probs = np.zeros(len(df))
            weights = self.config['weights']

            for name, weight in weights.items():
                df_features = self._features_consistency(df_features, name)

                tem_df = df_features[self.features[name]].copy()
                
                if name=='xgb':
                    booster = self.models[name].get_booster()
                    dmatrix = xgb.DMatrix(
                        tem_df.values,
                        feature_names=self.features[name],
                        enable_categorical=True
                    )
                    p =  booster.predict(dmatrix)
                else:
                    p = self.models[name].predict_proba(tem_df)[:, 1]
                probs += (p * weight)

        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            raise RuntimeError(f"Inference pipeline failed: {str(e)}")

        hard_threshold = self.config['threshold']

        actions = [self._get_action(p, soft_threshold, hard_threshold) for p in probs.tolist()]
        isfrauds = [1 if p >= hard_threshold else 0 for p in probs.tolist()]
        y_true = df['isFraud'].values.tolist() if 'isFraud' in df.columns else [0]*len(df)
        tnx_ids = df['TransactionID'].values.tolist()
        fraud_probs = [round(p, 4) for p in probs.tolist()]

        report = {
            "transaction_id": tnx_ids, "probabilities": fraud_probs, "is_frauds": isfrauds, "y_true": y_true, "actions": actions,
            "meta": {
                "threshold": hard_threshold,
                "timestamp": datetime.now().replace(microsecond=0).isoformat(), 
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        }
        return report

    def _get_features(self, model, model_type):
        if model_type == 'xgb':
            features = model.get_booster().feature_names
        elif model_type == 'cb':
            features = model.feature_names_
        elif model_type == 'lgb':
            features = model.feature_name_ 
        else:
            raise ValueError("Unsupported model type")
        return features

    def _features_consistency(self, data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Ensure consistent feature types for inference.
        """
        df = data.copy()
        missing_features = list(set(self.features[model_name]) - set(df.columns.tolist()))
        if missing_features:
            missing_data = {col: np.nan for col in missing_features}
            missing_df = pd.DataFrame(missing_data, index=df.index)
            df = pd.concat([df, missing_df], axis=1)
        return df

    def _type_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure consistent data types for inference.
        """
        INT_COLS = ['TransactionID', 'isFraud', 'TransactionDT', 'card1']
        OBJ_COLS = [
            'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 
            'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 
            'id_37', 'id_38', 'DeviceType', 'DeviceInfo'
            ]
        df = data.copy()
        FLOAT_COLS = [c for c in df.columns if c not in INT_COLS and c not in OBJ_COLS]

        for col in INT_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        for col in OBJ_COLS:
            if col in df.columns:
                df[col] = df[col].astype('object')

        df[FLOAT_COLS] = df[FLOAT_COLS].apply(pd.to_numeric, errors='coerce').astype('float64')

        return df

    def _get_action(self, prob: float, soft: float, hard: float) -> str:
        if prob >= hard: return "BLOCK"
        elif prob >= soft: return "REVIEW"
        return "APPROVE" 


if __name__ == "__main__":
    print("Sentinel Inference Module Loaded")
 