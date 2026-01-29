import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Any, List, Optional
import gc
import logging
import time
from datetime import datetime

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
            
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
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
                feat_path = self.model_dir / f"{model_name}_features.json"
                if not model_path.exists() or not feat_path.exists():
                    raise FileNotFoundError(f"Required model {model_name} or features not found at {model_path} or {feat_path}")
                
                self.models[model_name] = joblib.load(model_path)
                with open(feat_path, 'r') as f: features = json.load(f)
                self.features[model_name] = features
                logger.info(f"Loaded {model_name} model.")

            elapsed = time.time() - start_time
            logger.info(f"✅ Sentinel Inference initialized in {elapsed:.2f}s ")

        except Exception as e:
            logger.error(f"Failed to initialize Sentinel: {e}")
            raise RuntimeError(f"Artifact loading failed: {e}")
        

    def predict(self, data: Union[Dict, pd.DataFrame], soft_threshold: float = 0.12) -> Dict[str, Any]:
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
            is_batch = False
        else:
            df = data.copy()
            is_batch = True

        if 'TransactionID' not in df.columns:
            print('warning: no TransactionID column found, generating one...')
            df['TransactionID'] = f"tx_{int(time.time() * 1000)}"

        y_true = df['isFraud'] if 'isFraud' in df.columns else 0

        try:
            df_clean = self.preprocessor.transform(df)
            
            self.df_features = self.engineer.transform(df_clean)

            final_probs = np.zeros(len(self.df_features))
            weights = self.config['weights']
            
            for name, weight in weights.items():
                tem_df = self.df_features[self.features[name]]
                for col in self.cat_features:
                    if col in tem_df.columns:
                        tem_df[col] = tem_df[col].astype('category')
                p = self.models[name].predict_proba(tem_df)[:, 1]
                final_probs += p * weight
                del tem_df
                gc.collect()
            hard_threshold = self.config['threshold']

            if not is_batch:
                prob = float(final_probs[0])

                action = self._get_action(prob, soft_threshold, hard_threshold)
                is_fraud = 1 if prob >= hard_threshold else 0
                return {
                    "transaction_id": data.get("TransactionID", f"tx_{int(time.time() * 1000)}"),
                    "probability": round(prob, 4),
                    "is_fraud": is_fraud,
                    "y_true": y_true,
                    "action": action,
                    "meta": {
                        "model_version": self.config.get("selected_model", "ensemble"),
                        "threshold_used": hard_threshold,
                        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
                }

            else:
                actions = [self._get_action(p, soft_threshold, hard_threshold) for p in final_probs]
                is_frauds = [1 if p >= hard_threshold else 0 for p in final_probs]
                return {
                    "transaction_id": [data.get("TransactionID", f"tx_{int(time.time() * 1000) + i}") for i in range(len(final_probs))],
                    "probabilities": np.round(final_probs, 4).tolist(),
                    "is_frauds": is_frauds,
                    "y_true": y_true,
                    "actions": actions,
                    "meta": {
                        "batch_size": len(df),
                        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
                        "latency_ms": int((time.time() - start_time) * 1000)
                    }
                }

        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            raise RuntimeError(f"Inference pipeline failed: {str(e)}")


    def _get_feat4board(self, data: Union[Dict, pd.DataFrame]=None, 
                        features: List[str]=[
                            'TransactionAmt',      
                            'ProductCD',
                            'card_email_combo_fraud_rate',      
                            'P_emaildomain', 'R_emaildomain_is_free',  
                            'UID_velocity_24h',   
                            'dist1',              
                            'addr1', 'card1_freq_enc',                        
                            'D15',                
                            'device_vendor',        
                            'C13', 'C1', 'C14', 'UID_vel'
                        ]) -> Dict[str, Any]:
        """
        Get features for dashboard, prioritizing explainability and impact.
        """
        if data is None:
            data = self.df_features

        feat4board = {}
        if 'TransactionDT' in data:
             feat4board['hour_of_day'] = (data['TransactionDT'] // 3600) % 24

        for col in features:
            if col in data: 
                feat4board[col] = data[col].values.tolist()
                     
        return feat4board

    @staticmethod
    def _get_action(prob: float, soft: float, hard: float) -> str:
        if prob >= hard: return "BLOCK"
        elif prob >= soft: return "CHALLENGE"
        return "APPROVE" 


if __name__ == "__main__":
    print("Sentinel Inference Module Loaded")
