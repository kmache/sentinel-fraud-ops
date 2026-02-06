import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Any, List
import logging
import time
from datetime import datetime
import xgboost as xgb

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
    def __init__(self, model_dir: str, verbose:bool=True):
        """
        Args:
            model_dir (str): Path to the folder containing the model artifacts.
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.features = {}
        self.config = {}
        self._load_artifacts()

        self.verbose = verbose

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
            Dict with probability, action (APPROVE/REVIEW/BLOCK), and metadata.
        """
        start_time = time.time()
        df = self._to_df(data)
        df = self._type_consistency(df)

        try:
            df_clean = self.preprocessor.transform(df, verbose=self.verbose)
            df_features = self.engineer.transform(df_clean, verbose=self.verbose)

            probs = np.zeros(len(df))
            weights = self.config['weights']

            for name, weight in weights.items(): 

                model_cols = self.features[name]
                missing = list(set(model_cols) - set(df_features.columns.tolist()))
                if missing:
                    for c in missing: df_features[c] = np.nan

                tem_df = df_features[model_cols].copy()
                 
                if name=='xgb':
                    dmatrix = xgb.DMatrix(tem_df, feature_names=model_cols, enable_categorical=True)
                    p = self.models[name].get_booster().predict(dmatrix)
                else:
                    p = self.models[name].predict_proba(tem_df)[:, 1]
                probs += (p * weight)
                
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            raise RuntimeError(f"Inference pipeline failed: {str(e)}")

        hard_threshold = self.config['threshold']

        actions = [self._get_action(p, soft_threshold, hard_threshold) for p in probs.tolist()]
        isfrauds = [1 if p >= hard_threshold else 0 for p in probs.tolist()]
        fraud_probs = [round(p, 4) for p in probs.tolist()] 

        tnx_ids = df['TransactionID'].astype(str).tolist()
        y_true = df['isFraud'].tolist() if 'isFraud' in df.columns else [0]*len(df)
        
        data4dashboard = self._extract_data_for_dashboard(df, df_features)

        report = {
            "transaction_id": tnx_ids, 
            "probabilities": fraud_probs, 
            "is_frauds": isfrauds, 
            "y_true": y_true, 
            "actions": actions,
            "dashboard_data": data4dashboard,
            "meta": {
                "model_weights": weights,
                "threshold": hard_threshold,
                "timestamp": datetime.now().replace(microsecond=0).isoformat(), 
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        }
        return report


    def explain(self, data, model_name='xgb', n=10):
        """
        Calculates SHAP values using Native XGBoost (Fastest Method).
        """
        if model_name != 'xgb':
            raise ValueError("Only 'xgb' supported for SHAP at this time.")

        # Re-run engineering (Necessary as this runs in a separate thread usually)
        df = self._to_df(data)
        df = self._type_consistency(df)
        df_clean = self.preprocessor.transform(df, verbose=self.verbose)
        df_features = self.engineer.transform(df_clean, verbose=self.verbose)
        
        feature_names = self.features[model_name]
        
        # Ensure columns exist
        missing = list(set(feature_names) - set(df_features.columns.tolist()))
        if missing:
            for c in missing: df_features[c] = np.nan
            
        temp_df = df_features[feature_names].copy()
        dmatrix = xgb.DMatrix(temp_df, feature_names=feature_names, enable_categorical=True)
        shap_matrix = self.models[model_name].get_booster().predict(dmatrix, pred_contribs=True)
        
        batch_explanations = []
        
        for i in range(len(temp_df)):
            feature_impacts = shap_matrix[i][:-1]
            raw_values = temp_df.iloc[i].values
            
            impacts_list = []
            for name, val, impact in zip(feature_names, raw_values, feature_impacts):
                if abs(impact) > 1e-4:
                    impacts_list.append({
                        "feature": name,
                        "value": str(val),
                        "impact": round(float(impact), 4)
                    })
            
            top_n = sorted(impacts_list, key=lambda x: abs(x['impact']), reverse=True)[:n]
            batch_explanations.append(top_n)

        return batch_explanations


    def _to_df(self, data):
        if isinstance(data, dict): 
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame): 
            df = data.copy()
        else: 
            raise TypeError("data must be a dict, list of dicts, or pandas DataFrame")
        df = df.replace({None: np.nan})
        return df

    def _get_features(self, model, model_type):
        if model_type == 'xgb':
            features = model.get_booster().feature_names
        elif model_type == 'cb':
            features = model.feature_names_
        elif model_type == 'lgb':
            features = model.feature_name_ 
        else:
            raise ValueError("Unsupported model type")
        return list(features)

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

    def _extract_data_for_dashboard(self, data_original: pd.DataFrame, df_features: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract data for dashboard.
        """
        DASHBOARD_FEATURES = [
            'P_emaildomain', 'TransactionAmt', 'ProductCD','TransactionAmt_log', 'TransactionAmt_suspicious', 'cents_value',
            'country', 'composite_risk_score', 'DeviceType', 'os_browser_combo', 
            'UID_velocity_1h', 'UID_velocity_12h', 'UID_velocity_24h',
            'multi_entity_sharing', 'card_email_combo', 'device_info_combo', 
            'P_emaildomain_risk_score', 'email_match_status', 'P_emaildomain_is_free',
            'ProductCD_switch', 'user_amt_zscore', 'Amt_div_card1_mean', 
            'hour_of_day', 'day_of_week', 'time_gap_anomaly',
            'screen_area', 'addr1_fraud_rate', 'addr1_degree'
        ]

        orinigal_cols = [c for c in data_original.columns if c in DASHBOARD_FEATURES]
        feature_cols = [col for col in DASHBOARD_FEATURES if col not in orinigal_cols]

        missing_cols = [c for c in DASHBOARD_FEATURES if c not in data_original.columns and c not in df_features.columns]
        
        for col in missing_cols:
            df_features[col] = np.nan

        export_df = data_original[orinigal_cols].copy()
        eng_df = df_features[feature_cols].copy()
        dash_df = pd.concat([export_df, eng_df], axis=1)

        records = dash_df.replace({np.nan: None}).to_dict('records')

        return records

    def _get_action(self, prob: float, soft: float, hard: float) -> str:
        if prob >= hard: return "BLOCK" 
        elif prob >= soft: return "REVIEW"
        return "APPROVE" 

if __name__ == "__main__":
    print("Sentinel Inference Module Loaded")
 