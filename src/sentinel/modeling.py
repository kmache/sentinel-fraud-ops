from statistics import median
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
import json
import gc
import yaml
import optuna
from pathlib import Path
from typing import Dict, Optional, List, Union

# Internal Imports
from sentinel.preprocessing import SentinelPreprocessing
from sentinel.features import SentinelFeatureEngineering
from sentinel.evaluation import SentinelEvaluator

class SentinelTrainer:
    """
    Orchestrates Data Prep, Optuna Optimization, Training, and Business Reporting.
    """
    def __init__(self, 
                 config_path: str = "config/params.yaml",
                 experiment_name: str = "prod_v1",
                 base_path: str = "."):
        
        self.base_path = Path(base_path)
        self.config = self._load_config(config_path)
        self.eval_config = self.config.get('evaluation', {})
        self.experiment_name = experiment_name
        
        # Setup Paths
        self.data_dir = self.base_path / self.config.get('paths', {}).get('data', 'data/raw')
        self.models_dir = self.base_path / self.config.get('paths', {}).get('models', f'models/{experiment_name}')
        self.artifacts_dir = self.models_dir
        
        for p in [self.data_dir, self.models_dir]:
            p.mkdir(parents=True, exist_ok=True)
            
        # Data Placeholders
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_amounts = None
        self.test_amounts = None
        self.cat_cols = []
        self.train_raw = None
        self.test_raw = None

        # --- DEFAULT SEARCH SPACES (For Optuna) ---
        self.default_search_spaces = {
            'lgb': {
                'n_estimators': ('int', 1000, 5000),
                'learning_rate': ('float', 0.005, 0.2, True),
                'num_leaves': ('int', 20, 256),
                'max_depth': ('int', 5, 15),
                'subsample': ('float', 0.5, 1.0),
                'colsample_bytree': ('float', 0.5, 1.0),
                'reg_alpha': ('float', 1e-8, 10.0, True),
                'reg_lambda': ('float', 1e-8, 10.0, True),
                'min_child_samples': ('int', 10, 100)
                    },

            'cb': {
                'iterations': ('int', 1000, 5000),
                'learning_rate': ('float', 0.005, 0.2, True),
                'depth': ('int', 4, 10),
                'l2_leaf_reg': ('float', 1.0, 30.0, True),
                'subsample': ('float', 0.5, 1.0),
                'random_strength': ('float', 1e-9, 10.0, True),

            },

            'xgb': {
                'n_estimators': ('int', 1000, 5000),
                'learning_rate': ('float', 0.005, 0.2, True),
                'max_depth': ('int', 3, 10),
                'min_child_weight': ('int', 1, 10),
                'subsample': ('float', 0.5, 1.0),
                'colsample_bytree': ('float', 0.5, 1.0),
                'gamma': ('float', 1e-8, 5.0, True),
                'reg_alpha': ('float', 1e-8, 10.0, True),
                'reg_lambda': ('float', 1e-8, 10.0, True)
            }
        }

    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML file."""
        path = self.base_path / path
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ Config not found at {path}. Using defaults.")
            return {"evaluation": {"max_fpr": 0.01, "business_costs": {}}}

    # =========================================================================
    # 1. DATA PREPARATION
    # =========================================================================
    def prepare_data(self, df: pd.DataFrame, train_size: float = 0.85, 
                     nan_thresh: float = 0.99, corr_thresh: float = 0.98, 
                     cat_cols: Optional[List[str]] = None, verbose: bool = True):
        """
        Full pipeline: Temporal Split -> Preprocessing -> X/y Split -> Feature Engineering.
        """
        if not 0 < train_size < 1: raise ValueError(f"train_size must be between 0 and 1, got {train_size}")

        required_cols = ['TransactionDT', 'isFraud', 'TransactionAmt']
        missing = [c for c in required_cols if c not in df.columns]
        if missing: raise ValueError(f"Missing required columns: {missing}")
        if verbose: print(f"\n🚀 Starting Data Preparation (Train Split={train_size})")
        
        df_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
        train_idx = int(len(df_sorted) * train_size)
        
        train_raw = df_sorted.iloc[:train_idx].copy()
        test_raw = df_sorted.iloc[train_idx:].copy()

        self.train_raw = train_raw
        self.test_raw = test_raw
        
        self.train_amounts = train_raw['TransactionAmt'].values
        self.test_amounts = test_raw['TransactionAmt'].values
        
        test_raw.to_csv(self.data_dir / 'test_raw.csv', index=False)
        if verbose: 
            print(f"   Train size: {len(train_raw):,} ({train_size*100:.1f}%)")
            print(f"   Test size:  {len(test_raw):,} ({(1-train_size)*100:.1f}%)")

        if verbose: print(" \n  ⚙️  Running SentinelPreprocessing...")
        preprocessor = SentinelPreprocessing(start_date='2017-11-30', nan_threshold=nan_thresh, correlation_threshold=corr_thresh, verbose=verbose)
        preprocessor.fit(train_raw, y='isFraud')

        joblib.dump(preprocessor, self.artifacts_dir / 'sentinel_preprocessor.pkl')
            
        X_train_proc_raw = preprocessor.transform(train_raw)
        X_test_proc_raw = preprocessor.transform(test_raw)
        
        def split_X_y(processed_df, raw_df):
            y = raw_df['isFraud'].copy()
            X = processed_df.drop(columns=['isFraud'], errors='ignore')
            if 'TransactionDT' not in X.columns:
                print("⚠️ Warning: TransactionDT not found in processed_df, adding from raw_df")
                X['TransactionDT'] = raw_df['TransactionDT']
            return X, y

        X_train_split, self.y_train = split_X_y(X_train_proc_raw, train_raw)
        X_test_split, self.y_test = split_X_y(X_test_proc_raw, test_raw)

        if verbose: print("   🛠️  Running SentinelFeatureEngineering...")
        engineer = SentinelFeatureEngineering(time_col='TransactionDT', target_col='isFraud', smoothing=500.0, verbose=verbose)
        engineer.fit(X_train_split, self.y_train)
        
        joblib.dump(engineer, self.artifacts_dir / 'sentinel_engineer.pkl')
            
        self.X_train = engineer.transform(X_train_split)
        self.X_test = engineer.transform(X_test_split)
        
        try:
            pd.testing.assert_index_equal(self.X_train.index, self.y_train.index, obj="Train Index")
            pd.testing.assert_index_equal(self.X_test.index, self.y_test.index, obj="Test Index")
        except AssertionError as e:
            print(f"\n❌ CRITICAL ERROR: Feature/Target Misalignment detected!")
            raise e
        
        if cat_cols:
            self.cat_cols = [c for c in cat_cols if c in self.X_train.columns]
        else:
            print("\n   ℹ️  Warning: No categorical columns provided, automatically detected 'category' dtype columns.")
            self.cat_cols = [c for c in self.X_train.columns if self.X_train[c].dtype.name == 'category']
        
        with open(self.artifacts_dir / 'categorical_features.json', 'w') as f:
            json.dump(self.cat_cols, f) 
                
        del df_sorted, train_raw, test_raw, X_train_proc_raw, X_test_proc_raw, X_train_split, X_test_split
        gc.collect()
        
        if verbose: print(f"✅ Ready. Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    # =========================================================================
    # 2. TRAINING ENGINE
    # =========================================================================
    def train_model(self, model_type='lgb', params=None, n_folds=5, 
            top_n_features: Union[int, List[str], None] = None,
            full_data=False, cols_to_drop=None, save_artifacts=True, find_best_threshold='cost'):
        """
        Trains final model using TimeSeriesSplit validation.
        Optimizes Threshold based on Business Rules (FPR/Profit).
        """
        print(f"\n🚀 Training {model_type.upper()} Model...")

        available_features = list(self.X_train.columns)
        if cols_to_drop:
            available_features = [c for c in available_features if c not in cols_to_drop]

        feature_list = available_features
        if isinstance(top_n_features, list):
            feature_list = [f for f in top_n_features if f in available_features]
        elif isinstance(top_n_features, int):
            print(f"\n🔄 Auto-selecting Top {top_n_features} features via Quick-CV...")
            temp_res = self._train_loop(model_type, params, available_features, full_data=False, n_folds=3, verbose=True)
            imp_df = temp_res['importances']
            avg_imp = imp_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
            feature_list = avg_imp.head(top_n_features).index.tolist()
            print(f"   ✅ Selected Top {len(feature_list)} features.")

        print(f"\n🚀 Training Final {model_type.upper()} on {len(feature_list)} features | Full Data: {full_data}")
        result = self._train_loop(model_type, params, feature_list, full_data=full_data, n_folds=n_folds, find_best_threshold=find_best_threshold, verbose=True)

        if save_artifacts: self._save_artifacts(result['model'], model_type, result['threshold'], feature_list)
        return result

    def _train_loop(self, model_type, params, feature_list, full_data=False, n_folds=5, find_best_threshold='cost', verbose=True):
        """
        Internal training loop for model training.
        """
        if verbose: print(f"\n🚀 Training {model_type.upper()} Model...")

        X_sub = self.X_train[feature_list].copy()
        X_test_sub = self.X_test[feature_list].copy()

        curr_cats = [c for c in self.cat_cols if c in feature_list]
        for c in curr_cats:
            X_sub[c] = X_sub[c].astype('category')
            X_test_sub[c] = X_test_sub[c].astype('category')

        if params is None:
            params = self.config.get('models', {}).get(model_type, {}).get('params', {})

        if full_data:
            n_samples = len(X_sub)
            train_idx_len = int(n_samples * 0.85)
            train_idx = np.arange(0, train_idx_len)
            val_idx = np.arange(train_idx_len, n_samples)
            split_generator = [(train_idx, val_idx)] 
            actual_n_folds = 1
        else:
            folds = TimeSeriesSplit(n_splits=n_folds)
            split_generator = folds.split(X_sub)
            actual_n_folds = n_folds
            
        test_preds_accum = np.zeros(len(self.X_test))
        feature_importance_df = pd.DataFrame()
        best_model = None
        
        for fold, (train_idx, val_idx) in enumerate(split_generator):
            print(f"   -> Fold {fold+1}/{actual_n_folds}...")
            
            X_tr, y_tr = X_sub.iloc[train_idx], self.y_train.iloc[train_idx]
            X_val, y_val = X_sub.iloc[val_idx], self.y_train.iloc[val_idx]
            
            if verbose: print(f"   --- Size: Train={len(X_tr)}, Val={len(X_val)} ---")

            model = self._get_model_instance(model_type, params, y_tr)
            imp = np.zeros(len(feature_list))

            fold_test_preds = None

            if model_type == 'lgb':
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', categorical_feature=curr_cats, callbacks=[lgb.early_stopping(50, verbose=False)])
                fold_test_preds = model.predict_proba(X_test_sub, num_iteration=model.best_iteration_)[:, 1]
            
            elif model_type == 'cb':
                train_pool = Pool(X_tr, y_tr, cat_features=curr_cats)
                val_pool = Pool(X_val, y_val, cat_features=curr_cats)
                model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
                fold_test_preds = model.predict_proba(X_test_sub)[:, 1]
                
            elif model_type == 'xgb':
                for col in curr_cats:
                    train_cats = X_tr[col].cat.categories
                    X_val[col] = X_val[col].cat.set_categories(train_cats)
                    X_test_sub[col] = X_test_sub[col].cat.set_categories(train_cats)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                fold_test_preds = model.predict_proba(X_test_sub)[:, 1]

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            imp = self.get_importances(model)

            if full_data:
                test_preds_accum = fold_test_preds 
            else:
                test_preds_accum += fold_test_preds / actual_n_folds

            fold_imp = pd.DataFrame({'feature': feature_list, 'importance': imp, 'fold': fold+1})
            feature_importance_df = pd.concat([feature_importance_df, fold_imp], axis=0)

            best_model = model

        # --- FINAL BUSINESS EVALUATION ---
        print("\n🏁 Final Test Set Evaluation:")
        evaluator = SentinelEvaluator(self.y_test, test_preds_accum, self.test_amounts)

        cost_cfg = self.eval_config.get('business_costs', {})
        best_threshold = evaluator.find_best_threshold(method=find_best_threshold, **cost_cfg)
        impact = evaluator.report_business_impact(best_threshold)

        tier_cfg = self.eval_config.get('tiers', {})
        tiered_actions = evaluator.get_tiered_strategy(soft_threshold=tier_cfg.get('soft_threshold', 0.2),  hard_threshold=best_threshold)
        
        if verbose:
            print(f"\n🏁 Final Test Results ({model_type.upper()}):")
            print(f"   Optimum Threshold: {best_threshold:.4f}")
            print(f"   AUC: {impact['performance']['auc']:.4f}")
            print(f"   Precision: {impact['performance']['precision']:.2%}")
            print(f"   Recall: {impact['performance']['recall']:.2%}")
            print(f"   Net Profit Impact: ${impact['financials']['net_savings']:,.2f}")

        result = {
            'model': best_model,
            'model_type': model_type,
            'threshold': best_threshold,
            'features': feature_list,
            'importances': feature_importance_df,
            'impact': impact,
            'tiered_actions': tiered_actions,
            'test_auc': impact['performance']['auc'],
            'test_preds': test_preds_accum
        }
        return result

    def final_train(self, model_type, best_params, feature_list, verbose=True):
        if verbose: print(f"\n🚀 Training {model_type.upper()} Model...")
        X_sub = self.X_train[feature_list].copy()
        X_test_sub = self.X_test[feature_list].copy()

        model = self._get_model_instance(model_type, best_params, self.y_train)

        curr_cats = [c for c in self.cat_cols if c in feature_list]
        for c in curr_cats:
            X_sub[c] = X_sub[c].astype('category')
            X_test_sub[c] = X_test_sub[c].astype('category')

        if model_type == 'lgb':
            model.fit(X_sub, self.y_train, categorical_feature=curr_cats)
        
        elif model_type == 'cb':
            model.fit(X_sub, self.y_train, cat_features=curr_cats)
            
        elif model_type == 'xgb':
            model.fit(X_sub, self.y_train)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        y_test_preds = model.predict_proba(X_test_sub)[:, 1]
        evaluator = SentinelEvaluator(self.y_test, y_test_preds, self.test_amounts)
        cost_cfg = self.eval_config.get('business_costs', {})
        best_threshold = evaluator.find_best_threshold(method='cost', **cost_cfg)
        report = evaluator.report_business_impact(best_threshold)
        report[model_type] = model

        print(f"\n🏁 Final Test Results ({model_type.upper()}):")
        print(f"   Optimum Threshold: {best_threshold:.4f}") 
        print(f"   AUC: {report['performance']['auc']:.4f}")
        print(f"   Precision: {report['performance']['precision']:.2%}")
        print(f"   Recall: {report['performance']['recall']:.2%}")
        print(f"   Net Profit Impact: ${report['financials']['net_savings']:,.2f}")

        self._save_artifacts(model, model_type, best_threshold, feature_list)
        return report

    # =========================================================================
    # 3. OPTIMIZATION
    # =========================================================================
    def optimize_hyperparameters(self, model_type='lgb', n_trials=20, n_folds=3, top_n_features: Union[int, None] = None):

        FIXED_PARAMS = {
            'lgb': {'objective': 'binary', 'boosting_type': 'gbdt', 'metric': 'auc', 'n_jobs': -1, 'verbose': -1, 'random_state': 42},
            'cb': {'loss_function': 'Logloss', 'eval_metric': 'AUC', 'grow_policy': 'SymmetricTree', 'bootstrap_type': 'Bernoulli', 'task_type': 'GPU', 'thread_count': -1, 'verbose': 0, 'random_state': 42},
            'xgb': {'objective': 'binary:logistic', 'booster': 'gbtree', 'eval_metric': 'auc', 'grow_policy': 'depthwise', 
            'tree_method': 'hist', 'device': 'cuda', 'n_jobs': -1, 'verbosity': 0, 'early_stopping_rounds': 50, 'random_state': 42}
        }
        if isinstance(top_n_features, int):
            print(f"🔍 Feature Selection: Pre-selecting top {top_n_features} features...")
            temp_res = self._train_loop(model_type, None, list(self.X_train.columns), False, 3, False)
            avg_imp = temp_res['importances'].groupby('feature')['importance'].mean().sort_values(ascending=False)
            features = avg_imp.head(top_n_features).index.tolist()

        elif isinstance(top_n_features, list):
            features = top_n_features
        else:
            print("⚠️ Warning: Feature selection returned 0 features. Reverting to all.")
            features = list(self.X_train.columns)
        print(f"\n🔍 Optuna: Optimizing {model_type.upper()} on {len(features)} features, ({n_trials} trials)...")
        
        X_sub = self.X_train[features].copy()
        X_test_sub = self.X_test[features].copy()

        curr_cats = [c for c in self.cat_cols if c in features]
        for c in curr_cats:
            X_sub[c] = X_sub[c].astype('category')
            X_test_sub[c] = X_test_sub[c].astype('category')
        
        config_space = self.config.get('optimization', {}).get(model_type, {}) 
        default_space = self.default_search_spaces.get(model_type, {})
        target_space = config_space if config_space else default_space

        def objective(trial):
            params = {}
            for name, config in target_space.items():
                p_type = config[0]
                if p_type == 'int': 
                    params[name] = trial.suggest_int(name, config[1], config[2])
                elif p_type == 'float': 
                    log_scale = config[3] if len(config) > 3 else False
                    params[name] = trial.suggest_float(name, config[1], config[2], log=log_scale)
                elif p_type == 'categorical': 
                    params[name] = trial.suggest_categorical(name, config[1])

            params.update(FIXED_PARAMS[model_type])
            tscv = TimeSeriesSplit(n_splits=n_folds)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sub)):
                if len(train_idx) < 1000: continue
                
                X_tr, y_tr = X_sub.iloc[train_idx].copy(), self.y_train.iloc[train_idx]
                X_val, y_val = X_sub.iloc[val_idx].copy(), self.y_train.iloc[val_idx]
                
                model = self._get_model_instance(model_type, params, y_tr)
                
                if model_type == 'lgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', categorical_feature=curr_cats, callbacks=[lgb.early_stopping(50, verbose=False)])

                elif model_type == 'cb':
                    train_pool = Pool(X_tr, y_tr, cat_features=curr_cats)
                    val_pool = Pool(X_val, y_val, cat_features=curr_cats)
                    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

                elif model_type == 'xgb':
                    for col in curr_cats:
                        train_cats = X_tr[col].cat.categories
                        X_val[col] = X_val[col].cat.set_categories(train_cats)
                        X_test_sub[col] = X_test_sub[col].cat.set_categories(train_cats)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                else:
                    model.fit(X_tr, y_tr)

                preds = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
            
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        print(f"   ✅ Best params: {study.best_params}")
        return study.best_params

    # =========================================================================
    # 4. FIND BEST ENSEMBLE/MODEL
    # =========================================================================
    def select_best_model(self, models_to_consider: List[str] = ['lgb', 'cb', 'xgb']):
        """
        Loads models, tests WEIGHTED ensembles, and selects the winner 
        based on the FINANCIAL COST function in config.yaml.
        """
        print(f"\n🏆 Selecting Best Configuration based on Financial Impact...")
        preds = {}
        valid_models = []
        
        for m in models_to_consider:
            model_path = self.models_dir / f"{m}_model.pkl"
            feat_path = self.models_dir / f"{m}_features.json"
            if model_path.exists() and feat_path.exists():
                with open(feat_path, 'r') as f: features = json.load(f)
                model = joblib.load(model_path)
                X_sub = self.X_test[features].copy()
                curr_cats = [c for c in self.cat_cols if c in features]
                for c in curr_cats: X_sub[c] = X_sub[c].astype('category')
                p = model.predict_proba(X_sub)[:, 1]
                preds[m] = p
                valid_models.append(m)
        if not valid_models: 
            print("❌ No valid models found.")
            return

        candidates = []
        for m in valid_models: candidates.append({m: 1.0})
        if len(valid_models) >= 2:
            import itertools
            for m1, m2 in itertools.combinations(valid_models, 2):
                candidates.extend([{m1: 0.5, m2: 0.5}, {m1: 0.7, m2: 0.3}, {m1: 0.3, m2: 0.7}])
        if len(valid_models) >= 3:
            candidates.append({m: 1.0/len(valid_models) for m in valid_models})
        best_profit = -np.inf
        best_cfg = {}
        cost_cfg = self.eval_config.get('business_costs', {})

        print(f"\n   Evaluating {len(candidates)} combinations using Cost Strategy...")
        for weights in candidates:
            final_pred = np.zeros_like(list(preds.values())[0])
            for m, w in weights.items(): final_pred += preds[m] * w

            evaluator = SentinelEvaluator(self.y_test, final_pred, self.test_amounts)
            thresh = evaluator.find_best_threshold(method='cost', **cost_cfg)
            impact = evaluator.report_business_impact(thresh)
            profit = impact['financials']['net_savings']
            
            name = "+".join([f"{k}:{v}" for k,v in weights.items()])
            if profit > best_profit:
                best_profit = profit
                best_cfg = {'weights': weights, 'threshold': thresh, 'profit': profit, 'auc': evaluator.get_auc()}
                print(f"   ⭐ New Leader: [{name}] Net Savings=${profit:,.2f}")


        print(f"\n🎉 WINNER: {best_cfg['weights']}")
        print(f"   AUC: {best_cfg['auc']:.4f}")
        print(f"   Net Savings: ${best_cfg['profit']:,.2f}")
        print(f"   Optimal Threshold: {best_cfg['threshold']:.4f}")
        
        self._save_production_config(best_cfg)
        return best_cfg


    def _get_model_instance(self, model_type, params, y_train):
        neg = (y_train == 0).sum() 
        pos = (y_train == 1).sum() 
        scale = np.sqrt(neg / pos) if pos > 0 else 1.0
        if 'scale_pos_weight' in params:
            scale = params['scale_pos_weight']
            del params['scale_pos_weight']

        if model_type == 'lgb':
            return lgb.LGBMClassifier(scale_pos_weight=scale, **params)
        elif model_type == 'cb':
            return CatBoostClassifier(scale_pos_weight=scale, allow_writing_files=False, **params)
        elif model_type == 'xgb':
            return xgb.XGBClassifier(scale_pos_weight=scale, enable_categorical=True,  **params)
        else:
            raise ValueError(f"Unknown model: {model_type}")
    
    @staticmethod
    def get_importances(model):
        """
        Returns the raw feature importance array for CatBoost, XGBoost, or LGBM.
        """
        if hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
            
        raise ValueError(f"Model {type(model).__name__} has no feature importance info.")

    def _save_production_config(self, config):
        """Saves config for Inference Engine."""
        path = self.models_dir / "production_config.json"
        
        clean_config = {
            "weights": config['weights'],
            "threshold": float(config['threshold']),
            "meta": {
                "profit": float(config['profit']),
                "auc": float(config['auc'])
            }
        }
        
        with open(path, "w") as f: json.dump(clean_config, f, indent=4)
        print(f"💾 Saved production_config.json to {self.models_dir}")        

    def _save_artifacts(self, model, model_type, threshold, features):
        joblib.dump(model, self.models_dir / f"{model_type}_model.pkl")
        with open(self.models_dir / f"{model_type}_features.json", "w") as f: json.dump(features, f)
        with open(self.models_dir / f"{model_type}_threshold.json", "w") as f: json.dump(threshold, f)
        print(f"💾 Saved {model_type} model & {len(features)} features.")

if __name__ == "__main__":
    print("Sentinel Modeling Module Loaded")

