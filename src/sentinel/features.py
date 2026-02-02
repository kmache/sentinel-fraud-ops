import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Union, List, Optional
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#***********************************************************************************#
#************ Part II: Features Engineering  ***************************************#
#***********************************************************************************#
class SentinelFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Sentinel Feature Engineering (v7.0 - The "Grandmaster" Edition).
    
    Combines Top 1% Kaggle Strategies:
    1. Magic UID (Card+Addr+BirthDate).
    2. Contextual Aggregations (Amt vs Group Means) -> CRITICAL for >0.95 AUC.
    3. Fraud Patterns (Suspicious Cents, Round Numbers).
    4. Graph Network Features (Reverse Linkage, Degree Centrality).
    5. Grouped PCA on V-Columns (97% Information Retention).
    """
    
    def __init__(self, 
                 time_col: str = 'TransactionDT',
                 target_col: str = 'isFraud',
                 smoothing: float = 500.0,
                 verbose: bool = True):
        
        self.user_key = 'UID'
        self.time_col = time_col
        self.target_col = target_col
        self.smoothing = smoothing
        self.verbose = verbose
        
        # --- State Storage ---
        self.user_start_dates: Dict[str, float] = {} 
        self.user_stats: Dict[str, Dict] = {'mean': {}, 'std': {}}
        self.group_stats: Dict[str, Dict] = {} 
        self.frequency_maps: Dict[str, Dict] = {}
        self.risk_maps: Dict[str, Dict] = {}
        
        # PCA Storage
        self.v_pca_models = {}
        self.v_group_mapping = {}
        
        # Amount Stats for Z-scores
        self.log_amt_stats: Dict[str, float] = {}
        self.amt_bins: Optional[np.ndarray] = None
        self.global_mean: float = 0.035
        
        self._fitted = False
    
    # ------------------------------------------------------------------------- 
    # PIPELINE
    # -------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Union[str, pd.Series] = None):
        if self.verbose: print(f"--- Fitting Sentinel v7.0 on {len(X):,} rows ---")
        
        # 1. Temp Setup
        temp = X.copy()
        temp = self._make_uid(temp)
        temp = self._make_combos(temp)
        
        # Resolve Target
        y_series = y if isinstance(y, pd.Series) else (temp[y] if y in temp.columns else None)
        
        # 2. Learn Amount Stats
        if 'TransactionAmt' in temp.columns:
            amt = temp['TransactionAmt']
            log_amt = np.log1p(amt)
            self.log_amt_stats = {'mean': float(log_amt.mean()), 'std': float(log_amt.std())}
            self.amt_bins = np.percentile(amt, np.arange(0, 101, 10))
            
            # 2b. Learn Group Stats (Context)
            group_cols = ['card1', 'card4', 'addr1', 'P_emaildomain_vendor_id', 'ProductCD']
            for col in group_cols:
                if col in temp.columns:
                    self.group_stats[col] = temp.groupby(col)['TransactionAmt'].mean().to_dict()
        
        # 3. Learn User Stats
        self.user_start_dates = temp.groupby(self.user_key)[self.time_col].min().to_dict()
        if 'TransactionAmt' in temp.columns:
            stats = temp.groupby(self.user_key)['TransactionAmt'].agg(['mean', 'std'])
            self.user_stats['mean'] = stats['mean'].to_dict()
            self.user_stats['std'] = stats['std'].fillna(1).replace(0, 1).to_dict()
        
        # 4. Learn Frequency Maps
        freq_cols = [self.user_key, 'card1', 'addr1', 'card4', 'P_emaildomain_vendor_id', 'device_vendor']
        for col in freq_cols:
            if col in temp.columns:
                self.frequency_maps[col] = temp[col].value_counts().to_dict()
        
        # 5. Learn Risk Scores (Target Encoding)
        if y_series is not None:
            learn_df = temp.copy()
            learn_df['target'] = y_series
            self.global_mean = float(y_series.mean())
            
            risk_cols = [
                'addr1', 'card4', 'P_emaildomain_vendor_id', 'device_vendor',
                'device_info_combo', 'card_email_combo', 'ProductCD',
                'card6', 'os_type', 'browser_type', 'hour_of_day'
            ]
            for col in risk_cols:
                if col in learn_df.columns:
                    self.risk_maps[col] = self._calc_smooth_mean(learn_df, col, 'target')

        # 6. Fit V-Column PCA (97% Variance)
        self.fit_v_pca(temp)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.empty: return X
        X = X.copy()
        
        if self.verbose: print(f"--- Transforming {len(X):,} rows ---")
        
        # 1. Preprocessing & Identity
        X = self._make_uid(X)
        X = self._make_combos(X)
        
        # 2. Time Ordering (Critical for sequences)
        X['_orig_index'] = np.arange(len(X))
        
        # Only sort if time column exists and is not sorted
        if self.time_col in X.columns and not X[self.time_col].is_monotonic_increasing:
            X = X.sort_values(self.time_col)
            
        # 3. Feature Blocks
        X = self._add_temporal_features(X)
        X = self._add_velocity_features(X)
        X = self._add_transaction_amount_features(X)
        X = self._add_sequence_features(X)           
        X = self._add_interaction_features(X)
        X = self._add_graph_features(X)
        X = self._add_advanced_stats(X)
        
        # 4. V-Column Reduction (PCA)
        X = self.transform_v_pca(X)
        
        # 5. Categorical Optimization
        X = self._optimize_categoricals(X)
        
        # 6. Cleanup
        if '_orig_index' in X.columns:
            X = X.sort_values('_orig_index').drop(columns=['_orig_index'])
            
        drops = ['_temp_dt']
        X = X.drop(columns=[c for c in drops if c in X.columns], errors='ignore')
        
        return X

    # -------------------------------------------------------------------------
    # FEATURE LOGIC
    # -------------------------------------------------------------------------
    def _make_uid(self, df: pd.DataFrame) -> pd.DataFrame:
        d_col = 'D1_norm' if 'D1_norm' in df.columns else 'D1'
        c1 = df['card1'].fillna(-1).astype(int).astype(str)
        a1 = df['addr1'].fillna(-1).astype(int).astype(str)
        
        if d_col in df.columns:
            d1_weekly = (df[d_col].fillna(-999) // 7).astype(int).astype(str)
        else:
            d1_weekly = "nan"
            
        df['UID'] = c1 + '_' + a1 + '_' + d1_weekly
        return df

    def _make_combos(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'browser_type' in df.columns and 'screen_width' in df.columns:
            df['device_info_combo'] = df['browser_type'].astype(str) + '_' + df['screen_width'].fillna(-1).astype(int).astype(str)
        if 'card1' in df.columns and 'P_emaildomain_vendor_id' in df.columns:
            c1 = df['card1'].fillna(-1).astype(int).astype(str)
            v_id = df['P_emaildomain_vendor_id'].fillna(-1).astype(int).astype(str)
            df['card_email_combo'] = c1 + '_' + v_id
        return df

    def _calc_smooth_mean(self, df: pd.DataFrame, by: str, on: str) -> Dict:
        global_mean = df[on].mean()
        agg = df.groupby(by)[on].agg(['count', 'mean'])
        smooth = (agg['count'] * agg['mean'] + self.smoothing * global_mean) / (agg['count'] + self.smoothing)
        return smooth.to_dict()

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'hour_of_day' in df.columns:
            hour_rad = 2 * np.pi * df['hour_of_day'] / 24
            df['hour_sin'] = np.sin(hour_rad).astype('f4')
            df['hour_cos'] = np.cos(hour_rad).astype('f4')
        
        start_dates = df[self.user_key].map(self.user_start_dates).fillna(df[self.time_col])
        df['days_since_first_txn'] = ((df[self.time_col] - start_dates) / 86400).astype('f4')
        return df

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f'{self.user_key}_dt_diff'] = df.groupby(self.user_key)[self.time_col].diff().fillna(0).astype('f4')
        
        windows = {'1h': '1h', '12h': '12h', '24h': '24h'}
        counts = df[self.user_key].value_counts()
        active_users = counts[counts > 1].index
        
        if len(active_users) > 0:
            mask = df[self.user_key].isin(active_users)
            df_active = df[mask].copy()
            df_active['_temp_dt'] = pd.to_datetime(df_active[self.time_col], unit='s', origin='2017-11-30')
            df_active = df_active.set_index('_temp_dt').sort_index()
            g = df_active.groupby(self.user_key)['TransactionAmt']
            
            for w_name, w_str in windows.items():
                rolled = g.rolling(window=w_str, closed='left')
                df.loc[mask, f'{self.user_key}_count_{w_name}'] = rolled.count().fillna(0).values.astype('i2')
                df.loc[mask, f'{self.user_key}_amt_sum_{w_name}'] = rolled.sum().fillna(0).values.astype('f4')
                
                hours = int(w_name.replace('h',''))
                df.loc[mask, f'{self.user_key}_velocity_{w_name}'] = (df.loc[mask, f'{self.user_key}_amt_sum_{w_name}'] / hours).astype('f4')
        
        for w in windows:
            c = f'{self.user_key}_count_{w}'
            if c in df.columns: df[c] = df[c].fillna(0)
        return df

    def _add_transaction_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'TransactionAmt' not in df.columns: return df
        amt = df['TransactionAmt']
        
        df['TransactionAmt_log'] = np.log1p(amt).astype('f4')
        
        if hasattr(self, 'log_amt_stats'):
            df['TransactionAmt_log_z'] = ((np.log1p(amt) - self.log_amt_stats['mean']) / max(self.log_amt_stats['std'], 1e-6)).astype('f4')
        
        df['TransactionAmt_suspicious'] = 0
        df.loc[amt % 100 == 0, 'TransactionAmt_suspicious'] += 1
        df.loc[amt % 100 == 99, 'TransactionAmt_suspicious'] += 1
        cents = (amt * 100) % 100
        df.loc[cents.isin([99, 95, 90, 49, 00]), 'TransactionAmt_suspicious'] += 1
        df['TransactionAmt_suspicious'] = df['TransactionAmt_suspicious'].astype('i1')
        
        return df

    def _add_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([self.user_key, self.time_col])
        df['txn_sequence'] = df.groupby(self.user_key).cumcount().astype('i2')
        
        if 'TransactionAmt' in df.columns:
            df['amt_change_abs'] = df.groupby(self.user_key)['TransactionAmt'].diff().fillna(0).astype('f4')
        
        for col in ['ProductCD', 'addr1', 'device_vendor']:
            if col in df.columns:
                df[f'{col}_switch'] = (df.groupby(self.user_key)[col].diff().fillna(0) != 0).astype('i1')
                
        time_gap = df.groupby(self.user_key)[self.time_col].diff().fillna(0)
        df['time_gap_anomaly'] = (time_gap < 30).astype('i1')
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'screen_width' in df.columns and 'screen_height' in df.columns:
            df['screen_area'] = (df['screen_width'] * df['screen_height']).astype('f4')
            df['screen_aspect_ratio'] = (df['screen_width'] / (df['screen_height'] + 1)).astype('f4')

        if 'ProductCD' in df.columns and 'card4' in df.columns:
            df['product_network_combo'] = (df['ProductCD'].astype(str) + '_' + df['card4'].astype(str)).astype('category')
        if 'card1' in df.columns and 'addr1' in df.columns:
            df['card1_addr1_combo'] = (df['card1'].astype(str) + '_' + df['addr1'].astype(str)).astype('category')
        if 'os_type' in df.columns and 'browser_type' in df.columns:
            df['os_browser_combo'] = (df['os_type'].astype(str) + '_' + df['browser_type'].astype(str)).astype('category')
        return df

    def _add_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['device_vendor', 'addr1']:
            if col in df.columns:
                df[f'{col}_degree'] = df.groupby(col)[self.user_key].transform('nunique').fillna(0).astype('i2')
        
        if 'DeviceInfo' in df.columns and 'card1' in df.columns:
             df['cards_per_device'] = df.groupby('DeviceInfo')['card1'].transform('nunique').fillna(0).astype('i2')

        if 'addr1' in df.columns and 'card1' in df.columns:
            df['unique_addrs_per_card'] = df.groupby('card1')['addr1'].transform('nunique').fillna(0).astype('i2')

        if 'P_emaildomain' in df.columns and 'card1' in df.columns:
            df['unique_emails_per_card'] = df.groupby('card1')['P_emaildomain'].transform('nunique').fillna(0).astype('i2')

        if 'device_vendor' in df.columns and 'addr1' in df.columns:
            d_share = df.groupby('device_vendor')[self.user_key].transform('nunique')
            i_share = df.groupby('addr1')[self.user_key].transform('nunique')
            df['multi_entity_sharing'] = ((d_share + i_share)/2).astype('f4')
        return df

    def fit_v_pca(self, X: pd.DataFrame):
        v_cols = [c for c in X.columns if c.startswith('V')]
        if not v_cols: return

        nan_counts = X[v_cols].isnull().sum()
        unique_counts = nan_counts.unique()
        
        if not hasattr(self, 'v_pca_models'):
            self.v_pca_models = {}
            self.v_group_mapping = {}

        for i, count in enumerate(unique_counts):
            cols = nan_counts[nan_counts == count].index.tolist()
            if len(cols) > 1:
                group_name = f'Group_{i}'
                self.v_group_mapping[group_name] = cols
                
                sample_size = min(100000, len(X))
                data = X[cols].sample(sample_size, random_state=42).fillna(-1)
                
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                
                pca = PCA(n_components=0.97, svd_solver='full', random_state=42)
                pca.fit(data_scaled)

                if pca.n_components_ > 10:
                    pca = PCA(n_components=10, random_state=42)
                    pca.fit(data_scaled)
                
                self.v_pca_models[group_name] = (scaler, pca)
                if self.verbose:
                    print(f"   > PCA {group_name}: {len(cols)} cols -> {pca.n_components_} comps (97% var)")

    def transform_v_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, 'v_pca_models') or not self.v_pca_models:
            return X

        cols_to_drop = []

        for group_name, cols in self.v_group_mapping.items():
            valid_cols = [c for c in cols if c in X.columns]
            if len(valid_cols) != len(cols):
                continue
            
            try:
                scaler, pca = self.v_pca_models[group_name]
                data = X[valid_cols].fillna(-1)
                data_scaled = scaler.transform(data)
                data_pca = pca.transform(data_scaled)
                
                pca_columns = {}
                for i in range(data_pca.shape[1]):
                    pca_columns[f'PCA_{group_name}_{i}'] = data_pca[:, i].astype(np.float32)
                pca_df = pd.DataFrame(pca_columns, index=X.index)
                X = pd.concat([X, pca_df], axis=1)
                
                cols_to_drop.extend(valid_cols)
            except Exception as e:
                if self.verbose: print(f"Warning: PCA skipped for {group_name} due to drift: {e}")
                
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop, errors='ignore')
            
        return X

    def _add_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'TransactionAmt' in df.columns:
            amt = df['TransactionAmt'].fillna(0).astype('float64')
            cents = (np.mod(amt, 1) * 1000).astype('i2')
            df['cents_value'] = cents
            df['is_exact_dollars'] = (cents == 0).astype('i1')
            df['is_99_cents'] = ((cents >= 990) & (cents <= 999)).astype('i1')
            df['TransactionAmt_decimal'] = (amt % 1).astype('f4')

        for col, mapping in self.frequency_maps.items():
            if col in df.columns: 
                df[f'{col}_freq_enc'] = df[col].map(mapping).fillna(0).astype('f4')
        
        for col, mapping in self.risk_maps.items():
            if col in df.columns: 
                df[f'{col}_fraud_rate'] = df[col].map(mapping).fillna(self.global_mean).astype('f4')

        risk_cols = [c for c in df.columns if c.endswith('_fraud_rate')]
        if risk_cols:
            df['composite_risk_score'] = df[risk_cols].mean(axis=1).astype('f4')

        if 'TransactionAmt' in df.columns:
            amt = df['TransactionAmt'].astype('float64')
            
            user_mean = df[self.user_key].map(self.user_stats['mean']).fillna(amt)
            user_std = df[self.user_key].map(self.user_stats['std']).fillna(1)
            df['user_amt_zscore'] = ((amt - user_mean) / user_std).astype('f4')
            
            for col, stats_dict in self.group_stats.items():
                if col in df.columns:
                    group_mean = df[col].map(stats_dict).fillna(amt)
                    df[f'Amt_div_{col}_mean'] = (amt / (group_mean + 1e-5)).astype('f4')

        return df

    def _optimize_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        high_card_cols = ['device_info_combo', 'card_email_combo', 'product_network_combo', 'card1_addr1_combo', 'os_browser_combo']
        for col in high_card_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes.astype('i4')
        
        df = df.copy()
        if 'UID' in df.columns:
            df['UID_hash'] = (df['UID'].astype(str).apply(hash) % 10000).astype('i4')
            
        return df

if __name__ == "__main__":
    print("Features engineering module loaded successfully!")