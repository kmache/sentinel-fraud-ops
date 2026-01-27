from os import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx
from typing import List, Optional
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import re
#***********************************************************************************#
#************ Part III: Feature Quality & Drift Analysis ***************************#
#***********************************************************************************#
class SentinelFeatureQuality(BaseEstimator, TransformerMixin):
    """
    Sentinel Feature Quality Engine (v8.0).
    
    FOCUS: Pre-training validation.
    - Eliminates Noise (KS Statistic).
    - Eliminates Redundancy (Collinearity).
    - Detects Dangerous Drift (PSI & Adversarial Validation).
    - Visualizes Feature structure (Networks, Trends, KDE).
    TODO: Add more validation metrics.
    """
    
    def __init__(self, 
                 X_train: pd.DataFrame, 
                 y_train: pd.Series, 
                 X_test: Optional[pd.DataFrame] = None, 
                 time_col: str = 'TransactionDT',
                 verbose: bool = True):
        
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy() if X_test is not None else None
        
        self.time_col = time_col
        self.verbose = verbose
        self.colors = ['#1f77b4', '#d62728']  # Blue (Legit), Red (Fraud)
        sns.set_style("whitegrid")
        
        if verbose:
            print(f"📊 Sentinel Feature Quality Initialized")
            print(f"   Train: {len(self.X_train):,} rows | Fraud Rate: {self.y_train.mean():.4%}")
            if self.X_test is not None:
                print(f"   Test:  {len(self.X_test):,} rows (Drift Analysis Ready)")

    # ===========================================================================
    # 1. SIGNAL STRENGTH (Do these features actually work?)
    # ===========================================================================
    def score_features(self, features: List[str]) -> pd.DataFrame:
        """
        Calculates KS Statistic to measure how well a feature separates Fraud vs Legit.
        """
        results = []
        if self.verbose: print(f"\n🔍 Scoring {len(features)} Features for Signal Strength...")
        
        for col in tqdm(features, disable=not self.verbose):
            if col not in self.X_train.columns: continue
            
            try:
                # Sample if too large to speed up KS test
                df_sample = self.X_train[[col]].copy()
                df_sample['target'] = self.y_train
                
                if len(df_sample) > 100000:
                    df_sample = df_sample.sample(100000, random_state=42)

                neg = df_sample.loc[df_sample['target'] == 0, col].dropna()
                pos = df_sample.loc[df_sample['target'] == 1, col].dropna()
                
                if len(neg) < 10 or len(pos) < 10: continue

                # KS Statistic (Distance between distributions)
                if pd.api.types.is_numeric_dtype(neg):
                    ks_stat, _ = ks_2samp(neg, pos)
                else:
                    # Factorize categoricals for rough KS estimation
                    neg_codes = pd.factorize(neg, sort=True)[0]
                    pos_codes = pd.factorize(pos, sort=True)[0]
                    ks_stat, _ = ks_2samp(neg_codes, pos_codes)

                results.append({
                    'Feature': col,
                    'KS_Stat': round(ks_stat, 4),
                    'Fraud_Mean': round(pos.mean(), 4) if pd.api.types.is_numeric_dtype(pos) else 0,
                    'Legit_Mean': round(neg.mean(), 4) if pd.api.types.is_numeric_dtype(neg) else 0,
                    'Sparsity': round(self.X_train[col].isnull().mean(), 4)
                })
                
            except Exception:
                continue
            
        df_res = pd.DataFrame(results).sort_values('KS_Stat', ascending=False)
        if not df_res.empty:
            # Interpretation Guide
            df_res['Signal_Quality'] = pd.cut(df_res['KS_Stat'], 
                                            [-1, 0.05, 0.2, 0.4, 1.0], 
                                            labels=['Noise (Drop)', 'Weak', 'Good', 'Strong'])
        return df_res

    def check_collinearity(self, features: List[str], threshold: float = 0.98):
        """Identifies highly correlated features to reduce noise."""
        print(f"\n🧩 Checking for Redundant Features (> {threshold} correlation)...")
        valid_cols = [c for c in features if c in self.X_train.columns and pd.api.types.is_numeric_dtype(self.X_train[c])]
        if len(valid_cols) < 2: return []

        # Use sample for speed
        df_sample = self.X_train[valid_cols].fillna(-999)
        if len(df_sample) > 50000: df_sample = df_sample.sample(50000, random_state=42)
            
        corr_matrix = df_sample.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            print(f"   ⚠️ Found {len(to_drop)} redundant features. (e.g., {to_drop[:3]})")
            print("   (Consider dropping these to speed up training)")
        else:
            print("   ✅ No highly redundant features found.")
        return to_drop

    # ===========================================================================
    # 2. DRIFT & STABILITY (Will this break in production?)
    # ===========================================================================
    def check_stability(self, features: List[str]):
        """Prints PSI analysis (Population Stability Index)."""
        if self.X_test is None: return
        
        print(f"\n📉 Checking Train-Test Stability (PSI)...")
        results = []
        for col in tqdm(features, disable=not self.verbose):
            if col not in self.X_train.columns or col not in self.X_test.columns: continue
            
            psi = self._calculate_psi(self.X_train[col], self.X_test[col])
            
            status = '🟢 Stable'
            if psi > 0.1: status = '🟡 Warning'
            if psi > 0.2: status = '🔴 Critical'
            
            results.append({'Feature': col, 'PSI': psi, 'Status': status})
            
        df_psi = pd.DataFrame(results).sort_values('PSI', ascending=False)
        print(df_psi.head(15).to_string(index=False))
        return df_psi

    def _calculate_psi(self, expected, actual, buckets=10):
        try:
            expected = expected.dropna()
            actual = actual.dropna()
            if len(expected) == 0 or len(actual) == 0: return 0
            
            if pd.api.types.is_numeric_dtype(expected) and expected.nunique() > 20:
                breakpoints = np.percentile(expected, np.linspace(0, 100, buckets+1))
                breakpoints = np.unique(breakpoints)
                if len(breakpoints) < 2: return 0
                exp_percents = np.histogram(expected, breakpoints)[0] / len(expected)
                act_percents = np.histogram(actual, breakpoints)[0] / len(actual)
            else:
                cats = expected.unique()
                if len(cats) > 50: cats = expected.value_counts().head(50).index
                exp_percents = expected.value_counts(normalize=True).reindex(cats, fill_value=0).values
                act_percents = actual.value_counts(normalize=True).reindex(cats, fill_value=0).values

            # Avoid division by zero
            exp_percents = np.where(exp_percents == 0, 0.0001, exp_percents)
            act_percents = np.where(act_percents == 0, 0.0001, act_percents)
            
            return np.sum((exp_percents - act_percents) * np.log(exp_percents / act_percents))
        except:
            return 0


    def check_adversarial(self, features: List[str]):
        """
        Runs Adversarial Validation.
        Tries to train a model to predict: Is this row Train or Test?
        If AUC > 0.70, the feature set has significant time drift.
        """
        if self.X_test is None: return

        print(f"\n🕵️ Running Adversarial Validation (Drift Detection)...")
        cols = [f for f in features if f in self.X_train.columns and f in self.X_test.columns]
        if not cols: return

        # Downsample for speed
        train = self.X_train[cols].sample(min(20000, len(self.X_train)), random_state=42).fillna(-999)
        test = self.X_test[cols].sample(min(20000, len(self.X_test)), random_state=42).fillna(-999)
        
        X_adv = pd.concat([train, test])
        y_adv = [0]*len(train) + [1]*len(test)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_adv, y_adv)
        
        auc_score = roc_auc_score(y_adv, clf.predict_proba(X_adv)[:, 1])
        print(f"   Adversarial AUC: {auc_score:.4f}")
        
        if auc_score > 0.70:
            print("   ⚠️ High Drift! Model can easily distinguish Train vs Test.")
            print("   These features changed significantly over time (Top 5 Drifters):")
            print(pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False).head(5))
        else:
            print("   ✅ Low Drift. Train and Test data look statistically similar.")

    # ===========================================================================
    # 3. VISUAL INSPECTION
    # ===========================================================================
    def plot_feature_trend(self, feature: str, resample_rule: str = '1D'):
        """Visualizes the mean value of a feature over time."""
        if self.time_col not in self.X_train.columns: return
        if feature not in self.X_train.columns: return

        combined_X = pd.concat([self.X_train, self.X_test]) if self.X_test is not None else self.X_train
        df = combined_X[[self.time_col, feature]].copy()
        
        # Convert seconds to datetime
        df['dt'] = pd.to_datetime(df[self.time_col], unit='s', origin='2017-11-30')
        df = df.set_index('dt').sort_index()
        
        if pd.api.types.is_numeric_dtype(df[feature]):
            daily = df[feature].resample(resample_rule).agg(['mean', 'std'])
            plt.figure(figsize=(12, 5))
            plt.plot(daily.index, daily['mean'], color='black', label='Mean')
            plt.fill_between(daily.index, daily['mean']-daily['std'], daily['mean']+daily['std'], color='gray', alpha=0.2)
            
            if self.X_test is not None:
                split_date = pd.to_datetime(self.X_train[self.time_col].max(), unit='s', origin='2017-11-30')
                plt.axvline(split_date, color='red', linestyle='--', label='Train/Test Split')
                
            plt.title(f"Temporal Stability: {feature}")
            plt.legend()
            plt.show()


    def plot_signal_kde(self, feature: str, log_scale: bool = False):
        """KDE Plot to visualize separation between Fraud and Legit."""
        if feature not in self.X_train.columns: return
        plt.figure(figsize=(10, 4))
        try:
            sns.kdeplot(self.X_train.loc[self.y_train == 0, feature].dropna(), label='Legit', fill=True, color=self.colors[0], alpha=0.3)
            sns.kdeplot(self.X_train.loc[self.y_train == 1, feature].dropna(), label='Fraud', fill=True, color=self.colors[1], alpha=0.3)
            plt.title(f'Signal Density: {feature}', fontsize=12)
            if log_scale: plt.xscale('log')
            plt.legend()
            plt.show()
        except: pass


    def plot_fraud_network(self, user_col: str, device_col: str, top_n_rings: int = 3):
        """Visualizes Device Farms (Graph Feature Inspection)."""
        if user_col not in self.X_train.columns or device_col not in self.X_train.columns: return
        print(f"\n🕸️ Inspecting Fraud Network Features...")
        
        fraud_mask = (self.y_train == 1)
        df_fraud = self.X_train.loc[fraud_mask, [user_col, device_col]].dropna().copy()
        if len(df_fraud) > 5000: df_fraud = df_fraud.sample(5000, random_state=42)

        # Filter for meaningful connections (at least 2 connections)
        device_counts = df_fraud[device_col].value_counts()
        valid = device_counts[device_counts > 1].index
        df_fraud = df_fraud[df_fraud[device_col].isin(valid)]
        
        if len(df_fraud) == 0:
            print("   No obvious fraud rings found in sample.")
            return

        G = nx.Graph()
        df_fraud['u'] = 'U_' + df_fraud[user_col].astype(str)
        df_fraud['d'] = 'D_' + df_fraud[device_col].astype(str)
        G.add_edges_from(zip(df_fraud['u'], df_fraud['d']))
        
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        # Sort by size to find largest rings
        rings = sorted(components, key=len, reverse=True)
        
        for i in range(min(top_n_rings, len(rings))):
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(rings[i], k=0.5)
            cols = ['#d62728' if str(n).startswith('U_') else '#1f77b4' for n in rings[i].nodes()]
            nx.draw(rings[i], pos, node_color=cols, node_size=100, alpha=0.8, with_labels=False)
            plt.title(f"Detected Fraud Ring #{i+1} (Size: {len(rings[i].nodes)})")
            plt.show()

    # ===========================================================================
    # 4. ORCHESTRATOR
    # ===========================================================================
    def generate_report(self, features: List[str]):
        """Runs the full feature quality suite."""
        # 1. Signal Analysis
        scores = self.score_features(features)
        print(f"\n🏆 Top 10 Strongest Features (KS Statistic):")
        print(scores.head(10)[['Feature', 'KS_Stat', 'Signal_Quality']])
        
        # 2. Redundancy
        self.check_collinearity(features)
        
        # 3. Stability & Drift
        self.check_stability(features[:20]) # Check top 20 features for stability
        self.check_adversarial(features)
        
        # 4. Visuals (Top Feature)
        if not scores.empty:
            best_feat = scores.iloc[0]['Feature']
            print(f"\n📈 Visualizing Best Feature: {best_feat}")
            self.plot_signal_kde(best_feat)
            self.plot_feature_trend(best_feat)
        
        # 5. Network Visual (If graph features exist)
        if 'UID' in self.X_train.columns and 'device_vendor' in self.X_train.columns:
            self.plot_fraud_network('UID', 'device_vendor')

if name == "__main__":
    print("Validation module loaded successfully!")