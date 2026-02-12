import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union, Optional

#***********************************************************************************#
#************* Data Preprocessing **************************************************#
#***********************************************************************************#
class SentinelPreprocessing(BaseEstimator, TransformerMixin):
    """
    Sentinel Fraud Detection Preprocessing (v4.0 Final).
    
    Includes:
    - Robust 'Float-to-Int' casting for ID columns.
    - Rich Email Feature Extraction (Length, Suffix, Digits, Free Providers).
    - Detailed Device Parsing (OS, Browser, Vendor).
    - Time Drift Correction (D-Column Normalization).
    - Target Leakage Prevention in fit().
    """
    
    def __init__(self, start_date: str = '2017-11-30', nan_threshold: float = 0.90,
                 correlation_threshold: float = 0.98, verbose: bool = True):
        
        self.start_date = pd.to_datetime(start_date)
        self.nan_threshold = nan_threshold
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose
        
        # State
        self.cols_to_drop: List[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Union[str, pd.Series] = None):
        if self.verbose: print("--- Fitting Sentinel Preprocessor ---✅done")
        
        # Identify columns that are almost entirely null
        nan_drops = [c for c in X.columns if X[c].isnull().mean() > 0.99]
        
        if y is not None:
             nan_drops += self._identify_nan_drops(X, y)
             
        # Deduplicate and save 
        self.cols_to_drop = list(set(nan_drops))
        
        if self.verbose: print(f"Dropped {len(self.cols_to_drop)} columns")
        self._fitted = True
        if self.verbose: print("--- Sentinel Preprocessor Fitted ---")
        return self
 
    def transform(self, X: pd.DataFrame, verbose:bool=True) -> pd.DataFrame:
        X = X.copy()
        if verbose: print(f"--- Transforming {len(X)} rows --✅done")         
        
        # 1. Drop Empty Columns
        if self.cols_to_drop: 
            X = X.drop(columns=self.cols_to_drop, errors='ignore')
            
        # 2. Geography
        X = self._set_country(X)
        
        # 3. Parse & Extract
        X = self._process_time_features(X)
        X = self._process_emails(X)           # Rich features added (Risk Scores, Digits)
        X = self._process_m_columns(X)
        X = self._process_card_columns(X)     
        X = self._process_id_columns(X)       # Detailed parsing (OS, Browser)
        X = self._process_product_features(X)
        
        # 4. Structural Fixes
        X = self._process_high_cardinality(X) 
        
        # 5. Statistical Fixes (D-Column Normalization)
        X = self._process_d_columns(X) 
        
        # 6. Cleanup
        X = X.replace([np.inf, -np.inf], np.nan)
        X = self._reduce_memory_usage(X)
        return X
    
    # -------------------------------------------------------------------------
    # FEATURE GROUPS
    # -------------------------------------------------------------------------
    def _process_emails(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses emails and extracts metadata (Length, Free Provider, Digits).
        Includes robust Risk Scoring and Country Mismatch logic.
        """
        # Config
        fix_map = {'gmail': 'gmail.com', 'nan': 'unknown'}
        vendor_map = {
            'gmail': 'google', 'googlemail': 'google',
            'hotmail': 'microsoft', 'outlook': 'microsoft', 'msn': 'microsoft', 'live': 'microsoft',
            'yahoo': 'yahoo', 'ymail': 'yahoo', 'rocketmail': 'yahoo',
            'icloud': 'apple', 'me': 'apple', 'mac': 'apple',
            'protonmail': 'privacy', 'anonymous': 'privacy', 'mailinator': 'privacy',
            'comcast': 'isp', 'verizon': 'isp', 'att': 'isp', 'sbcglobal': 'isp'
        }
        free_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
                        'aol.com', 'mail.ru', 'yandex.ru', 'protonmail.com'}
        
        # Mappers
        vendor_enc = {'unknown': 0, 'google': 1, 'microsoft': 2, 'yahoo': 3, 'apple': 4, 'isp': 5, 'privacy': 6}
        suffix_enc = {'unknown': 0, 'com': 1, 'net': 2, 'edu': 3, 'org': 4, 'mx': 5, 'es': 6, 'de': 7}
        country_map = {'mx': 1, 'es': 2, 'de': 3, 'fr': 4, 'uk': 5, 'jp': 6, 'br': 7, 'ru': 8}

        for col in ['P_emaildomain', 'R_emaildomain']:
            if col in df.columns:
                # 1. Cleaning
                clean_val = df[col].fillna('nan').astype(str).str.lower().replace(fix_map)
                lengths = clean_val.str.len()
                lengths.loc[clean_val == 'nan'] = np.nan
                
                # 2. Extract Structure
                df[f'{col}_length'] = lengths.astype('Int16')
                
                # Digits in email is a huge fraud signal
                df[f'{col}_has_digits'] = clean_val.str.contains(r'\d', regex=True).astype(np.int8) 
                df[f'{col}_is_free'] = clean_val.isin(free_domains).astype(np.int8)
                
                # 3. Parsing
                temp_vendor = clean_val.astype(str).str.split('.', n=1).str[0]
                temp_vendor = temp_vendor.replace('nan', np.nan)

                temp_suffix = clean_val.astype(str).str.rsplit('.', n=1).str[-1]
                temp_suffix = temp_suffix.replace('nan', np.nan) 
                
                group_col = temp_vendor.map(vendor_map).fillna('unknown')
                
                # 4. Encoding
                df[f'{col}_vendor_id'] = group_col.map(vendor_enc).fillna(0).astype(np.int8)
                df[f'{col}_suffix_id'] = temp_suffix.map(suffix_enc).fillna(0).astype(np.int8)
                df[f'{col}_country_id'] = temp_suffix.map(country_map).fillna(0).astype(np.int8)
                
                # 5. Risk Scoring (Heuristic)
                # Base score: Privacy=5, ISP=-1
                risk_map = {'google': 0, 'microsoft': 0, 'yahoo': 0, 'apple': 0, 'isp': -1, 'edu': -2, 'privacy': 5, 'unknown': 1}
                df[f'{col}_risk_score'] = group_col.map(risk_map).fillna(1).astype(np.int8)
                
                # Add risk for bad patterns
                df.loc[df[f'{col}_has_digits'] == 1, f'{col}_risk_score'] += 1
                df.loc[temp_vendor.str.contains('temp|fake|trash', regex=True, na=False), f'{col}_risk_score'] += 3

                # Keep clean string for Feature Engineering phase
                df[col] = clean_val 

        # 6. Interactions (Cross-Border Checks)
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            df['email_match_status'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(np.int8)
            
            p_ctry = df['P_emaildomain_country_id']
            r_ctry = df['R_emaildomain_country_id']
            # If Payer and Recipient countries are known but different
            df['email_country_mismatch'] = ((p_ctry != 0) & (r_ctry != 0) & (p_ctry != r_ctry)).astype(np.int8)

        return df


    def _extract_country_from_domain(self, domain):
        """Parses P_emaildomain suffix to infer geographic origin."""
        if pd.isna(domain) or str(domain).lower() in ['nan', 'none', 'unknown']:
            return "Unknown"
        try:
            # Extract suffix
            suffix = str(domain).lower().strip().split('.')[-1]
            
            # Mapping
            mapping = {
                'mx': 'Mexico', 'es': 'Spain', 'fr': 'France', 'uk': 'United Kingdom',
                'de': 'Germany', 'jp': 'Japan', 'br': 'Brazil', 'ca': 'Canada',
                'ru': 'Russia', 'it': 'Italy', 'au': 'Australia',
                'com': 'USA/Global', 'net': 'USA/Global', 'org': 'USA/Global', 
                'edu': 'USA/Global'
            }
            return mapping.get(suffix, "USA/Global")
        except:
            return "Unknown"


    def _set_country(self, df):
        """Creates the 'country' column based on available email domains."""
        # Check P_emaildomain first
        if 'P_emaildomain' in df.columns:
            df['country'] = df['P_emaildomain'].apply(self._extract_country_from_domain)
        # Fallback to R_emaildomain
        elif 'R_emaildomain' in df.columns:
            df['country'] = df['R_emaildomain'].apply(self._extract_country_from_domain)
        # Default if neither exists
        else:
            if self.verbose: 
                print(f'   Info: No email domain found. Setting country to Unknown.')
            df['country'] = 'Unknown'
        return df
    

    def _process_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses OS, Browser, Screen, Vendor into integers."""
        
        # 1. Resolution
        if 'id_33' in df.columns:
            res = df['id_33'].astype(str).str.split('x', expand=True)
            if res.shape[1] == 2:
                df['screen_width'] = pd.to_numeric(res[0], errors='coerce')
                df['screen_height'] = pd.to_numeric(res[1], errors='coerce')
            df = df.drop(columns=['id_33'], errors='ignore')

        # 2. OS Parsing
        if 'id_30' in df.columns:
            df['id_30'] = df['id_30'].astype(str).str.lower()
            df['os_type'] = 0 
            os_map = {'ios': 1, 'android': 2, 'windows': 3, 'mac': 4, 'linux': 5}
            for os_name, val in os_map.items():
                pat = 'mac os' if os_name == 'mac' else os_name
                df.loc[df['id_30'].str.contains(pat, na=False), 'os_type'] = val
            df['os_type'] = df['os_type'].astype(np.int8)
            df = df.drop(columns=['id_30'], errors='ignore')

        # 3. Browser Parsing 
        if 'id_31' in df.columns:
            df['id_31'] = df['id_31'].astype(str).str.lower()
            df['browser_type'] = 0 # Unknown/Other
            
            # Group 1: High Risk / Niche / Privacy Browsers
            risk_browsers = 'puffin|maxthon|comodo|iron|silk|line|aol|cyberfox|waterfox|palemoon|seamonkey'
            df.loc[df['id_31'].str.contains(risk_browsers, na=False), 'browser_type'] = 1
            
            # Group 2: Samsung (Browser or Device signature)
            df.loc[df['id_31'].str.contains('samsung', na=False), 'browser_type'] = 2
            
            # Group 3: Android Native / WebView / Generic / Manufacturers
            native_patterns = 'android|generic|zte|lg/|mot|huawei|lanix|blade|nokia|m4tel|inco|blu'
            df.loc[df['id_31'].str.contains(native_patterns, na=False), 'browser_type'] = 3
            
            # Group 4: Opera
            df.loc[df['id_31'].str.contains('opera', na=False), 'browser_type'] = 4
            
            # Group 5: Firefox (Standard)
            df.loc[(df['browser_type'] == 0) & df['id_31'].str.contains('firefox|mozilla', na=False), 'browser_type'] = 5
            
            # Group 6: IE / Edge (Microsoft)
            df.loc[df['id_31'].str.contains('edge|ie |trident|msie|microsoft', na=False), 'browser_type'] = 6
            
            # Group 7: Chrome (Mobile vs Desktop)
            mask_chrome = (df['browser_type'] == 0) & df['id_31'].str.contains('chrome|chromium|google', na=False)
            mask_mobile = df['id_31'].str.contains('mobile|android|ios', na=False)
            
            df.loc[mask_chrome & mask_mobile, 'browser_type'] = 7  # Chrome Mobile
            df.loc[mask_chrome & ~mask_mobile, 'browser_type'] = 8 # Chrome Desktop
            
            # Group 8: Safari (Mobile vs Desktop)
            mask_safari = (df['browser_type'] == 0) & df['id_31'].str.contains('safari', na=False)
            
            df.loc[mask_safari & mask_mobile, 'browser_type'] = 9  # Safari Mobile
            df.loc[mask_safari & ~mask_mobile, 'browser_type'] = 10 # Safari Desktop

            df['browser_type'] = df['browser_type'].astype(np.int8)
            df = df.drop(columns=['id_31'], errors='ignore')

        # 4. Status/Proxy Mappings
        status_map = {'Found': 1, 'New': 1, 'NotFound': 0, 'Unknown': 0, 'nan': -1}
        for col in ['id_12', 'id_15', 'id_16', 'id_27', 'id_28', 'id_29']:
            if col in df.columns:
                df[col] = df[col].astype(str).map(status_map).fillna(-1).astype(np.int8)

        if 'id_23' in df.columns:
            proxy_map = {'IP_PROXY:TRANSPARENT': 1, 'IP_PROXY:ANONYMOUS': 2, 'IP_PROXY:HIDDEN': 3, 'nan': -1}
            df['id_23'] = df['id_23'].astype(str).map(proxy_map).fillna(-1).astype(np.int8)
            
        if 'id_34' in df.columns:
             df['id_34'] = df['id_34'].astype(str).str.split(':', expand=True).iloc[:, -1]
             df['id_34'] = pd.to_numeric(df['id_34'], errors='coerce').fillna(-1).astype(np.int8)

        id_map = {'nan': -1, 'T': 1, 'F': 0}
        for col in ['id_35', 'id_36', 'id_37', 'id_38']:    
            if col in df.columns:
                df[col] = df[col].astype(str).map(id_map).fillna(-1).astype(np.int8)

        # 5. Device Info (Vendor)
        if 'DeviceInfo' in df.columns:
            df['DeviceInfo'] = df['DeviceInfo'].astype(str).str.lower()
            df['device_vendor'] = 0 
            v_map = {
                'samsung': 1, 'sm-': 1, 'gt-': 1, 'ios': 2, 'apple': 2, 'iphone': 2,
                'huawei': 3, 'honor': 3, 'moto': 4, 'lg': 5, 'sony': 6, 'zte': 7, 
                'pixel': 8, 'lenovo': 9, 'alcatel': 10, 'redmi': 11, 'windows': 12, 
                'rv:': 13, 'htc': 14, 'asus': 15
            }
            for pattern, val in v_map.items():
                df.loc[df['DeviceInfo'].str.contains(pattern, na=False, regex=False), 'device_vendor'] = val
            df['device_vendor'] = df['device_vendor'].astype(np.int8)
            df = df.drop(columns=['DeviceInfo'], errors='ignore')

        if 'DeviceType' in df.columns:
            d_type_map = {'mobile': 1, 'desktop': 0, 'nan': -1}
            df['DeviceType'] = df['DeviceType'].astype(str).map(d_type_map).fillna(-1).astype(np.int8)

        return df

    def _process_high_cardinality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles Float-looking IDs safely."""
        cat_cols = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']
        for col in cat_cols:
            if col in df.columns:
                # 1. Fill NaNs
                df[col] = df[col].fillna(-1)
                
                # 2. Safe Float->Int cast
                if pd.api.types.is_float_dtype(df[col]):
                    # Only cast if safe
                    if (df[col] % 1 == 0).all():
                        df[col] = df[col].astype(np.int32)
        # Distances stay as float
        if 'dist1' in df.columns: df['dist1'] = df['dist1'].astype(np.float32)
        if 'dist2' in df.columns: df['dist2'] = df['dist2'].astype(np.float32)

        return df

    # --- Standard Helpers ---
    def _process_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dt_series = self.start_date + pd.to_timedelta(df['TransactionDT'], unit='s')
        df['hour_of_day'] = dt_series.dt.hour.astype(np.int8)
        df['day_of_week'] = dt_series.dt.dayofweek.astype(np.int8)
        df['day_of_month'] = dt_series.dt.day.astype(np.int8)
        df['month_year'] = (dt_series.dt.year * 100) + dt_series.dt.month
        return df

    def _process_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        product_map = {'W': 1, 'H': 2, 'C': 3, 'S': 4, 'R': 5}
        df['ProductCD'] = df['ProductCD'].map(product_map).fillna(0).astype(np.int8)
        return df

    def _process_m_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        m_cols = [c for c in df.columns if c.startswith('M')]
        mapping = {'T': 1, 'F': 0, 'M0': 0, 'M1': 1, 'M2': 2}
        for col in m_cols:
            df[col] = df[col].map(mapping).fillna(-1).astype(np.int8)
        return df
    
    def _process_card_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'card4' in df.columns:
            c4_map = {'visa': 1, 'mastercard': 2, 'american express': 3, 'discover': 4}
            df['card4'] = df['card4'].astype(str).str.lower().map(c4_map).fillna(-1).astype(np.int8)
        if 'card6' in df.columns:
            c6_map = {'debit': 1, 'credit': 2, 'debit or credit': 3, 'charge card': 4}
            df['card6'] = df['card6'].astype(str).str.lower().map(c6_map).fillna(-1).astype(np.int8)
        return df

    def _process_d_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        d_cols = [c for c in df.columns if c.startswith('D') and c[1:].isdigit()]
        dt_days = (df['TransactionDT'] / 86400).astype(np.float32)
        for col in d_cols:
            # Normalize to "Account Start Date"
            df[f'{col}_norm'] = dt_days - df[col].astype(np.float32)
        return df

    def _identify_nan_drops(self, df: pd.DataFrame, y: Union[str, pd.Series]) -> List[str]:
        to_drop = []
        target = df[y] if isinstance(y, str) else y
        for col in df.columns:
            if col in ['TransactionID', 'TransactionDT', 'TransactionAmt'] or col not in df.columns: continue
            mask_null = df[col].isnull()
            if mask_null.mean() > self.nan_threshold:
                if mask_null.all() or (~mask_null).sum() < 100: 
                    to_drop.append(col)
                    continue
                if abs(target[mask_null].mean() - target[~mask_null].mean()) < 0.01:
                    to_drop.append(col)
        return to_drop


    def _reduce_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            # 1. THE ROBUST CHECK: Only process numeric columns.
            # This automatically skips 'object', 'string', 'category', and 'datetime'
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # 2. Double check: Skip if it's a Categorical type
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                continue

            c_min = df[col].min()
            c_max = df[col].max()

            # 3. Check for Nulls/Infinites before math
            if pd.isna(c_min) or pd.isna(c_max):
                # If numeric but has NaNs, we can still downcast to float32
                if df[col].dtype != np.float32:
                    df[col] = df[col].astype(np.float32)
                continue

            # 4. Math check (Safe now because we know it's numeric)
            if (df[col] % 1 == 0).all():
                if c_min >= 0:
                    if c_max < 255: df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535: df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295: df[col] = df[col].astype(np.uint32)
                    else: df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > -128 and c_max < 127: df[col] = df[col].astype(np.int8)
                    elif c_min > -32768 and c_max < 32767: df[col] = df[col].astype(np.int16)
                    elif c_min > -2147483648 and c_max < 2147483647: df[col] = df[col].astype(np.int32)
                    else: df[col] = df[col].astype(np.int64)
            else:
                if df[col].dtype != np.float32:
                    df[col] = df[col].astype(np.float32)
                    
        return df

if __name__ == "__main__":
    print("Sentinel Preprocessing Module")