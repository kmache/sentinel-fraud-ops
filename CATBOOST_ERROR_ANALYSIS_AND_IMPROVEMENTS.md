# Sentinel Fraud Detection System - CatBoost Error Analysis & Improvement Recommendations

## Executive Summary

This document provides a comprehensive analysis of the CatBoost categorical feature error and suggests improvements across the entire codebase to enhance robustness, maintainability, and production reliability.

## Error Analysis

### Root Cause: CatBoost Categorical Feature Mismatch

**Error Message**: `Invalid new DataFrame. The data type doesn't match the one used in the training dataset. Both should be either numeric or categorical. For a categorical feature, the index type must match between the training and test set.`

**Primary Issues Identified**:

1. **Categorical Feature Inconsistency**: The system uses XGBoost in production (`weights: {"xgb": 1.0}`) but has CatBoost model artifacts and categorical feature definitions that may not align with current data processing.

2. **Feature Schema Drift**: The categorical features list contains 33 features, but the actual inference pipeline may generate different feature names or data types.

3. **Data Type Mismatch**: Features that were categorical during training may be numeric during inference, or vice versa.

## Detailed Code Analysis & Recommendations

### 1. Inference Pipeline (`src/sentinel/inference.py`)

#### Current Issues:
- **Line 131**: Categorical conversion is fragile: `tem_df[col] = tem_df[col].astype(str).replace(['nan', 'None', '<NA>'], np.nan).astype('category')`
- **Line 123-125**: Missing features are filled with `np.nan` without considering categorical requirements
- **Line 179**: Generic exception handling masks specific CatBoost errors

#### Recommended Improvements:

```python
# Enhanced categorical handling with validation
def _validate_categorical_features(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Validate and fix categorical features for specific model requirements."""
    required_features = self.features[model_name]
    
    for col in required_features:
        if col not in df.columns:
            logger.warning(f"Missing feature {col} for model {model_name}")
            # Create missing feature with appropriate default
            if col in self.cat_features:
                df[col] = 'UNKNOWN'  # Categorical default
            else:
                df[col] = 0.0  # Numeric default
    
    # Ensure categorical features have consistent data types
    for col in self.cat_features:
        if col in df.columns:
            # Convert to string first to handle mixed types
            df[col] = df[col].astype(str).replace(['nan', 'None', '<NA>', 'NaN'], 'UNKNOWN')
            df[col] = df[col].astype('category')
    
    return df

# Enhanced error handling with specific CatBoost diagnostics
def predict(self, data: Union[Dict, pd.DataFrame], soft_threshold: float = 0.12) -> Dict[str, Any]:
    try:
        # ... existing code ...
        
        for name, weight in weights.items():
            required_features = self.features[name]
            
            # Validate feature schema before prediction
            tem_df = self._validate_categorical_features(self.df_features, name)
            tem_df = tem_df[required_features].copy()
            
            # Additional validation for CatBoost
            if name == 'cb':  # CatBoost specific handling
                self._validate_catboost_schema(tem_df, name)
            
            p = self.models[name].predict_proba(tem_df)[:, 1]
            final_probs += p * weight
            
    except Exception as e:
        # Enhanced error reporting
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "model_name": name if 'name' in locals() else "unknown",
            "feature_count": len(tem_df.columns) if 'tem_df' in locals() else 0,
            "categorical_features": len([c for c in tem_df.columns if c in self.cat_features]) if 'tem_df' in locals() else 0
        }
        logger.error(f"Prediction failed: {error_details}")
        raise RuntimeError(f"Inference pipeline failed: {error_details}")
```

### 2. Preprocessing Pipeline (`src/sentinel/preprocessing.py`)

#### Current Issues:
- **Line 119**: Email processing creates inconsistent categorical encodings
- **Lines 130-132**: Hard-coded integer mappings may not match training schema
- **Line 374-379**: Memory optimization may change data types unexpectedly

#### Recommended Improvements:

```python
# Enhanced categorical feature consistency
def _process_emails(self, df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced email processing with consistent categorical handling."""
    
    # Define consistent categorical mappings (should match training)
    CATEGORICAL_MAPPINGS = {
        'vendor_enc': {'unknown': 0, 'google': 1, 'microsoft': 2, 'yahoo': 3, 'apple': 4, 'isp': 5, 'privacy': 6},
        'suffix_enc': {'unknown': 0, 'com': 1, 'net': 2, 'edu': 3, 'org': 4, 'mx': 5, 'es': 6, 'de': 7},
        'country_map': {'mx': 1, 'es': 2, 'de': 3, 'fr': 4, 'uk': 5, 'jp': 6, 'br': 7, 'ru': 8}
    }
    
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            # Consistent cleaning and categorization
            clean_val = df[col].fillna('unknown').astype(str).str.lower()
            
            # Create categorical features with consistent categories
            df[f'{col}_vendor_id'] = pd.Categorical(
                clean_val.str.split('.', n=1).str[0].map(CATEGORICAL_MAPPINGS['vendor_enc']).fillna('unknown'),
                categories=list(CATEGORICAL_MAPPINGS['vendor_enc'].keys())
            )
            
            # Ensure consistent data types for production
            df[f'{col}_vendor_id'] = df[f'{col}_vendor_id'].cat.codes.astype('int8')
    
    return df

# Enhanced memory optimization with categorical preservation
def _reduce_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
    """Memory optimization that preserves categorical data types."""
    for col in df.columns:
        # Skip categorical columns entirely
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            continue
            
        # Skip string/object columns that should remain categorical
        if col in self.cat_features or any(col.endswith(suffix) for suffix in ['_id', '_type', '_combo']):
            continue
            
        # Existing numeric optimization logic...
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # ... rest of existing logic ...
    
    return df
```

### 3. Feature Engineering (`src/sentinel/features.py`)

#### Current Issues:
- **Line 263**: Creates new categorical combos without consistent schema
- **Line 393**: High-cardinality categoricals are converted to codes without validation
- **Line 396**: UID hashing creates inconsistent values across runs

#### Recommended Improvements:

```python
# Enhanced categorical feature creation with schema validation
def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features with consistent categorical schemas."""
    
    # Define expected categorical schemas (should match training)
    EXPECTED_CATEGORIES = {
        'product_network_combo': ['W_visa', 'W_mastercard', 'H_visa', 'C_discover', 'S_american express'],
        'card1_addr1_combo': None,  # High cardinality - use codes
        'os_browser_combo': ['3_8', '3_10', '2_8', '1_8', '4_10']  # Example categories
    }
    
    if 'ProductCD' in df.columns and 'card4' in df.columns:
        combo = df['ProductCD'].astype(str) + '_' + df['card4'].astype(str)
        
        if 'product_network_combo' in EXPECTED_CATEGORIES and EXPECTED_CATEGORIES['product_network_combo']:
            # Use predefined categories for consistency
            df['product_network_combo'] = pd.Categorical(
                combo,
                categories=EXPECTED_CATEGORIES['product_network_combo']
            )
        else:
            # High cardinality - use consistent encoding
            df['product_network_combo'] = combo.astype('category')
    
    return df

# Enhanced UID generation with consistency
def _make_uid(self, df: pd.DataFrame) -> pd.DataFrame:
    """Generate consistent UID across training and inference."""
    d_col = 'D1_norm' if 'D1_norm' in df.columns else 'D1'
    
    # Use consistent string formatting and hashing
    c1 = df['card1'].fillna(-1).astype(int).astype(str).str.zfill(8)
    a1 = df['addr1'].fillna(-1).astype(int).astype(str).str.zfill(6)
    
    if d_col in df.columns:
        d1_weekly = (df[d_col].fillna(-999) // 7).astype(int).astype(str).str.zfill(6)
    else:
        d1_weekly = "000000"
        
    df['UID'] = c1 + '_' + a1 + '_' + d1_weekly
    
    # Add consistent hash for high-cardinality handling
    df['UID_hash'] = (df['UID'].apply(lambda x: abs(hash(str(x)) % 10000))).astype('int16')
    
    return df
```

### 4. Worker Processor (`worker/processor.py`)

#### Current Issues:
- **Lines 178-187**: Data type conversion may not match model expectations
- **Line 189**: No validation of input schema before inference
- **Lines 191-196**: Error handling loses specific diagnostic information

#### Recommended Improvements:

```python
# Enhanced data preparation with schema validation
def prepare_inference_data(self, raw_data):
    """Prepare data with schema validation for inference."""
    df = pd.DataFrame(raw_data)
    df = df.replace({None: np.nan})
    
    # Define expected schema (should match training)
    EXPECTED_SCHEMA = {
        'TransactionID': 'Int64',
        'TransactionAmt': 'float64',
        'ProductCD': 'object',
        'card4': 'object',
        'P_emaildomain': 'object'
    }
    
    # Validate and fix data types
    for col, expected_type in EXPECTED_SCHEMA.items():
        if col in df.columns:
            try:
                if expected_type == 'Int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif expected_type == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                elif expected_type == 'object':
                    df[col] = df[col].astype(str)
            except Exception as e:
                logger.warning(f"Failed to convert {col} to {expected_type}: {e}")
    
    return df

# Enhanced batch processing with detailed error reporting
def flush_batch(self, messages):
    if not messages: 
        return
    
    try:
        raw_data = [m.value for m in messages]
        df = self.prepare_inference_data(raw_data)
        
        # Validate input schema before inference
        self._validate_input_schema(df)
        
        results = self.model_engine.predict(df)
        
    except Exception as e:
        # Enhanced error reporting with batch diagnostics
        error_info = {
            "batch_size": len(messages),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "sample_data": raw_data[:2] if raw_data else [],  # First 2 samples for debugging
            "data_shape": df.shape if 'df' in locals() else None,
            "data_columns": list(df.columns) if 'df' in locals() else []
        }
        
        log_event("Inference Batch Failed", **error_info)
        logger.error("❌ CRITICAL INFERENCE ERROR:")
        logger.error(traceback.format_exc())
        
        self.send_to_dlq(raw_data, "inference_batch_error", e)
        return

def _validate_input_schema(self, df):
    """Validate input data schema against expected model requirements."""
    REQUIRED_COLUMNS = ['TransactionID', 'TransactionAmt', 'ProductCD']
    
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for data type issues
    if 'TransactionAmt' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['TransactionAmt']):
            raise ValueError("TransactionAmt must be numeric")
```

## System-Wide Improvements

### 1. Configuration Management

```python
# config/model_schema.py
"""Centralized model schema configuration."""

MODEL_SCHEMAS = {
    'xgb': {
        'categorical_features': [
            'ProductCD', 'card6', 'product_network_combo', 'P_emaildomain',
            'M3', 'id_38', 'R_emaildomain', 'device_info_combo', 'M1',
            'os_type', 'M4', 'card2', 'card1_addr1_combo', 'card4'
        ],
        'numeric_features': [
            'TransactionAmt', 'card1', 'addr1', 'dist1', 'C1', 'C13', 'C14'
        ],
        'required_columns': ['TransactionID', 'TransactionAmt', 'ProductCD']
    },
    'cb': {
        'categorical_features': [
            'ProductCD', 'card6', 'product_network_combo', 'P_emaildomain',
            'M3', 'id_38', 'R_emaildomain', 'device_info_combo', 'M1',
            'os_type', 'M4', 'card2', 'card1_addr1_combo', 'card4'
        ],
        'numeric_features': [
            'card_email_combo_fraud_rate', 'card1_freq_enc', 'C13', 'C1'
        ],
        'required_columns': ['TransactionID', 'TransactionAmt', 'ProductCD']
    }
}
```

### 2. Enhanced Validation Pipeline

```python
# src/sentinel/validation.py
class ProductionValidator:
    """Validate data schema and model compatibility."""
    
    def __init__(self, model_config_path: str):
        with open(model_config_path, 'r') as f:
            self.model_config = json.load(f)
    
    def validate_inference_data(self, df: pd.DataFrame, model_name: str) -> bool:
        """Validate data schema for specific model."""
        schema = MODEL_SCHEMAS.get(model_name, {})
        
        # Check required columns
        missing_cols = set(schema.get('required_columns', [])) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for {model_name}: {missing_cols}")
        
        # Validate categorical features
        cat_features = schema.get('categorical_features', [])
        for col in cat_features:
            if col in df.columns:
                if not isinstance(df[col].dtype, pd.CategoricalDtype):
                    logger.warning(f"Categorical feature {col} is not categorical type")
        
        return True
    
    def check_feature_drift(self, df: pd.DataFrame, reference_stats: dict) -> dict:
        """Check for feature drift compared to training data."""
        drift_report = {}
        
        for col, stats in reference_stats.items():
            if col in df.columns:
                current_mean = df[col].mean()
                ref_mean = stats.get('mean', 0)
                
                # Simple drift detection
                drift_pct = abs(current_mean - ref_mean) / (ref_mean + 1e-6)
                drift_report[col] = {
                    'drift_percentage': drift_pct,
                    'current_mean': current_mean,
                    'reference_mean': ref_mean,
                    'status': 'high' if drift_pct > 0.1 else 'low'
                }
        
        return drift_report
```

### 3. Monitoring and Alerting

```python
# src/sentinel/monitoring.py
class ModelMonitor:
    """Monitor model performance and data drift in production."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def log_prediction_stats(self, predictions: dict, input_features: dict):
        """Log prediction statistics for monitoring."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'prediction_count': len(predictions.get('probabilities', [])),
            'avg_probability': np.mean(predictions.get('probabilities', [])),
            'fraud_rate': np.mean(predictions.get('is_frauds', [])),
            'feature_count': len(input_features),
            'categorical_feature_count': len([f for f in input_features.keys() 
                                           if f in MODEL_SCHEMAS['xgb']['categorical_features']])
        }
        
        self.redis.lpush('model_stats', json.dumps(stats))
        self.redis.ltrim('model_stats', 0, 999)  # Keep last 1000 entries
    
    def check_model_health(self) -> dict:
        """Check overall model health metrics."""
        recent_stats = self.redis.lrange('model_stats', 0, 99)  # Last 100 entries
        
        if not recent_stats:
            return {'status': 'no_data', 'message': 'No recent statistics available'}
        
        # Parse recent stats
        stats_data = [json.loads(s) for s in recent_stats]
        
        avg_fraud_rate = np.mean([s['fraud_rate'] for s in stats_data])
        avg_probability = np.mean([s['avg_probability'] for s in stats_data])
        
        health_status = {
            'status': 'healthy',
            'avg_fraud_rate': avg_fraud_rate,
            'avg_probability': avg_probability,
            'prediction_count': sum([s['prediction_count'] for s in stats_data]),
            'alerts': []
        }
        
        # Simple health checks
        if avg_fraud_rate > 0.1:  # Unusually high fraud rate
            health_status['alerts'].append('High fraud rate detected')
            health_status['status'] = 'warning'
        
        if avg_probability > 0.5:  # Unusually high prediction confidence
            health_status['alerts'].append('High prediction confidence - possible drift')
            health_status['status'] = 'warning'
        
        return health_status
```

## Implementation Priority

### High Priority (Immediate - Fixes CatBoost Error)
1. **Fix categorical feature consistency in inference.py** (Lines 130-131)
2. **Add schema validation before model prediction**
3. **Enhance error handling with specific diagnostics**

### Medium Priority (Next Sprint)
1. **Implement centralized configuration management**
2. **Add production data validation pipeline**
3. **Enhance monitoring and alerting**

### Low Priority (Future Enhancements)
1. **Automated drift detection**
2. **Model performance monitoring dashboard**
3. **Automated retraining triggers**

## Testing Recommendations

### 1. Unit Tests for Schema Validation
```python
def test_categorical_feature_consistency():
    """Test that categorical features are consistent between training and inference."""
    # Load training schema
    training_schema = load_training_schema()
    
    # Process sample data
    preprocessor = SentinelPreprocessing()
    sample_data = create_sample_transaction_data()
    processed_data = preprocessor.transform(sample_data)
    
    # Validate categorical features
    for cat_feature in training_schema['categorical_features']:
        assert cat_feature in processed_data.columns
        assert isinstance(processed_data[cat_feature].dtype, pd.CategoricalDtype)
```

### 2. Integration Tests for Inference Pipeline
```python
def test_inference_pipeline_robustness():
    """Test inference pipeline with various data quality issues."""
    model_engine = SentinelInference(model_dir='test_models')
    
    # Test with missing features
    incomplete_data = {'TransactionID': 1, 'TransactionAmt': 100}
    result = model_engine.predict(incomplete_data)
    assert 'error' not in result
    
    # Test with wrong data types
    wrong_types_data = {'TransactionID': 'abc', 'TransactionAmt': '100', 'ProductCD': 123}
    result = model_engine.predict(wrong_types_data)
    assert 'error' not in result
```

## Conclusion

The CatBoost error is primarily caused by categorical feature schema mismatches between training and inference. The recommended improvements focus on:

1. **Immediate fixes** for categorical feature handling
2. **Enhanced validation** to prevent schema drift
3. **Better error handling** with specific diagnostics
4. **Long-term monitoring** to detect issues early

Implementing these recommendations will significantly improve the system's robustness and prevent similar errors in production.
