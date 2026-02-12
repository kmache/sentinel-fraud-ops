import numpy as np

def generate_feature_baseline(df, feature_cols):
    baseline = {}
    for col in feature_cols:
        null_count = df[col].isna().sum()
        total_count = len(df)
        null_rate = null_count / total_count

        clean_values = df[col].dropna().values
        clean_values = clean_values[np.isfinite(clean_values)]

        if len(clean_values) > 100:
            counts, bin_edges = np.histogram(clean_values, bins=10)
            percents = (counts / len(clean_values)).tolist()
            edges = bin_edges.tolist()
        else:
            percents = []
            edges = []

        baseline[col] = {
            "expected_null_rate": float(null_rate),
            "expected_pct": percents,
            "bin_edges": edges,
            "is_sparse": bool(null_rate > 0.8)
        }
    return baseline


def calculate_psi(expected_pct, actual_values, bin_edges):
    """
    Calculates the Population Stability Index (PSI).
    
    Args:
        expected_pct (list/np.array): Pre-calculated proportions from training baseline.
        actual_values (list/np.array): Raw numerical values from the live stream.
        bin_edges (list/np.array): The exact boundaries created during training.
        
    Returns:
        float: The PSI score.
    """
    try:
        expected_pct = np.array(expected_pct)
        actual_values = np.array(actual_values)
        bin_edges = np.array(bin_edges)

        actual_values = actual_values[np.isfinite(actual_values)]
        
        if len(actual_values) == 0:
            return 0.0
        actual_counts, _ = np.histogram(actual_values, bins=bin_edges)
        
        actual_pct = actual_counts / len(actual_values)


        expected_pct = np.clip(expected_pct, 1e-6, None)
        actual_pct = np.clip(actual_pct, 1e-6, None)

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi_val)

    except Exception as e:
        print(f"Error calculating PSI: {e}")
        return 0.0
    
def check_feature_drift(live_data_array, baseline_item):
    """
    Check drift for a single feature, handling high NaN cases.
    """
    # 1. Check Availability Drift (Null Rate Change)
    live_null_rate = np.isnan(live_data_array).sum() / len(live_data_array)
    null_drift = abs(live_null_rate - baseline_item['expected_null_rate'])
    
    # 2. Check Distribution Drift (PSI)
    psi_score = 0.0
    if not baseline_item['is_sparse'] and len(baseline_item['bin_edges']) > 0:
        # Only compute PSI if the feature isn't too sparse
        clean_live = live_data_array[np.isfinite(live_data_array)]
        if len(clean_live) > 50:
            psi_score = calculate_psi(
                np.array(baseline_item['expected_pct']),
                clean_live,
                np.array(baseline_item['bin_edges'])
            )

    return {
        "psi": psi_score,
        "null_drift": null_drift,
        "status": "RED" if (psi_score > 0.2 or null_drift > 0.1) else "GREEN"
    }