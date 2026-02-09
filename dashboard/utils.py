import pandas as pd
from config import COLUMN_MAPPING, StandardColumns

def format_currency(value: float) -> str:
    """Formats large numbers into readable currency (e.g., $1.2M)."""
    if value is None: 
        return "$0.00"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.2f}"

def format_percent(value: float) -> str:
    """Formats a decimal into a clean percentage string."""
    if value is None:
        return "0%"
    return f"{value:.1%}"

def clean_dataframe(data) -> pd.DataFrame:
    """
    Standardizes a DataFrame from the API.
    1. Converts list of dicts to DataFrame if needed.
    2. Renames columns based on config.
    3. Enforces types (Timestamps).
    """
    # 1. Convert to DataFrame if it's a list
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    if df.empty:
        return df

    # Work on a copy
    cleaned = df.copy()

    # 2. Apply mapping from config.py
    # This looks at the columns in 'cleaned' and renames them if they exist in mapping
    rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in cleaned.columns}
    cleaned = cleaned.rename(columns=rename_dict)

    # 3. Ensure timestamp is a datetime object
    ts_col = StandardColumns.TIMESTAMP
    if ts_col in cleaned.columns:
        cleaned[ts_col] = pd.to_datetime(cleaned[ts_col], errors='coerce')

    # 4. Fill NaNs for display safety
    if StandardColumns.SCORE in cleaned.columns:
        cleaned[StandardColumns.SCORE] = cleaned[StandardColumns.SCORE].fillna(0)

    # Deduplicate and reset index
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
    cleaned = cleaned.reset_index(drop=True)

    return cleaned