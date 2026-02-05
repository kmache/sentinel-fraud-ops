import pandas as pd
from config import COLUMN_MAPPING

def format_currency(value: float) -> str:
    """Formats large numbers into readable currency (e.g., $1.2M)."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.2f}"

def format_percent(value: float) -> str:
    """Formats a decimal into a clean percentage string."""
    return f"{value:.1%}"

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes a DataFrame from the API.
    1. Renames columns based on config.
    2. Removes duplicates.
    3. Resets index for crash-free alignment.
    """
    if df.empty:
        return df

    # Work on a copy to avoid mutating session state
    cleaned = df.copy()

    # Apply mapping from config.py
    rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in cleaned.columns}
    cleaned = cleaned.rename(columns=rename_dict)

    # Ensure timestamp is a datetime object for Plotly
    if 'timestamp' in cleaned.columns:
        cleaned['timestamp'] = pd.to_datetime(cleaned['timestamp'])

    # Deduplicate and fix index
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
    cleaned = cleaned.reset_index(drop=True)

    return cleaned

    