import sys
import os
import time
import logging
import traceback
import pandas as pd
import streamlit as st

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import setup_page, COLORS, render_top_banner
from api_client import SentinelClient  
from views import executive, ops, ml, strategy, forensics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")

# ==============================================================================
# 1. SETUP & STATE
# ==============================================================================
setup_page("Sentinel Ops Center")

if 'api_client' not in st.session_state:
    st.session_state.api_client = SentinelClient()
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False

# Initialize Cache
if 'last_stats' not in st.session_state: st.session_state.last_stats = {}
if 'last_df' not in st.session_state: st.session_state.last_df = pd.DataFrame()

client = st.session_state.api_client

# ==============================================================================
# 2. SIDEBAR (NAVIGATION & CONTROLS)
# ==============================================================================
with st.sidebar:
    st.markdown(f"<h1 style='text-align: center; color: {COLORS['highlight']}; letter-spacing: 2px; margin-bottom: 0;'>Sentinel Fraud Ops</h1>", unsafe_allow_html=True)
    st.caption("Real-time Fraud Detection System")
    st.markdown("---")
    
    # 2.1 Navigation
    page = st.radio(
        "MODULES", 
        ["Executive View", "Ops Center", "ML Monitor", "Strategy", "Forensics"],
        index=0
    )
    
    st.markdown("---")
    
    # 2.2 Preferences
    with st.expander("⚙️ View Settings", expanded=True):
        refresh_rate = st.slider("Refresh Rate (s)", 1, 60, 5)
        # We only pass data_limit to the API, so this is efficient
        data_limit = st.select_slider("History Depth", options=[100, 500, 1000, 2000], value=500)
    
    st.markdown("---")

    # 2.3 Live Control
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏸ PAUSE" if not st.session_state.is_paused else "▶ RESUME"):
            st.session_state.is_paused = not st.session_state.is_paused
            st.rerun()
            
    with c2:
        if st.button("🔄 REFRESH"):
            st.session_state.is_paused = False 
            st.cache_data.clear()
            st.rerun()

    if st.session_state.is_paused:
        st.warning("⚠️ Feed Paused")
    else:
        st.success("🟢 Feed Active")

    # 2.4 Backend Status
    st.markdown("---")
    try:
        # Use the lightweight health check
        if client.get_system_health():
            st.caption("🟢 Backend: Online")
            # Only fetch metrics if backend is alive
            sys_metrics = client.get_system_metrics()
            cpu = sys_metrics.get('cpu_usage_percent', 0)
            mem = sys_metrics.get('memory_usage_mb', 0)
            st.progress(min(cpu/100, 1.0), f"CPU: {cpu}% | Mem: {mem}MB")
        else:
            st.error("❌ Backend: Offline")
    except Exception:
        st.error("❌ Backend: Unreachable")

# ==============================================================================
# 3. ROBUST DATA LOADING
# ==============================================================================
def load_data(limit):
    """
    Fetches all data needed for the dashboard.
    Returns: (stats, df, timeseries, curve_df, alerts_df)
    """
    # 1. If paused, strictly use cache
    if st.session_state.is_paused:
        return (
            st.session_state.get('last_stats', {}),
            st.session_state.get('last_df', pd.DataFrame()),
            st.session_state.get('last_series', pd.DataFrame()),
            st.session_state.get('last_curve', pd.DataFrame()),
            st.session_state.get('last_alert', pd.DataFrame()),
        )

    try:
        # 2. Attempt Fetch from API
        stats = client.get_dashboard_stats()
        df = client.get_recent_transactions(limit=limit)
        timeseries = client.get_financial_timeseries()
        curve_df = client.get_threshold_optimization_curve()
        alerts_df = client.get_alerts(limit=100)
        
        # 3. Update Cache
        st.session_state.last_stats = stats
        st.session_state.last_df = df
        st.session_state.last_series = timeseries
        st.session_state.last_curve = curve_df
        
        return stats, df, timeseries, curve_df, alerts_df

    except Exception as e:
        logger.error(f"Data Fetch Error: {e}")
        # On error, return cache as fallback
        return (
            st.session_state.get('last_stats', {}),
            st.session_state.get('last_df', pd.DataFrame()),
            st.session_state.get('last_series', pd.DataFrame()),
            st.session_state.get('last_curve', pd.DataFrame())
        )

# ==============================================================================
# 4. MAIN CONTROLLER
# ==============================================================================
def main():
    try:
        #render_top_banner()

        stats, df, timeseries_df, curve_df, alerts_df = load_data(limit=data_limit)
        
        # Default threshold if not provided by backend
        current_threshold = stats.get('threshold', 0.5)

        if page == "Executive View":
            executive.render_page(
                recent_df=df, 
                metrics=stats, 
                threshold=current_threshold, 
                timeseries_df=timeseries_df,
                curve_df=curve_df
                )
            
        elif page == "Ops Center":
            ops.load_view() # Using standard naming convention from your first prompt
            
        elif page == "ML Monitor":
            ml.load_view()
            
        elif page == "Strategy":
            strategy.load_view()
            
        elif page == "Forensics":
            forensics.load_view()

        # Auto-refresh logic
        if not st.session_state.is_paused:
            time.sleep(refresh_rate)
            st.rerun()

    except Exception as e:
        st.error("🚨 An unexpected error occurred in the dashboard controller.")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

