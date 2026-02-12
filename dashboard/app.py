import sys
import os
import time
import logging
import traceback
import pandas as pd
import streamlit as st

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
    
    PAGES = ["Executive View", "Ops Center", "ML Monitor", "Strategy", "Forensics"]
    
    if "page" in st.query_params:
        url_page = st.query_params["page"]
        if url_page in PAGES:
            st.session_state.current_page = url_page

    def update_url():
        st.query_params["page"] = st.session_state.current_page

    page = st.radio(
        "MODULES", 
        PAGES,
        key="current_page", 
        on_change=update_url
    )
    
    st.markdown("---")
    
    with st.expander("⚙️ View Settings", expanded=True):
        refresh_rate = st.slider("Refresh Rate (s)", 1, 60, 5)
        data_limit = st.select_slider("History Depth", options=[100, 500, 1000, 2000], value=500)
    
    st.markdown("---")

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

    st.markdown("---")
    try:

        if client.get_system_health():
            st.caption("🟢 Backend: Online")
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
            st.session_state.get('last_explain', pd.DataFrame()),
            st.session_state.get('last_drift', {}),
            st.session_state.get('last_lookup', {}),
            st.session_state.get('last_calibration', {})
        )

    try:
        # 2. Attempt Fetch from API
        stats = client.get_dashboard_stats()
        df = client.get_recent_transactions(limit=limit)
        timeseries = client.get_financial_timeseries()
        curve_df = client.get_threshold_optimization_curve()
        alerts_df = client.get_alerts(limit=100)
        explain_df = client.get_global_feature_importance()
        drift_dict = client.get_feature_drift_report()
        look_up_table = client.get_performance_lookup()
        calibration_data = client.get_calibration_report()
        
        # 3. Update Cache
        st.session_state.last_stats = stats
        st.session_state.last_df = df
        st.session_state.last_series = timeseries
        st.session_state.last_explain = explain_df
        st.session_state.last_curve = curve_df
        st.session_state.last_drift = drift_dict
        st.session_state.last_alert = alerts_df
        st.session_state.last_lookup = look_up_table
        st.session_state.last_calibration = calibration_data
        
        return stats, df, timeseries, curve_df, alerts_df, explain_df, drift_dict, look_up_table, calibration_data

    except Exception as e:
        logger.error(f"Data Fetch Error: {e}")
        # On error, return cache as fallback
        return (
            st.session_state.get('last_stats', {}),
            st.session_state.get('last_df', pd.DataFrame()),
            st.session_state.get('last_series', pd.DataFrame()),
            st.session_state.get('last_curve', pd.DataFrame()),
            st.session_state.get('last_alert', pd.DataFrame()),
            st.session_state.get('last_explain', pd.DataFrame()),
            st.session_state.get('last_drift', {}),
            st.session_state.get('last_lookup', {}),
            st.session_state.get('last_calibration', {})
        )

# ==============================================================================
# 4. MAIN CONTROLLER
# ==============================================================================
def main():
    try:
        stats, df, timeseries_df, curve_df, alerts_df, explain_df, drift_dict, lookup_table, calibration_data = load_data(limit=data_limit)

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
            ops.render_page(
                recent_df=df, 
                alerts_df=alerts_df, 
                metrics=stats
            ) 
            
        elif page == "ML Monitor":
            ml.render_page(
            recent_df=df, 
            explain_df=explain_df, 
            metrics=stats, 
            drift_dict=drift_dict, 
            lookup_table=lookup_table,
            calibration_data=calibration_data
        )
            
        elif page == "Strategy":
            strategy.render_page(
                recent_df=df,
                alerts_df=alerts_df,
                metrics=stats,
                curve_df=curve_df
            )
            
        elif page == "Forensics":
            forensics.load_view()

        if not st.session_state.is_paused:
            time.sleep(refresh_rate)
            st.rerun()

    except Exception as e:
        st.error(f"🚨 An unexpected error occurred in the dashboard controller: {e}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

