"""
Sentinel Fraud Ops - Main Dashboard Controller
"""
import sys
import os
import time
import logging
import traceback
import pandas as pd
import streamlit as st

#sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import setup_page, COLORS
from api_client import SentinelApiClient
from views import executive, ops, ml, strategy, forensics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")

# ==============================================================================
# 1. SETUP & STATE
# ==============================================================================
setup_page("Sentinel Ops Center")

if 'api_client' not in st.session_state:
    st.session_state.api_client = SentinelApiClient()
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False

if 'last_stats' not in st.session_state:
    st.session_state.last_stats = {}
if 'last_df' not in st.session_state:
    st.session_state.last_df = pd.DataFrame()
if 'last_perf' not in st.session_state:
    st.session_state.last_perf = {}

client = st.session_state.api_client

# ==============================================================================
# 2. SIDEBAR (NAVIGATION & CONTROLS)
# ==============================================================================
with st.sidebar:
    st.markdown(f"<h1 style='text-align: center; color: {COLORS['highlight']}; letter-spacing: 2px; margin-bottom: 0;'>SENTINEL</h1>", unsafe_allow_html=True)
    st.caption("Enterprise Fraud Detection System")
    st.markdown("---")
    
    # 2.1 Navigation
    page = st.radio(
        "MODULES", 
        ["Executive View", "Ops Center", "ML Monitor", "Strategy", "Forensics"],
        index=0
    )
    
    st.markdown("---")
    
    # 2.2 Preferences (New from Review)
    with st.expander("⚙️ View Settings", expanded=True):
        refresh_rate = st.slider("Refresh Rate (s)", 1, 60, 3)
        data_limit = st.select_slider("History Depth", options=[100, 500, 1000, 2000], value=1000)
    
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
        if client.is_backend_alive():
            st.caption("✅ Backend: Online")
            sys_metrics = client.get_system_metrics()
            cpu = sys_metrics.get('cpu_usage_percent', 0)
            st.progress(min(cpu/100, 1.0), f"CPU Load: {cpu}%")
        else:
            st.error("❌ Backend: Offline")
    except:
        st.error("❌ Backend: Unreachable")

# ==============================================================================
# 3. ROBUST DATA LOADING
# ==============================================================================
def load_data(limit):
    """
    Fetches data with Error Fallback.
    If API fails, returns cached data and shows a warning.
    """
    # 1. If paused, strictly use cache
    if st.session_state.is_paused:
        return (
            st.session_state.last_stats, 
            st.session_state.last_df, 
            st.session_state.last_perf
        )

    try:
        # 2. Attempt Fetch
        stats = client.get_dashboard_stats()
        df = client.get_recent_transactions(limit=limit)
        perf_report = client.get_model_performance(threshold=None)
        
        # 3. Update Cache
        st.session_state.last_stats = stats
        st.session_state.last_df = df
        st.session_state.last_perf = perf_report
        
        return stats, df, perf_report

    except Exception as e:
        logger.error(f"Data Fetch Error: {e}")
        if not st.session_state.last_df.empty:
            st.toast(f"⚠️ Connection Issue. Using cached data. ({str(e)})", icon="⚠️")
            return (
                st.session_state.last_stats, 
                st.session_state.last_df, 
                st.session_state.last_perf
            )
        else:
            st.error("Unable to connect to Backend API.")
            return {}, pd.DataFrame(), {}

# ==============================================================================
# 4. MAIN CONTROLLER
# ==============================================================================
def main():
    try:
        stats, df, perf_report = load_data(limit=data_limit)
        current_threshold = perf_report.get('config', {}).get('threshold_used', 0.5)

        if page == "Executive View":
            executive.render_page(df, perf_report, current_threshold)
            
        elif page == "Ops Center":
            ops.render_page(df, current_threshold)
            
        elif page == "ML Monitor":
            ml.render_page(df, current_threshold)
            
        elif page == "Strategy":
            strategy.render_page(df)
            
        elif page == "Forensics":
            forensics.render_page(df)

        if not st.session_state.is_paused:
            time.sleep(refresh_rate)
            st.rerun()

    except Exception as e:
        st.error("🚨 An unexpected error occurred in the dashboard controller.")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()