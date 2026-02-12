import sys
import os
import time
import logging
import traceback
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import setup_page, COLORS
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
PAGES = ["Executive View", "Ops Center", "ML Monitor", "Strategy", "Forensics"]

def update_params():
    st.query_params["page"] = st.session_state.current_page
    st.query_params["refresh_rate"] = st.session_state.refresh_rate
    st.query_params["data_limit"] = st.session_state.data_limit
    
    st.session_state["needs_hard_refresh"] = True

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]

if "page" in st.query_params:
    url_page = st.query_params["page"]
    if url_page in PAGES:
        st.session_state.current_page = url_page

with st.sidebar:
    logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
    with logo_col2:
        st.image("logo.png", width='stretch')

    st.markdown(f"""
        <h1 style='text-align: center; color: {COLORS['highlight']}; 
        letter-spacing: 2px; margin-top: -15px; margin-bottom: 0;'>
        Sentinel Fraud Ops
        </h1>
    """, unsafe_allow_html=True)
    
    st.caption("<p style='text-align: center;'>Real-time Fraud Detection System</p>", unsafe_allow_html=True)
    st.markdown("---")

    if "refresh_rate" not in st.session_state:
        st.session_state.refresh_rate = int(st.query_params.get("refresh_rate", 5))
        
    if "data_limit" not in st.session_state:
        st.session_state.data_limit = int(st.query_params.get("data_limit", 500))

    page = st.radio(
        "MODULES", 
        PAGES,
        key="current_page", 
        on_change=update_params
    )

    st.markdown("---")
    
    with st.expander("⚙️ View Settings", expanded=True):
        refresh_rate = st.slider("Refresh Rate (s)", 1, 60, key="refresh_rate", on_change=update_params)
        data_limit = st.select_slider("History Depth", options=[100, 500, 1000, 2000], key="data_limit", on_change=update_params)
    
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏸ PAUSE" if not st.session_state.get('is_paused', False) else "▶ RESUME"):
            st.session_state.is_paused = not st.session_state.get('is_paused', False)
            st.rerun()
    with c2:
        if st.button("🔄 REFRESH"):
            st.session_state.is_paused = False 
            st.cache_data.clear()
            st.rerun()

    if st.session_state.get('is_paused', False):
        st.warning("⚠️ Feed Paused")
    else:
        st.success("🟢 Feed Active")
    
    st.markdown("---")
    try:
        st.caption("🟢 Backend: Online")
    except:
        st.error("❌ Backend: Offline")

# ==============================================================================
# 3. DATA LOADING
# ==============================================================================
def load_data(limit):
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
        stats = client.get_dashboard_stats()
        df = client.get_recent_transactions(limit=limit)
        timeseries = client.get_financial_timeseries()
        curve_df = client.get_threshold_optimization_curve()
        alerts_df = client.get_alerts(limit=100)
        explain_df = client.get_global_feature_importance()
        drift_dict = client.get_feature_drift_report()
        look_up_table = client.get_performance_lookup()
        calibration_data = client.get_calibration_report()
        
        st.session_state.last_stats = stats
        st.session_state.last_df = df
        st.session_state.last_series = timeseries
        st.session_state.last_curve = curve_df
        st.session_state.last_alert = alerts_df
        st.session_state.last_explain = explain_df
        st.session_state.last_drift = drift_dict
        st.session_state.last_lookup = look_up_table
        st.session_state.last_calibration = calibration_data
        
        return stats, df, timeseries, curve_df, alerts_df, explain_df, drift_dict, look_up_table, calibration_data
    except Exception as e:
        logger.error(f"Data Fetch Error: {e}")
        return (
            st.session_state.get('last_stats', {}),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, {}
        )

# ==============================================================================
# 4. MAIN CONTROLLER
# ==============================================================================
def main():
    if st.session_state.get("needs_hard_refresh", False):
        st.session_state.needs_hard_refresh = False
        
        js = "<script>window.parent.location.reload();</script>"
        components.html(js, height=0, width=0)
        
        st.stop()
    # -------------------------------------------
    try:
        stats, df, timeseries_df, curve_df, alerts_df, explain_df, drift_dict, lookup_table, calibration_data = load_data(limit=st.session_state.data_limit)
        
        current_page = st.session_state.get("current_page", "Executive View")
        current_threshold = stats.get('threshold', 0.5)

        if current_page == "Executive View":
            executive.render_page(
                recent_df=df, metrics=stats, threshold=current_threshold, 
                timeseries_df=timeseries_df, curve_df=curve_df
            )
        
        elif current_page == "Ops Center":
            ops.render_page(
                recent_df=df, alerts_df=alerts_df, metrics=stats
            ) 
        
        elif current_page == "ML Monitor":
            ml.render_page(
                recent_df=df, explain_df=explain_df, metrics=stats, 
                drift_dict=drift_dict, lookup_table=lookup_table, calibration_data=calibration_data
            )
        
        elif current_page == "Strategy":
            strategy.render_page(
                recent_df=df, alerts_df=alerts_df, metrics=stats, curve_df=curve_df
            )
        
        elif current_page == "Forensics":
            forensics.load_view()

        if not st.session_state.get('is_paused', False):
            time.sleep(st.session_state.refresh_rate)
            st.rerun()

    except Exception as e:
        st.error("🚨 An unexpected error occurred.")
        with st.expander("Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()


