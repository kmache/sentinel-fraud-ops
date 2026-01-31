"""
Sentinel Fraud Ops - Main Dashboard Controller
"""
import sys
import os
import time
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from styles import apply_custom_css, COLORS
from api_client import SentinelApiClient
from pages import executive, ops, ml, strategy, forensics

# ==============================================================================
# 1. SETUP
# ==============================================================================
st.set_page_config(
    page_title="Sentinel Ops",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

if 'api_client' not in st.session_state:
    st.session_state.api_client = SentinelApiClient()
    # Initialize Pause State
    st.session_state.is_paused = False

client = st.session_state.api_client

# ==============================================================================
# 2. SIDEBAR (NAVIGATION & CONTROLS)
# ==============================================================================
with st.sidebar:
    st.markdown(f"<h2 style='text-align: center; color: {COLORS['highlight']}; letter-spacing: 2px;'>SENTINEL</h2>", unsafe_allow_html=True)
    st.caption("Enterprise Fraud Detection System")
    st.markdown("---")
    
    # 2.1 Navigation
    page = st.radio(
        "MODULES", 
        ["Executive View", "Ops Center", "ML Monitor", "Strategy", "Forensics"],
        index=0
    )
    
    st.markdown("---")
    
    # 2.2 Live Control (The "Lightness" Fix)
    st.markdown("**LIVE FEED CONTROL**")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏸ PAUSE" if not st.session_state.is_paused else "▶ RESUME"):
            st.session_state.is_paused = not st.session_state.is_paused
            st.rerun()
            
    with c2:
        if st.button("🔄 REFRESH"):
            st.session_state.is_paused = False # Unpause on manual refresh
            st.rerun()

    if st.session_state.is_paused:
        st.warning("⚠️ Live Feed Paused")
    else:
        st.success("🟢 Live Feed Active")

    # 2.3 System Health
    st.markdown("---")
    alive = client.is_backend_alive()
    st.caption(f"System Status: {'Online ✅' if alive else 'Offline ❌'}")

# ==============================================================================
# 3. DATA LOADING
# ==============================================================================
def load_data():
    if st.session_state.is_paused and 'last_df' in st.session_state:
        # Return cached data if paused
        return st.session_state.last_stats, st.session_state.last_df

    stats = client.get_dashboard_stats()
    df = client.get_recent_transactions(limit=500)
    
    # Cache for pause functionality
    st.session_state.last_stats = stats
    st.session_state.last_df = df
    
    return stats, df

# ==============================================================================
# 4. MAIN RENDER
# ==============================================================================
def main():
    stats, df = load_data()
    
    # Routing
    if page == "Executive View":
        executive.render_page(stats, df)
    elif page == "Ops Center":
        ops.render_page(stats, df)
    elif page == "ML Monitor":
        ml.render_page(stats, df)
    elif page == "Strategy":
        strategy.render_page(df)
    elif page == "Forensics":
        forensics.render_page(df)

    # Auto-Refresh Logic (Only if not paused)
    if not st.session_state.is_paused:
        time.sleep(5) # Slower refresh for smoother UX
        st.rerun()

if __name__ == "__main__":
    main() 