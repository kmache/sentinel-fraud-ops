import os
import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# Ensure imports work from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, apply_plot_style, render_header
# Assuming kpi_card is available, or we use a similar HTML generator for Ops specific cards

# ==============================================================================
# 1. COMPONENT FUNCTIONS (The "Rows")
# ==============================================================================

def _render_ops_kpis(recent_df: pd.DataFrame, alerts_df: pd.DataFrame, metrics: dict):
    """
    Top Row: Operational Efficiency Metrics.
    Uses custom HTML to match the Executive KPI style but with Ops-specific indicators.
    """
    c1, c2, c3, c4 = st.columns(4)

    # 1. Active Alerts
    with c1:
        active_alerts = len(alerts_df) 
        color = COLORS['danger'] if active_alerts > 20 else COLORS['warning'] if active_alerts > 10 else COLORS['safe']
        
        st.markdown(f"""
        <div style="background-color: {color}15; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 0.9rem; color: #aaa;">Active Alerts</p>
            <h2 style="margin: 0; font-size: 1.8rem; color: #fff;">{active_alerts}</h2>
            <p style="margin: 0; font-size: 0.8rem; color: {color};">
                {'🚨 Critical' if active_alerts > 20 else '⚠️ Elevated' if active_alerts > 10 else '✅ Normal Queue'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. MTTR (Estimated)
    with c2:
        mttr = 0
        if not alerts_df.empty and 'timestamp' in alerts_df.columns:
            alerts_df['dt'] = pd.to_datetime(alerts_df['timestamp'])
            oldest = (pd.Timestamp.now() - alerts_df['dt'].min()).total_seconds() / 60
            mttr = max(2, oldest / 2) # Mock logic
        
        color = COLORS['danger'] if mttr > 30 else COLORS['warning'] if mttr > 15 else COLORS['safe']
        st.markdown(f"""
        <div style="background-color: {color}15; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 0.9rem; color: #aaa;">Est. Resolution Time</p>
            <h2 style="margin: 0; font-size: 1.8rem; color: #fff;">{mttr:.0f} <span style="font-size:1rem">min</span></h2>
            <p style="margin: 0; font-size: 0.8rem; color: {color};">
                {'🚨 Backlog' if mttr > 30 else '⚡ On Track'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Risk Velocity
    with c3:
        velocity = 0
        if not recent_df.empty and 'score' in recent_df.columns:
            high_risk_now = len(recent_df[recent_df['score'] > 0.8])
            # Mocking "previous" as 80% of current for visual demo if no historic data passed
            velocity = int(high_risk_now * 0.1) 
        
        color = COLORS['highlight']
        icon = "📈" if velocity >= 0 else "📉"
        
        st.markdown(f"""
        <div style="background-color: {color}15; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 0.9rem; color: #aaa;">Risk Velocity (15m)</p>
            <h2 style="margin: 0; font-size: 1.8rem; color: #fff;">{icon} +{velocity}</h2>
            <p style="margin: 0; font-size: 0.8rem; color: {color};">New High-Risk Events</p>
        </div>
        """, unsafe_allow_html=True)

    # 4. Auto-Block Ratio
    with c4:
        ratio = None
        if not recent_df.empty:
            # Mock calculation if 'action' column exists, otherwise use threshold
            df_high_risk = recent_df[recent_df["action"].isin(["BLOCK"])] 
            total_high_risk = len(recent_df[recent_df['score'] > metrics.get('threshold', 0.5)])
            if total_high_risk > 0:
                ratio = round(100 * (len(df_high_risk) / total_high_risk), 2) if total_high_risk > 0 else None
        
        st.markdown(f"""
        <div style="background-color: {COLORS['safe']}15; padding: 15px; border-radius: 10px; border-left: 5px solid {COLORS['safe']}; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 0.9rem; color: #aaa;">Auto-Resolution</p>
            <h2 style="margin: 0; font-size: 1.8rem; color: #fff;">{ratio}%</h2>
            <p style="margin: 0; font-size: 0.8rem; color: {COLORS['safe']};">AI Confidence Level</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


def _render_alert_queue(alerts_df: pd.DataFrame):
    """
    Middle Row: The Interactive Alert Queue.
    """
    col_header, col_tools = st.columns([2, 1])
    with col_header:
        st.subheader("📋 Priority Alert Queue")
    
    with col_tools:
        # Mini toolbar
        c_a, c_b = st.columns(2)
        with c_a: st.button("✅ Approve All", width='stretch')
        with c_b: st.button("📥 Export", type="secondary", width='stretch')

    if alerts_df.empty:
        st.success("🎉 Queue is empty. No anomalies detected.")
        return

    # Processing for display
    display_df = alerts_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
    
    # Sort by Score
    display_df = display_df.sort_values('score', ascending=False).head(50)
    final_df = display_df[['timestamp', 'transaction_id', 'score', 'TransactionAmt', 'ProductCD', 'card4', 'P_emaildomain', 'action']].copy()
    
    # Styling logic
    def highlight_risk(val):
        # Determine color
        if val > 0.9: color = '#ff4b4b'      # Red
        elif val > 0.7: color = '#ffa726'    # Orange
        else: color = '#00e676'              # Green
        return f'color: {color}; font-weight: bold'
    
    styled_df = final_df.style.map(highlight_risk, subset=['score'])

    st.dataframe(
        styled_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(
                "Date & Time", 
                format="YYYY-MM-DD HH:mm:ss",  # <--- CHANGED HERE
                width="medium"
            ),
            "transaction_id": "ID",
            "score": st.column_config.ProgressColumn(
                "Risk Score", 
                format="%.3f", 
                min_value=0, 
                max_value=1,
                help="Model Confidence"
            ),
            "TransactionAmt": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "ProductCD": "Product",
            "card4": "Card",
            "P_emaildomain": "Email Domain",
            "action": "Action Taken"
        },
        width='stretch',
        height=350,
        hide_index=True
    )
    st.markdown("<br>", unsafe_allow_html=True)


def _render_charts_row(recent_df: pd.DataFrame, alerts_df: pd.DataFrame, metrics: dict):
    """
    Bottom Row: Risk Distribution and Timeline.
    """
    c1, c2 = st.columns(2)

    # --- CHART 1: Risk Distribution (Histogram) ---
    with c1:
        if not recent_df.empty and 'score' in recent_df.columns:
            fig_hist = go.Figure()
            
            # Create buckets
            counts, bins = pd.cut(recent_df['score'], bins=10, retbins=True)
            counts = counts.value_counts().sort_index()
            
            # Color logic based on risk severity
            colors = [COLORS['safe'] if b.right < 0.5 else COLORS['warning'] if b.right < 0.85 else COLORS['danger'] for b in counts.index]

            fig_hist.add_trace(go.Bar(
                x=[f"{i.left:.1f}-{i.right:.1f}" for i in counts.index],
                y=counts.values,
                marker_color=colors,
                name="Transactions"
            ))

            fig_hist = apply_plot_style(fig_hist, title="Live Risk Distribution")
            fig_hist.update_layout(
                xaxis_title="Risk Score Probability",
                yaxis_title="Volume",
                bargap=0.1
            )
                        
            st.plotly_chart(fig_hist, width='stretch')
        else:
            st.info("Waiting for transaction data...")

    # --- CHART 2: Alert Volume (Hourly) ---
    with c2:
        if not alerts_df.empty and 'timestamp' in alerts_df.columns:
            alerts_df['hour'] = pd.to_datetime(alerts_df['timestamp']).dt.hour
            hourly_counts = alerts_df.groupby('hour').size()

            fig_time = go.Figure()
            
            fig_time.add_trace(go.Scatter(
                x=hourly_counts.index,
                y=hourly_counts.values,
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color=COLORS['highlight'], width=3),
                marker=dict(size=8, color=COLORS['text']),
                name="Alerts"
            ))

            fig_time = apply_plot_style(fig_time, title="Alert Volume (24h)")
            fig_time.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Alert Count",
                hovermode="x unified"
            )
            st.plotly_chart(fig_time, width='stretch')
        else:
            st.info("Waiting for alert history...")

# ==============================================================================
# 2. MAIN RENDER FUNCTION
# ==============================================================================

def render_page(recent_df: pd.DataFrame, alerts_df: pd.DataFrame, metrics: dict):
    """
    Main controller for the Ops Center.
    Matches the signature expected by app.py (if updated) or called similarly to executive.
    """
    render_header("Ops Center", "Real-time Monitoring & Alert Management")
    
    # 1. Top Row: KPIs
    if metrics:
        _render_ops_kpis(recent_df, alerts_df, metrics)
    else:
        st.warning("⚠️ No metrics available yet.")
    
    # 2. Middle Row: The Work Queue
    _render_alert_queue(alerts_df)
    
    # 3. Bottom Row: Analytics
    _render_charts_row(recent_df, alerts_df, metrics)