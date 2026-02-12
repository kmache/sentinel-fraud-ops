import os
import sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, apply_plot_style, kpi_card, render_header

# ==============================================================================
# 1. COMPONENT FUNCTIONS (The "Rows")
# ==============================================================================
def _render_ops_kpis(recent_df: pd.DataFrame, alerts_df: pd.DataFrame, metrics: dict):
    """
    Top Row: Operational Efficiency Metrics.
    Now uses shared kpi_card for consistency.
    """
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        active_alerts = len(alerts_df)
        if active_alerts > 20:
            status = "🚨 Critical"
            color = COLORS['danger']
        elif active_alerts > 10:
            status = "⚠️ Elevated"
            color = COLORS['warning']
        else:
            status = "✅ Normal Queue"
            color = COLORS['safe']
            
        st.markdown(
            kpi_card("Active Alerts", f"{active_alerts}", status, color),
            unsafe_allow_html=True
        )

    with c2:
        mttr = 0
        if not alerts_df.empty and 'timestamp' in alerts_df.columns:
            alerts_df['dt'] = pd.to_datetime(alerts_df['timestamp'])
            oldest = (pd.Timestamp.now() - alerts_df['dt'].min()).total_seconds() / 60
            mttr = max(2, oldest / 2)

        if mttr > 30:
            status = "🚨 Backlog"
            color = COLORS['danger']
        elif mttr > 15:
            status = "⚠️ Behind"
            color = COLORS['warning']
        else:
            status = "⚡ On Track"
            color = COLORS['safe']

        st.markdown(
            kpi_card("Est. Resolution Time", f"{mttr:.0f} min", status, color),
            unsafe_allow_html=True
        )

    with c3:
        velocity = 0
        if not recent_df.empty and 'score' in recent_df.columns:
            high_risk_now = len(recent_df[recent_df['score'] > 0.8])
            velocity = int(high_risk_now * 0.1) 

        icon = "📈" if velocity >= 0 else "📉"
        status = f"{icon} New High-Risk Events"
        color = COLORS['highlight']
        
        st.markdown(
            kpi_card("Risk Velocity (15m)", f"{icon} +{velocity}", status, color),
            unsafe_allow_html=True
        )

    with c4:
        ratio = 0
        if not recent_df.empty and 'action' in recent_df.columns and 'score' in recent_df.columns:
            df_high_risk = recent_df[recent_df["action"].isin(["BLOCK"])]
            total_high_risk = len(recent_df[recent_df['score'] > metrics.get('threshold', 0.5)])
            if total_high_risk > 0:
                ratio = round(100 * (len(df_high_risk) / total_high_risk), 2)

        status = "AI Confidence Level"
        color = COLORS['safe']
        
        st.markdown(
            kpi_card("Auto-Resolution", f"{ratio}%", status, color),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)


def _render_alert_queue(alerts_df: pd.DataFrame):
    """
    Middle Row: The Interactive Alert Queue.
    """
    col_header, col_tools = st.columns([2, 1])
    with col_header:
        st.subheader("📋 Priority Alert Queue")
    
    with col_tools:
        c_a, c_b = st.columns(2)
        with c_a: st.button("✅ Approve All", width='stretch')
        with c_b: st.button("📥 Export", type="secondary", width='stretch')

    if alerts_df.empty:
        st.success("🎉 Queue is empty. No anomalies detected.")
        return

    display_df = alerts_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
    
    display_df = display_df.sort_values('score', ascending=False).head(50).reset_index(drop=True)
    
    final_df = display_df[['timestamp', 'transaction_id', 'score', 'TransactionAmt', 'ProductCD', 'card4', 'P_emaildomain']]

    def highlight_risk(val):
        if val > 0.9: color = '#ff4b4b'
        elif val > 0.7: color = '#ffa726'
        else: color = '#00e676'
        return f'color: {color}; font-weight: bold'

    styled_df = final_df.style.map(highlight_risk, subset=['score'])

    selection = st.dataframe(
        styled_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm:ss"),
            "transaction_id": "ID",
            "score": st.column_config.ProgressColumn("Risk Score", format="%.3f", min_value=0, max_value=1),
            "TransactionAmt": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "ProductCD": "Product",
            "card4": "Card",
            "P_emaildomain": "Email"
        },
        width='stretch',
        height=350,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun", 
        key="alert_queue_table"
    )

    if len(selection.selection.rows) > 0:
        selected_index = selection.selection.rows[0]
        selected_row = final_df.iloc[selected_index]
        tx_id = selected_row['transaction_id']
        amt = selected_row['TransactionAmt']
        
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid {COLORS['highlight']}; border-radius: 5px; background-color: {COLORS['card_bg']}; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>🎯 <b>Selected:</b> {tx_id} (${amt:,.2f})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_investigate, c_approve, c_ignore = st.columns([1, 1, 2])
        
        with c_investigate:
            if st.button("🕵️‍♂️ Investigate Case", type="primary", width='stretch'):
                
                st.query_params["page"] = "Forensics"
                st.query_params["case_id"] = tx_id

                st.session_state['selected_case'] = tx_id

                st.session_state["needs_hard_refresh"] = True
                
                st.rerun()
        
        with c_approve:
            if st.button("✅ Quick Approve", width='stretch'):
                st.toast(f"Transaction {tx_id} approved!", icon="✅")
                
    st.markdown("<br>", unsafe_allow_html=True)

def _render_charts_row(recent_df: pd.DataFrame, alerts_df: pd.DataFrame, metrics: dict):
    """
    Bottom Row: Risk Distribution and Timeline.
    """
    c1, c2 = st.columns(2)

    with c1:
        if not recent_df.empty and 'score' in recent_df.columns:
            fig_hist = go.Figure()
            
            counts, bins = pd.cut(recent_df['score'], bins=10, retbins=True)
            counts = counts.value_counts().sort_index()
            
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
                        
            st.plotly_chart(fig_hist, width='stretch', key="ops_risk_hist")
        else:
            st.info("Waiting for transaction data...")

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
            st.plotly_chart(fig_time, width='stretch', key="ops_alert_timeline")
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

    if metrics:
        _render_ops_kpis(recent_df, alerts_df, metrics)
    else:
        st.warning("⚠️ No metrics available yet.")

    _render_alert_queue(alerts_df)

    _render_charts_row(recent_df, alerts_df, metrics)