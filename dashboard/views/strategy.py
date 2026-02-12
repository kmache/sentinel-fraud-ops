import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, kpi_card, apply_plot_style, render_header

# ==============================================================================
# ROW 1: BUSINESS CONVERSION KPIs
# ==============================================================================
def _render_conversion_kpis(metrics: dict, alerts_df: pd.DataFrame):
    """
    Focuses on "Customer Friction" and "Operational Load".
    """
    st.markdown("### 📈 Conversion & Operational Load")
    c1, c2, c3, c4 = st.columns(4)
    
    total = metrics.get('total_processed', 1)
    if total == 0: total = 1
    
    alert_count = len(alerts_df)
    
    with c1:
        val = (1 - (alert_count / total))
        st.markdown(kpi_card("Approval Rate", f"{val:.1%}", "Smooth Customer Path", COLORS['safe']), unsafe_allow_html=True)
    
    with c2:
        val = (alert_count / total)
        st.markdown(kpi_card("Challenge Rate", f"{val:.1%}", "Friction / 2FA Rate", COLORS['warning']), unsafe_allow_html=True)
    
    with c3:
        if not alerts_df.empty and 'TransactionAmt' in alerts_df.columns:
            val = alerts_df['TransactionAmt'].sum()
        else:
            val = 0
        st.markdown(kpi_card("Revenue at Risk", f"${val/1000:,.1f}K", "Pending Investigation", COLORS['danger']), unsafe_allow_html=True)
        
    with c4:
        missed = metrics.get('fraud_missed_val', 0)
        ops_cost = alert_count * 15 
        total_cost = (missed + ops_cost)
        st.markdown(kpi_card("Total Risk Cost", f"${total_cost/1000:,.1f}K", "Loss + Review Expense", COLORS['neutral']), unsafe_allow_html=True)

    st.markdown("---")

# ==============================================================================
# ROW 2: THE ROI FRONTIER (Cost Curve)
# ==============================================================================
def _render_cost_optimization(curve_df: pd.DataFrame, current_threshold: float):
    """
    Visualizes the "Sweet Spot" where loss is minimized.
    """
    st.subheader("🎯 Profit Maximization (ROI Frontier)")
    
    if curve_df.empty:
        st.info("ℹ️ Calculating ROI Cost Curve... (Waiting for more data)")
        return
    
    optimal_idx = curve_df['total_loss'].idxmin()
    best_t = curve_df.loc[optimal_idx, 'threshold']
    min_loss = curve_df.loc[optimal_idx, 'total_loss']

    current_idx = (curve_df['threshold'] - current_threshold).abs().idxmin()
    current_loss = curve_df.loc[current_idx, 'total_loss']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curve_df['threshold'], y=curve_df['total_loss'],
        mode='lines', name='Total Economic Loss',
        line=dict(color=COLORS['highlight'], width=4),
        fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.05)'
    ))

    fig.add_trace(go.Scatter(
        x=[best_t], y=[min_loss],
        mode='markers+text', name='Theoretical Optimal',
        text=["MAX PROFIT"], textposition="bottom center",
        marker=dict(color=COLORS['safe'], size=15, symbol="star")
    ))

    fig.add_trace(go.Scatter(
        x=[current_threshold], y=[current_loss],
        mode='markers+text', name='Current Setting',
        text=["ACTIVE"], textposition="top center",
        marker=dict(color=COLORS['danger'], size=12, symbol="circle")
    ))

    fig = apply_plot_style(fig, title="Total Cost of Fraud vs. Risk Threshold")
    fig.update_layout(
        xaxis_title="Aggressiveness (Threshold)",
        yaxis_title="Financial Impact ($)",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width='stretch')

    diff = current_threshold - best_t
    savings = current_loss - min_loss
    
    if abs(diff) < 0.05:
        st.success(f"✅ **Strategy Insight:** Current threshold ({current_threshold:.2f}) is aligned with the economic optimal.")
    elif diff > 0:
        st.warning(f"⚠️ **Strategy Insight:** You are being too **Conservative**. Lowering the threshold to {best_t:.2f} could improve ROI by ${savings:,.0f}.")
    else:
        st.error(f"🚨 **Strategy Insight:** You are being too **Aggressive**. Raising the threshold to {best_t:.2f} would reduce operational costs by ${savings:,.0f}.")

    st.markdown("---")

# ==============================================================================
# ROW 3: HEURISTIC RULE PERFORMANCE
# ==============================================================================
def _render_rule_performance(recent_df: pd.DataFrame):
    """
    Analyzes which high-level segments are performing poorly.
    """
    st.subheader("🛡️ Segment Strategy Analysis")
    
    if recent_df.empty:
        st.info("Waiting for data...")
        return

    c1, c2 = st.columns(2)

    with c1:
        if 'ProductCD' in recent_df.columns:
            target_col = 'ground_truth' if 'ground_truth' in recent_df.columns else 'is_fraud'
            
            prod_perf = recent_df.groupby('ProductCD').agg({
                'score': 'mean',
                target_col: 'mean'
            }).reset_index()
            prod_perf.columns = ['Product', 'Avg Risk', 'Fraud Rate']
            
            fig_prod = px.bar(
                prod_perf, x='Product', y='Fraud Rate', 
                color='Avg Risk', 
                title="Fraud Exposure by Product Line",
                color_continuous_scale='Reds'
            )
            fig_prod = apply_plot_style(fig_prod, "")

            st.plotly_chart(fig_prod, width='stretch')
        else:
            st.warning("Product data not available.")

    with c2:
        if 'card4' in recent_df.columns:

            df_clean = recent_df.copy()
            df_clean['card4'] = df_clean['card4'].fillna("Unknown")
            
            card_perf = df_clean.groupby('card4').agg({
                'TransactionAmt': 'sum'
            }).reset_index()
            card_perf.columns = ['Network', 'Total Value']
            
            fig_card = px.pie(
                card_perf, values='Total Value', names='Network', 
                title="Value Distribution by Network",
                hole=0.4, 
                color_discrete_sequence=px.colors.sequential.GnBu
            )
            fig_card = apply_plot_style(fig_card, "")

            st.plotly_chart(fig_card, width='stretch')
        else:
            st.warning("Card Network data not available.")

# ==============================================================================
# MAIN RENDERER
# ==============================================================================
def render_page(recent_df: pd.DataFrame, alerts_df: pd.DataFrame, metrics: dict, curve_df: pd.DataFrame):
    render_header("Strategy Center", "Threshold Optimization & Rule Performance")
    
    current_t = metrics.get('threshold', 0.5)

    _render_conversion_kpis(metrics, alerts_df)
    
    _render_cost_optimization(curve_df, current_t)

    _render_rule_performance(recent_df)