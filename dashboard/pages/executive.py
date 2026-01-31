import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from styles import COLORS, kpi_card, apply_plot_style, render_header

def render_page(df: pd.DataFrame, metrics: dict, threshold: float):
    render_header("Executive Overview", "Financial Impact & ROI Analysis")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # --- KPI ROW ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        st.markdown(kpi_card("Net Business Benefit", f"${metrics['net_benefit']/1000:,.1f}K", "Saved - Costs - Missed", COLORS['safe']), unsafe_allow_html=True)
    with c2: 
        st.markdown(kpi_card("Total Fraud Prevented", f"${metrics['fraud_prevented']/1000:,.1f}K", "Gross Value Protected", COLORS['safe']), unsafe_allow_html=True)
    with c3:
        ratio_color = COLORS['safe'] if metrics['fp_ratio'] < 3 else COLORS['danger']
        st.markdown(kpi_card("False Positive Ratio", f"1 : {metrics['fp_ratio']}", "Target < 1:3", ratio_color), unsafe_allow_html=True)
    with c4: 
        st.markdown(kpi_card("Global Recall", f"{metrics['recall']:.1%}", "Of known patterns", COLORS['warning']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- FINANCIAL CHART ---
    daily = df.copy().sort_values('timestamp')
    # Cumulative savings
    daily['cum_saved'] = daily[daily['ground_truth']==1]['TransactionAmt'].cumsum().ffill().fillna(0)
    
    # Realized Loss (Missed Fraud)
    daily['missed'] = (daily['ground_truth']==1) & (daily['composite_risk_score'] <= threshold)
    daily['cum_loss'] = daily[daily['missed']]['TransactionAmt'].cumsum().ffill().fillna(0)

    from plotly.subplots import make_subplots
    fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
    fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_saved'], name="Savings", fill='tozeroy', line=dict(color=COLORS['safe'])), secondary_y=False)
    fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_loss'], name="Loss", line=dict(color=COLORS['danger'], dash='dot')), secondary_y=True)

    fig_fin = apply_plot_style(fig_fin, title="Cumulative Value vs. Realized Loss")
    st.plotly_chart(fig_fin, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- COMPOSITION & SENSITIVITY ---
    c1, c2 = st.columns(2)
    with c1:
        # Fraud Composition
        fraud_only = df[df['ground_truth']==1]
        available_cols = [c for c in ['ProductCD', 'device_vendor', 'card_type'] if c in fraud_only.columns]
        
        if not fraud_only.empty and available_cols:
            # Use the first available column for simple sunburst or specific ones
            path = available_cols[:2] 
            fig_sun = px.sunburst(fraud_only, path=path, values='TransactionAmt', color_discrete_sequence=px.colors.sequential.RdBu)
            fig_sun = apply_plot_style(fig_sun, title=f"Fraud Composition ({' > '.join(path)})")
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.markdown(kpi_card("Fraud Composition", "No Data", "Waiting for fraud patterns", COLORS['neutral']), unsafe_allow_html=True)
            
    with c2:
        # Cost Curve Simulation
        x = np.linspace(0, 1, 100)
        # Fake cost curve equation for demo visualization
        y = 1000 * ((x - 0.6)**2 * 20 + 2) 
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=x, y=y, name="Cost Function", line=dict(color="white")))
        
        current_y = 1000 * ((threshold - 0.6)**2 * 20 + 2)
        fig_curve.add_trace(go.Scatter(x=[threshold], y=[current_y], mode='markers', 
                                     marker=dict(color=COLORS['warning'], size=15), name="Current Threshold"))
        
        fig_curve = apply_plot_style(fig_curve, title="Cost-Benefit Sensitivity Analysis")
        fig_curve.update_layout(xaxis_title="Threshold", yaxis_title="Est. Operational Cost ($)")
        st.plotly_chart(fig_curve, use_container_width=True)