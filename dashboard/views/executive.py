import os
import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, kpi_card, apply_plot_style, render_header

def render_page(df: pd.DataFrame, metrics: dict, threshold: float):
    render_header("Executive Overview", "Financial Impact & ROI Analysis")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # Normalize metrics keys just in case
    m = metrics.get('metrics', metrics) 
    
    # --- KPI ROW ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        val = m.get('net_benefit', 0)
        st.markdown(kpi_card("Net Business Benefit", f"${val/1000:,.1f}K", "Saved - Costs - Missed", COLORS['safe']), unsafe_allow_html=True)
    with c2: 
        val = m.get('fraud_prevented', 0)
        st.markdown(kpi_card("Total Fraud Prevented", f"${val/1000:,.1f}K", "Gross Value Protected", COLORS['safe']), unsafe_allow_html=True)
    with c3:
        fp_ratio = m.get('fp_ratio', 0)
        ratio_color = COLORS['safe'] if fp_ratio < 3 else COLORS['danger']
        st.markdown(kpi_card("False Positive Ratio", f"1 : {fp_ratio}", "Target < 1:3", ratio_color), unsafe_allow_html=True)
    with c4: 
        recall = m.get('recall', 0)
        st.markdown(kpi_card("Global Recall", f"{recall:.1%}", "Of known patterns", COLORS['warning']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- FINANCIAL CHART ---
    if 'timestamp' in df.columns:
        daily = df.copy().sort_values('timestamp')
        
        # FIX: Reset index to ensure uniqueness.
        # This prevents "ValueError: cannot reindex on an axis with duplicate labels"
        # which occurs if the upstream dataframe was created via concatenation without ignore_index=True.
        daily = daily.reset_index(drop=True)
        
        # Determine Amount Column
        amt_col = 'amount' if 'amount' in df.columns else 'TransactionAmt'
        fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'ground_truth'
        score_col = 'score' if 'score' in df.columns else 'composite_risk_score'

        # Cumulative savings
        daily['cum_saved'] = daily[daily[fraud_col]==1][amt_col].cumsum().ffill().fillna(0)
        
        # Realized Loss (Missed Fraud: Fraud exists but score < threshold)
        # This boolean operation requires a unique index
        daily['missed'] = (daily[fraud_col]==1) & (daily[score_col] <= threshold)
        daily['cum_loss'] = daily[daily['missed']][amt_col].cumsum().ffill().fillna(0)

        fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_saved'], name="Savings", fill='tozeroy', line=dict(color=COLORS['safe'])), secondary_y=False)
        fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_loss'], name="Realized Loss", line=dict(color=COLORS['danger'], dash='dot')), secondary_y=True)

        fig_fin = apply_plot_style(fig_fin, title="Cumulative Value vs. Realized Loss")
        st.plotly_chart(fig_fin, use_container_width=True)
    else:
        st.warning("Timestamp data missing for financial chart.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- COMPOSITION & SENSITIVITY ---
    c1, c2 = st.columns(2)
    with c1:
        # Fraud Composition
        fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'ground_truth'
        fraud_only = df[df[fraud_col]==1]
        
        # Check which enriched columns exist
        candidates = ['ProductCD', 'device_vendor', 'card4', 'P_emaildomain']
        available_cols = [c for c in candidates if c in fraud_only.columns]
        
        if not fraud_only.empty and len(available_cols) >= 1:
            # Use up to 2 available columns for sunburst
            path = available_cols[:2] 
            amt_col = 'amount' if 'amount' in df.columns else 'TransactionAmt'
            
            fig_sun = px.sunburst(fraud_only, path=path, values=amt_col, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_sun = apply_plot_style(fig_sun, title=f"Fraud Composition ({' > '.join(path)})")
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.markdown(kpi_card("Fraud Composition", "No Data", "Waiting for fraud patterns", COLORS['neutral']), unsafe_allow_html=True)
            
    with c2:
        # Cost Curve Simulation
        x = np.linspace(0, 1, 100)
        y = 1000 * ((x - 0.6)**2 * 20 + 2) 
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=x, y=y, name="Est. Cost Function", line=dict(color="white")))
        
        current_y = 1000 * ((threshold - 0.6)**2 * 20 + 2)
        fig_curve.add_trace(go.Scatter(x=[threshold], y=[current_y], mode='markers', 
                                     marker=dict(color=COLORS['warning'], size=15), name="Current Threshold"))
        
        fig_curve = apply_plot_style(fig_curve, title="Cost-Benefit Sensitivity Analysis")
        fig_curve.update_layout(xaxis_title="Threshold", yaxis_title="Est. Operational Cost ($)")
        st.plotly_chart(fig_curve, use_container_width=True)


import os
import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, kpi_card, apply_plot_style, render_header

def render_page(df: pd.DataFrame, metrics: dict, threshold: float):
    render_header("Executive Overview", "Financial Impact & ROI Analysis")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # Normalize metrics keys just in case
    m = metrics.get('metrics', metrics) 
    
    # --- KPI ROW ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        val = m.get('net_benefit', 0)
        st.markdown(kpi_card("Net Business Benefit", f"${val/1000:,.1f}K", "Saved - Costs - Missed", COLORS['safe']), unsafe_allow_html=True)
    with c2: 
        val = m.get('fraud_prevented', 0)
        st.markdown(kpi_card("Total Fraud Prevented", f"${val/1000:,.1f}K", "Gross Value Protected", COLORS['safe']), unsafe_allow_html=True)
    with c3:
        fp_ratio = m.get('fp_ratio', 0)
        ratio_color = COLORS['safe'] if fp_ratio < 3 else COLORS['danger']
        st.markdown(kpi_card("False Positive Ratio", f"1 : {fp_ratio}", "Target < 1:3", ratio_color), unsafe_allow_html=True)
    with c4: 
        recall = m.get('recall', 0)
        st.markdown(kpi_card("Global Recall", f"{recall:.1%}", "Of known patterns", COLORS['warning']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- FINANCIAL CHART ---
    if 'timestamp' in df.columns:
        daily = df.copy().sort_values('timestamp')
        
        # Determine Amount Column
        amt_col = 'amount' if 'amount' in df.columns else 'TransactionAmt'
        fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'ground_truth'
        score_col = 'score' if 'score' in df.columns else 'composite_risk_score'

        # Cumulative savings
        daily['cum_saved'] = daily[daily[fraud_col]==1][amt_col].cumsum().ffill().fillna(0)
        
        # Realized Loss (Missed Fraud: Fraud exists but score < threshold)
        daily['missed'] = (daily[fraud_col]==1) & (daily[score_col] <= threshold)
        daily['cum_loss'] = daily[daily['missed']][amt_col].cumsum().ffill().fillna(0)

        fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_saved'], name="Savings", fill='tozeroy', line=dict(color=COLORS['safe'])), secondary_y=False)
        fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_loss'], name="Realized Loss", line=dict(color=COLORS['danger'], dash='dot')), secondary_y=True)

        fig_fin = apply_plot_style(fig_fin, title="Cumulative Value vs. Realized Loss")
        st.plotly_chart(fig_fin, use_container_width=True)
    else:
        st.warning("Timestamp data missing for financial chart.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- COMPOSITION & SENSITIVITY ---
    c1, c2 = st.columns(2)
    with c1:
        # Fraud Composition
        fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'ground_truth'
        fraud_only = df[df[fraud_col]==1]
        
        # Check which enriched columns exist
        candidates = ['ProductCD', 'device_vendor', 'card4', 'P_emaildomain']
        available_cols = [c for c in candidates if c in fraud_only.columns]
        
        if not fraud_only.empty and len(available_cols) >= 1:
            # Use up to 2 available columns for sunburst
            path = available_cols[:2] 
            amt_col = 'amount' if 'amount' in df.columns else 'TransactionAmt'
            
            fig_sun = px.sunburst(fraud_only, path=path, values=amt_col, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_sun = apply_plot_style(fig_sun, title=f"Fraud Composition ({' > '.join(path)})")
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.markdown(kpi_card("Fraud Composition", "No Data", "Waiting for fraud patterns", COLORS['neutral']), unsafe_allow_html=True)
            
    with c2:
        # Cost Curve Simulation
        x = np.linspace(0, 1, 100)
        y = 1000 * ((x - 0.6)**2 * 20 + 2) 
        
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=x, y=y, name="Est. Cost Function", line=dict(color="white")))
        
        current_y = 1000 * ((threshold - 0.6)**2 * 20 + 2)
        fig_curve.add_trace(go.Scatter(x=[threshold], y=[current_y], mode='markers', 
                                     marker=dict(color=COLORS['warning'], size=15), name="Current Threshold"))
        
        fig_curve = apply_plot_style(fig_curve, title="Cost-Benefit Sensitivity Analysis")
        fig_curve.update_layout(xaxis_title="Threshold", yaxis_title="Est. Operational Cost ($)")
        st.plotly_chart(fig_curve, use_container_width=True)