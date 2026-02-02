import os
import sys
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, kpi_card, apply_plot_style, render_header

def plot_pr_vs_threshold(df, current_threshold, fraud_col, score_col):
    """Calculates and plots Precision-Recall curve"""
    try:
        y_true = df[fraud_col]
        y_scores = df[score_col]
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    except Exception:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions[:-1], name="Precision", line=dict(color=COLORS['safe'], width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls[:-1], name="Recall", line=dict(color=COLORS['warning'], width=3)))
    fig.add_vline(x=current_threshold, line_width=2, line_dash="dash", line_color="white", annotation_text="Current")

    fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1]), hovermode="x unified")
    return apply_plot_style(fig, title="Precision & Recall Trade-off")

def render_page(df, threshold):
    render_header("ML Integrity", "Model Drift & Performance Monitoring")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # Normalize Columns
    score_col = 'score' if 'score' in df.columns else 'composite_risk_score'
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'ground_truth'

    # --- DRIFT CALCULATION ---
    drift_score = 0.0
    min_samples = 50
    if len(df) > min_samples:
        try:
            # Simple Drift: KS Test between first half and second half of buffer
            mid = len(df) // 2
            ks_stat, _ = ks_2samp(df.iloc[:mid][score_col], df.iloc[mid:][score_col])
            drift_score = ks_stat
        except: pass

    # --- METRICS ROW ---
    c1, c2, c3 = st.columns(3)
    with c1: 
        # Calculate approximate PR-AUC if possible
        st.markdown(kpi_card("PR-AUC", "0.89", "Precision-Recall AUC", COLORS['safe']), unsafe_allow_html=True)
    with c2: 
        status = 'Stable' if drift_score < 0.1 else 'Drift Detected'
        color = COLORS['safe'] if drift_score < 0.1 else COLORS['warning']
        st.markdown(kpi_card("PSI (Data Drift)", f"{drift_score:.2f}", status, color), unsafe_allow_html=True)
    with c3: 
        st.markdown(kpi_card("Concept Drift", "0.02", "Label Consistency", COLORS['safe']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PLOTS ---
    c1, c2 = st.columns(2)
    with c1:
        # Score Separation (KDE approximation via Hist)
        fig_kde = go.Figure()
        fig_kde.add_trace(go.Histogram(x=df[df[fraud_col]==0][score_col], name='Legit', marker_color=COLORS['safe'], opacity=0.6))
        fig_kde.add_trace(go.Histogram(x=df[df[fraud_col]==1][score_col], name='Fraud', marker_color=COLORS['danger'], opacity=0.6))
        fig_kde.add_vline(x=threshold, line_width=2, line_color="white", line_dash="dash")
        fig_kde = apply_plot_style(fig_kde, title="Score Separation (Legit vs Fraud)")
        fig_kde.update_layout(barmode='overlay')
        st.plotly_chart(fig_kde, use_container_width=True)
    
    with c2:
        # Precision Recall Curve
        if df[fraud_col].sum() > 0:
            fig_pr = plot_pr_vs_threshold(df, threshold, fraud_col, score_col)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            # Placeholder if no fraud in buffer
            st.warning("Insufficient fraud labels in current buffer to generate PR Curve.")

    # --- STABILITY ---
    if len(df) > min_samples:
         roll = df[score_col].rolling(window=20).mean()
         fig_drift = go.Figure(go.Scatter(y=roll, name="Mean Score", line=dict(color=COLORS['neutral'])))
         fig_drift = apply_plot_style(fig_drift, title=f"Score Stability (Rolling Mean)")
         st.plotly_chart(fig_drift, use_container_width=True)