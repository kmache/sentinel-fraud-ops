import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, kpi_card, apply_plot_style, render_header

# ==============================================================================
# ROW 1: MODEL EFFICACY (The "Truth" Layer)
# ==============================================================================
def _render_efficacy_kpi_row(metrics: dict):
    st.markdown("### 📉 Model Efficacy")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        val = metrics.get('precision', 0.88)
        st.markdown(kpi_card("Precision", f"{val:.1%}", 
                             "Trustworthiness", COLORS['safe']), 
                             unsafe_allow_html=True)
    
    with c2:
        val = metrics.get('recall', 0.76)
        st.markdown(kpi_card("Recall", f"{val:.1%}", 
                             "Fraud Catch Rate", COLORS['highlight']),
                               unsafe_allow_html=True)
    
    with c3:
        val = metrics.get('f1_score', 0.81)
        st.markdown(kpi_card("F1-Score", f"{val:.2f}", 
                             "Overall Balance", COLORS['neutral']), 
                             unsafe_allow_html=True)
        
    with c4:
        val = metrics.get('live_latency_ms', 45)
        color = COLORS['safe'] if val < 100 else COLORS['danger']
        st.markdown(kpi_card("Avg Latency", f"{val:.0f}ms", 
                             "API Response", color), 
                             unsafe_allow_html=True)

    st.markdown("---")

# ==============================================================================
# ROW 2: STABILITY & CALIBRATION (The "Consistency" Layer)
# ==============================================================================
def _render_stability_row(recent_df: pd.DataFrame, metrics: dict, calibration_data: dict):
    """
    Scientific analysis of model outputs (Density & Calibration).
    """
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("🧠 Score Distribution (Density)")
        if not recent_df.empty and 'score' in recent_df.columns:
            scores = recent_df['score'].dropna()
            
            fig_dist = go.Figure()

            fig_dist.add_trace(go.Histogram(
                x=scores,
                nbinsx=50,
                name="Density",
                histnorm='probability density',
                marker_color=COLORS['highlight'],
                opacity=0.4
            ))

            try:
                kde = gaussian_kde(scores)
                x_range = np.linspace(0, 1, 100)
                y_kde = kde(x_range)
                
                fig_dist.add_trace(go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode='lines',
                    name='KDE',
                    line=dict(color=COLORS['highlight'], width=3)
                ))
            except Exception as e:
                pass

            threshold = metrics.get('threshold', 0.5)
            fig_dist.add_vline(
                x=threshold, 
                line_dash="dash", 
                line_color=COLORS['danger'], 
                annotation_text="Active Threshold"
            )

            fig_dist = apply_plot_style(fig_dist, title="")
            fig_dist.update_layout(
                xaxis_title="Risk Score (0.0 = Safe, 1.0 = Fraud)",
                yaxis_title="Density",
                showlegend=False,
                margin=dict(t=20, b=20),
                bargap=0.05
            )
            
            st.plotly_chart(fig_dist, width='stretch', key="ml_density_kde")
            st.caption("💡 **Interpretation:** A healthy model is 'polarized' (peaks at 0 and 1). A growing hump near 0.5 indicates model confusion.")
        else:
            st.info("Waiting for inference data...")

    with c2:
        st.subheader("🎯 Calibration")
        fig_cal = go.Figure()

        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Ideal'
        ))

        if calibration_data and 'prob_pred' in calibration_data:
            fig_cal.add_trace(go.Scatter(
                x=calibration_data['prob_pred'], 
                y=calibration_data['prob_true'], 
                mode='lines+markers',
                line=dict(color=COLORS['safe'], width=3),
                marker=dict(size=8, symbol="diamond"),
                name='Current Model'
            ))
            st.caption("💡 **Interpretation:** Points below the diagonal mean the model is **Overconfident**.")
        else:
            st.info("ℹ️ Waiting for calibration data...")

        fig_cal = apply_plot_style(fig_cal, title="")
        fig_cal.update_layout(
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Fraud Rate",
            height=300,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_cal, width='stretch', key="ml_calibration_chart")

    st.markdown("---")

# ==============================================================================
# ROW 3: THRESHOLD STRATEGY SIMULATOR (The "Tool")
# ==============================================================================
def _render_threshold_simulator(lookup_table: dict, current_metrics: dict):
    """
    Interactive tool using TRUE historical data to simulate impact.
    Includes a Chart to visualize the trade-off.
    """
    st.subheader("🎛️ Threshold Strategy Simulator")
    
    if not lookup_table:
        st.info("ℹ️ Waiting for performance simulation table...")
        return
    
    try:
        rows = []
        for t_str, vals in lookup_table.items():
            rows.append({
                'threshold': float(t_str),
                'precision': vals['p'],
                'recall': vals['r'],
                'insult': vals['insult']
            })
        sim_df = pd.DataFrame(rows).sort_values('threshold')
    except Exception as e:
        st.error(f"Error parsing simulation table: {e}")
        return

    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.markdown("<br>", unsafe_allow_html=True)

        current_threshold = current_metrics.get('threshold', 0.5)
        sim_threshold = st.slider(
            "Simulate Risk Threshold", 
            min_value=0.0, max_value=1.0, value=current_threshold, step=0.01, 
            help="Adjust to see how the model WOULD HAVE performed on this data."
        )

        idx = (sim_df['threshold'] - sim_threshold).abs().idxmin()
        row = sim_df.loc[idx]

        baseline_p = current_metrics.get('precision', 0)
        baseline_r = current_metrics.get('recall', 0)

        st.markdown("<br>", unsafe_allow_html=True)
        
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Proj. Precision", f"{row['precision']:.1%}", delta=f"{(row['precision'] - baseline_p):.1%}")
        with k2:
            st.metric("Proj. Recall", f"{row['recall']:.1%}", delta=f"{(row['recall'] - baseline_r):.1%}", delta_color="inverse")
            
        st.metric(
            "Customer Insult Rate", 
            f"{row['insult']:.2%}", 
            delta="Friction" if row['insult'] > 0.05 else "Smooth",
            delta_color="inverse",
            help="% of Legitimate Customers blocked"
        )

    with col_viz:
        fig_sim = go.Figure()
        
        fig_sim.add_trace(go.Scatter(
            x=sim_df['threshold'], y=sim_df['precision'],
            mode='lines', name='Precision',
            line=dict(color=COLORS['safe'], width=3)
        ))
        
        fig_sim.add_trace(go.Scatter(
            x=sim_df['threshold'], y=sim_df['recall'],
            mode='lines', name='Recall',
            line=dict(color=COLORS['warning'], width=3)
        ))
        
        fig_sim.add_vline(x=sim_threshold, line_dash="dash", line_color="white", annotation_text="Selected")

        fig_sim = apply_plot_style(fig_sim, title="Trade-off Curve")
        fig_sim.update_layout(
            xaxis_title="Threshold",
            yaxis_title="Score",
            hovermode="x unified",
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sim, width='stretch', key="threshold_simulator")

    st.markdown("---")

# ==============================================================================
# ROW 4: FEATURE HEALTH (The "Diagnosis" Layer)
# ==============================================================================
def _render_feature_analysis_row(explain_df: pd.DataFrame, drift_dict: dict):
    """
    Combines Model Sensitivity (SHAP) and Data Drift (PSI) into a single row.
    """
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🔑 Global Feature Importance")
        
        if explain_df.empty:
            st.info("ℹ️ Waiting for SHAP importance data...")
        else:
            df_plot = explain_df.sort_values(by='Importance', ascending=True).tail(15)
            
            fig_feat = go.Figure(go.Bar(
                x=df_plot['Importance'],
                y=df_plot['Feature'],
                orientation='h',
                marker_color=COLORS['text'],
                opacity=0.8,
                text=df_plot['Importance'].apply(lambda x: f"{x:.3f}"),
                textposition='outside'
            ))
            
            fig_feat = apply_plot_style(fig_feat, title="")
            fig_feat.update_layout(
                height=400, 
                margin=dict(l=0, r=40, t=10, b=0),
                xaxis_title="Avg Absolute SHAP Value"
            )
            st.plotly_chart(fig_feat, width='stretch', key="feature_importance")
            st.caption("Top factors currently driving model decisions.")

    with c2:
        st.subheader("⚠️ Feature Drift (PSI)")
        
        if not drift_dict:
            st.info("ℹ️ No drift data available. Waiting for Metrics Worker...")
        else:
            df_drift = pd.DataFrame(list(drift_dict.items()), columns=['Feature', 'PSI'])
            
            df_drift = df_drift.sort_values(by='PSI', ascending=True).tail(15)

            colors = [
                COLORS['danger'] if x > 0.2 else COLORS['warning'] if x > 0.1 else COLORS['safe'] 
                for x in df_drift['PSI']
            ]
            
            fig_drift = go.Figure(go.Bar(
                x=df_drift['PSI'],
                y=df_drift['Feature'],
                orientation='h',
                marker_color=colors,
                text=df_drift['PSI'].apply(lambda x: f"{x:.3f}"),
                textposition='auto'
            ))
            
            fig_drift.add_vline(x=0.2, line_dash="dot", line_color=COLORS['danger'], 
                                annotation_text="Critical Drift")
            
            fig_drift = apply_plot_style(fig_drift, title="")
            fig_drift.update_layout(
                height=400, 
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="PSI Score"
            )
            st.plotly_chart(fig_drift, width='stretch', key="feature_drift")
            st.caption("Measures if live data has shifted from training baseline.")

# ==============================================================================
# MAIN PAGE CONTROLLER
# ==============================================================================
def render_page(recent_df: pd.DataFrame, explain_df: pd.DataFrame, metrics: dict, drift_dict: dict, lookup_table: dict, calibration_data: dict):
    render_header("ML Monitor", "Model Performance, Drift & Calibration")

    _render_efficacy_kpi_row(metrics)
    
    _render_stability_row(recent_df, metrics, calibration_data)       

    _render_threshold_simulator(lookup_table, metrics)

    _render_feature_analysis_row(explain_df, drift_dict)

    with st.expander("📝 View Raw Inference Logs"):
        if not recent_df.empty:
            st.dataframe(
                recent_df[['timestamp', 'transaction_id', 'score', 'TransactionAmt']].head(50), 
                width='stretch'
            )
        else:
            st.info("No inference logs available.")

