import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Ensure imports work from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, kpi_card, apply_plot_style, render_header
from config import StandardColumns

# ==============================================================================
# ROW 1: MODEL EFFICACY (The "Truth" Layer)
# ==============================================================================
def _render_efficacy_kpi_row(metrics: dict):
    """
    Displays high-level statistical health.
    """
    st.markdown("### 📉 Model Efficacy")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        # Precision: TP / (TP + FP)
        # Low precision = High "Insult Rate" (Customer Friction)
        val = metrics.get('precision', 0.88)
        st.markdown(kpi_card(
            "Precision", 
            f"{val:.1%}", 
            "Trustworthiness of Alerts", 
            COLORS['safe']
        ), unsafe_allow_html=True)
    
    with c2:
        # Recall: TP / (TP + FN)
        # Low recall = High Financial Loss
        val = metrics.get('recall', 0.76)
        st.markdown(kpi_card(
            "Recall", 
            f"{val:.1%}", 
            "Fraud Catch Rate", 
            COLORS['highlight']
        ), unsafe_allow_html=True)
    
    with c3:
        # F1 Score: Harmonic Mean
        val = metrics.get('f1_score', 0.81)
        st.markdown(kpi_card(
            "F1-Score", 
            f"{val:.2f}", 
            "Overall Balance", 
            COLORS['neutral']
        ), unsafe_allow_html=True)
        
    with c4:
        # Inference Latency
        # Must be < 200ms for real-time auth
        val = metrics.get('live_latency_ms', 45)
        color = COLORS['safe'] if val < 100 else COLORS['danger']
        st.markdown(kpi_card(
            "Avg Latency", 
            f"{val:.0f}ms", 
            "API Response Time", 
            color
        ), unsafe_allow_html=True)

    st.markdown("---")


# ==============================================================================
# ROW 2: STABILITY & CALIBRATION (The "Consistency" Layer)
# ==============================================================================
def _render_stability_row(recent_df: pd.DataFrame, metrics: dict):
    """
    Scientific analysis of model outputs (Density & Calibration).
    """
    c1, c2 = st.columns([2, 1])

    # --- CHART 1: Probability Density (KDE) ---
    with c1:
        st.subheader("🧠 Score Distribution (Density)")
        if not recent_df.empty and 'score' in recent_df.columns:
            hist_data = [recent_df['score'].values]
            group_labels = ['Model Confidence'] 
            
            try:
                fig_dist = ff.create_distplot(
                    hist_data, group_labels, 
                    bin_size=.025, 
                    colors=[COLORS['highlight']],
                    show_rug=True,
                    show_curve=True
                )
            except: 
                fig_dist = px.histogram(recent_df, x='score', nbins=50)

            # Add Threshold Line
            threshold = metrics.get('threshold', 0.5)
            fig_dist.add_vline(x=threshold, line_dash="dash", line_color=COLORS['danger'], annotation_text="Active Threshold")

            fig_dist = apply_plot_style(fig_dist, title="")
            fig_dist.update_layout(
                xaxis_title="Risk Score (0.0 = Safe, 1.0 = Fraud)",
                yaxis_title="Density",
                showlegend=False,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_dist, width='stretch')
            
            st.caption("💡 **Interpretation:** A healthy model is 'polarized' (peaks at 0 and 1). A growing hump near 0.5 indicates model confusion/drift.")
        else:
            st.info("Waiting for inference data...")

    # --- CHART 2: Calibration Curve ---
    with c2:
        st.subheader("🎯 Calibration")
        #TODO: Replace with real calibration data from metrics when available
        
        # Mocking a Reliability Diagram
        # X = Predicted Prob, Y = Actual Fraction of Fraud
        x_axis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        y_perfect = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        y_model =   [0.01, 0.15, 0.35, 0.65, 0.88, 0.99] # Slightly over/under confident

        fig_cal = go.Figure()
        
        # Perfect Calibration (Reference)
        fig_cal.add_trace(go.Scatter(
            x=x_axis, y=y_perfect,
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Perfectly Calibrated'
        ))

        # Actual Model Performance
        fig_cal.add_trace(go.Scatter(
            x=x_axis, y=y_model,
            mode='lines+markers',
            line=dict(color=COLORS['safe'], width=3),
            marker=dict(size=8),
            name='Current Model'
        ))

        fig_cal = apply_plot_style(fig_cal, title="")
        fig_cal.update_layout(
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Fraud Rate",
            height=300,
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_cal, width='stretch')
        
        st.caption("💡 **Interpretation:** If the green line is **above** the dashed line, the model is 'Underconfident'. If below, it is 'Overconfident'.")

    st.markdown("---")


# ==============================================================================
# ROW 3: THRESHOLD STRATEGY SIMULATOR (The "Tool")
# ==============================================================================
def _render_threshold_simulator(lookup_table: dict, current_metrics: dict):
    """
    Interactive tool using TRUE historical data to simulate impact.
    """
    st.subheader("🎛️ Threshold Strategy Simulator")
    
    # Get baseline values for delta calculation
    baseline_p = current_metrics.get('precision', 0.88)
    baseline_r = current_metrics.get('recall', 0.76)

    col_input, col_kpi = st.columns([1, 3])
    
    with col_input:
        st.markdown("<br>", unsafe_allow_html=True)
        sim_threshold = st.slider(
            "Simulate Risk Threshold", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01, # Finer steps
            help="Adjust to see how the model WOULD HAVE performed on this data."
        )
    
    with col_kpi:
        # We find the key in the dict closest to the slider value
        t_key = f"{sim_threshold:.2f}"
        
        if t_key in lookup_table:
            data = lookup_table[t_key]
            sim_prec = data['p']
            sim_rec = data['r']
            sim_insult = data['insult']
            
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric(
                    "Proj. Precision", 
                    f"{sim_prec:.1%}", 
                    delta=f"{(sim_prec - baseline_p):.1%}"
                )
            with k2:
                st.metric(
                    "Proj. Recall", 
                    f"{sim_rec:.1%}", 
                    delta=f"{(sim_rec - baseline_r):.1%}", 
                    delta_color="inverse"
                )
            with k3:
                # Color the delta red if friction increases
                st.metric(
                    "Customer Insult Rate", 
                    f"{sim_insult:.2%}", 
                    help="% of Legitimate Customers who would have been blocked.",
                    delta="Friction" if sim_insult > 0.05 else "Smooth"
                )
        else:
            st.warning("Simulation data for this threshold is not yet computed.")

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
            st.plotly_chart(fig_feat, width='stretch')
            st.caption("Top factors currently driving model decisions.")

    # --- COLUMN 2: Data Drift (PSI) ---
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
            st.plotly_chart(fig_drift, width='stretch')
            st.caption("Measures if live data has shifted from training baseline.")

# ==============================================================================
# MAIN PAGE CONTROLLER
# ==============================================================================
def render_page(recent_df: pd.DataFrame, explain_df: pd.DataFrame, metrics: dict, drift_dict: dict, lookup_table: dict):
    render_header("ML Monitor", "Model Performance, Drift & Calibration")

    # 1. Efficacy (KPIs)
    _render_efficacy_kpi_row(metrics)
    
    # 2. Stability (Plots)
    _render_stability_row(recent_df, metrics)       
    
    # 3. Strategy (Simulator)
    _render_threshold_simulator(lookup_table, metrics)
    
    # 4. Diagnosis (Features)
    #_render_feature_health(explain_df)
    _render_feature_analysis_row(explain_df, drift_dict)

    # 5. Raw Logs
    with st.expander("📝 View Raw Inference Logs"):
        if not recent_df.empty:
            st.dataframe(
                recent_df[['timestamp', 'transaction_id', 'score', 'TransactionAmt']].head(50), 
                width='stretch'
            )
        else:
            st.info("No inference logs available.")


# def _render_threshold_simulator():
#     """
#     Interactive tool to simulate business impact of threshold changes.
#     """
#     st.subheader("🎛️ Threshold Strategy Simulator")
    
#     col_input, col_kpi = st.columns([1, 3])
    
#     with col_input:
#         st.markdown("<br>", unsafe_allow_html=True)
#         sim_threshold = st.slider(
#             "Simulate Risk Threshold", 
#             min_value=0.0, max_value=1.0, value=0.5, step=0.05,
#             help="Adjust to see impact on Precision vs. Recall"
#         )
    
#     with col_kpi:
#         # Mock Simulation Logic (Standard Precision-Recall Trade-off)
#         # T increases -> Precision Increases, Recall Decreases, Insults Decrease
#         sim_prec = min(0.99, 0.5 + (sim_threshold * 0.45))
#         sim_rec = max(0.1, 0.95 - (sim_threshold * 0.8))
        
#         # Insult Rate: (False Positives / Total Legitimate)
#         # Decreases exponentially as threshold rises
#         sim_insult = max(0.001, 0.05 * (1 - sim_threshold)**2)

#         k1, k2, k3 = st.columns(3)
#         with k1:
#             st.metric("Proj. Precision", f"{sim_prec:.1%}", delta=f"{(sim_prec-0.88)*100:.1f}%")
#         with k2:
#             st.metric("Proj. Recall", f"{sim_rec:.1%}", delta=f"{(sim_rec-0.76)*100:.1f}%", delta_color="inverse")
#         with k3:
#             # Highlight this metric as it's critical for business
#             st.metric(
#                 "Customer Insult Rate", 
#                 f"{sim_insult:.2%}", 
#                 help="% of Legitimate Customers who will get blocked.",
#                 delta=" Friction" if sim_insult > 0.02 else " Smooth"
#             )

#     st.markdown("---")

def _render_feature_health(explain_df: pd.DataFrame):
    """
    Global Feature Importance & Drift Detection (PSI).
    """
    c1, c2 = st.columns(2)
    
    # --- CHART 1: Feature Importance (SHAP) ---
    with c1:
        df_plot = explain_df.sort_values(by='Importance', ascending=True)
        st.subheader("🔑 Global Feature Importance")
        # Mock Data: Represents SHAP values (contribution to fraud score)        
        fig_feat = go.Figure(go.Bar(
            x=df_plot['Importance'],
            y=df_plot['Feature'],
            orientation='h',
            marker_color=COLORS['text'],
            opacity=0.8
        ))
        fig_feat = apply_plot_style(fig_feat, title="")
        fig_feat.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_feat, width='stretch')
        st.caption("Top factors driving the model's decisions right now.")

    # --- CHART 2: Data Drift (PSI) ---
    with c2:
        st.subheader("⚠️ Feature Drift (PSI)")
        # Mock PSI Data: >0.2 is Critical Drift, >0.1 is Warning
        drift_data = pd.DataFrame({
            'Feature': ['TransactionAmt', 'P_emaildomain', 'DeviceType', 'card4', 'dist_from_home'],
            'PSI': [0.05, 0.28, 0.02, 0.12, 0.01] 
        })
        
        # Color Logic
        colors = [
            COLORS['danger'] if x > 0.2 else COLORS['warning'] if x > 0.1 else COLORS['safe'] 
            for x in drift_data['PSI']
        ]
        
        fig_drift = go.Figure(go.Bar(
            x=drift_data['Feature'],
            y=drift_data['PSI'],
            marker_color=colors
        ))
        
        # Critical Threshold Line
        fig_drift.add_hline(y=0.2, line_dash="dot", line_color=COLORS['danger'], annotation_text="Critical Drift (>0.2)")
        
        fig_drift = apply_plot_style(fig_drift, title="")
        fig_drift.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_drift, width='stretch')
        st.caption("Measures if live data distribution has shifted from training data. High PSI = Retraining needed.")


def sync_data(self):
        """Chronologically syncs new data from Redis into local RAM."""
        current_len = self.r.llen('stats:hist_y_prob') 
        
        if current_len > self.last_idx:
            new_probs = self.r.lrange('stats:hist_y_prob', self.last_idx, -1)
            new_trues = self.r.lrange('stats:hist_y_true', self.last_idx, -1)
            new_amts = self.r.lrange('stats:hist_amounts', self.last_idx, -1)
 
            self.y_prob.extend([float(x) for x in new_probs])
            self.y_true.extend([int(x) for x in new_trues])
            self.amounts.extend([float(x) for x in new_amts])

            self.last_idx = current_len
            return True
        return False

    def _optimize_business_strategy(self, evaluator):
        """
        Performs the heavy-duty calculations: 
        1. Generates the full Cost Curve for the Dashboard.
        2. Finds the mathematically optimal threshold.
        """
        logger.info(f"🎯 Running Strategy Optimization...")
        
        # 1. Compute and Save Cost Curve (for the plot)
        cost_curve = evaluator.get_cost_curve(self.cost_params)
        self.r.set("stats:threshold_cost_curve", json.dumps(cost_curve))
        
        # 2. Compute and Save Best Threshold (for the system)
        new_optimal_t = evaluator.find_best_threshold(method='cost', **self.cost_params)
        self.current_threshold = new_optimal_t
        
        self.r.set('config:threshold', self.current_threshold)
        
        logger.info(f"✅ Strategy Updated. New Optimal Threshold: {new_optimal_t}")


    def run(self):
        logger.info(f"📈 Global Metrics Worker Started. Initial Threshold: {self.current_threshold}")

        while True:
            try:
                has_new_data = self.sync_data()
                total_count = len(self.y_true)

                if total_count > 0:
                    evaluator = SentinelEvaluator(self.y_true, self.y_prob, self.amounts)
                    
                    delta = total_count - self.last_optimized_count
                    
                    if delta >= self.OPTIMIZE_EVERY or self.last_optimized_count == 0:
                        self._optimize_business_strategy(evaluator)
                        self.last_optimized_count = total_count

                    full_report = evaluator.report_business_impact(threshold=self.current_threshold)
                    
                    full_report['meta'] = {
                        "threshold": self.current_threshold,
                        "total_count": total_count,
                        "next_optimization_at": self.last_optimized_count + self.OPTIMIZE_EVERY,
                        "updated_at": datetime.now().replace(microsecond=0).isoformat()
                    }

                    # Save the report the FastAPI /stats endpoint is looking for
                    self.r.set("stats:stat_bussiness_report", json.dumps(full_report))

            except Exception as e:
                logger.error(f"❌ Worker Error: {e}")

            time.sleep(REFRESH_INTERVAL)


    def get_cost_curve(self, cost_params: dict = None):
        """
        Returns a JSON-serializable list of dicts for the threshold-loss plot.
        """
        # Default values if none provided
        params = cost_params or {
            'cb_fee': 25.0, 
            'support_cost': 15.0, 
            'churn_factor': 0.1
        }
        
        curve = []
        # 50 points is the standard for a smooth UI curve without heavy CPU usage
        candidates = np.linspace(0.01, 0.99, 50) 
        
        for t in candidates:
            preds = (self.y_prob >= t).astype(int)
            fn_mask = (self.y_true == 1) & (preds == 0)
            fp_mask = (self.y_true == 0) & (preds == 1)
            
            # Calculate Costs
            fn_loss = self.amounts[fn_mask].sum() + (fn_mask.sum() * params['cb_fee'])
            fp_loss = (fp_mask.sum() * params['support_cost']) + \
                    (self.amounts[fp_mask].sum() * params['churn_factor'])
            
            curve.append({
                "threshold": round(float(t), 3),
                "total_loss": round(float(fn_loss + fp_loss), 2)
            })
        return curve

# styles.py old
# import streamlit as st
# import plotly.graph_objects as go

# # ==============================================================================
# # 1. COLOR PALETTE
# # ==============================================================================
# COLORS = {
#     "background": "#0E1117",      
#     "card_bg": "#181b21",         
#     "text": "#FFFFFF",            
#     "safe": "#00CC96",            
#     "danger": "#EF553B",          
#     "warning": "#FFA15A",         
#     "neutral": "#A0A4B0",         
#     "border": "#2b3b4f",          
#     "highlight": "#00CC96"        
# }

# # ==============================================================================
# # 2. PAGE SETUP & CSS
# # ==============================================================================
# def setup_page(title="Sentinel Dashboard"):
#     st.set_page_config(
#         page_title=title,
#         page_icon="🛡️",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     # Inject CSS
#     st.markdown(f"""
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

#         /* 1. GLOBAL APP STYLING */
#         .stApp {{
#             background-color: {COLORS['background']};
#             color: {COLORS['text']};
#             font-family: 'Inter', sans-serif;
#         }}

#         /* 2. SIDEBAR STYLING */
#         [data-testid="stSidebar"] {{
#             background-color: #11141a;
#             border-right: 1px solid {COLORS['border']};
#         }}
#         [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
#             color: {COLORS['highlight']} !important;
#         }}

#         /* 3. HIDE DEFAULT ELEMENTS */
#         .stDeployButton {{ display: none; }}
#         #MainMenu {{ visibility: hidden; }}
#         footer {{ visibility: hidden; }}
#         header {{ background: rgba(0,0,0,0); }}

#         /* 4. KPI CARDS */
#         .kpi-card {{
#             background-color: {COLORS['card_bg']};
#             border: 1px solid {COLORS['border']};
#             border-radius: 8px;
#             padding: 20px;
#             text-align: center;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.2);
#             margin-bottom: 15px;
#             min-height: 140px; 
#             display: flex;
#             flex-direction: column;
#             justify-content: center;
#         }}
#         .kpi-card h4 {{
#             font-size: 14px !important; 
#             font-weight: 400 !important; 
#             color: {COLORS['neutral']} !important; 
#             margin: 0 !important;
#         }}
#         .kpi-value {{
#             font-size: 32px; 
#             font-weight: 700; 
#             margin: 10px 0;
#         }}
#         .kpi-subtext {{
#             font-size: 12px; 
#             color: {COLORS['neutral']}; 
#             margin: 0;
#         }}

#         /* 5. INPUTS & BUTTONS */
#         div.stButton > button {{
#             background-color: {COLORS['card_bg']};
#             color: {COLORS['text']};
#             border: 1px solid {COLORS['border']};
#             border-radius: 5px;
#             width: 100%;
#         }}
#         div.stButton > button:hover {{
#             border-color: {COLORS['highlight']};
#             color: {COLORS['highlight']};
#         }}
#     </style>
#     """, unsafe_allow_html=True)

# # ==============================================================================
# # 3. UI HELPERS
# # ==============================================================================
# def render_header(title, subtitle=""):
#     st.markdown(f"""
#     <div style="border-bottom: 1px solid {COLORS['border']}; padding-bottom: 10px; margin-bottom: 25px;">
#         <h2 style="margin:0; color:{COLORS['text']};">{title}</h2>
#         <p style="margin:0; color:{COLORS['neutral']}; font-size:14px;">{subtitle}</p>
#     </div>
#     """, unsafe_allow_html=True)

# def kpi_card(title, value, subtext, value_color="safe"):
#     # Resolve color
#     color_hex = COLORS.get(value_color, COLORS['safe'])
    
#     return f"""
#     <div class="kpi-card">
#         <h4>{title}</h4>
#         <div class="kpi-value" style="color: {color_hex}">{value}</div>
#         <div class="kpi-subtext">{subtext}</div>
#     </div>
#     """

# def apply_plot_style(fig, title="", height=350):
#     fig.update_layout(
#         template="plotly_dark",
#         title={{
#             'text': f"<b>{title}</b>" if title else "",
#             'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
#             'font': {{'size': 14, 'color': COLORS['text'], 'family': "Inter, sans-serif"}}
#         }},
#         height=height,
#         font=dict(color=COLORS['neutral'], family="Inter, sans-serif"),
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         margin=dict(l=20, r=20, t=60, b=20),
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#     )
#     fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
#     fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
#     return fig



import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# 1. COLOR PALETTE
# ==============================================================================
COLORS = {
    "background": "#0E1117",      # Main App Background
    "card_bg": "#181b21",         # Card Background
    "text": "#FAFAFA",
    "safe": "#00CC96",            # Green
    "danger": "#EF553B",          # Red
    "warning": "#FFA15A",         # Amber
    "neutral": "#8b92a1",         # Subtext Gray
    "border": "#2b3b4f",          # Card Border
    "highlight": "#00CC96"        # Title Color
}

# ==============================================================================
# 2. CSS INJECTION (Optimized)
# ==============================================================================
# In dashboard/styles.py, update the apply_custom_css function:
# def apply_custom_css():
#     st.markdown(f"""
#     <style>
#         /* MAIN BACKGROUND */
#         .stApp {{
#             background-color: {COLORS['background']};
#         }}
        
#         /* HIDE DEFAULT MENU */
#         #MainMenu {{ visibility: hidden; }}
#         footer {{ visibility: hidden; }}
#         header {{ visibility: hidden; }}

#         /* CARD STYLING (Glassmorphism Lite) */
#         .kpi-card {{
#             background-color: rgba(24, 27, 33, 0.7); /* Slight transparency */
#             border: 1px solid {COLORS['border']};
#             border-radius: 8px;
#             padding: 20px;
#             text-align: center;
#             backdrop-filter: blur(5px); /* The Blur Effect */
#             box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#             transition: transform 0.2s;
#         }}
#         .kpi-card:hover {{
#             transform: translateY(-5px);
#             border-color: {COLORS['highlight']};
#         }}
        
#         /* HEADERS */
#         h1, h2, h3 {{
#             font-family: 'Inter', sans-serif;
#             font-weight: 700;
#             color: {COLORS['text']};
#         }}
        
#         /* TABS (If you use them later) */
#         .stTabs [data-baseweb="tab-list"] {{
#             gap: 10px;
#         }}
#         .stTabs [data-baseweb="tab"] {{
#             background-color: {COLORS['card_bg']};
#             border-radius: 4px;
#             color: {COLORS['neutral']};
#         }}
#         .stTabs [data-baseweb="tab"][aria-selected="true"] {{
#             background-color: {COLORS['highlight']};
#             color: #000;
#         }}
#     </style>
#     """, unsafe_allow_html=True)


def apply_custom_css():
    """Injects global CSS styles into the Streamlit app."""
    st.markdown(f"""
    <style>
        /* ---------------------------------------------------------------------
           RESET & MOBILE
           --------------------------------------------------------------------- */
        .stDeployButton {{ display: none; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        
        @media (max-width: 768px) {{
            .kpi-card {{ min-height: 120px; padding: 10px; }}
            .kpi-value {{ font-size: 22px; }}
            .global-title {{ font-size: 2rem; }}
        }}

        /* ---------------------------------------------------------------------
           GLOBAL THEME
           --------------------------------------------------------------------- */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        .block-container {{
            padding-top: 2rem; 
            padding-bottom: 2rem;
        }}

        /* ---------------------------------------------------------------------
           TYPOGRAPHY
           --------------------------------------------------------------------- */
        .global-title {{
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 5px;
            color: {COLORS['highlight']};
            line-height: 1.2;
            text-shadow: 0px 0px 10px rgba(0, 204, 150, 0.3);
        }}
        
        .page-header {{
            text-align: left;
            margin-top: 10px; margin-bottom: 15px;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 5px;
        }}
        .page-header h2 {{
            font-size: 22px; font-weight: 700;
            color: {COLORS['text']}; margin: 0;
        }}
        
        /* ---------------------------------------------------------------------
           KPI CARDS
           --------------------------------------------------------------------- */
        .kpi-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 10px;
            height: 100%; 
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 140px; 
        }}
        .kpi-title {{ font-size: 14px; font-weight: 600; color: {COLORS['text']}; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; }}
        .kpi-subtext {{ font-size: 11px; color: {COLORS['neutral']}; font-style: italic; }}

        /* ---------------------------------------------------------------------
           STATUS INDICATORS & UTILS
           --------------------------------------------------------------------- */
        .status-indicator {{
            height: 12px; width: 12px; border-radius: 50%;
            display: inline-block; margin-right: 8px;
        }}
        .status-green {{ background-color: {COLORS['safe']}; box-shadow: 0 0 6px {COLORS['safe']}; }}
        .status-orange {{ background-color: {COLORS['warning']}; box-shadow: 0 0 6px {COLORS['warning']}; }}
        .status-red {{ background-color: {COLORS['danger']}; box-shadow: 0 0 6px {COLORS['danger']}; }}
        
        div.stButton > button {{
            width: 100%;
            background-color: {COLORS['card_bg']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            height: 45px; font-weight: 600;
        }}
        div.stButton > button:hover {{
            border-color: {COLORS['safe']}; color: {COLORS['safe']};
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. UI HELPERS
# ==============================================================================
def render_header(title, subtitle=""):
    """Renders a consistent section header."""
    sub_html = f"<div style='font-size:12px; color:{COLORS['neutral']}; margin-top:2px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <h2>{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def kpi_card(title, value, subtext, value_color=COLORS['safe']):
    """Returns HTML for a styled KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {value_color}">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """

def apply_plot_style(fig, title="", height=350):
    """Applies the Dashboard Dark Theme to any Plotly figure."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 14, 'color': COLORS['text'], 'family': "Helvetica Neue, sans-serif"}
        },
        height=height,
        font=dict(color=COLORS['neutral'], family="Helvetica Neue, sans-serif"),
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=COLORS['text'])
        )
    )
    # Subtle Grid
    grid_style = dict(showgrid=True, gridcolor="rgba(43, 59, 79, 0.5)", linecolor=COLORS['border'], zeroline=False)
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(**grid_style)
    return fig
 
#dashboard/api_client.py
"""
ROLE: API Client (Adapter)
RESPONSIBILITIES:
1.  Abstracts HTTP requests to the Backend Service.
2.  Handles connection errors and timeouts gracefully.
3.  Converts raw JSON responses into Pandas DataFrames for Streamlit.
4.  Provides fallback data structures to prevent UI crashes.
"""
import os
import requests
import logging
import pandas as pd
from typing import Dict, Any, Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Default to 'backend' service name in Docker Compose network
API_BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000")
REQUEST_TIMEOUT = 3  # Seconds

logger = logging.getLogger("ApiClient")

class SentinelApiClient:
    """
    Client for interacting with the Sentinel Fraud Ops Backend.
    """

    def __init__(self):
        self.base_url = API_BASE_URL
        self.session = requests.Session()
        logger.info(f"🔌 API Client initialized pointing to: {self.base_url}")

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Internal helper to perform GET requests with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Connection Error: Could not reach {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"⏳ Timeout: Backend did not respond in {REQUEST_TIMEOUT}s")
            return None
        except Exception as e:
            logger.error(f"⚠️ API Error ({endpoint}): {e}")
            return None

    # ==========================================================================
    # SYSTEM HEALTH
    # ==========================================================================
    def is_backend_alive(self) -> bool:
        """Checks if the backend is reachable and Redis is connected."""
        data = self._get("/health")
        return data is not None and data.get("status") == "healthy"

    def get_system_metrics(self) -> Dict[str, Any]:
        """Fetches CPU and Memory usage of the backend service."""
        default = {"memory_usage_mb": 0, "cpu_usage_percent": 0, "redis_connected": False}
        data = self._get("/metrics")
        return data if data else default

    # ==========================================================================
    # DATA & STATISTICS
    # ==========================================================================
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Fetches aggregated statistics (Fraud Rate, Total Count, etc.).
        Returns zeroed-out dict on failure.
        """
        default = {
            "total_processed": 0,
            "fraud_detected": 0,
            "legit_transactions": 0,
            "fraud_rate": 0.0,
            "queue_depth": 0,
            "updated_at": "N/A"
        }
        data = self._get("/stats")
        return data if data else default

    def get_recent_transactions(self, limit: int = 20) -> pd.DataFrame:
        """
        Fetches the latest transactions stream.
        Returns: Pandas DataFrame suitable for Streamlit display.
        """
        data = self._get("/recent", params={"limit": limit})
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Data Formatting for UI
        if not df.empty:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            cols = ['timestamp', 'transaction_id', 'amount', 'score', 'is_fraud', 'action']
            existing_cols = [c for c in cols if c in df.columns]
            extra_cols = [c for c in df.columns if c not in cols]
            df = df[existing_cols + extra_cols]

        return df

    def get_fraud_alerts(self, limit: int = 10) -> pd.DataFrame:
        """
        Fetches only high-risk transactions (Fraud Alerts).
        Returns: Pandas DataFrame.
        """
        data = self._get("/alerts", params={"limit": limit})
        
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    
    
#dashboard/pages/executive.py
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

#dashboard/pages/forensics.py
import streamlit as st
import pandas as pd
import plotly.express as px
from styles import COLORS, apply_plot_style, render_header

def render_page(df: pd.DataFrame):
    render_header("Forensics & Search", "Deep Dive into Transaction Details")

    if df.empty:
        st.info("Waiting for data...")
        return

    # --- 1. SEARCH BAR ---
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            search_term = st.text_input("Search Transaction ID, Product, or Amount", placeholder="e.g., tx_1234 or 150.00")
        with c2:
            st.markdown("<br>", unsafe_allow_html=True) 
            filter_fraud_only = st.checkbox("Show Fraud Only", value=False)

    # --- 2. FILTER LOGIC ---
    filtered_df = df.copy()
    
    if filter_fraud_only:
        filtered_df = filtered_df[filtered_df['is_fraud'] == 1]

    if search_term:
        # Simple string matching across columns
        mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
        filtered_df = filtered_df[mask]

    # --- 3. RESULTS AREA ---
    st.markdown(f"**Found {len(filtered_df)} transactions**")

    if not filtered_df.empty:
        # Split screen: Table on Left, Details on Right
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.dataframe(
                filtered_df[['transaction_id', 'timestamp', 'amount', 'score', 'is_fraud', 'ProductCD']],
                use_container_width=True,
                height=500,
                hide_index=True
            )

        with c2:
            st.markdown("### 🔍 Risk Analysis")
            if len(filtered_df) == 1:
                # Detail View for Single Record
                record = filtered_df.iloc[0]
                
                score_color = COLORS['danger'] if record['score'] > 0.5 else COLORS['safe']
                st.markdown(f"""
                <div style="background-color: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                    <h1 style="color: {score_color}; margin:0;">{record['score']:.2f}</h1>
                    <div style="color: {COLORS['neutral']}; margin-bottom: 20px;">Risk Score</div>
                    
                    <p><b>ID:</b> {record['transaction_id']}</p>
                    <p><b>Amount:</b> ${record['amount']}</p>
                    <p><b>Product:</b> {record['ProductCD']}</p>
                    <p><b>Timestamp:</b> {record['timestamp']}</p>
                    <hr style="border-color: {COLORS['border']}">
                    <p><b>Action Taken:</b> {record.get('action', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Aggregate View for Multiple Records
                avg_score = filtered_df['score'].mean()
                total_amt = filtered_df['amount'].sum()
                
                fig = px.pie(filtered_df, names='is_fraud', title="Fraud vs Legit in Search",
                             color='is_fraud', color_discrete_map={0: COLORS['safe'], 1: COLORS['danger']})
                fig = apply_plot_style(fig, height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Total Amount in View", f"${total_amt:,.2f}")
                st.metric("Average Risk Score", f"{avg_score:.2f}")

    else:
        st.warning("No transactions match your search criteria.")

#dashboard/pages/ml.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve
from styles import COLORS, kpi_card, apply_plot_style, render_header

def plot_pr_vs_threshold(df, current_threshold):
    """Calculates and plots Precision-Recall curve"""
    try:
        y_true = df['ground_truth']
        y_scores = df['composite_risk_score']
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    except Exception:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions[:-1], name="Precision", line=dict(color=COLORS['safe'], width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls[:-1], name="Recall", line=dict(color=COLORS['warning'], width=3)))
    fig.add_vline(x=current_threshold, line_width=2, line_dash="dash", line_color="white")

    fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1]), hovermode="x unified")
    return apply_plot_style(fig, title="Precision & Recall Trade-off")

def render_page(df, threshold):
    render_header("ML Integrity", "Model Drift & Performance Monitoring")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # --- DRIFT CALCULATION ---
    drift_score = 0.0
    min_samples = 50
    if len(df) > min_samples:
        try:
            # Simple Drift: KS Test between first half and second half of buffer
            mid = len(df) // 2
            ks_stat, _ = ks_2samp(df.iloc[:mid]['composite_risk_score'], df.iloc[mid:]['composite_risk_score'])
            drift_score = ks_stat
        except: pass

    # --- METRICS ROW ---
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(kpi_card("PR-AUC", "0.89", "Precision-Recall AUC", COLORS['safe']), unsafe_allow_html=True)
    with c2: 
        status = 'Stable' if drift_score < 0.1 else 'Drift Detected'
        color = COLORS['safe'] if drift_score < 0.1 else COLORS['warning']
        st.markdown(kpi_card("PSI (Data Drift)", f"{drift_score:.2f}", status, color), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Concept Drift", "0.02", "Label Consistency", COLORS['safe']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PLOTS ---
    c1, c2 = st.columns(2)
    with c1:
        # Score Separation (KDE approximation via Hist)
        fig_kde = go.Figure()
        fig_kde.add_trace(go.Histogram(x=df[df['ground_truth']==0]['composite_risk_score'], name='Legit', marker_color=COLORS['safe'], opacity=0.6))
        fig_kde.add_trace(go.Histogram(x=df[df['ground_truth']==1]['composite_risk_score'], name='Fraud', marker_color=COLORS['danger'], opacity=0.6))
        fig_kde.add_vline(x=threshold, line_width=3, line_color="white")
        fig_kde = apply_plot_style(fig_kde, title="Score Separation (Legit vs Fraud)")
        fig_kde.update_layout(barmode='overlay')
        st.plotly_chart(fig_kde, use_container_width=True)
    
    with c2:
        # Precision Recall Curve
        if df['ground_truth'].sum() > 0:
            fig_pr = plot_pr_vs_threshold(df, threshold)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            # Placeholder if no fraud in buffer
            st.info("Insufficient fraud labels in buffer to generate PR Curve.")

    # --- STABILITY ---
    if len(df) > min_samples:
         roll = df['composite_risk_score'].rolling(window=20).mean()
         fig_drift = go.Figure(go.Scatter(y=roll, name="Mean Score", line=dict(color=COLORS['neutral'])))
         fig_drift = apply_plot_style(fig_drift, title=f"Score Stability (Rolling Mean)")
         st.plotly_chart(fig_drift, use_container_width=True)

#dashboard/pages/ops.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from styles import COLORS, kpi_card, apply_plot_style, render_header

def render_page(df, threshold):
    render_header("Real-Time Operations", "SOC Monitoring & Case Management")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # --- OPERATIONAL METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # Calculate Risk Index
        risk_mean = df.tail(50)['composite_risk_score'].mean() if 'composite_risk_score' in df.columns else 0
        risk_index = int(risk_mean * 100)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = risk_index,
            number = {'font': {'color': COLORS['text'], 'size': 24}, 'suffix': "%"}, 
            gauge = {
                'axis': {'range': [None, 100], 'visible': False}, 
                'bar': {'color': "rgba(0,0,0,0)"}, 
                'steps': [
                    {'range': [0, 40], 'color': COLORS['safe']},
                    {'range': [40, 75], 'color': COLORS['warning']},
                    {'range': [75, 100], 'color': COLORS['danger']}
                ],
                'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': risk_index}
            }
        ))
        fig_gauge = apply_plot_style(fig_gauge, title="Live Risk Index", height=155)
        fig_gauge.update_layout(margin=dict(l=25, r=25, t=35, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    
    # Latency simulation if column missing
    proc_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 45
    
    with c2: st.markdown(kpi_card("Mean Latency", f"{proc_time:.0f}ms", "SLA: < 100ms", COLORS['safe']), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Throughput", f"{len(df)}", "Transactions Buffered", COLORS['text']), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Analyst Overturn", "1.2%", "Label Correction Rate", COLORS['warning']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TRAFFIC & BOT HUNTER ---
    c1, c2 = st.columns([1.5, 1])
    with c1:
        stream_df = df.tail(100).copy()
        stream_df['legit_vol'] = 1 
        stream_df['blocked_vol'] = stream_df['composite_risk_score'].apply(lambda x: 1 if x > threshold else 0)
        
        fig_pulse = go.Figure()
        fig_pulse.add_trace(go.Scatter(x=stream_df['timestamp'], y=stream_df['legit_vol'], stackgroup='one', name='Legit', line=dict(color=COLORS['safe'])))
        fig_pulse.add_trace(go.Scatter(x=stream_df['timestamp'], y=stream_df['blocked_vol'], stackgroup='one', name='Blocked', line=dict(color=COLORS['danger'])))
        fig_pulse = apply_plot_style(fig_pulse, title="Traffic Pulse (Rolling Window)")
        st.plotly_chart(fig_pulse, use_container_width=True)
        
    with c2:
        # Scatter Plot: Velocity vs Amount
        plot_df = df.tail(200).copy()
        if 'UID_velocity_24h' not in plot_df.columns: plot_df['UID_velocity_24h'] = 0
            
        fig_scatter = px.scatter(
            plot_df, 
            x='UID_velocity_24h', 
            y='TransactionAmt', 
            color='composite_risk_score',
            color_continuous_scale='Reds',
            size='TransactionAmt',
            size_max=15,
            labels={'UID_velocity_24h': 'Velocity', 'TransactionAmt': 'Amt ($)'}
        )
        fig_scatter.add_vline(x=40, line_dash="dash", line_color=COLORS['warning'])
        fig_scatter.update_layout(coloraxis_colorbar=dict(title="Risk", orientation="v", title_side="right"))
        fig_scatter = apply_plot_style(fig_scatter, title="Bot Hunter (Vel vs Amt)")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # --- ALERT QUEUE ---
    st.markdown("<h3 style='text-align:left'>🔥 High-Priority Investigation Queue</h3>", unsafe_allow_html=True)
    queue = df[df['composite_risk_score'] > threshold].copy()
    queue = queue.sort_values('composite_risk_score', ascending=False).head(10)

    if not queue.empty:
        queue['time_formatted'] = queue['timestamp'].dt.strftime('%H:%M:%S')
        queue['Action'] = 'REVIEW'
        
        # Select relevant columns for the analyst
        cols = ['transaction_id', 'time_formatted', 'composite_risk_score', 'TransactionAmt', 'Action']
        display_queue = queue[[c for c in cols if c in queue.columns]]
        
        st.dataframe(display_queue.style.background_gradient(subset=['composite_risk_score'], cmap="Reds"), use_container_width=True)
    else:
        st.success("✅ Queue is empty. System healthy.")

#dashboard/pages/strategy.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from styles import COLORS, apply_plot_style, render_header

def render_page(df: pd.DataFrame):
    render_header("Strategy & Product", "Risk Profiling & Link Analysis")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    c1, c2 = st.columns(2)

    # --- DEVICE RISK ---
    with c1:
        if 'device_vendor' in df.columns:
            # Simple aggregation
            dev_risk = df.groupby('device_vendor')['ground_truth'].mean().reset_index()
            # Filter for visualisation
            dev_risk = dev_risk.sort_values('ground_truth', ascending=True).tail(10)
            
            fig_dev = px.bar(dev_risk, y='device_vendor', x='ground_truth', orientation='h', 
                             color='ground_truth', color_continuous_scale='Reds', 
                             labels={'ground_truth': 'Fraud Rate'})
            fig_dev.update_layout(coloraxis_colorbar=dict(title="Rate", orientation="v", title_side="right"))
            fig_dev = apply_plot_style(fig_dev, title="Risk by Device Vendor")
            st.plotly_chart(fig_dev, use_container_width=True)
        else:
            st.warning("Device data not available in stream.")

    # --- EMAIL DOMAIN RISK (Mock if column missing) ---
    with c2:
        # Mocking this specific chart if data is missing, as per original dashboard intent
        email_data = pd.DataFrame({
            'Domain': ['Proton', 'TempMail', 'Gmail', 'Yahoo', 'Corporate'], 
            'Risk_Rate': [0.85, 0.95, 0.02, 0.03, 0.01]
        }).sort_values('Risk_Rate')
        
        fig_email = px.bar(email_data, y='Domain', x='Risk_Rate', orientation='h', 
                           color='Risk_Rate', color_continuous_scale='Reds')
        fig_email = apply_plot_style(fig_email, title="Risk by Email Domain (Global Stats)")
        st.plotly_chart(fig_email, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- NETWORK GRAPH (FRAUD RINGS) ---
    try:
        # Simulated Network Graph
        G = nx.Graph()
        center = "Bad_Actor_X"
        G.add_node(center, type='User')
        for i in range(5):
            ip = f"Device_{i}" 
            G.add_node(ip, type='Device')
            G.add_edge(center, ip)
            linked = f"User_{i}" 
            G.add_node(linked, type='User')
            G.add_edge(ip, linked)
            
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        node_x, node_y, node_color, node_text = [], [], [], []
        
        for node in G.nodes():
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            node_text.append(node)
            if node == center: node_color.append(COLORS['danger'])
            elif "Device" in node: node_color.append(COLORS['warning'])
            else: node_color.append(COLORS['neutral'])
            
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, 
                                marker=dict(showscale=False, color=node_color, size=20))
                                
        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net = apply_plot_style(fig_net, title="Fraud Ring Analysis (Linked via C13/Device)")
        fig_net.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig_net, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render Network Graph: {e}")

#and dashboard/app.py
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




#styles.py
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# 1. COLOR PALETTE
# ==============================================================================
COLORS = {
    "background": "#0E1117",      # Main App Background
    "card_bg": "#181b21",         # Card Background
    "text": "#FAFAFA",
    "safe": "#00CC96",            # Green
    "danger": "#EF553B",          # Red
    "warning": "#FFA15A",         # Amber
    "neutral": "#8b92a1",         # Subtext Gray
    "border": "#2b3b4f",          # Card Border
    "highlight": "#00CC96"        # Title Color
}

# ==============================================================================
# 2. CSS INJECTION (Optimized)
# ==============================================================================
# In dashboard/styles.py, update the apply_custom_css function:
# def apply_custom_css():
#     st.markdown(f"""
#     <style>
#         /* MAIN BACKGROUND */
#         .stApp {{
#             background-color: {COLORS['background']};
#         }}
        
#         /* HIDE DEFAULT MENU */
#         #MainMenu {{ visibility: hidden; }}
#         footer {{ visibility: hidden; }}
#         header {{ visibility: hidden; }}

#         /* CARD STYLING (Glassmorphism Lite) */
#         .kpi-card {{
#             background-color: rgba(24, 27, 33, 0.7); /* Slight transparency */
#             border: 1px solid {COLORS['border']};
#             border-radius: 8px;
#             padding: 20px;
#             text-align: center;
#             backdrop-filter: blur(5px); /* The Blur Effect */
#             box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#             transition: transform 0.2s;
#         }}
#         .kpi-card:hover {{
#             transform: translateY(-5px);
#             border-color: {COLORS['highlight']};
#         }}
        
#         /* HEADERS */
#         h1, h2, h3 {{
#             font-family: 'Inter', sans-serif;
#             font-weight: 700;
#             color: {COLORS['text']};
#         }}
        
#         /* TABS (If you use them later) */
#         .stTabs [data-baseweb="tab-list"] {{
#             gap: 10px;
#         }}
#         .stTabs [data-baseweb="tab"] {{
#             background-color: {COLORS['card_bg']};
#             border-radius: 4px;
#             color: {COLORS['neutral']};
#         }}
#         .stTabs [data-baseweb="tab"][aria-selected="true"] {{
#             background-color: {COLORS['highlight']};
#             color: #000;
#         }}
#     </style>
#     """, unsafe_allow_html=True)


def apply_custom_css():
    """Injects global CSS styles into the Streamlit app."""
    st.markdown(f"""
    <style>
        /* ---------------------------------------------------------------------
           RESET & MOBILE
           --------------------------------------------------------------------- */
        .stDeployButton {{ display: none; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        
        @media (max-width: 768px) {{
            .kpi-card {{ min-height: 120px; padding: 10px; }}
            .kpi-value {{ font-size: 22px; }}
            .global-title {{ font-size: 2rem; }}
        }}

        /* ---------------------------------------------------------------------
           GLOBAL THEME
           --------------------------------------------------------------------- */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        .block-container {{
            padding-top: 2rem; 
            padding-bottom: 2rem;
        }}

        /* ---------------------------------------------------------------------
           TYPOGRAPHY
           --------------------------------------------------------------------- */
        .global-title {{
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 5px;
            color: {COLORS['highlight']};
            line-height: 1.2;
            text-shadow: 0px 0px 10px rgba(0, 204, 150, 0.3);
        }}
        
        .page-header {{
            text-align: left;
            margin-top: 10px; margin-bottom: 15px;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 5px;
        }}
        .page-header h2 {{
            font-size: 22px; font-weight: 700;
            color: {COLORS['text']}; margin: 0;
        }}
        
        /* ---------------------------------------------------------------------
           KPI CARDS
           --------------------------------------------------------------------- */
        .kpi-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 10px;
            height: 100%; 
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 140px; 
        }}
        .kpi-title {{ font-size: 14px; font-weight: 600; color: {COLORS['text']}; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; }}
        .kpi-subtext {{ font-size: 11px; color: {COLORS['neutral']}; font-style: italic; }}

        /* ---------------------------------------------------------------------
           STATUS INDICATORS & UTILS
           --------------------------------------------------------------------- */
        .status-indicator {{
            height: 12px; width: 12px; border-radius: 50%;
            display: inline-block; margin-right: 8px;
        }}
        .status-green {{ background-color: {COLORS['safe']}; box-shadow: 0 0 6px {COLORS['safe']}; }}
        .status-orange {{ background-color: {COLORS['warning']}; box-shadow: 0 0 6px {COLORS['warning']}; }}
        .status-red {{ background-color: {COLORS['danger']}; box-shadow: 0 0 6px {COLORS['danger']}; }}
        
        div.stButton > button {{
            width: 100%;
            background-color: {COLORS['card_bg']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            height: 45px; font-weight: 600;
        }}
        div.stButton > button:hover {{
            border-color: {COLORS['safe']}; color: {COLORS['safe']};
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. UI HELPERS
# ==============================================================================
def render_header(title, subtitle=""):
    """Renders a consistent section header."""
    sub_html = f"<div style='font-size:12px; color:{COLORS['neutral']}; margin-top:2px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <h2>{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def kpi_card(title, value, subtext, value_color=COLORS['safe']):
    """Returns HTML for a styled KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {value_color}">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """

def apply_plot_style(fig, title="", height=350):
    """Applies the Dashboard Dark Theme to any Plotly figure."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 14, 'color': COLORS['text'], 'family': "Helvetica Neue, sans-serif"}
        },
        height=height,
        font=dict(color=COLORS['neutral'], family="Helvetica Neue, sans-serif"),
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=COLORS['text'])
        )
    )
    # Subtle Grid
    grid_style = dict(showgrid=True, gridcolor="rgba(43, 59, 79, 0.5)", linecolor=COLORS['border'], zeroline=False)
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(**grid_style)
    return fig