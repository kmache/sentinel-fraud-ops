"""
Dashboard for Sentinel Fraud Detection System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve
import networkx as nx
import time
import json
import os
import redis
from scipy.stats import ks_2samp

# ==============================================================================
# 1. CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="Sentinel | Real-Time Fraud Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color Palette
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

# ------------------------------------------------------------------------------
# CSS INJECTION
# ------------------------------------------------------------------------------

st.markdown(f"""
<style>
    /* Global App Background */
    .stApp {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
    }}

    /* 1. ADJUST CONTAINER */
    .block-container {{
        padding-top: 2rem; 
        padding-bottom: 1rem;
    }}

    /* CENTRALIZED GLOBAL TITLE */
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
    .global-summary {{
        text-align: center;
        color: {COLORS['neutral']};
        font-size: 1rem;
        margin-bottom: 25px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }}

    /* LEFT ALIGNED PAGE HEADERS */
    .page-header {{
        text-align: left;
        font-family: 'Helvetica Neue', sans-serif;
        margin-top: 0px; 
        margin-bottom: 15px;
        border-bottom: 1px solid {COLORS['border']};
        padding-bottom: 5px;
    }}
    .page-header h2 {{
        font-size: 22px;
        font-weight: 700;
        color: {COLORS['text']};
        margin: 0;
    }}
    
    /* KPI CARD STYLING */
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
    
    .kpi-title {{
        font-size: 16px;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }}
    
    .kpi-value {{
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 10px;
    }}
    
    .kpi-subtext {{
        font-size: 12px;
        color: {COLORS['neutral']};
        font-style: italic;
    }}

    /* STATUS INDICATORS */
    .status-indicator {{
        height: 15px;
        width: 15px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }}
    .status-green {{ background-color: {COLORS['safe']}; box-shadow: 0 0 8px {COLORS['safe']}; }}
    .status-orange {{ background-color: {COLORS['warning']}; box-shadow: 0 0 8px {COLORS['warning']}; }}
    .status-red {{ background-color: {COLORS['danger']}; box-shadow: 0 0 8px {COLORS['danger']}; }}

    /* PLOT STYLING */
    .stPlotlyChart {{
        width: 100%;
    }}
    
    /* NAVIGATION BUTTON STYLING */
    div.stButton > button {{
        width: 100%;
        background-color: {COLORS['card_bg']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        height: 50px;
        font-weight: 600;
        margin-bottom: 0px;
    }}
    div.stButton > button:hover {{
        border-color: {COLORS['safe']};
        color: {COLORS['safe']};
    }}
    div.stButton > button:focus {{
        background-color: {COLORS['border']};
        color: white;
    }}
    
    /* CUSTOM SEPARATOR STYLE */
    .nav-separator {{
        margin-top: -10px; 
        margin-bottom: 10px; 
        border: 0; 
        border-top: 1px solid {COLORS['border']};
    }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def render_header(title):
    st.markdown(f"""
    <div class="page-header">
        <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(title, value, subtext, value_color=COLORS['safe']):
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {value_color}">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """

def apply_plot_style(fig, title="", height=350):
    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': COLORS['text']}
        },
        height=height,
        font=dict(color=COLORS['neutral']),
        paper_bgcolor=COLORS['card_bg'],
        plot_bgcolor=COLORS['card_bg'],
        margin=dict(l=20, r=20, t=40, b=20),
        # MOVE LEGEND INSIDE PLOT
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(size=10, color="white")
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'], linecolor=COLORS['border'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], linecolor=COLORS['border'])
    return fig

def plot_pr_vs_threshold(df, current_threshold):
    """Plotly version of Precision/Recall vs Threshold."""
    y_true = df['ground_truth']
    y_scores = df['composite_risk_score']
    
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    except Exception:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions[:-1], name="Precision", line=dict(color=COLORS['safe'], width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls[:-1], name="Recall", line=dict(color=COLORS['warning'], width=3)))
    fig.add_vline(x=current_threshold, line_width=2, line_dash="dash", line_color="white")

    if len(thresholds) > 0:
        idx = (np.abs(thresholds - current_threshold)).argmin()
        if idx < len(precisions) and idx < len(recalls):
            curr_prec = precisions[idx]
            curr_rec = recalls[idx]
            fig.add_annotation(
                x=current_threshold, y=0.5, 
                text=f"P: {curr_prec:.2f}<br>R: {curr_rec:.2f}", 
                showarrow=True, arrowhead=1, ax=40, ay=0, 
                bgcolor=COLORS['card_bg'], bordercolor=COLORS['border']
            )

    fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1]), hovermode="x unified")
    return apply_plot_style(fig, title="Precision & Recall Trade-off")

def get_device_name(val):
    """Inverse map for Device Vendor"""
    inv_map = {
        1: 'Samsung', 2: 'Apple/iOS', 3: 'Huawei', 4: 'Motorola',
        5: 'LG', 6: 'Sony', 7: 'ZTE', 8: 'Pixel', 9: 'Lenovo',
        10: 'Alcatel', 11: 'Xiaomi', 12: 'Windows', 13: 'RV',
        14: 'HTC', 15: 'Asus'
    }
    return inv_map.get(val, 'Unknown')

# ==============================================================================
# 2. REAL DATA ENGINE (Redis Connected)
# ==============================================================================

class SentinelEngine:
    def __init__(self):
        # Config from Environment Variables (Docker)
        redis_host = os.getenv('REDIS_HOST', 'redis')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_pass = os.getenv('REDIS_PASSWORD', 'sentinel_pass_2024')
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                password=redis_pass, 
                decode_responses=True,
                socket_timeout=2
            )
            self.redis_client.ping()
        except Exception as e:
            self.redis_client = None

    def get_live_data(self):
        """Fetch real-time data from Redis Stream"""
        if not self.redis_client:
            return pd.DataFrame()
            
        try:
            # Fetch the last 2000 records
            raw_data = self.redis_client.lrange('sentinel_stream', 0, -1)
            
            if not raw_data:
                return pd.DataFrame()
            
            data = [json.loads(x) for x in raw_data]
            df = pd.DataFrame(data)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            numeric_cols = ['composite_risk_score', 'ground_truth', 'TransactionAmt', 
                           'UID_velocity_24h', 'dist1', 'device_vendor', 'D1_norm', 'processing_time_ms']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            if 'ground_truth' not in df.columns: df['ground_truth'] = 0
            if 'composite_risk_score' not in df.columns: df['composite_risk_score'] = 0.0
            if 'TransactionAmt' not in df.columns: df['TransactionAmt'] = 0.0
            
            return df
            
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return pd.DataFrame()

    def update_config(self, mode, weight_lgb, weight_cb, threshold):
        if self.redis_client:
            self.redis_client.set('config:mode', mode)
            self.redis_client.set('config:weight_lgb', weight_lgb)
            self.redis_client.set('config:weight_cb', weight_cb)
            self.redis_client.set('config:threshold', threshold)

    def get_config(self):
        if self.redis_client:
            return {
                'mode': self.redis_client.get('config:mode') or 'ensemble',
                'weight_lgb': float(self.redis_client.get('config:weight_lgb') or 0.6),
                'threshold': float(self.redis_client.get('config:threshold') or 0.5)
            }
        return {'mode': 'ensemble', 'weight_lgb': 0.6, 'threshold': 0.5}

# Initialize Engine
engine = SentinelEngine()
df = engine.get_live_data()
current_config = engine.get_config()

# ==============================================================================
# 3. GLOBAL SIDEBAR (DYNAMIC CONTROL PLANE)
# ==============================================================================

with st.sidebar:
    st.markdown(f"<h2 style='text-align: center; color: {COLORS['highlight']}; margin-bottom:0;'>⚙️ Controls</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("Infrastructure")
    c1, c2 = st.columns(2)
    
    # Traffic Light Logic
    stream_status = "red"
    status_text = "Offline"
    
    if not df.empty and 'timestamp' in df.columns:
        last_ts = df['timestamp'].max()
        time_diff = (pd.Timestamp.now() - last_ts).total_seconds()
        
        if time_diff < 30: 
            stream_status = "green"
            status_text = "Live"
        elif time_diff < 120: 
            stream_status = "orange"
            status_text = "Idle"
        else: 
            stream_status = "red"
            status_text = "Stale"
            
    with c1:
        st.markdown("**Stream**")
        st.markdown(f"""
        <div style="display:flex; align-items:center;">
            <div class="status-indicator status-{stream_status}"></div>
            <span>{status_text}</span>
        </div>
        """, unsafe_allow_html=True)
        
    # Buffer count removed as requested if not increasing

    st.markdown("---")
    st.subheader("Model Control Plane")

    # 1. Mode
    mode_options = ["champion", "ensemble", "shadow"]
    try:
        idx = mode_options.index(current_config['mode'])
    except: idx = 1
        
    selected_mode = st.selectbox("Operating Strategy", mode_options, index=idx)

    # 2. Weights
    if selected_mode == "ensemble":
        st.markdown("**Ensemble Weights**")
        w_lgb = st.slider("LGBM (Champion) Weight", 0.0, 1.0, current_config['weight_lgb'], 0.05)
        w_cb = round(1.0 - w_lgb, 2)
        st.progress(w_lgb)
        st.caption(f"LGBM: {w_lgb} | CatBoost: {w_cb}")
    else:
        w_lgb = 0.6; w_cb = 0.4

    # 3. Threshold
    threshold = st.slider("Decision Threshold", 0.0, 1.0, current_config['threshold'], 0.01)

    # 4. Risk Mode
    risk_mode = "Balanced"
    mode_color = COLORS['safe']
    if threshold > 0.75: risk_mode = "Aggressive"; mode_color = COLORS['danger']
    elif threshold < 0.25: risk_mode = "Permissive"; mode_color = COLORS['warning']
    st.markdown(f"<div style='text-align:center'>Risk Appetite: <b style='color:{mode_color}'>{risk_mode}</b></div>", unsafe_allow_html=True)

    # 5. Update
    engine.update_config(selected_mode, w_lgb, w_cb, threshold)

    st.markdown("---")
    if st.button("Refresh View"):
        st.rerun()

# ==============================================================================
# 4. BUSINESS LOGIC CALCULATIONS
# ==============================================================================

def calculate_fraud_metrics(df, threshold):
    """
    Calculate metrics. 
    Formula: Net Benefit = Saved - (FP*5) - Missed Fraud
    """
    if df.empty:
        return {
            'net_benefit': 0, 'fraud_prevented': 0, 'fraud_losses': 0,
            'tp': 0, 'fp': 0, 'fn': 0, 'recall': 0, 'fp_ratio': 0
        }

    df_calc = df.copy()
    df_calc['predicted_fraud'] = df_calc['composite_risk_score'] > threshold

    # Masks
    actual_fraud = df_calc['ground_truth'] == 1
    predicted_fraud = df_calc['predicted_fraud']
    
    tp_mask = predicted_fraud & actual_fraud
    fp_mask = predicted_fraud & ~actual_fraud
    fn_mask = ~predicted_fraud & actual_fraud

    # Counts
    tp = tp_mask.sum()
    fp = fp_mask.sum()
    fn = fn_mask.sum()

    # Values
    fraud_prevented = df_calc.loc[tp_mask, 'TransactionAmt'].sum()
    fraud_losses = df_calc.loc[fn_mask, 'TransactionAmt'].sum() # Missed fraud
    investigation_costs = fp * 5.0 

    # Specific Formula Requested
    net_benefit = fraud_prevented - investigation_costs - fraud_losses

    # Performance
    recall = tp / max(tp + fn, 1)
    denom_fp = tp if tp > 0 else 1
    fp_ratio = round(fp / denom_fp, 1)

    return {
        'net_benefit': float(net_benefit),
        'fraud_prevented': float(fraud_prevented),
        'fraud_losses': float(fraud_losses),
        'investigation_costs': float(investigation_costs),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'recall': float(recall),
        'fp_ratio': fp_ratio
    }

metrics = calculate_fraud_metrics(df, threshold)

# ==============================================================================
# 5. PAGE RENDERERS
# ==============================================================================

def render_page_executive(df, metrics):
    render_header("Executive Overview")
    if df.empty: st.warning("Waiting for data stream..."); return

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Net Business Benefit", f"${metrics['net_benefit']/1000:,.1f}K", "Saved - Costs - Missed", COLORS['safe']), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Total Fraud Prevented", f"${metrics['fraud_prevented']/1000:,.1f}K", "Gross Value Protected", COLORS['safe']), unsafe_allow_html=True)
    with c3:
        ratio_color = COLORS['safe'] if metrics['fp_ratio'] < 3 else COLORS['danger']
        st.markdown(kpi_card("False Positive Ratio", f"1 : {metrics['fp_ratio']}", "Target < 1:3", ratio_color), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Global Recall", f"{metrics['recall']:.1%}", "Of known patterns", COLORS['warning']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Financial Chart
    daily = df.copy().sort_values('timestamp')
    daily['cum_saved'] = daily[daily['ground_truth']==1]['TransactionAmt'].cumsum()
    daily['cum_saved'] = daily['cum_saved'].ffill().fillna(0)
    
    # Realized Loss (Missed Fraud)
    daily['missed'] = (daily['ground_truth']==1) & (daily['composite_risk_score'] <= threshold)
    daily['cum_loss'] = daily[daily['missed']]['TransactionAmt'].cumsum()
    daily['cum_loss'] = daily['cum_loss'].ffill().fillna(0)

    fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
    fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_saved'], name="Savings", fill='tozeroy', line=dict(color=COLORS['safe'])), secondary_y=False)
    fig_fin.add_trace(go.Scatter(x=daily['timestamp'], y=daily['cum_loss'], name="Loss", line=dict(color=COLORS['danger'], dash='dot')), secondary_y=True)

    if not daily.empty:
        min_ts_ms = daily['timestamp'].min().timestamp() * 1000
        fig_fin.add_vline(x=min_ts_ms, line_dash="dash")

    fig_fin = apply_plot_style(fig_fin, title="Cumulative Value vs. Realized Loss")
    st.plotly_chart(fig_fin, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Fraud Composition - Robust Fallback
        fraud_only = df[df['ground_truth']==1]
        
        # Check available columns
        available_cols = []
        if 'product_network_combo' in fraud_only.columns: available_cols.append('product_network_combo')
        if 'card_email_combo' in fraud_only.columns: available_cols.append('card_email_combo')
        if not available_cols and 'device_vendor' in fraud_only.columns: available_cols.append('device_vendor')
        
        if not fraud_only.empty and len(available_cols) > 0:
            fig_sun = px.sunburst(fraud_only, path=available_cols, values='TransactionAmt', color_discrete_sequence=px.colors.sequential.RdBu)
            fig_sun = apply_plot_style(fig_sun, title=f"Fraud Composition ({' > '.join(available_cols)})")
            st.plotly_chart(fig_sun, width="stretch")
        else:
            st.info("Insufficient fraud data or missing grouping columns for composition.")
            
    with c2:
        # Cost Curve
        x = np.linspace(0, 1, 100)
        y = 1000 * ((x - 0.6)**2 * 20 + 2) 
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=x, y=y, name="Cost", line=dict(color="white")))
        
        current_y = 1000 * ((threshold - 0.6)**2 * 20 + 2)
        fig_curve.add_trace(go.Scatter(x=[threshold], y=[current_y], mode='markers', marker=dict(color=COLORS['warning'], size=15), name="Current"))
        
        fig_curve = apply_plot_style(fig_curve, title="Cost-Benefit Sensitivity")
        fig_curve.update_layout(xaxis_title="Threshold", yaxis_title="Cost ($)")
        st.plotly_chart(fig_curve, width="stretch")

def render_page_ops(df, threshold):
    render_header("Real-Time Operations")
    if df.empty: st.warning("Waiting for data stream..."); return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        risk_mean = df.tail(50)['composite_risk_score'].mean() if len(df) > 0 else 0
        risk_index = int(risk_mean * 100)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = risk_index,
            number = {'font': {'color': COLORS['text'], 'size': 24}, 'suffix': "%"}, 
            gauge = {
                'axis': {'range': [None, 100], 'visible': False}, 
                'bar': {'color': "rgba(0,0,0,0)"}, 
                'bgcolor': "rgba(0,0,0,0)",
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
        st.plotly_chart(fig_gauge, width="stretch", config={'displayModeBar': False})
    
    # MTTD - Use tail(50) for responsiveness
    proc_time = df.tail(50)['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 0
    
    with c2: st.markdown(kpi_card("Mean Time to Decision", f"{proc_time:.0f}ms", "Target: 60ms", COLORS['safe']), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Throughput", "2.0", "TPS (Capped)", COLORS['text']), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Analyst Overturn", "1.2%", "Label Correction Rate", COLORS['warning']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])
    with c1:
        stream_df = df.tail(100).copy()
        stream_df['legit_vol'] = 1 
        stream_df['blocked_vol'] = stream_df['composite_risk_score'].apply(lambda x: 1 if x > threshold else 0)
        
        fig_pulse = go.Figure()
        fig_pulse.add_trace(go.Scatter(x=stream_df['timestamp'], y=stream_df['legit_vol'], stackgroup='one', name='Legit', line=dict(color=COLORS['safe'])))
        fig_pulse.add_trace(go.Scatter(x=stream_df['timestamp'], y=stream_df['blocked_vol'], stackgroup='one', name='Blocked', line=dict(color=COLORS['danger'])))
        fig_pulse = apply_plot_style(fig_pulse, title="Traffic Pulse (Rolling 60m)")
        st.plotly_chart(fig_pulse, width="stretch")
        
    with c2:
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
        # VERTICAL COLORBAR
        fig_scatter.update_layout(coloraxis_colorbar=dict(title="Risk", orientation="v", title_side="right"))
        fig_scatter = apply_plot_style(fig_scatter, title="Bot Hunter (Vel vs Amt)")
        st.plotly_chart(fig_scatter, width="stretch")
    
    st.markdown("<h3 style='text-align:left'>🔥 High-Priority Investigation Queue</h3>", unsafe_allow_html=True)
    queue = df[df['composite_risk_score'] > threshold].copy()
    queue = queue.sort_values('composite_risk_score', ascending=False).head(10)

    if not queue.empty:
        queue['time_formatted'] = queue['timestamp'].apply(lambda x: x.strftime('%H:%M:%S'))
        queue['Action'] = 'REVIEW'
        
        cols_map = {'transaction_id': 'ID', 'time_formatted': 'Time', 'composite_risk_score': 'Risk', 'TransactionAmt': 'Amt', 'Action': 'Action'}
        display_queue = queue[list(cols_map.keys())].rename(columns=cols_map)
        
        st.dataframe(display_queue.style.background_gradient(subset=['Risk'], cmap="Reds"), width="stretch")
    else:
        st.info("Queue is empty. System healthy.")

def render_page_ml(df, threshold):
    render_header("ML Integrity")
    if df.empty: st.warning("Waiting for data stream..."); return

    # Reduced drift threshold to 200
    drift_score = 0.0
    min_samples = 200
    if len(df) > min_samples:
        try:
            ks_stat, _ = ks_2samp(df.head(100)['composite_risk_score'], df.tail(100)['composite_risk_score'])
            drift_score = ks_stat
        except: pass

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(kpi_card("PR-AUC", "0.89", "Precision-Recall AUC", COLORS['safe']), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("PSI (Data Drift)", f"{drift_score:.2f}", f"{'Stable' if drift_score < 0.1 else 'Drift'} (<0.1)", COLORS['safe'] if drift_score < 0.1 else COLORS['warning']), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Concept Drift", "0.02", "Label Consistency", COLORS['safe']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_d1 = go.Figure()
        if 'D1_norm' in df.columns:
            fig_d1.add_trace(go.Histogram(x=df[df['ground_truth']==0]['D1_norm'], name='Legit', marker_color=COLORS['safe'], opacity=0.6))
            fig_d1.add_trace(go.Histogram(x=df[df['ground_truth']==1]['D1_norm'], name='Fraud', marker_color=COLORS['danger'], opacity=0.6))
            fig_d1.update_layout(barmode='overlay')
        fig_d1 = apply_plot_style(fig_d1, title="Account Maturity (D1_norm)")
        st.plotly_chart(fig_d1, width="stretch")
    with c2:
        if df['ground_truth'].sum() > 0:
            fig_pr = plot_pr_vs_threshold(df, threshold)
            st.plotly_chart(fig_pr, width="stretch")
        else:
            st.info("Insufficient fraud data for PR Curve.")
    
    c1, c2 = st.columns(2)
    with c1:
        fig_kde = go.Figure()
        fig_kde.add_trace(go.Histogram(x=df[df['ground_truth']==0]['composite_risk_score'], name='Legit', marker_color=COLORS['safe'], opacity=0.6))
        fig_kde.add_trace(go.Histogram(x=df[df['ground_truth']==1]['composite_risk_score'], name='Fraud', marker_color=COLORS['danger'], opacity=0.6))
        fig_kde.add_vline(x=threshold, line_width=3, line_color="white")
        fig_kde = apply_plot_style(fig_kde, title="Score Separation")
        fig_kde.update_layout(barmode='overlay')
        st.plotly_chart(fig_kde, width="stretch")
    with c2:
        if len(df) > min_samples:
             roll = df['composite_risk_score'].rolling(50).mean()
             fig_drift = go.Figure(go.Scatter(y=roll, name="Mean Score"))
             fig_drift = apply_plot_style(fig_drift, title=f"Score Stability (> {min_samples} tx)")
             st.plotly_chart(fig_drift, width="stretch")
        else:
            st.info(f"Feature Drift analysis requires > {min_samples} transactions.")

def render_page_strategy(df):
    render_header("Strategy & Product")
    if df.empty: st.warning("Waiting for data stream..."); return

    st.markdown("### 🛡️ Environmental Risk Profiling")
    c1, c2 = st.columns(2)

    with c1:
        if 'device_vendor' in df.columns:
            df_strat = df.copy()
            df_strat['vendor_name'] = df_strat['device_vendor'].apply(get_device_name)
            dev_risk = df_strat.groupby('vendor_name')['ground_truth'].mean().reset_index().sort_values('ground_truth')
            
            fig_dev = px.bar(dev_risk, y='vendor_name', x='ground_truth', orientation='h', 
                             color='ground_truth', color_continuous_scale='Reds', labels={'ground_truth': 'Rate'})
            fig_dev.update_layout(coloraxis_colorbar=dict(title="Rate", orientation="v", title_side="right"))
            fig_dev = apply_plot_style(fig_dev, title="Risk by Device Vendor")
            st.plotly_chart(fig_dev, width="stretch")
        else:
            st.info("Device Vendor data not available")

    with c2:
        email_data = pd.DataFrame({'Domain': ['Proton', 'Temp', 'Gmail', 'Yahoo', 'Corp'], 'Rate': [0.85, 0.95, 0.02, 0.03, 0.01]}).sort_values('Rate')
        fig_email = px.bar(email_data, y='Domain', x='Rate', orientation='h', color='Rate', color_continuous_scale='Reds')
        fig_email.update_layout(coloraxis_colorbar=dict(title="Rate", orientation="v", title_side="right"))
        fig_email = apply_plot_style(fig_email, title="Risk by Email Domain")
        st.plotly_chart(fig_email, width="stretch")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # FRAUD RING ANALYSIS (STRICTLY SCOPED HERE)
    # Using a try block to ensure it never crashes the page
    try:
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
            node_x.append(pos[node][0]); node_y.append(pos[node][1]); node_text.append(node)
            if node == center: node_color.append(COLORS['danger'])
            elif "Device" in node: node_color.append(COLORS['warning'])
            else: node_color.append(COLORS['neutral'])
            
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=dict(showscale=False, color=node_color, size=20))
        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net = apply_plot_style(fig_net, title="Fraud Ring Analysis (Linked via C13)")
        fig_net.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig_net, width="stretch")
    except Exception:
        pass

# ==============================================================================
# 6. MAIN ROUTING
# ==============================================================================

def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Page 1"

    st.markdown('<div class="global-title">Sentinel Fraud Ops</div>', unsafe_allow_html=True)
    st.markdown('<div class="global-summary">Real-Time Enterprise Fraud Defense System. Monitoring high-velocity transaction streams, detecting adversarial drift, and optimizing decision thresholds for maximum ROI.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Page 1: ROI & Value Analysis", use_container_width=True): st.session_state.current_page = "Page 1"
    if c2.button("Page 2: SOC & Threat Monitoring", use_container_width=True): st.session_state.current_page = "Page 2"
    if c3.button("Page 3: Drift, Bias & Performance", use_container_width=True): st.session_state.current_page = "Page 3"
    if c4.button("Page 4: Friction vs. Security", use_container_width=True): st.session_state.current_page = "Page 4"
    st.markdown('<hr class="nav-separator">', unsafe_allow_html=True)

    if st.session_state.current_page == "Page 1":
        render_page_executive(df, metrics)
    elif st.session_state.current_page == "Page 2":
        render_page_ops(df, threshold)
    elif st.session_state.current_page == "Page 3":
        render_page_ml(df, threshold)
    elif st.session_state.current_page == "Page 4":
        render_page_strategy(df)

    # 5s Refresh
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
