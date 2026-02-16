import os
import sys
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, apply_plot_style, render_header

# ==============================================================================
# 1. SEARCH INTERFACE
# ==============================================================================
def _render_search_interface(default_val=""):
    """
    Handles case lookup.
    """
    st.markdown(f"""
    <div style="background-color: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="margin: 0 0 10px 0;">🕵️‍♂️ Case Investigation</h4>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        tx_id_input = st.text_input("Transaction ID", value=default_val, placeholder="Enter UUID...")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Analyze Case", type="primary", width='stretch')
    
    return tx_id_input, search_clicked

# ==============================================================================
# 2. CASE HEADER
# ==============================================================================
def _render_case_header(data: dict):
    """
    Top-level summary of the transaction status.
    """
    is_fraud = data.get('is_fraud', 0) == 1
    score = data.get('score', 0)
    
    color = COLORS['danger'] if is_fraud else COLORS['safe']
    status = "🛑 HIGH RISK: BLOCKED" if is_fraud else "✅ LOW RISK: APPROVED"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, {color}22 0%, rgba(0,0,0,0) 100%);
        border-left: 5px solid {color};
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; color: #fff;">{status}</h2>
                <p style="margin: 5px 0 0 0; color: #aaa; font-family: monospace;">ID: {data.get('transaction_id')}</p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 2.5rem; font-weight: bold; color: {color};">{score:.3f}</span>
                <br><span style="color: #aaa;">Risk Score</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. SHAP EVIDENCE CHART
# ==============================================================================
def _render_shap_waterfall(explanations: list):
    """
    Visualizes specific evidence for this transaction.
    """
    st.subheader("🧠 Transaction Risk Evidence")
    
    if not explanations:
        st.info("ℹ️ No explanation data available for this transaction.")
        return

    data = sorted(explanations, key=lambda x: x['impact'])
    features = [d['feature'] for d in data]
    impacts = [d['impact'] for d in data]

    values = []
    for d in data:
        val = d['value']
        values.append(f"{val:.2f}" if isinstance(val, float) else str(val))
    labels = [f"<b>{f}</b> ({v})" for f, v in zip(features, values)]
    
    colors = ['#EF553B' if x > 0 else '#00CC96' for x in impacts]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=impacts,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.2f}" for x in impacts],
        textposition='outside'
    ))

    fig.update_layout(
        title="<sup>What pushed the model toward Fraud vs Legit</sup>",
        xaxis_title="Impact on Score (Log-Odds)",
        yaxis=dict(ticksuffix="  "),
        height=max(400, len(features) * 40),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )
    
    fig.add_vline(x=0, line_width=2, line_color="white", line_dash="dash")
    fig = apply_plot_style(fig, title="")
    st.plotly_chart(fig, width='stretch')

# ==============================================================================
# 4. ENTITY PROFILE
# ==============================================================================
def _render_entity_profile(data: dict):
    """
    Contextual information about the user/device.
    """
    st.subheader("👤 Identity Profile")
    
    profile = {
        "Card Info": f"{data.get('card4', 'Unknown')} ({data.get('card6', '')})",
        "Email Domain": data.get('P_emaildomain', 'N/A'),
        "Product Code": data.get('ProductCD', 'N/A'),
        "Amount": f"${float(data.get('TransactionAmt', 0)):,.2f}",
        "Device": data.get('DeviceType', 'Unknown'),
        "IP Address": data.get('addr1', 'Unknown (Hidden)')
    }
    
    for k, v in profile.items():
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #333; padding: 8px 0;">
            <span style="color: #888;">{k}</span>
            <span style="color: #fff; font-weight: 500;">{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View Raw JSON Payload"):
        st.json(data)

# ==============================================================================
# MAIN VIEW CONTROLLER
# ==============================================================================
def render_page():
    """
    Main entry point for Forensics Page.
    """
    if 'api_client' not in st.session_state:
        st.error("API Client not initialized.")
        return
        
    client = st.session_state.api_client
    
    render_header("Forensics Deep-Dive", "Transaction Investigation & Explanation")
    
    target_id = ""

    if st.session_state.get('selected_case'):
        target_id = st.session_state.get('selected_case')
        st.session_state['selected_case'] = ""
        st.query_params["case_id"] = target_id

    elif "case_id" in st.query_params:
        target_id = st.query_params["case_id"]

    tx_id_input, search_clicked = _render_search_interface(default_val=target_id)
    
    if search_clicked and tx_id_input:
        target_id = tx_id_input
        st.query_params["case_id"] = target_id

    should_fetch = False
    
    if target_id:
        last_loaded = st.session_state.get('last_loaded_case_id')
        if last_loaded != target_id:
            should_fetch = True
        elif 'current_case_data' not in st.session_state:
            should_fetch = True

    if should_fetch:
        try:
            with st.spinner(f"Running forensics on {target_id}..."):
                case_data = client.get_transaction_detail(target_id)
                st.session_state['current_case_data'] = case_data
                st.session_state['last_loaded_case_id'] = target_id
        except Exception as e:
            st.error(f"Transaction {target_id} not found or expired.")
            return

    if 'current_case_data' in st.session_state:
        data = st.session_state['current_case_data']

        _render_case_header(data)
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            _render_shap_waterfall(data.get('explanations', []))
        with c2:
            _render_entity_profile(data)
