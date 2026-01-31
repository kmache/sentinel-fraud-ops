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