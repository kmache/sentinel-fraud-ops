import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, apply_plot_style, render_header

def render_page(df: pd.DataFrame):
    render_header("Forensics & Search", "Deep Dive into Transaction Details")

    if df.empty:
        st.info("Waiting for data...")
        return
    
    # Normalize Column Names
    score_col = 'score' if 'score' in df.columns else 'composite_risk_score'
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'ground_truth'
    amt_col = 'amount' if 'amount' in df.columns else 'TransactionAmt'
    id_col = 'transaction_id' if 'transaction_id' in df.columns else 'TransactionID'

    # --- 1. SEARCH BAR ---
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            search_term = st.text_input("Search Transaction ID, Product, or Amount", placeholder="e.g., mock_1234 or 150.00")
        with c2:
            st.markdown("<br>", unsafe_allow_html=True) 
            filter_fraud_only = st.checkbox("Show Fraud Only", value=False)

    # --- 2. FILTER LOGIC ---
    filtered_df = df.copy()
    
    if filter_fraud_only and fraud_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[fraud_col] == 1]

    if search_term:
        # Simple string matching across all columns
        mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
        filtered_df = filtered_df[mask]

    # --- 3. RESULTS AREA ---
    st.markdown(f"**Found {len(filtered_df)} transactions**")

    if not filtered_df.empty:
        # Split screen: Table on Left, Details on Right
        c1, c2 = st.columns([2, 1])
        
        # Columns to display in table
        display_cols = [id_col, 'timestamp', amt_col, score_col, fraud_col]
        if 'ProductCD' in filtered_df.columns: display_cols.append('ProductCD')
        
        with c1:
            st.dataframe(
                filtered_df[display_cols],
                use_container_width=True,
                height=500,
                hide_index=True
            )

        with c2:
            st.markdown("### 🔍 Risk Analysis")
            if len(filtered_df) == 1:
                # Detail View for Single Record
                record = filtered_df.iloc[0]
                s_val = record.get(score_col, 0)
                score_color = COLORS['danger'] if s_val > 0.5 else COLORS['safe']
                
                prod = record.get('ProductCD', 'N/A')
                action = record.get('action', 'Unknown')
                
                st.markdown(f"""
                <div style="background-color: {COLORS['card_bg']}; padding: 20px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                    <h1 style="color: {score_color}; margin:0;">{s_val:.4f}</h1>
                    <div style="color: {COLORS['neutral']}; margin-bottom: 20px; font-size: 12px;">RISK SCORE</div>
                    
                    <p style="margin: 5px 0;"><b>ID:</b> {record.get(id_col)}</p>
                    <p style="margin: 5px 0;"><b>Amount:</b> ${record.get(amt_col)}</p>
                    <p style="margin: 5px 0;"><b>Product:</b> {prod}</p>
                    <p style="margin: 5px 0;"><b>Timestamp:</b> {record.get('timestamp')}</p>
                    <hr style="border-color: {COLORS['border']}; margin: 15px 0;">
                    <p style="margin: 5px 0;"><b>Action Taken:</b> <span style="color:{COLORS['text']}">{action}</span></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Aggregate View for Multiple Records
                avg_score = filtered_df[score_col].mean()
                total_amt = filtered_df[amt_col].sum()
                
                if fraud_col in filtered_df.columns:
                    fig = px.pie(filtered_df, names=fraud_col, title="Fraud vs Legit in Search",
                                 color=fraud_col, color_discrete_map={0: COLORS['safe'], 1: COLORS['danger']})
                    fig = apply_plot_style(fig, height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Total Amount in View", f"${total_amt:,.2f}")
                st.metric("Average Risk Score", f"{avg_score:.2f}")

    else:
        st.warning("No transactions match your search criteria.")