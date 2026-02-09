import os
import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

# Ensure imports work from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, kpi_card, apply_plot_style, render_header
from config import StandardColumns

# ==============================================================================
# 1. COMPONENT FUNCTIONS (The "Rows")
# ==============================================================================

def _render_kpi_row(metrics: dict):
    """Simplified KPI row with 4 business-focused metrics"""
    c1, c2, c3, c4 = st.columns(4)
    
    with c1: 
        val = metrics.get('net_savings', 0)
        color = COLORS['safe'] if val >= 0 else COLORS['danger']
        st.markdown(kpi_card(
            "Net Savings", 
            f"${val/1000:,.1f}K",
            "Fraud Stopped - Ops Cost",
            color
        ), unsafe_allow_html=True)
    
    with c2: 
        val = metrics.get('fraud_stopped_val', 0)
        st.markdown(kpi_card(
            "Fraud Blocked", 
            f"${val/1000:,.1f}K",
            "Gross Value Protected",
            COLORS['safe']
        ), unsafe_allow_html=True)
    
    with c3:
        val = metrics.get('fraud_missed_val', 0)
        st.markdown(kpi_card(
            "Fraud Missed", 
            f"${val/1000:,.1f}K",
            "Value of Undetected Fraud",
            COLORS['danger']
        ), unsafe_allow_html=True)
    
    with c4: 
        val = metrics.get('fraud_rate', 0)
        st.markdown(kpi_card(
            "Fraud Rate", 
            f"{val:.2f}%",
            "Percentage of Fraudulent Tx",
            COLORS['warning']
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)



def _render_financial_chart(timeseries_df: pd.DataFrame):
    """Renders the Cumulative Savings vs Loss chart."""
    if timeseries_df.empty or 'timestamp' not in timeseries_df.columns:
        st.info("ℹ️ Waiting for financial history data...")
        return

    df = timeseries_df.sort_values('timestamp')
    
    # Create figure with secondary y-axis
    fig_fin = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Cumulative Savings
    fig_fin.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['cumulative_savings'], 
        name="Cumulative Savings", 
        fill='tozeroy', 
        line=dict(color=COLORS['safe'])
    ), secondary_y=False)
    
    # 2. Cumulative Loss
    fig_fin.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['cumulative_loss'], 
        name="Realized Fraud Loss", 
        line=dict(color=COLORS['danger'], dash='dot')
    ), secondary_y=True)

    # Apply standard styles first
    fig_fin = apply_plot_style(fig_fin, title="Cumulative Value vs. Realized Loss")
    
    # --- CUSTOM LAYOUT OPTIMIZATION ---
    fig_fin.update_layout(
        # Move Legend INSIDE the chart (Top Left)
        legend=dict(
            yanchor="top",
            y=0.99,            # 99% from the bottom (near top)
            xanchor="left",
            x=0.01,            # 1% from the left
            bgcolor="rgba(0,0,0,0)", # Transparent background
            bordercolor="rgba(0,0,0,0)"
        ),
        # Reduce whitespace around the chart
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified" # Shows both values when hovering over a timestamp
    )

    fig_fin.update_yaxes(title_text="Savings ($)", secondary_y=False)
    fig_fin.update_yaxes(title_text="Loss ($)", secondary_y=True)
    
    st.plotly_chart(fig_fin, key="fin_chart", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)


def _render_analysis_row(recent_df: pd.DataFrame, threshold: float, curve_df: pd.DataFrame):
    """Renders the bottom row: Fraud Composition and Sensitivity Curve."""
    c1, c2 = st.columns(2)
    
    # --- CHART 1: Fraud Composition ---
    with c1:
        fraud_col = 'is_fraud' 
        amt_col = 'TransactionAmt'
        path_cols = ['ProductCD', 'DeviceType']
        
        if not recent_df.empty and fraud_col in recent_df.columns:
            fraud_only = recent_df[recent_df[fraud_col] == 1].copy()
            
            for col in path_cols:
                if col in fraud_only.columns:
                    fraud_only[col] = fraud_only[col].fillna("Unknown").replace("", "Unknown")
            
            available_path = [c for c in path_cols if c in fraud_only.columns]
            
            if not fraud_only.empty and available_path:
                fig_sun = px.sunburst(
                    fraud_only, 
                    path=available_path, 
                    values=amt_col,
                    color=amt_col,
                    color_continuous_scale='Reds'
                )
                
                fig_sun.update_traces(
                    textinfo="label+percent entry",
                    insidetextorientation='radial'
                )

                fig_sun.update_layout(
                    # --- CENTRALIZE TITLE ---
                    title={
                        'text': "Fraud Distribution: Product > Device",
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    height=420, 
                    margin=dict(t=80, b=50, l=10, r=10), # Increased top margin for title
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"),
                    coloraxis_showscale=False 
                )
                
                total_fraud = fraud_only[amt_col].sum()
                fig_sun.add_annotation(
                    text=f"<b>Total Fraud Captured</b><br>${total_fraud:,.0f} ({len(fraud_only)} Alerts)",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=12, color=COLORS.get('highlight', '#00e676')),
                    align="center"
                )
                
                st.plotly_chart(fig_sun, use_container_width=True)
                
                # --- CENTRALIZE CAPTION ---
                if 'ProductCD' in fraud_only.columns:
                    top_p = fraud_only.groupby('ProductCD')[amt_col].sum().idxmax()
                    st.markdown(
                        f"""
                        <p style='text-align: center; font-size: 0.85rem; color: #888;'>
                            📊 <b>Insight:</b> Most fraud volume is concentrated in <b>{top_p}</b> transactions.
                        </p>
                        """, 
                        unsafe_allow_html=True
                    )
            else:
                st.info("ℹ️ No recent fraud cases to decompose.")
        else:
            st.warning("⚠️ Classification data (is_fraud) not found.")

    # --- CHART 2: Sensitivity Curve ---
    with c2:
        if curve_df.empty:
            st.info("Calculating threshold optimization curve...")
        else:
            fig_curve = go.Figure()
            # 1. The Cost Curve
            fig_curve.add_trace(go.Scatter(
                x=curve_df['threshold'], 
                y=curve_df['total_loss'], 
                name="Total Financial Loss",
                line=dict(color=COLORS['highlight'], width=3)
            ))

            # 2. Add the Diamond marker at the CURRENT threshold
            # We find the y-value (loss) closest to our current threshold
            # to place the marker exactly on the line
            idx = (curve_df['threshold'] - threshold).abs().idxmin()
            current_loss = curve_df.loc[idx, 'total_loss']
            fig_curve.add_trace(go.Scatter(
                x=[threshold], y=[current_loss], 
                mode='markers+text', 
                text=["ACTIVE"], textposition="top center",
                marker=dict(color=COLORS['warning'], size=15, symbol="diamond"),
                name="Current Setting"
            ))
            fig_curve = apply_plot_style(fig_curve, title="Optimization: Risk vs. Cost")
            fig_curve.update_layout(
                xaxis_title="Risk Threshold",
                yaxis_title="Potential Financial Impact ($)",
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_curve, use_container_width=True, key="threshold_curve")


# ==============================================================================
# 2. MAIN RENDER FUNCTION
# ==============================================================================

def render_page(recent_df: pd.DataFrame, metrics: dict, threshold: float, timeseries_df: pd.DataFrame, curve_df:pd.DataFrame):
    """
    Main controller for the Executive View.
    """
    render_header("Executive Overview", "Financial Impact & ROI Analysis")
    
    if metrics:
        _render_kpi_row(metrics)
    else:
        st.warning("⚠️ No metrics available yet.")

    # Pass the specialized timeseries dataframe here
    _render_financial_chart(timeseries_df)

    # Pass the recent transactions for composition analysis
    _render_analysis_row(recent_df, threshold, curve_df)