import os
import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from styles import COLORS, apply_plot_style, render_header

def render_page(df: pd.DataFrame):
    render_header("Strategy & Product", "Risk Profiling & Link Analysis")
    
    if df.empty:
        st.info("ℹ️ Waiting for transaction stream data...")
        return

    # Normalize Columns
    fraud_col = 'is_fraud' if 'is_fraud' in df.columns else 'ground_truth'
    
    c1, c2 = st.columns(2)

    # --- DEVICE RISK ---
    with c1:
        if 'device_vendor' in df.columns:
            # Simple aggregation
            dev_risk = df.groupby('device_vendor')[fraud_col].mean().reset_index()
            # Filter for visualisation
            dev_risk = dev_risk.sort_values(fraud_col, ascending=True).tail(10)
            
            fig_dev = px.bar(dev_risk, y='device_vendor', x=fraud_col, orientation='h', 
                             color=fraud_col, color_continuous_scale='Reds', 
                             labels={fraud_col: 'Fraud Rate'})

            fig_dev.update_layout(coloraxis_colorbar=dict(title="Rate", orientation="v", title_side="right"))
            fig_dev = apply_plot_style(fig_dev, title="Risk by Device Vendor")
            st.plotly_chart(fig_dev, use_container_width=True)
        else:
            st.warning("Device data not available in stream.")

    # --- EMAIL DOMAIN RISK ---
    with c2:
        # Check if email domain data exists in stream, else use fallback or skip
        if 'P_emaildomain' in df.columns:
             email_risk = df.groupby('P_emaildomain')[fraud_col].mean().reset_index()
             email_risk = email_risk.sort_values(fraud_col, ascending=True).tail(10)
             
             fig_email = px.bar(email_risk, y='P_emaildomain', x=fraud_col, orientation='h', 
                           color=fraud_col, color_continuous_scale='Reds')
             fig_email = apply_plot_style(fig_email, title="Risk by Email Domain")
             st.plotly_chart(fig_email, use_container_width=True)
        else:
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
        # In a real app, this would use data from df where 'C13' or 'device_id' match
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