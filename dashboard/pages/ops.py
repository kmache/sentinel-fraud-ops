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