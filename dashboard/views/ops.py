import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

def render_page(recent_df: pd.DataFrame, alerts_df: pd.DataFrame, metrics: dict):
    """
    Ops Center - Operational Efficiency & Alert Management
    
    Args:
        recent_df: Recent transactions from /recent
        alerts_df: High-risk alerts from /alerts  
        metrics: System metrics from /stats and /health
    """
    
    # ==========================================================================
    # PAGE HEADER
    # ==========================================================================
    st.title("🚨 Ops Center")
    st.markdown("### Real-time Alert Management & Operational Efficiency")
    
    # Refresh controls
    col_refresh, col_info, col_export = st.columns([1, 2, 1])
    with col_refresh:
        if st.button("🔄 Refresh Now", type="secondary", use_container_width=True):
            st.rerun()
    with col_info:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} • Auto-refresh: 10s")
    with col_export:
        if st.button("📥 Export Queue", type="secondary", use_container_width=True):
            st.info("Exporting alert queue...")
    
    st.divider()
    
    # ==========================================================================
    # 1. TOP ROW: OPERATIONAL EFFICIENCY KPIs
    # ==========================================================================
    st.subheader("📊 Operational Efficiency Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Active Alerts
        active_alerts = len(alerts_df) if not alerts_df.empty else 0
        alert_color = COLORS['danger'] if active_alerts > 20 else COLORS['warning'] if active_alerts > 10 else COLORS['safe']
        
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: {alert_color}15; border-radius: 10px; border-left: 4px solid {alert_color};">
            <h2 style="margin: 0; color: {COLORS['text']};">{active_alerts}</h2>
            <p style="margin: 0; font-size: 12px; color: {COLORS['text_secondary']};">Active Alerts</p>
            <small style="color: {alert_color}; font-size: 10px;">
                {'🚨 Critical' if active_alerts > 20 else '⚠️ Elevated' if active_alerts > 10 else '✅ Normal'}
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Mean Time to Resolve (MTTR) - Estimated
        if not alerts_df.empty and 'timestamp' in alerts_df.columns:
            # Simple MTTR estimation: oldest alert age
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
            current_time = pd.Timestamp.now()
            oldest_alert_age = (current_time - alerts_df['timestamp'].min()).total_seconds() / 60  # minutes
            
            # Mock MTTR (in reality, you'd track resolution times)
            mttr = max(5, oldest_alert_age / 2)  # Simplified calculation
            mttr_color = COLORS['danger'] if mttr > 30 else COLORS['warning'] if mttr > 15 else COLORS['safe']
            
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {mttr_color}15; border-radius: 10px; border-left: 4px solid {mttr_color};">
                <h2 style="margin: 0; color: {COLORS['text']};">{mttr:.0f} min</h2>
                <p style="margin: 0; font-size: 12px; color: {COLORS['text_secondary']};">Avg Resolution Time</p>
                <small style="color: {mttr_color}; font-size: 10px;">
                    {'🚨 Slow' if mttr > 30 else '⚠️ Moderate' if mttr > 15 else '✅ Fast'}
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {COLORS['neutral']}15; border-radius: 10px;">
                <h2 style="margin: 0; color: {COLORS['text']};">--</h2>
                <p style="margin: 0; font-size: 12px; color: {COLORS['text_secondary']};">MTTR</p>
                <small style="color: {COLORS['neutral']}; font-size: 10px;">No alert data</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # High Risk Velocity (spike detection)
        if not recent_df.empty and 'score' in recent_df.columns:
            # Count high-risk transactions in last 15 minutes
            recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
            last_15min = recent_df[recent_df['timestamp'] > datetime.now() - timedelta(minutes=15)]
            
            high_risk_now = len(last_15min[last_15min['score'] > 0.7])
            high_risk_prev = len(recent_df[
                (recent_df['timestamp'] > datetime.now() - timedelta(minutes=30)) &
                (recent_df['timestamp'] <= datetime.now() - timedelta(minutes=15)) &
                (recent_df['score'] > 0.7)
            ])
            
            velocity = high_risk_now - high_risk_prev
            velocity_color = COLORS['danger'] if velocity > 5 else COLORS['warning'] if velocity > 2 else COLORS['safe']
            velocity_icon = "📈" if velocity > 0 else "📉" if velocity < 0 else "➡️"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {velocity_color}15; border-radius: 10px; border-left: 4px solid {velocity_color};">
                <h2 style="margin: 0; color: {COLORS['text']};">{velocity_icon} {velocity:+d}</h2>
                <p style="margin: 0; font-size: 12px; color: {COLORS['text_secondary']};">Risk Velocity</p>
                <small style="color: {velocity_color}; font-size: 10px;">
                    {high_risk_now} high-risk in last 15min
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {COLORS['neutral']}15; border-radius: 10px;">
                <h2 style="margin: 0; color: {COLORS['text']};">--</h2>
                <p style="margin: 0; font-size: 12px; color: {COLORS['text_secondary']};">Risk Velocity</p>
                <small style="color: {COLORS['neutral']}; font-size: 10px;">No score data</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Auto-Block vs Manual Review Ratio
        if not recent_df.empty and 'action' in recent_df.columns and 'score' in recent_df.columns:
            threshold = metrics.get('threshold', 0.5)
            
            auto_block = len(recent_df[(recent_df['score'] > threshold) & (recent_df['action'] == 'BLOCK')])
            manual_review = len(recent_df[(recent_df['score'] > threshold) & (recent_df['action'] != 'BLOCK')])
            total = auto_block + manual_review
            
            auto_ratio = (auto_block / total * 100) if total > 0 else 0
            ratio_color = COLORS['safe'] if auto_ratio > 70 else COLORS['warning'] if auto_ratio > 30 else COLORS['danger']
            
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {ratio_color}15; border-radius: 10px; border-left: 4px solid {ratio_color};">
                <h2 style="margin: 0; color: {COLORS['text']};">{auto_ratio:.0f}%</h2>
                <p style="margin: 0; font-size: 12px; color: {COLORS['text_secondary']};">Auto-Block Rate</p>
                <small style="color: {ratio_color}; font-size: 10px;">
                    {auto_block} auto • {manual_review} manual
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {COLORS['neutral']}15; border-radius: 10px;">
                <h2 style="margin: 0; color: {COLORS['text']};">--</h2>
                <p style="margin: 0; font-size: 12px; color: {COLORS['text_secondary']};">Auto-Block Ratio</p>
                <small style="color: {COLORS['neutral']}; font-size: 10px;">No action data</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # ==========================================================================
    # 2. MIDDLE ROW: ALERT QUEUE (MAIN TABLE)
    # ==========================================================================
    st.subheader("📋 Alert Queue - Requiring Immediate Attention")
    
    if not alerts_df.empty:
        # Define threshold for high risk (can be configurable)
        high_risk_threshold = st.slider(
            "Set Alert Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.05,
            help="Transactions with scores above this threshold appear in the alert queue"
        )
        
        # Filter alerts by threshold
        high_risk_alerts = alerts_df[alerts_df['score'] >= high_risk_threshold].copy()
        
        if not high_risk_alerts.empty:
            # Sort by score descending
            high_risk_alerts = high_risk_alerts.sort_values('score', ascending=False)
            
            # Create display dataframe with key columns
            display_cols = ['timestamp', 'transaction_id', 'score', 'TransactionAmt', 'ProductCD']
            
            # Add risk level indicator
            high_risk_alerts['risk_level'] = pd.cut(
                high_risk_alerts['score'],
                bins=[high_risk_threshold, 0.8, 0.9, 1.0],
                labels=['Medium', 'High', 'Critical']
            )
            
            # Add quick look indicator (color-coded dots)
            def get_risk_color(level):
                if level == 'Critical':
                    return '🔴'
                elif level == 'High':
                    return '🟠'
                else:
                    return '🟡'
            
            high_risk_alerts['risk_indicator'] = high_risk_alerts['risk_level'].apply(get_risk_color)
            
            # Prepare display columns
            display_df = high_risk_alerts[['risk_indicator'] + display_cols].copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
            
            # Format columns
            display_df = display_df.rename(columns={
                'risk_indicator': ' ',
                'transaction_id': 'ID',
                'TransactionAmt': 'Amount',
                'ProductCD': 'Product',
                'score': 'Risk Score'
            })
            
            # Style the dataframe
            def color_row(row):
                score = row['Risk Score']
                if score >= 0.9:
                    return ['background-color: #FEE2E2'] * len(row)
                elif score >= 0.8:
                    return ['background-color: #FEF3C7'] * len(row)
                else:
                    return ['background-color: #F0F9FF'] * len(row)
            
            # Display with pagination
            items_per_page = 15
            total_items = len(display_df)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
            
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="alert_page")
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)
            
            # Display the styled dataframe
            st.dataframe(
                display_df.iloc[start_idx:end_idx].style.apply(color_row, axis=1).format({
                    'Amount': '${:,.2f}',
                    'Risk Score': '{:.3f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Quick actions for selected alerts
            st.markdown("##### ⚡ Quick Actions")
            
            col_action1, col_action2, col_action3 = st.columns(3)
            
            with col_action1:
                if st.button("📋 Assign to Me", use_container_width=True):
                    st.success(f"Assigned {min(5, len(high_risk_alerts))} alerts to your queue")
            
            with col_action2:
                if st.button("✅ Mark as Reviewed", use_container_width=True):
                    st.info(f"Marked {len(high_risk_alerts)} alerts as reviewed")
            
            with col_action3:
                if st.button("🚨 Escalate All", type="secondary", use_container_width=True):
                    st.warning("Escalated all alerts to security team")
            
            # Individual alert actions
            st.markdown("##### 🔍 Investigate Individual Alerts")
            
            # Display first 5 alerts with expandable details
            for idx, alert in high_risk_alerts.head(5).iterrows():
                with st.expander(f"{get_risk_color(alert['risk_level'])} Alert #{alert['transaction_id'][:8]} - ${alert['TransactionAmt']:,.2f} at {pd.Timestamp(alert['timestamp']).strftime('%H:%M')}", expanded=False):
                    col_info, col_action = st.columns([2, 1])
                    
                    with col_info:
                        st.markdown(f"""
                        **Details:**
                        - **Risk Score:** {alert['score']:.3f}
                        - **Product:** {alert.get('ProductCD', 'N/A')}
                        - **Country:** {alert.get('country', 'N/A')}
                        - **Email Domain:** {alert.get('P_emaildomain', 'N/A')}
                        - **Device:** {alert.get('DeviceType', 'N/A')}
                        """)
                    
                    with col_action:
                        if st.button("Investigate", key=f"invest_{alert['transaction_id']}", use_container_width=True):
                            st.session_state['selected_case'] = alert['transaction_id']
                            st.switch_page("pages/forensics.py")
                        if st.button("Approve", key=f"approve_{alert['transaction_id']}", type="secondary", use_container_width=True):
                            st.success(f"Approved alert #{alert['transaction_id'][:8]}")
                        if st.button("Block", key=f"block_{alert['transaction_id']}", type="secondary", use_container_width=True):
                            st.error(f"Blocked transaction #{alert['transaction_id'][:8]}")
            
            if len(high_risk_alerts) > 5:
                st.caption(f"... and {len(high_risk_alerts) - 5} more alerts")
        
        else:
            st.success("🎉 No alerts above the current threshold!")
    
    else:
        st.success("""
        🎉 **No Active Alerts!**
        
        *Great news! The system is currently detecting no high-risk transactions 
        requiring immediate attention. Monitor the risk distribution below for early signals.*
        """)
    
    st.divider()
    
    # ==========================================================================
    # 3. BOTTOM ROW: RISK DISTRIBUTION & ANALYST LOAD
    # ==========================================================================
    st.subheader("📈 Risk Analysis & Operational Trends")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Score Distribution Histogram
        st.markdown("##### 📊 Risk Score Distribution")
        
        if not recent_df.empty and 'score' in recent_df.columns:
            # Create buckets: 0-0.1, 0.1-0.5, 0.5-0.9, 0.9-1.0
            bins = [0, 0.1, 0.5, 0.9, 1.0]
            labels = ['Low (0-0.1)', 'Medium (0.1-0.5)', 'High (0.5-0.9)', 'Critical (0.9-1.0)']
            
            recent_df['risk_bucket'] = pd.cut(recent_df['score'], bins=bins, labels=labels, include_lowest=True)
            bucket_counts = recent_df['risk_bucket'].value_counts().reindex(labels).fillna(0)
            
            # Create bar chart
            fig_dist = go.Figure()
            
            colors = [COLORS['safe'], COLORS['warning'], COLORS['danger'], '#7F1D1D']  # Dark red for critical
            
            for label, color in zip(labels, colors):
                count = bucket_counts.get(label, 0)
                fig_dist.add_trace(go.Bar(
                    x=[label],
                    y=[count],
                    name=label,
                    marker_color=color,
                    text=[f"{count}"],
                    textposition='auto'
                ))
            
            fig_dist.update_layout(
                title="Transactions by Risk Category",
                showlegend=False,
                xaxis_title="Risk Category",
                yaxis_title="Number of Transactions",
                height=350,
                bargap=0.3
            )
            
            # Add threshold line annotation
            threshold = metrics.get('threshold', 0.5)
            fig_dist.add_vline(
                x=2.5,  # Between High and Medium buckets (0.5 threshold)
                line_dash="dash",
                line_color=COLORS['text'],
                annotation_text=f"Threshold: {threshold}",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Summary statistics
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Avg Score", f"{recent_df['score'].mean():.3f}")
            with col_stats2:
                above_threshold = len(recent_df[recent_df['score'] > threshold])
                st.metric("Above Threshold", above_threshold)
            with col_stats3:
                critical_count = len(recent_df[recent_df['score'] > 0.9])
                st.metric("Critical", critical_count)
        
        else:
            st.info("No score data available for distribution analysis")
    
    with col_right:
        # Alerts Over Time (Hourly)
        st.markdown("##### ⏰ Alert Volume Timeline")
        
        if not alerts_df.empty and 'timestamp' in alerts_df.columns:
            # Convert timestamp and extract hour
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
            alerts_df['hour'] = alerts_df['timestamp'].dt.floor('h')
            
            # Group by hour for last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_alerts = alerts_df[alerts_df['timestamp'] > cutoff_time]
            
            if not recent_alerts.empty:
                # Create hourly bins
                hours = pd.date_range(
                    start=cutoff_time,
                    end=datetime.now(),
                    freq='h'
                )
                
                # Count alerts per hour
                alert_counts = recent_alerts.groupby('hour').size().reindex(hours, fill_value=0)
                
                # Create bar chart
                fig_timeline = go.Figure()
                
                fig_timeline.add_trace(go.Bar(
                    x=alert_counts.index,
                    y=alert_counts.values,
                    name='Alerts',
                    marker_color=COLORS['danger'],
                    opacity=0.7
                ))
                
                # Add moving average line
                window_size = 3
                if len(alert_counts) >= window_size:
                    moving_avg = alert_counts.rolling(window=window_size, center=True).mean()
                    fig_timeline.add_trace(go.Scatter(
                        x=moving_avg.index,
                        y=moving_avg.values,
                        mode='lines',
                        name=f'{window_size}-hour MA',
                        line=dict(color=COLORS['highlight'], width=3)
                    ))
                
                fig_timeline.update_layout(
                    title="Alert Volume by Hour (Last 24h)",
                    xaxis_title="Time",
                    yaxis_title="Number of Alerts",
                    height=350,
                    showlegend=True,
                    hovermode="x unified"
                )
                
                # Format x-axis to show hours
                fig_timeline.update_xaxes(
                    tickformat="%H:00",
                    tickangle=45
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Peak hour analysis
                if len(alert_counts) > 0:
                    peak_hour = alert_counts.idxmax()
                    peak_count = alert_counts.max()
                    
                    col_peak1, col_peak2 = st.columns(2)
                    with col_peak1:
                        st.metric("Peak Hour", peak_hour.strftime('%H:00'))
                    with col_peak2:
                        st.metric("Peak Alerts", peak_count)
            
            else:
                st.info("No alerts in the last 24 hours")
        
        else:
            st.info("No timestamp data available for timeline analysis")
    
    # ==========================================================================
    # FOOTER: SYSTEM HEALTH & REFRESH
    # ==========================================================================
    st.divider()
    
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        # System health status
        redis_status = metrics.get('redis_connected', False)
        memory_usage = metrics.get('memory_usage_mb', 0)
        
        health_icon = "✅" if redis_status and memory_usage < 100 else "⚠️"
        health_text = "Healthy" if redis_status and memory_usage < 100 else "Degraded"
        
        st.caption(f"{health_icon} System: {health_text} ({memory_usage:.0f}MB RAM)")
    
    with col_footer2:
        # Processing stats
        total_processed = metrics.get('total_processed', 0)
        queue_depth = metrics.get('queue_depth', 0)
        
        st.caption(f"📊 Processed: {total_processed:,} • Queue: {queue_depth}")
    
    with col_footer3:
        # Manual refresh reminder
        if st.button("🔄 Manual Refresh", key="footer_refresh", use_container_width=True):
            st.rerun()



# import os
# import sys
# import streamlit as st
# import plotly.graph_objects as go
# import plotly.express as px

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from styles import COLORS, kpi_card, apply_plot_style, render_header

# def render_page(df, threshold):
#     render_header("Real-Time Operations", "SOC Monitoring & Case Management")
    
#     if df.empty:
#         st.info("ℹ️ Waiting for transaction stream data...")
#         return

#     # Normalize Columns
#     score_col = 'score' if 'score' in df.columns else 'composite_risk_score'
#     amt_col = 'amount' if 'amount' in df.columns else 'TransactionAmt'
#     id_col = 'transaction_id' if 'transaction_id' in df.columns else 'TransactionID'

#     # --- OPERATIONAL METRICS ---
#     c1, c2, c3, c4 = st.columns(4)
#     with c1:
#         # Calculate Risk Index
#         risk_mean = df.tail(50)[score_col].mean() if score_col in df.columns else 0
#         risk_index = int(risk_mean * 100)
        
#         fig_gauge = go.Figure(go.Indicator(
#             mode = "gauge+number", value = risk_index,
#             number = {'font': {'color': COLORS['text'], 'size': 24}, 'suffix': "%"}, 
#             gauge = {
#                 'axis': {'range': [None, 100], 'visible': False}, 
#                 'bar': {'color': "rgba(0,0,0,0)"}, 
#                 'steps': [
#                     {'range': [0, 40], 'color': COLORS['safe']},
#                     {'range': [40, 75], 'color': COLORS['warning']},
#                     {'range': [75, 100], 'color': COLORS['danger']}
#                 ],
#                 'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': risk_index}
#             }
#         ))
#         fig_gauge = apply_plot_style(fig_gauge, title="Live Risk Index", height=155)
#         fig_gauge.update_layout(margin=dict(l=25, r=25, t=35, b=10))
#         st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    
#     # Latency simulation if column missing (common in mock data)
#     proc_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 45
    
#     with c2: st.markdown(kpi_card("Mean Latency", f"{proc_time:.0f}ms", "SLA: < 100ms", COLORS['safe']), unsafe_allow_html=True)
#     with c3: st.markdown(kpi_card("Throughput", f"{len(df)}", "Transactions Buffered", COLORS['text']), unsafe_allow_html=True)
#     with c4: st.markdown(kpi_card("Analyst Overturn", "1.2%", "Label Correction Rate", COLORS['warning']), unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     # --- TRAFFIC & BOT HUNTER ---
#     c1, c2 = st.columns([1.5, 1])
#     with c1:
#         stream_df = df.tail(100).copy()
#         stream_df['legit_vol'] = 1 
#         stream_df['blocked_vol'] = stream_df[score_col].apply(lambda x: 1 if x > threshold else 0)
        
#         fig_pulse = go.Figure()
#         fig_pulse.add_trace(go.Scatter(x=stream_df['timestamp'], y=stream_df['legit_vol'], stackgroup='one', name='Legit', line=dict(color=COLORS['safe'])))
#         fig_pulse.add_trace(go.Scatter(x=stream_df['timestamp'], y=stream_df['blocked_vol'], stackgroup='one', name='Blocked', line=dict(color=COLORS['danger'])))
#         fig_pulse = apply_plot_style(fig_pulse, title="Traffic Pulse (Rolling Window)")
#         st.plotly_chart(fig_pulse, use_container_width=True)
        
#     with c2:
#         # Scatter Plot: Velocity vs Amount
#         plot_df = df.tail(200).copy()
        
#         # Check for velocity column variations
#         vel_col = 'UID_velocity_24h'
#         if vel_col not in plot_df.columns:
#             vel_col = 'UID_vel' if 'UID_vel' in plot_df.columns else None
            
#         if vel_col:
#             fig_scatter = px.scatter(
#                 plot_df, 
#                 x=vel_col, 
#                 y=amt_col, 
#                 color=score_col,
#                 color_continuous_scale='Reds',
#                 size=amt_col,
#                 size_max=15,
#                 labels={vel_col: 'Velocity', amt_col: 'Amt ($)'}
#             )
#             fig_scatter.add_vline(x=40, line_dash="dash", line_color=COLORS['warning'])
#             fig_scatter.update_layout(coloraxis_colorbar=dict(title="Risk", orientation="v", title_side="right"))
#             fig_scatter = apply_plot_style(fig_scatter, title="Bot Hunter (Vel vs Amt)")
#             st.plotly_chart(fig_scatter, use_container_width=True)
#         else:
#             st.warning("Velocity data missing for Bot Hunter chart.")
    
#     # --- ALERT QUEUE ---
#     st.markdown("<h3 style='text-align:left'>🔥 High-Priority Investigation Queue</h3>", unsafe_allow_html=True)
    
#     # Filter High Risk
#     queue = df[df[score_col] > threshold].copy()
#     queue = queue.sort_values(score_col, ascending=False).head(10)

#     if not queue.empty:
#         queue['time_formatted'] = queue['timestamp'].dt.strftime('%H:%M:%S')
#         queue['Action'] = 'REVIEW'
        
#         # Select relevant columns for the analyst
#         cols_to_show = [id_col, 'time_formatted', score_col, amt_col, 'Action']
#         # Add enriched info if available
#         if 'device_vendor' in queue.columns: cols_to_show.insert(3, 'device_vendor')
            
#         display_queue = queue[[c for c in cols_to_show if c in queue.columns]]
        
#         st.dataframe(display_queue.style.background_gradient(subset=[score_col], cmap="Reds"), use_container_width=True)
#     else:
#         st.success("✅ Queue is empty. System healthy.")