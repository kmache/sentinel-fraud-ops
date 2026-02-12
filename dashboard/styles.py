import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# 1. COLOR PALETTE
# ==============================================================================
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

# ==============================================================================
# 2. PAGE SETUP & CSS
# ==============================================================================
def setup_page(title="Sentinel Dashboard", layout="wide"):
    st.set_page_config(
        page_title=title,
        page_icon="logo.png",
        layout=layout,
        initial_sidebar_state="expanded" 
    )
    
    st.markdown(f"""
    <style>
        /* Global App Background */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }}

        /* RESET & PADDING */
        .stDeployButton {{ display: none; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        
        .block-container {{
            padding-top: 3.5rem; 
            padding-bottom: 1rem;
        }}

        /* CENTRALIZED GLOBAL TITLE */
        .global-title {{
            text-align: center;
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
            margin-top: 0px; 
            margin-bottom: 20px;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 10px;
        }}
        .page-header h2 {{
            font-size: 24px;
            font-weight: 700;
            color: {COLORS['text']};
            margin: 0;
        }}
        .page-header p {{
            font-size: 14px;
            color: {COLORS['neutral']};
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
            transition: transform 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
            border-color: {COLORS['highlight']};
        }}
        
        .kpi-title {{
            font-size: 14px;
            font-weight: 600;
            color: {COLORS['neutral']};
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .kpi-value {{
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 5px;
        }}
        
        .kpi-subtext {{
            font-size: 12px;
            color: {COLORS['neutral']};
            font-style: italic;
        }}

        /* STATUS INDICATORS (Useful for Sidebar/System health) */
        .status-indicator {{
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }}
        .status-green {{ background-color: {COLORS['safe']}; box-shadow: 0 0 8px {COLORS['safe']}; }}
        .status-orange {{ background-color: {COLORS['warning']}; box-shadow: 0 0 8px {COLORS['warning']}; }}
        .status-red {{ background-color: {COLORS['danger']}; box-shadow: 0 0 8px {COLORS['danger']}; }}

        /* BUTTON STYLING */
        div.stButton > button {{
            width: 100%;
            background-color: {COLORS['card_bg']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 5px;
            height: 45px;
            font-weight: 600;
        }}
        div.stButton > button:hover {{
            border-color: {COLORS['safe']};
            color: {COLORS['safe']};
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. UI HELPERS
# ==============================================================================
def render_top_banner():
    """Renders the main Global Header centralized at the top."""
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 40px; padding-top: 10px;">
        <h1 style="
            color: {COLORS['highlight']}; 
            font-size: 3.5rem; 
            margin-bottom: 0; 
            letter-spacing: -1.5px; 
            font-weight: 800;
            text-shadow: 0px 0px 15px rgba(0, 204, 150, 0.2);
        ">
            Sentinel Fraud Ops
        </h1>
        <div style="
            color: {COLORS['neutral']}; 
            font-size: 1.1rem; 
            margin-top: 5px; 
            font-weight: 300; 
            letter-spacing: 1px;
            text-transform: uppercase;
        ">
            Real-time AI Security & Forensics System
        </div>
        <div style="
            margin: 15px auto 0 auto; 
            width: 80px; 
            height: 3px; 
            background: {COLORS['highlight']}; 
            border-radius: 2px;
            opacity: 0.6;
        "></div>
    </div>
    """, unsafe_allow_html=True)

def render_header(title, subtitle=""):
    """Renders a smaller section header."""
    sub_html = f"<div style='font-size:13px; color:{COLORS['neutral']}; margin-top:-5px; margin-bottom:15px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom: 15px;">
        <h3 style="margin:0; padding:0; color:{COLORS['text']};">{title}</h3>
        {sub_html}
        <hr style="margin: 5px 0 15px 0; border: 0; border-top: 1px solid {COLORS['border']};">
    </div>
    """, unsafe_allow_html=True)

def kpi_card(title, value, subtext="", value_color="safe"):
    color_hex = COLORS.get(value_color, value_color)
    if not color_hex.startswith("#") and value_color not in COLORS:
         color_hex = COLORS['text']

    return f"""
    <div class="kpi-card">
        <h4>{title}</h4>
        <div class="kpi-value" style="color: {color_hex}">{value}</div>
        <p class="kpi-subtext">{subtext}</p>
    </div>
    """

def apply_plot_style(fig, title="", height=350):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title={
            'text': f"<b>{title}</b>" if title else "",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 16, 'color': COLORS['text'], 'family': "Inter, sans-serif"}
        },
        font=dict(color=COLORS['neutral'], family="Inter, sans-serif"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        
        hoverlabel=dict(bgcolor=COLORS['card_bg'], font_size=12, font_family="Inter", bordercolor=COLORS['border']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    grid_style = dict(showgrid=True, gridcolor="rgba(43, 59, 79, 0.4)", linecolor=COLORS['border'], zeroline=False)
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(**grid_style)
    return fig

    