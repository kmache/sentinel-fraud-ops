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