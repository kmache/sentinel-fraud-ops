# import streamlit as st
# import plotly.graph_objects as go

# # ==============================================================================
# # 1. COLOR PALETTE
# # ==============================================================================
# COLORS = {
#     "background": "#0E1117",      # Main App Background
#     "card_bg": "#181b21",         # Card Background
#     "text": "#FFFFFF",            # Main Text (Brightened for contrast)
#     "safe": "#00CC96",            # Green (Success)
#     "danger": "#EF553B",          # Red (Fraud/Danger)
#     "warning": "#FFA15A",         # Amber (Alert)
#     "neutral": "#A0A4B0",         # Subtext Gray (Brightened for readability)
#     "border": "#2b3b4f",          # Card Borders
#     "highlight": "#00CC96"        # Brand Color
# }

# # ==============================================================================
# # 2. PAGE SETUP & CSS
# # ==============================================================================
# def setup_page(title="Sentinel Dashboard", layout="wide"):
#     """
#     Configures the page settings and injects global CSS.
#     Call this at the very top of your app.py.
#     """
#     st.set_page_config(
#         page_title=title,
#         page_icon="🛡️",
#         layout=layout,
#         initial_sidebar_state="collapsed"
#     )
    
#     # Inject CSS
#     st.markdown(f"""
#     <style>
#         /* IMPORT FONT (Inter) */
#         @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

#         /* ---------------------------------------------------------------------
#            RESET & LAYOUT
#            --------------------------------------------------------------------- */
#         .stDeployButton {{ display: none; }}
#         #MainMenu {{ visibility: hidden; }}
#         footer {{ visibility: hidden; }}
#         header {{ visibility: hidden; }}
        
#         .block-container {{
#             padding-top: 1.5rem; 
#             padding-bottom: 2rem;
#         }}

#         /* ---------------------------------------------------------------------
#            GLOBAL THEME
#            --------------------------------------------------------------------- */
#         .stApp {{
#             background-color: {COLORS['background']};
#             color: {COLORS['text']};
#             font-family: 'Inter', sans-serif; /* Applied Globally */
#         }}
        
#         h1, h2, h3, h4, h5, h6 {{
#             color: {COLORS['text']} !important;
#             font-family: 'Inter', sans-serif;
#             font-weight: 700;
#         }}

#         /* ---------------------------------------------------------------------
#            KPI CARDS
#            --------------------------------------------------------------------- */
#         .kpi-card {{
#             background-color: {COLORS['card_bg']};
#             border: 1px solid {COLORS['border']};
#             border-radius: 8px;
#             padding: 20px;
#             text-align: center;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.2);
#             margin-bottom: 10px;
#             height: 100%; 
#             display: flex;
#             flex-direction: column;
#             justify-content: center;
#             align-items: center;
#             min-height: 130px; 
#             transition: transform 0.2s;
#         }}
#         .kpi-card:hover {{
#             border-color: {COLORS['highlight']};
#             transform: translateY(-2px);
#         }}
        
#         /* Semantic styling for KPI internals */
#         .kpi-card h4 {{
#             font-size: 14px; 
#             font-weight: 600; 
#             color: {COLORS['text']}; 
#             margin: 0 0 5px 0;
#             padding: 0;
#         }}
#         .kpi-value {{
#             font-size: 28px; 
#             font-weight: 700; 
#             margin: 5px 0;
#         }}
#         .kpi-subtext {{
#             font-size: 12px; 
#             color: {COLORS['neutral']}; 
#             margin: 0;
#         }}

#         /* ---------------------------------------------------------------------
#            CUSTOM TABS
#            --------------------------------------------------------------------- */
#         .stTabs [data-baseweb="tab-list"] {{
#             gap: 8px;
#         }}
#         .stTabs [data-baseweb="tab"] {{
#             background-color: {COLORS['card_bg']};
#             border: 1px solid {COLORS['border']};
#             border-radius: 4px;
#             color: {COLORS['neutral']};
#             padding: 8px 16px;
#         }}
#         .stTabs [data-baseweb="tab"]:hover {{
#             color: {COLORS['highlight']};
#             border-color: {COLORS['highlight']};
#         }}
#         .stTabs [data-baseweb="tab"][aria-selected="true"] {{
#             background-color: {COLORS['highlight']} !important;
#             color: #000000 !important;
#             border-color: {COLORS['highlight']};
#             font-weight: 600;
#         }}

#         /* ---------------------------------------------------------------------
#            MOBILE RESPONSIVENESS
#            --------------------------------------------------------------------- */
#         @media (max-width: 768px) {{
#             .kpi-card {{ min-height: 110px; padding: 10px; }}
#             .kpi-value {{ font-size: 22px; }}
#         }}
#     </style>
#     """, unsafe_allow_html=True)

# # ==============================================================================
# # 3. UI HELPERS
# # ==============================================================================
# def render_header(title, subtitle=""):
#     """Renders a consistent section header."""
#     sub_html = f"<div style='font-size:13px; color:{COLORS['neutral']}; margin-top:-5px; margin-bottom:15px;'>{subtitle}</div>" if subtitle else ""
#     st.markdown(f"""
#     <div style="border-bottom: 1px solid {COLORS['border']}; padding-bottom: 5px; margin-bottom: 15px;">
#         <h3 style="margin:0; padding:0; color:{COLORS['text']};">{title}</h3>
#         {sub_html}
#     </div>
#     """, unsafe_allow_html=True)

# def kpi_card(title, value, subtext, value_color=COLORS['safe']):
#     """Returns HTML for a styled KPI card using semantic tags."""
#     # Robust check: if value_color is a key in COLORS, use that hex, else use raw
#     color_hex = COLORS.get(value_color, value_color)
    
#     return f"""
#     <div class="kpi-card">
#         <h4>{title}</h4>
#         <div class="kpi-value" style="color: {color_hex}">{value}</div>
#         <p class="kpi-subtext">{subtext}</p>
#     </div>
#     """

# def apply_plot_style(fig, title="", height=350):
#     """
#     Applies the Dashboard Dark Theme to any Plotly figure.
#     Includes custom hover labels for a professional finish.
#     """
#     fig.update_layout(
#         template="plotly_dark",
#         title={
#             'text': f"<b>{title}</b>" if title else "",
#             'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
#             'font': {'size': 16, 'color': COLORS['text'], 'family': "Inter, sans-serif"}
#         },
#         height=height,
#         font=dict(color=COLORS['neutral'], family="Inter, sans-serif"),
#         paper_bgcolor=COLORS['card_bg'],
#         plot_bgcolor=COLORS['card_bg'],
#         margin=dict(l=40, r=40, t=50, b=40),
        
#         # Hover Label Styling (Review Recommendation)
#         hoverlabel=dict(
#             bgcolor=COLORS['card_bg'],
#             font_size=12,
#             font_family="Inter, sans-serif",
#             bordercolor=COLORS['border']
#         ),
        
#         legend=dict(
#             orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
#             bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=COLORS['text'])
#         )
#     )
    
#     # Custom Grid
#     grid_style = dict(
#         showgrid=True, 
#         gridcolor="rgba(43, 59, 79, 0.4)", 
#         linecolor=COLORS['border'], 
#         zeroline=False
#     )
#     fig.update_xaxes(**grid_style)
#     fig.update_yaxes(**grid_style)
    
#     return fig


# import streamlit as st
# import plotly.graph_objects as go

# # ==============================================================================
# # 1. COLOR PALETTE
# # ==============================================================================
# COLORS = {
#     "background": "#0E1117",      
#     "card_bg": "#181b21",         
#     "text": "#FFFFFF",            
#     "safe": "#00CC96",            
#     "danger": "#EF553B",          
#     "warning": "#FFA15A",         
#     "neutral": "#A0A4B0",         
#     "border": "#2b3b4f",          
#     "highlight": "#00CC96"        
# }

# # ==============================================================================
# # 2. PAGE SETUP & CSS
# # ==============================================================================
# def setup_page(title="Sentinel Dashboard"):
#     st.set_page_config(
#         page_title=title,
#         page_icon="🛡️",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     # Inject CSS
#     st.markdown(f"""
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

#         /* 1. GLOBAL APP STYLING */
#         .stApp {{
#             background-color: {COLORS['background']};
#             color: {COLORS['text']};
#             font-family: 'Inter', sans-serif;
#         }}

#         /* 2. SIDEBAR STYLING */
#         [data-testid="stSidebar"] {{
#             background-color: #11141a;
#             border-right: 1px solid {COLORS['border']};
#         }}
#         [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
#             color: {COLORS['highlight']} !important;
#         }}

#         /* 3. HIDE DEFAULT ELEMENTS */
#         .stDeployButton {{ display: none; }}
#         #MainMenu {{ visibility: hidden; }}
#         footer {{ visibility: hidden; }}
#         header {{ background: rgba(0,0,0,0); }}

#         /* 4. KPI CARDS */
#         .kpi-card {{
#             background-color: {COLORS['card_bg']};
#             border: 1px solid {COLORS['border']};
#             border-radius: 8px;
#             padding: 20px;
#             text-align: center;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.2);
#             margin-bottom: 15px;
#             min-height: 140px; 
#             display: flex;
#             flex-direction: column;
#             justify-content: center;
#         }}
#         .kpi-card h4 {{
#             font-size: 14px !important; 
#             font-weight: 400 !important; 
#             color: {COLORS['neutral']} !important; 
#             margin: 0 !important;
#         }}
#         .kpi-value {{
#             font-size: 32px; 
#             font-weight: 700; 
#             margin: 10px 0;
#         }}
#         .kpi-subtext {{
#             font-size: 12px; 
#             color: {COLORS['neutral']}; 
#             margin: 0;
#         }}

#         /* 5. INPUTS & BUTTONS */
#         div.stButton > button {{
#             background-color: {COLORS['card_bg']};
#             color: {COLORS['text']};
#             border: 1px solid {COLORS['border']};
#             border-radius: 5px;
#             width: 100%;
#         }}
#         div.stButton > button:hover {{
#             border-color: {COLORS['highlight']};
#             color: {COLORS['highlight']};
#         }}
#     </style>
#     """, unsafe_allow_html=True)

# # ==============================================================================
# # 3. UI HELPERS
# # ==============================================================================
# def render_header(title, subtitle=""):
#     st.markdown(f"""
#     <div style="border-bottom: 1px solid {COLORS['border']}; padding-bottom: 10px; margin-bottom: 25px;">
#         <h2 style="margin:0; color:{COLORS['text']};">{title}</h2>
#         <p style="margin:0; color:{COLORS['neutral']}; font-size:14px;">{subtitle}</p>
#     </div>
#     """, unsafe_allow_html=True)

# def kpi_card(title, value, subtext, value_color="safe"):
#     # Resolve color
#     color_hex = COLORS.get(value_color, COLORS['safe'])
    
#     return f"""
#     <div class="kpi-card">
#         <h4>{title}</h4>
#         <div class="kpi-value" style="color: {color_hex}">{value}</div>
#         <div class="kpi-subtext">{subtext}</div>
#     </div>
#     """

# def apply_plot_style(fig, title="", height=350):
#     fig.update_layout(
#         template="plotly_dark",
#         title={{
#             'text': f"<b>{title}</b>" if title else "",
#             'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
#             'font': {{'size': 14, 'color': COLORS['text'], 'family': "Inter, sans-serif"}}
#         }},
#         height=height,
#         font=dict(color=COLORS['neutral'], family="Inter, sans-serif"),
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         margin=dict(l=20, r=20, t=60, b=20),
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#     )
#     fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
#     fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
#     return fig