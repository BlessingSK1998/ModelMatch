"""Presentation layer for ModelMatch. No analysis or model behaviour lives here."""
from html import escape
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

COLORS=['#536dfe','#12b8a6','#a78bfa','#f4b45e','#ed7092','#38bdf8']
pio.templates['modelmatch']=go.layout.Template(layout=go.Layout(
    font=dict(family='Inter, ui-sans-serif, system-ui, sans-serif',color='#334155',size=12),
    paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',colorway=COLORS,
    title=dict(font=dict(size=18,color='#17233d'),x=.02,xanchor='left'),
    xaxis=dict(gridcolor='#eef2f8',zerolinecolor='#e2e8f0',showline=False),
    yaxis=dict(gridcolor='#eef2f8',zerolinecolor='#e2e8f0',showline=False),
    hoverlabel=dict(bgcolor='#14213b',font_color='white',bordercolor='#14213b'),
    margin=dict(l=24,r=24,t=65,b=35),legend=dict(orientation='h',y=-.2),
))
pio.templates.default='modelmatch'

CSS='''<style>
:root{--ink:#17233d;--muted:#78849b;--accent:#536dfe;--border:#e7ecf4}
.stApp{background:#f5f7fb}
.block-container{padding:2.1rem 2.7rem 3rem;max-width:1540px}
h1,h2,h3{color:var(--ink);letter-spacing:-.7px;font-weight:700!important}
h3{font-size:1.24rem!important;margin-top:.5rem}
p,li{line-height:1.65}
[data-testid="stHeader"]{background:rgba(245,247,251,.94);border-bottom:1px solid #e7ecf4}
[data-testid="stSidebar"]{background:#111c32;border-right:none;min-width:265px;max-width:290px}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:1.1rem}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stCaption{color:#c9d4e8!important}
[data-testid="stSidebar"] hr{border-color:#29364e;margin:1.5rem 0}
[data-testid="stSidebar"] [role="radiogroup"]{gap:6px}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:11px 14px;border-radius:10px;transition:background .15s;width:100%;margin:0!important}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#293b67;box-shadow:inset 3px 0 #8698ff}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#1d2c47}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:14px;font-weight:500;color:#dce5f6!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#1d2c47!important;border-color:#34445f!important;color:#eef2ff!important}
[data-testid="stSidebar"] [data-baseweb="select"] input{color:#eef2ff!important}
[data-testid="stSidebar"] .stButton button{background:#1d2c47;border-color:#34445f;color:#dbe5f7}
.brand{display:flex;align-items:center;gap:11px;padding:8px 0 16px}
.brand-icon{background:linear-gradient(145deg,#8192ff,#536dfe);color:white;font-size:27px;border-radius:12px;display:flex;align-items:center;justify-content:center;width:43px;height:43px;box-shadow:0 7px 20px #0003}
.brand-name{color:white;font-weight:750;font-size:24px;letter-spacing:-1px}.brand-name span{color:#9eaeff}
.brand-tag{color:#8697b6;font-size:10px;letter-spacing:1.8px;margin-top:3px}
.sidebar-label{color:#7589aa;font-size:10px;letter-spacing:1.6px;margin:12px 0 7px;font-weight:700}
.breadcrumb{font-size:12px;color:#8190a7;letter-spacing:.2px;display:flex;justify-content:space-between;align-items:center;margin-bottom:19px;gap:10px;flex-wrap:wrap}
.session-badge{display:inline-flex;align-items:center;gap:7px;font-size:11px;color:#327d71;background:#e7f5f0;border:1px solid #d2e9e2;padding:5px 10px;border-radius:99px}.session-dot{height:6px;width:6px;background:#159d83;border-radius:50%}
.hero{background:linear-gradient(115deg,#152640 0%,#223b68 65%,#304a83 100%);position:relative;overflow:hidden;padding:33px 36px;border-radius:19px;color:white;margin-bottom:20px;box-shadow:0 12px 32px #182c4f12;border:1px solid #ffffff12}
.hero:after{content:'';position:absolute;right:0;top:-90px;width:370px;height:370px;border:1px solid #a9bcff25;box-shadow:0 0 0 40px #92aeff08,0 0 0 80px #92aeff06;border-radius:50%;pointer-events:none}
.hero h1{color:white!important;font-size:clamp(27px,3vw,39px)!important;line-height:1.18;letter-spacing:-1.2px;max-width:850px;margin:9px 0 11px!important;padding:0!important;position:relative;z-index:1}
.hero p{color:#c1cde4;font-size:14px;max-width:720px;margin:0;position:relative;z-index:1}.pill{font-size:10px;letter-spacing:2.2px;font-weight:700;color:#aabaff}
.dataset-strip{color:#748199;font-size:12px;padding:0 2px 20px}.dataset-strip strong{color:#33465f;font-weight:600}
[data-testid="stMetric"]{background:white;border:1px solid var(--border);border-radius:14px;padding:20px 22px;box-shadow:0 3px 10px #17233d03;min-height:117px}
[data-testid="stMetricLabel"]{color:#78849b;font-size:12px!important;font-weight:500}
[data-testid="stMetricValue"]{color:#17233d;font-size:30px;font-weight:650;letter-spacing:-.9px}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-color:var(--border)!important;border-radius:14px!important;background:white}
[data-testid="stExpander"]{background:white;border:1px solid var(--border)!important;border-radius:12px!important;overflow:hidden}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden}
[data-testid="stPlotlyChart"]{border:1px solid var(--border);border-radius:14px;background:white;overflow:hidden;padding:6px}
.stButton>button,.stDownloadButton>button{border-radius:9px;font-size:13px;font-weight:600;border-color:#dfe5f0;min-height:40px;transition:transform .15s,box-shadow .15s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#8495fd;box-shadow:0 4px 12px #536dfe12}
.stButton>button[kind="primary"]{background:#536dfe;color:white;border:none;box-shadow:0 4px 12px #536dfe25}
[data-baseweb="input"],[data-baseweb="select"]>div{border-radius:9px!important;border-color:#e1e7f0!important}
[data-testid="stTabs"] [role="tablist"]{gap:24px;border-bottom:1px solid #e2e8f0}
[data-testid="stTabs"] [role="tab"]{font-size:13px;font-weight:600;padding:12px 0}
[data-testid="stAlert"]{border-radius:12px}
.section-kicker{font-size:10px;font-weight:700;color:#7b8ca7;letter-spacing:1.8px;margin-top:26px;margin-bottom:3px}
.insight-head{color:#536dfe;font-size:11px;font-weight:700;letter-spacing:1.3px;padding-bottom:9px}
.step-label{font-size:10px;letter-spacing:1.4px;color:#8c9bb2;font-weight:700}.step-title{font-size:16px;font-weight:700;color:#1f3050;margin:5px 0}.step-copy{font-size:12px;color:#7a879b;line-height:1.7;min-height:43px;margin-bottom:13px}
.field-row{font-size:12px;color:#51627d;border-bottom:1px solid #f0f3f8;padding:8px 0;overflow-wrap:anywhere}.field-row span{font-weight:700;font-size:11px;color:#8b9ac0;margin-right:8px}.canvas-empty{text-align:center;padding:60px 14px 35px}.canvas-empty p{color:#8b97aa;font-size:13px}.canvas-symbol{font-size:48px;color:#8f9fe9;margin-bottom:10px}.workbench-bar{display:flex;justify-content:space-between;background:#15233d;color:#c3d0ef;padding:16px 22px;border-radius:12px;font-size:11px;letter-spacing:1.3px;margin:15px 0 20px}[data-testid="stTextArea"] textarea{font-family:ui-monospace,SFMono-Regular,Consolas,monospace!important;font-size:13px!important;line-height:1.8!important;background:#14213b!important;color:#dae6ff!important;padding:20px!important;border:1px solid #2c4164!important;border-radius:12px!important}
@media(max-width:800px){.block-container{padding:1.4rem 1rem}.hero{padding:24px}.hero:after{opacity:.35}[data-testid="stMetric"]{padding:15px}.brand-name{font-size:22px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>'''

PAGES={
'Overview':('YOUR ANALYTICS WORKSPACE','Your data. A clearer picture.','Everything you need to move from a dataset to an informed decision, in one workspace.'),
'Data Quality':('START WITH TRUST','Better data. Stronger decisions.','Review quality signals, confirm column roles and preview every change before you apply it.'),
'Explore':('FOLLOW YOUR CURIOSITY','Find the story in your data.','Explore relationships, compare groups and turn questions into interactive visuals.'),
'Dashboard':('YOUR LIVE DATA CANVAS','The bigger picture, at a glance.','Build a focused dashboard. Every filter updates your charts, metrics and records together.'),
'BI Studio':('DESIGN YOUR OWN ANALYSIS','A canvas for your next big idea.','Build reports, write SQL or pandas expressions, and turn your results into visuals.'),
'Model Lab':('FROM QUESTION TO MODEL','Test ideas. Compare the evidence.','Choose a goal, compare against a baseline and evaluate on data kept separate from model selection.'),
'Prediction Studio':('PUT YOUR MODEL TO WORK','Explore what could come next.','Make individual or batch predictions, test scenarios and keep model limitations in view.'),
'Exports':('BUILT TO GO WITH YOU','Your work, ready for the next step.','Download your analysis, dashboards, models and reproducible code.'),
}

def style():st.markdown(CSS,unsafe_allow_html=True)
def brand():st.markdown('<div class="brand"><div class="brand-icon">◈</div><div><div class="brand-name">Model<span>Match</span></div><div class="brand-tag">DATA TO DECISIONS</div></div></div>',unsafe_allow_html=True)
def header(page,name,rows,columns):
    kicker,title,description=PAGES[page]
    st.markdown(f'<div class="breadcrumb"><span>Workspace &nbsp;/&nbsp; <strong>{escape(page)}</strong></span><span class="session-badge"><span class="session-dot"></span> Private session</span></div><div class="hero"><div class="pill">{kicker}</div><h1>{title}</h1><p>{description}</p></div><div class="dataset-strip"><strong>{escape(name)}</strong> &nbsp;·&nbsp; {rows:,} records &nbsp;·&nbsp; {columns} columns &nbsp;·&nbsp; Dashboard filters do not change model training data</div>',unsafe_allow_html=True)
def polish(fig):
    fig.update_layout(template='modelmatch',font=dict(family='Inter, system-ui, sans-serif',color='#334155'),paper_bgcolor='white',plot_bgcolor='white',colorway=COLORS)
    return fig
