import os
import time
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(page_title="Bike Rental Demand Prediction", page_icon="🚲", layout="wide")


# Improved CSS (modern card-like style + header gradient)
APP_CSS = """
<style>
:root{
    --bg-1: #0f1724; /* deep navy */
    --bg-2: #0b1220; /* darker */
    --card: rgba(255,255,255,0.03);
    --muted: #9aa4b2;
    --accent: #00d4ff; /* cyan */
    --accent-2: #7b61ff; /* purple */
    --accent-3: #ff7ab6; /* pink */
}
/* App background gradient to give depth */
html, body, .stApp, .reportview-container, .main, .block-container {
    background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%) !important;
    color: #e6eef6 !important;
}
body { margin: 0 !important; padding: 0 !important; }

.app-header {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: white; padding: 18px 24px; border-radius: 10px; font-size: 26px; font-weight: 800;
    box-shadow: 0 8px 30px rgba(11,18,32,0.6);
}

.top-fixed-header { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }

.model-card, .input-card, .result-card {
    background: var(--card); border-radius: 12px; padding: 18px; margin-bottom: 14px;
    box-shadow: 0 6px 30px rgba(2,6,23,0.6); transition: transform .18s ease, box-shadow .18s ease;
}
.model-card:hover, .input-card:hover, .result-card:hover { transform: translateY(-4px); box-shadow: 0 10px 40px rgba(2,6,23,0.75); }

.big-pred { font-size: 42px; font-weight:800; color: var(--accent-3); }
.pred-badge { display:inline-block; padding:8px 14px; border-radius:20px; font-weight:700; color:#081022; background: linear-gradient(90deg,var(--accent),var(--accent-2)); box-shadow: 0 6px 18px rgba(123,97,255,0.18); }
.muted { color: var(--muted); }

/* Inputs: make them subtle in dark mode */
input[type="text"], input[type="number"], input[type="search"], textarea, .stTextInput>div>div, div[data-baseweb="input"] {
    background: rgba(255,255,255,0.02) !important; color: #e6eef6 !important; border: 1px solid rgba(255,255,255,0.04) !important; box-shadow: none !important;
}

/* Sidebar polish */
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#071023,#0b1220) !important; color: #d5e6f5 !important; }
div[data-testid="stSidebar"] .app-header { background: transparent !important; box-shadow: none !important; padding: 8px 12px; }
/* Make sidebar wider and add padding to match screenshot */
div[data-testid="stSidebar"] > div[style] { width: 320px !important; padding-left: 20px !important; padding-right: 20px !important; }

/* Hero tile in sidebar: large gradient box with stacked title */
.sidebar-hero { width: 220px; height: 160px; border-radius: 14px; background: linear-gradient(135deg,var(--accent),var(--accent-2)); display:flex; flex-direction:column; justify-content:center; padding:18px; box-shadow: 0 14px 40px rgba(11,18,32,0.5); color:#fff; font-weight:800; font-size:20px; }
.sidebar-hero .line { line-height:1.0; font-size:22px; }

/* Make the sidebar input groups look like inner cards */
div[data-testid="stSidebar"] .input-card { background: rgba(255,255,255,0.02) !important; padding:12px !important; border-radius:10px !important; }

/* Thumbnail spacer rounded bars (the screenshot shows rounded empty boxes) */
.top-rounded-spacer { height:34px; border-radius:18px; background: rgba(255,255,255,0.02); margin-bottom:18px; }

/* Tidy top header removal for multiple Streamlit versions */
header, [data-testid="stToolbar"], [data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stAppViewContainer"] > header { display:none !important; }

/* Reduce top spacing across common containers */
div.block-container, main, main > div { padding-top: 88px !important; }

/* Nice small helpers */
.small-muted { color: var(--muted); font-size:12px; }

/* Responsive header: stack on narrow screens */
@media (max-width: 780px) {
    .top-fixed-header div[style] { flex-direction: column; align-items: flex-start; gap:8px; }
    .pred-badge { display: none; }
    div.block-container, main, main > div { padding-top: 140px !important; }
}

</style>
"""

# Additional CSS fallback to remove card/column spacers that can appear in some Streamlit versions
EXTRA_CSS = """
<style>
/* Ensure our card components sit flush at the top of their columns */
.model-card, .input-card, .result-card {
    margin-top: 0 !important;
    padding-top: 8px !important;
}

/* Column and section wrappers used by Streamlit */
section[data-testid="stColumns"] > div, div[data-testid="column"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Remove spacer wrappers that sometimes appear as first children */
div[data-testid="stBlock"] > div:first-child, div[data-testid="stAppViewContainer"] main > div > div > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Hide stray horizontal rules inserted by Streamlit markdown */
hr {
    display: none !important;
}

</style>
"""

st.markdown(EXTRA_CSS, unsafe_allow_html=True)

st.markdown(APP_CSS, unsafe_allow_html=True)

# Darken default input widgets and remove bright white rounded boxes
INPUT_CSS = """
<style>
/* Generic inputs and textareas (fall-back for Streamlit widget internals) */
input[type="text"], input[type="number"], input[type="search"], textarea, .stTextInput>div>div, div[data-baseweb="input"] {
    background: rgba(255,255,255,0.04) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: none !important;
}
/* Make empty rounded boxes subtler */
div.stTextInput, div.stNumberInput, div.stSelectbox, div.stSlider {
    background: transparent !important;
}
/* Ensure table cells remain readable */
table, th, td { color: #ddd !important; }
</style>
"""

st.markdown(INPUT_CSS, unsafe_allow_html=True)

# Sidebar specific styling to remove white rounded boxes and match dark theme
SIDEBAR_CSS = """
<style>
/* Sidebar background and text colors */
._sidebar, .css-1d391kg, .css-1lcbmhc { background: linear-gradient(180deg,#2b2f36, #33363b) !important; color: #ddd !important; }

/* Remove large white rounded boxes which Streamlit sometimes inserts */
div[data-testid="stSidebar"] .css-1d391kg, div[data-testid="stSidebar"] .css-1lcbmhc {
    background: transparent !important; box-shadow: none !important; border: none !important;
}

/* Tidy sidebar header spacing */
div[data-testid="stSidebar"] .app-header { margin-bottom: 8px !important; }

</style>
"""

st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

# Add a fixed top header that visually covers any stray white bars and provides a consistent title area
HEADER_HTML = """
<div class="top-fixed-header" role="banner">
    <div style="display:flex;align-items:center;gap:14px;padding:14px 24px;">
        <div style="width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent-2));display:flex;align-items:center;justify-content:center;font-size:20px;">🚲</div>
        <div style="display:flex;flex-direction:column;">
            <div style="font-size:20px;font-weight:800;color:white;">Bike Rental Demand</div>
            <div style="color:rgba(255,255,255,0.9);font-size:12px;">Interactive forecasting demo — adjust inputs on the left</div>
        </div>
                <div style="margin-left:auto;display:flex;align-items:center;gap:10px;">
                        <button id="sidebar-toggle" aria-label="Open sidebar" style="background:rgba(255,255,255,0.06);border:none;color:#081022;padding:6px 10px;border-radius:8px;font-weight:700;">☰</button>
                        <div id="preview-badge" class="pred-badge" style="cursor:pointer;background: linear-gradient(90deg,#00d4ff,#ff7ab6);">Preview</div>
                </div>
    </div>
</div>
"""

# Small JS to toggle Streamlit sidebar (defensive selectors)
sidebar_toggle_js = r"""
<script>
(() => {
    function tryToggle(){
        // Try common button selectors
        const selectors = [
            'button[title="Toggle sidebar"]',
            'button[aria-label*="sidebar"]',
            'button[title*="Sidebar"]',
            'button[data-testid="stSidebarToggleButton"]',
            'div[data-testid="collapsedControl"] button'
        ];
        for(const s of selectors){
            const el = document.querySelector(s);
            if(el){ el.click(); return true; }
        }
        // fallback: toggle 'class' on body used by older Streamlit versions
        try{
            const side = document.querySelector('div[data-testid="stSidebar"]');
            if(side){
                const visible = getComputedStyle(side).display !== 'none';
                side.style.display = visible ? 'none' : '';
                return true;
            }
        }catch(e){}
        return false;
    }

    document.addEventListener('click', e => {
        const target = e.target;
        if(!target) return;
        if(target.id === 'preview-badge' || target.id === 'sidebar-toggle' || target.closest('#preview-badge') ){
            e.preventDefault();
            tryToggle();
        }
    });

    // add keyboard accessibility
    const pb = document.getElementById('preview-badge');
    if(pb){ pb.setAttribute('tabindex', '0'); pb.addEventListener('keydown', (ev)=>{ if(ev.key==='Enter'){ tryToggle(); } }); }

})();
</script>
"""
components.html(sidebar_toggle_js, height=1)

# Floating open-sidebar button (always visible) and its JS handler
floating_sidebar_html = r"""
<div id="floating-open-sidebar" style="position:fixed;left:12px;top:86px;z-index:100001;">
    <button aria-label="Open sidebar" style="width:46px;height:46px;border-radius:12px;background:linear-gradient(135deg,#00d4ff,#7b61ff);border:none;color:#041226;font-weight:800;box-shadow:0 8px 24px rgba(11,18,32,0.6);cursor:pointer;">☰</button>
</div>
<script>
(function(){
    function tryToggle(){
        const selectors = [
            'button[title="Toggle sidebar"]',
            'button[aria-label*="sidebar"]',
            'button[title*="Sidebar"]',
            'button[data-testid="stSidebarToggleButton"]',
            'div[data-testid="collapsedControl"] button'
        ];
        for(const s of selectors){
            const el = document.querySelector(s);
            if(el){ el.click(); return true; }
        }
        try{
            const side = document.querySelector('div[data-testid="stSidebar"]');
            if(side){
                const visible = getComputedStyle(side).display !== 'none';
                side.style.display = visible ? 'none' : '';
                return true;
            }
        }catch(e){}
        return false;
    }

    const fb = document.getElementById('floating-open-sidebar');
    if(fb){ fb.addEventListener('click', (e)=>{ e.preventDefault(); tryToggle(); }); }

    // small observer to ensure button remains visible when Streamlit redraws
    const obs = new MutationObserver(()=>{ const b = document.getElementById('floating-open-sidebar'); if(!b){ document.body.appendChild(document.createRange().createContextualFragment(`` + document.getElementById('floating-open-sidebar')?.outerHTML || '')); }});
    obs.observe(document.body, { childList:true, subtree:true });
})();
</script>
"""
components.html(floating_sidebar_html, height=1)

HEADER_CSS = """
<style>
.top-fixed-header{ position: fixed; left: 0; right: 0; top: 0; height: 68px; z-index: 100000 !important; 
    background: linear-gradient(90deg,#1f77b4,#ff7f0e); box-shadow: 0 6px 18px rgba(0,0,0,0.25); border-bottom-left-radius:0; border-bottom-right-radius:0; pointer-events:auto; }
/* push the main content down so it doesn't hide under the fixed header */
div.block-container, main, main > div { padding-top: 90px !important; }

/* ensure sidebar content is pushed below the fixed header */
section[data-testid="stSidebar"], div[data-testid="stSidebar"] { padding-top: 90px !important; z-index: 1000 !important; }
</style>
"""

st.markdown(HEADER_CSS + HEADER_HTML, unsafe_allow_html=True)


# -----------------------------
# Utilities
# -----------------------------
MODEL_PATH = os.path.join("models", "bike_model.pkl")


@st.cache_resource
def load_model(path: str):
    """Load and return the trained model. Cached for faster reloads."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def format_input_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # Keep the original numeric columns as-is. (Model expects same columns used during training.)
    return df


def demand_level_and_style(count: int):
    if count < 50:
        return "Very Low", "#2b6cb0", "🔵"
    if count < 100:
        return "Low", "#2f855a", "🟢"
    if count < 200:
        return "Moderate", "#ed8936", "🟠"
    if count < 300:
        return "High", "#e53e3e", "🔴"
    return "Very High", "#6b46c1", "🟣"


# -----------------------------
# User input (left column / sidebar)
# -----------------------------
with st.sidebar:
    # Large gradient hero tile (stacked title) to match screenshot
    st.markdown('''
    <div style="margin-top:8px;margin-bottom:18px;">
      <div class="sidebar-hero">
        <div class="line">🚲 Bike</div>
        <div class="line">Rental</div>
        <div class="line">Demand</div>
        <div class="line">Prediction</div>
      </div>
    </div>
    <div class="top-rounded-spacer"></div>
    ''', unsafe_allow_html=True)
    st.markdown("""
    <div class='muted' style="padding-right:8px;">Predict hourly bike demand using weather, time and calendar features. Ideal for presenting as a mini-product in portfolios.</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.subheader("Date & Time")
    mnth = st.slider("Month", 1, 12, 6)
    hr = st.slider("Hour of Day", 0, 23, 12)
    weekday = st.select_slider("Day of Week", options=[0, 1, 2, 3, 4, 5, 6],
                              format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x],
                              value=2)

    st.subheader("Weather & Conditions")
    weathersit = st.select_slider("Weather Condition", options=[1, 2, 3, 4],
                                  format_func=lambda x: ["Clear", "Misty", "Light Rain", "Heavy Rain/Snow"][x - 1],
                                  value=1)
    temp = st.slider("Temperature (°C)", -10, 40, 20)
    atemp = st.slider("Feels Like (°C)", -10, 50, 22)
    hum = st.slider("Humidity (%)", 0, 100, 60)
    windspeed = st.slider("Windspeed (km/h)", 0, 100, 10)

    st.subheader("Other Factors")
    season = st.radio("Season", options=[1, 2, 3, 4], index=1,
                      format_func=lambda x: ["Spring", "Summer", "Fall", "Winter"][x - 1])
    holiday = st.radio("Is it a holiday?", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
    workingday = st.radio("Is it a working day?", options=[0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("</div>", unsafe_allow_html=True)

    # Build input dataframe
    input_data = pd.DataFrame({
        'season': [season],
        'mnth': [mnth],
        'hr': [hr],
        'holiday': [holiday],
        'weekday': [weekday],
        'workingday': [workingday],
        'weathersit': [weathersit],
        'temp': [temp],
        'atemp': [atemp],
        'hum': [hum],
        'windspeed': [windspeed]
    })
    # Optional client-side cleanup: hides stray white spacer elements.
    st.markdown('---')
    js_cleanup = st.checkbox('Run client-side cleanup (hide stray white bars)', value=True)

    if js_cleanup:
        # refined JS: hide small top elements with near-white backgrounds
        cleanup_js = r"""
        <script>
        (function(){
            try {
                const isNearWhite = (r,g,b,a) => {
                    if (a !== undefined && a < 0.03) return false;
                    // simple luminance check
                    const lum = 0.2126*r + 0.7152*g + 0.0722*b;
                    return lum > 230; // near white
                }
                function rgbFromStyle(s){
                    if (!s) return null;
                    const m = s.match(/rgba?\(([^)]+)\)/);
                    if(!m) return null;
                    const parts = m[1].split(',').map(p=>parseFloat(p.trim()));
                    return parts;
                }
                const els = Array.from(document.body.querySelectorAll('*'));
                els.forEach(el => {
                    try {
                        const r = el.getBoundingClientRect();
                        if (!r) return;
                        if (r.top >=0 && r.top < 60 && r.height>6 && r.height < 80) {
                            const cs = window.getComputedStyle(el);
                            const bg = cs.backgroundColor || cs.background;
                            const rgb = rgbFromStyle(bg);
                            if (rgb && isNearWhite(rgb[0], rgb[1], rgb[2], rgb[3])) {
                                el.style.display = 'none';
                            }
                        }
                    } catch(e){}
                });
            } catch(e){console && console.log && console.log(e)}
        })();
        </script>
        """
        components.html(cleanup_js, height=1)

        # Persistent observer: hides newly added white spacer elements during re-renders
        observer_js = r"""
        <script>
        (function(){
            function isNearWhiteColor(str){
                if(!str) return false;
                const m = str.match(/rgba?\(([^)]+)\)/);
                if(!m) return false;
                const parts = m[1].split(',').map(x=>parseFloat(x));
                const r = parts[0], g = parts[1], b = parts[2];
                const a = parts.length>3?parts[3]:1;
                if(a < 0.03) return false;
                const lum = 0.2126*r + 0.7152*g + 0.0722*b;
                return lum > 230;
            }

            function hideCandidates(root=document.body){
                const nodes = root.querySelectorAll('*');
                for(const el of nodes){
                    try{
                        const rect = el.getBoundingClientRect();
                        if(!rect) continue;
                        // candidates located near top and with some height
                        if(rect.top >=0 && rect.top < 70 && rect.height>6 && rect.height < 120){
                            const style = window.getComputedStyle(el);
                            const bg = style.backgroundColor || style.background;
                            const br = parseFloat(style.borderRadius) || 0;
                            if(br>6 && isNearWhiteColor(bg)){
                                el.style.display = 'none';
                            }
                        }
                    }catch(e){}
                }
            }

            // initial run
            setTimeout(()=>hideCandidates(), 100);

            // observe DOM changes
            const obs = new MutationObserver((mutations)=>{
                for(const m of mutations){
                    if(m.addedNodes && m.addedNodes.length>0){
                        m.addedNodes.forEach(n=>{ if(n.nodeType===1) hideCandidates(n); });
                    }
                }
            });
            obs.observe(document.body, { childList:true, subtree:true });

            // safety interval for a few seconds to catch delayed renders
            let runs = 0;
            const iv = setInterval(()=>{ hideCandidates(); runs++; if(runs>20) clearInterval(iv); }, 300);
        })();
        </script>
        """
        components.html(observer_js, height=1)

        # Recolor any large rounded white pills near the top (instead of hiding) to match dark theme
        recolor_js = r"""
        <script>
        (function(){
            try{
                const nodes = Array.from(document.body.querySelectorAll('*'));
                for(const el of nodes){
                    try{
                        const rect = el.getBoundingClientRect();
                        if(!rect) continue;
                        if(rect.top >= 0 && rect.top < 120 && rect.width > 80 && rect.height > 18){
                            const style = window.getComputedStyle(el);
                            const bg = style.backgroundColor || style.background;
                            // detect near-white background
                            const m = bg && bg.match(/rgba?\(([^)]+)\)/);
                            if(m){
                                const parts = m[1].split(',').map(p=>parseFloat(p));
                                const r = parts[0], g = parts[1], b = parts[2];
                                const lum = 0.2126*r + 0.7152*g + 0.0722*b;
                                if(lum > 240){
                                    el.style.background = 'rgba(20,20,20,0.6)';
                                    el.style.border = '1px solid rgba(255,255,255,0.06)';
                                    el.style.color = '#fff';
                                    el.style.boxShadow = 'none';
                                    // if contains input children, style them too
                                    el.querySelectorAll('input, textarea').forEach(i=>{
                                        i.style.background = 'transparent';
                                        i.style.color = '#fff';
                                        i.style.border = 'none';
                                    });
                                }
                            }
                        }
                    }catch(e){}
                }
            }catch(e){console && console.log && console.log(e)}
        })();
        </script>
        """
        components.html(recolor_js, height=1)


# -----------------------------
# Main area: model info, inputs and results
# -----------------------------
left, right = st.columns([2, 3])

with left:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.subheader("Model Info")
    try:
        model = load_model(MODEL_PATH)
        model_loaded = True
        model_type = type(model).__name__
        model_detail = "" if model_type == 'Pipeline' else model_type
        st.success(f"Model loaded: {os.path.basename(MODEL_PATH)}")
        st.write("Model type:", model_type)
    except FileNotFoundError as e:
        model_loaded = False
        st.error(str(e))
        st.stop()
    except Exception as e:
        model_loaded = False
        st.error(f"Error loading model: {e}")
        st.stop()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("Input Summary")
    disp = input_data.copy()
    disp_display = disp.rename(columns={
        'season': 'Season', 'mnth': 'Month', 'hr': 'Hour', 'holiday': 'Holiday',
        'weekday': 'Day', 'workingday': 'Working Day', 'weathersit': 'Weather',
        'temp': 'Temp (°C)', 'atemp': 'Feels Like (°C)', 'hum': 'Humidity (%)', 'windspeed': 'Windspeed'
    })
    st.table(disp_display.T.rename(columns={0: 'Value'}))
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Prediction")

    # Make prediction
    try:
        formatted = format_input_dataframe(input_data)
        preds = model.predict(formatted)
        predicted_count = int(np.round(preds[0]))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    level, color, emoji = demand_level_and_style(predicted_count)

    st.markdown(f"<div class='big-pred'>{emoji} {predicted_count} bikes</div>", unsafe_allow_html=True)
    st.markdown(f"<div class=\"muted\">Estimated demand level: <strong style='color:{color}'>{level}</strong></div>", unsafe_allow_html=True)

    # Compact bar visualization
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh([0], [predicted_count], color=color, alpha=0.85)
    ax.set_xlim(0, max(500, predicted_count * 1.2))
    ax.set_yticks([])
    ax.set_xlabel('Number of Bikes')
    ax.set_title('Predicted Demand')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#dddddd')
    st.pyplot(fig)

    # Quick insights
    st.markdown("---")
    st.subheader("Quick Insights")
    if input_data['weathersit'].iloc[0] >= 3:
        st.warning("Weather may reduce rentals today.")
    if input_data['holiday'].iloc[0] == 1:
        st.info("Holiday: patterns may deviate from typical working day demand.")
    if input_data['hr'].iloc[0] in [7, 8, 9, 17, 18, 19]:
        st.info("Peak commuting hour — expect higher demand in stations near transit hubs.")
    if input_data['temp'].iloc[0] < 5:
        st.info("Cold temperature may reduce rideability and demand.")

    st.markdown("</div>", unsafe_allow_html=True)


# Footer - resume friendly summary and run instructions
st.markdown("---")
st.markdown("### About this demo")
st.markdown(
    "This interactive demo showcases a trained regression model for hourly bike demand forecasting. "
    "It includes a clean UI, model metadata, and quick actionable insights — ideal for portfolio or resume demos."
)

st.caption("Tip: package the `models/bike_model.pkl` (trained pipeline), `app.py` and `requirements.txt` to reproduce this demo.")
