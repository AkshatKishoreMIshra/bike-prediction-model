import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Minimal, compact Streamlit app (single-file)
st.set_page_config(page_title="Bike Rental Demand", page_icon="🚲", layout="wide")
MODEL_PATH = os.path.join("models", "bike_model.pkl")


# Compact CSS + small header with safe JS toggle
APP_CSS = """
<style>
    /* Theme variables */
    :root {
        --bg: #071226;                 /* page background (deep blue) */
        --card: rgba(255,255,255,0.03); /* translucent card background */
        --muted: #9aa4b2;              /* subdued text color */
    }

    /* Page background and base text color */
    html, body, .stApp {
        background: linear-gradient(180deg, var(--bg), #071522) !important;
        color: #e6eef6;
    }

    /* Fixed top header */
    .top-fixed {
        position: fixed;
        left: 0;
        right: 0;
        top: 0;
        height: 66px;
        background: linear-gradient(90deg, #0b3d91, #205081);
        z-index: 9999;
        box-shadow: 0 6px 18px rgba(0,0,0,0.28);
    }

    .top-fixed .wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 20px;
    }

    /* Card style used in panels */
    .card {
        background: var(--card);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 6px 22px rgba(2,6,23,0.55);
    }

    /* Large numeric prediction style */
    .big {
        font-size: 34px;
        font-weight: 800;
        color: #ff7ab6;
    }

    /* Muted / helper text */
    .muted { color: var(--muted); }

    /* Make room under the fixed header so content isn't hidden */
    div.block-container, main, main > div {
        padding-top: 92px !important;
    }

    @media (max-width: 700px) {
        .big { font-size: 28px; }
    }
</style>

<!-- header HTML (simple, accessible banner) -->
<div class="top-fixed" role="banner">
    <div class="wrap">
        <div style="width:40px;height:40px;border-radius:8px;background:#fff;color:#0b3d91;display:flex;align-items:center;justify-content:center;font-weight:800">🚲</div>
        <div>
            <div style="font-weight:700;font-size:18px">Bike Rental Demand</div>
            <div class="muted" style="font-size:12px">Hourly forecast demo</div>
        </div>
        <div style="margin-left:auto">
            <button id="sb-toggle" style="background:transparent;border:1px solid rgba(255,255,255,0.08);color:#fff;padding:6px 10px;border-radius:8px">☰</button>
        </div>
    </div>
</div>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

# Small, defensive JS to toggle the Streamlit sidebar (keeps app portable)
components.html(r"""
<script>
(function(){
  function t(){
    var b=document.querySelector('button[data-testid="stSidebarToggleButton"]')||document.querySelector('button[aria-label*="sidebar"]');
    if(b){b.click();return}
    var s=document.querySelector('div[data-testid="stSidebar"]'); if(s) s.style.display=(getComputedStyle(s).display==='none')?'':'none';
  }
  document.addEventListener('click',function(e){var t0=e.target; if(!t0) return; if(t0.id==='sb-toggle' || t0.closest('#sb-toggle')){e.preventDefault(); t();}})
})();
</script>
""", height=1)


# Helpers - single canonical implementations
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def demand_style(n: int):
    if n < 50: return "Very Low", "#2b6cb0", "🔵"
    if n < 100: return "Low", "#2f855a", "🟢"
    if n < 200: return "Moderate", "#ed8936", "🟠"
    if n < 300: return "High", "#e53e3e", "🔴"
    return "Very High", "#6b46c1", "🟣"


# Sidebar inputs
with st.sidebar:
    st.markdown('<div class="card" style="width:220px;background:linear-gradient(135deg,#0b3d91,#7b61ff);color:#fff;font-weight:700;padding:12px;border-radius:8px">🚲 Bike Rental</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Predict hourly demand using weather, time and calendar features.</div>', unsafe_allow_html=True)
    st.markdown('---')

    mnth = st.slider('Month', 1, 12, 6)
    hr = st.slider('Hour', 0, 23, 12)
    weekday = st.select_slider('Weekday', options=list(range(7)), format_func=lambda x: ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][x], value=2)
    weathersit = st.select_slider('Weather', options=[1,2,3,4], format_func=lambda x: ['Clear','Misty','Light Rain','Heavy Rain/Snow'][x-1], value=1)
    temp = st.slider('Temp (°C)', -10, 40, 20)
    atemp = st.slider('Feels Like (°C)', -10, 50, 22)
    hum = st.slider('Humidity (%)', 0, 100, 60)
    windspeed = st.slider('Windspeed (km/h)', 0, 100, 10)

    season = st.radio('Season', [1,2,3,4], index=1, format_func=lambda x: ['Spring','Summer','Fall','Winter'][x-1])
    holiday = st.radio('Holiday?', [0,1], index=0, format_func=lambda x: 'Yes' if x else 'No')
    workingday = st.radio('Working day?', [0,1], index=1, format_func=lambda x: 'Yes' if x else 'No')

    input_df = pd.DataFrame({
        'season':[season],'mnth':[mnth],'hr':[hr],'holiday':[holiday],'weekday':[weekday],'workingday':[workingday],
        'weathersit':[weathersit],'temp':[temp],'atemp':[atemp],'hum':[hum],'windspeed':[windspeed]
    })


# Main layout
left, right = st.columns([2,3])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Model')
    try:
        model = load_model(MODEL_PATH)
        st.success(f'Loaded: {os.path.basename(MODEL_PATH)}')
        st.write('Type:', type(model).__name__)
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Inputs')
    disp = input_df.rename(columns={'mnth':'Month','hr':'Hour','weekday':'Day','weathersit':'Weather','temp':'Temp (°C)','atemp':'Feels Like (°C)','hum':'Humidity (%)','windspeed':'Windspeed'})
    st.table(disp.T.rename(columns={0:'Value'}))
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Prediction')
    try:
        preds = model.predict(input_df)
        n = int(np.round(preds[0]))
    except Exception as e:
        st.error(f'Prediction failed: {e}')
        st.stop()

    level, color, emoji = demand_style(n)
    st.markdown(f'<div class="big">{emoji} {n} bikes</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">Level: <strong style="color:{color}">{level}</strong></div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7,1.2))
    ax.barh([0],[n], color=color, alpha=0.9)
    ax.set_xlim(0, max(500, n*1.2)); ax.set_yticks([]); ax.set_xlabel('Bikes')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color('#dddddd')
    st.pyplot(fig)

    st.markdown('---')
    st.subheader('Quick insights')
    if input_df['weathersit'].iloc[0] >= 3: st.warning('Weather may reduce rentals today.')
    if input_df['holiday'].iloc[0] == 1: st.info('Holiday: patterns may deviate from typical working day demand.')
    if input_df['hr'].iloc[0] in [7,8,9,17,18,19]: st.info('Peak commuting hour — expect higher demand nearby transit hubs.')
    if input_df['temp'].iloc[0] < 5: st.info('Cold temperature may reduce rideability and demand.')
    st.markdown('</div>', unsafe_allow_html=True)


# Footer
st.markdown('---')
st.markdown('### About')
st.write('Interactive demo of an hourly bike demand regression model — ideal for portfolio demos.')
st.caption('Include `models/bike_model.pkl`, `app.py` and `requirements.txt` to reproduce.')

