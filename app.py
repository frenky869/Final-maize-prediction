import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import json

# --- Page Config ---
st.set_page_config(page_title="Farmer Price Guide", page_icon="⌰", layout="wide")

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')

# --- Custom Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
html, body, [class*=\"css\"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #14532d 0%, #166534 45%, #ca8a04 100%);
    padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem; text-align: center;
}
.kpi-card {
    background: #fdfcf0; border: 2px solid #ca8a04; border-radius: 15px;
    padding: 1.5rem; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.price-val { font-size: 2.8rem; font-weight: 700; color: #14532d; }
.action-advice { font-size: 1.5rem; font-weight: 700; padding: 10px; border-radius: 8px; }
.welcome-box { background: #f0fdf4; padding: 3rem; border-radius: 20px; text-align: center; border: 2px solid #16a34a; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_data():
    p = lambda f: os.path.join(DASHBOARD_DIR, f)
    data = {}
    data['panel'] = pd.read_csv(p("panel_clean.csv"), parse_dates=["week_start"])
    data['forecast'] = pd.read_csv(p("forecast.csv"), parse_dates=["week_start"])
    with open(p("metadata.json")) as f: data['meta'] = json.load(f)
    return data

try:
    bundle = load_data()
    panel_data = bundle['panel']
    forecast_data = bundle['forecast']
    counties = sorted(panel_data['county'].unique())
except:
    st.error("☑ Data missing. Please run the export cell in the notebook.")
    st.stop()

# --- Session State for Onboarding ---
if 'onboarding_step' not in st.session_state:
    st.session_state.onboarding_step = 1
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'user_role' not in st.session_state:
    st.session_state.user_role = "Farmer"

# --- ONBOARDING FLOW ---
if st.session_state.onboarding_step == 1:
    st.markdown("""
    <div class='welcome-box'>
        <h1>🌽 Karibu! Welcome to the Farmer Price Guide</h1>
        <p style='font-size: 1.2rem;'>This tool helps you know maize prices today and in the future so you can make more profit.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("##")
    if st.button("Get Started / Anza Sasa ➔", use_container_width=True):
        st.session_state.onboarding_step = 2
        st.rerun()

elif st.session_state.onboarding_step == 2:
    st.markdown("<h2 style='text-align: center;'>Let's set up your profile</h2>", unsafe_allow_html=True)
    with st.container():
        col_left, col_mid, col_right = st.columns([1, 2, 1])
        with col_mid:
            name = st.text_input("Your Name / Jina Lako", value=st.session_state.user_name)
            role = st.selectbox("Your Role / Kazi Yako", ["Farmer", "Trader", "Cooperative Member"])
            if st.button("Finish Setup / Maliza ➔", use_container_width=True):
                if name.strip():
                    st.session_state.user_name = name
                    st.session_state.user_role = role
                    st.session_state.onboarding_step = 3
                    st.rerun()
                else:
                    st.warning("Please enter your name to continue.")

# --- MAIN DASHBOARD ---
elif st.session_state.onboarding_step == 3:
    # Sidebar info
    st.sidebar.markdown(f"### 👤 Profile\n**Name:** {st.session_state.user_name}\n**Role:** {st.session_state.user_role}")
    if st.sidebar.button("Reset Profile"):
        st.session_state.onboarding_step = 1
        st.rerun()

    # Header
    st.markdown(f"""
    <div class='main-header'>
        <h1>⌰ Welcome, {st.session_state.user_name}</h1>
        <p>Know Maize Prices Before You Sell | Plan your sales with confidence</p>
    </div>
    """, unsafe_allow_html=True)

    # User Selection
    selected_county = st.selectbox(" Chagua Eneo Lako / Select Your Area", counties)

    PEAK_MONTHS = {
        "Kiambu": "Mar (Month 3)", "Kirinyaga": "Jun (Month 6)", 
        "Mombasa": "Mar (Month 3)", "Nairobi": "Jul (Month 7)", "Uasin-Gishu": "Jul (Month 7)"
    }

    hist = panel_data[panel_data['county'] == selected_county].sort_values('week_start')
    fore = forecast_data[forecast_data['county'] == selected_county].sort_values('week_start')
    current_price = hist['agri_price'].iloc[-1]
    future_price = fore['predicted_price'].iloc[4] # 4 weeks ahead
    trend_val = future_price - current_price

    # KPI Section
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi-card'><div style='color:#6b7280'>Bei ya Sasa<br>(Price Today)</div><div class='price-val'>KES {current_price:.1f}</div></div>", unsafe_allow_html=True)
    with c2:
        color = "#16a34a" if trend_val > 0 else "#dc2626"
        st.markdown(f"<div class='kpi-card'><div style='color:#6b7280'>Bei ya Baadaye<br>(Expected Soon)</div><div class='price-val' style='color:{color}'>KES {future_price:.1f}</div></div>", unsafe_allow_html=True)
    with c3:
        if trend_val > 2:
            advice, bg = "ፃ WEKA GHALA (Store & Wait)", "#dcfce7"
        elif trend_val < -2:
            advice, bg = "☑ UZA SASA (Sell Now)", "#fee2e2"
        else:
            advice, bg = "⚖ TULIA (Stable)", "#fef3c7"
        st.markdown(f"<div class='kpi-card' style='background:{bg}'><div style='color:#6b7280'>Ushauri Wetu<br>(Our Advice)</div><div class='action-advice'>{advice}</div></div>", unsafe_allow_html=True)

    # Interactive Chart
    st.write("### ን Mwongozo wa Bei / Price Guide")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['week_start'].tail(20), y=hist['agri_price'].tail(20), name="Past Prices", line=dict(color='#14532d', width=4)))
    fig.add_trace(go.Scatter(x=fore['week_start'], y=fore['predicted_price'], name="Future Prediction", line=dict(dash='dash', color='#ca8a04', width=4)))
    fig.update_layout(xaxis_title="Date", yaxis_title="KES/kg", height=400, margin=dict(l=0,r=0,b=0,t=40))
    st.plotly_chart(fig, use_container_width=True)

    # County Tip
    peak_info = PEAK_MONTHS.get(selected_county, "N/A")
    st.info(f"ፄ **Insight:** In {selected_county}, prices typically peak in **{peak_info}**. Plan your harvest and storage accordingly.")

    # WhatsApp Share
    st.button("ሀ Shiriki na kikundi cha WhatsApp / Share with Group")
