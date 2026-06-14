import os, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')

st.set_page_config(page_title="Farmer Price Guide", page_icon="⌰", layout="wide")

# Farmer-First Styling: High Contrast and Actionable
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
except:
    st.error("☑ Data missing. Please run the export cell in the notebook to sync the dashboard.")
    st.stop()

# Header
st.markdown("""
<div class='main-header'>
    <h1>⌰ Know Maize Prices Before You Sell</h1>
    <p>Panga mauzo yako kwa ujasiri | Plan your sales with confidence</p>
</div>
""", unsafe_allow_html=True)

# User Selection
counties = sorted(bundle['panel']['county'].unique())
selected_county = st.selectbox("ጁ Chagua Eneo Lako / Select Your Area", counties)

# Peak Month Summary (Specific Data)
PEAK_MONTHS = {
    "Kiambu": "Mar (Month 3)",
    "Kirinyaga": "Jun (Month 6)",
    "Mombasa": "Mar (Month 3)",
    "Nairobi": "Jul (Month 7)",
    "Uasin-Gishu": "Jul (Month 7)"
}

# Logic for Actionable Advice
hist = bundle['panel'][bundle['panel']['county'] == selected_county].sort_values('week_start')
fore = bundle['forecast'][bundle['forecast']['county'] == selected_county].sort_values('week_start')
current_price = hist['agri_price'].iloc[-1]
future_price = fore['predicted_price'].iloc[4] # Look 4 weeks ahead
trend_val = future_price - current_price

# Dashboard Layout
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='kpi-card'><div style='color:#6b7280'>Bei ya Sasa<br>(Price Today)</div><div class='price-val'>KES {current_price:.1f}</div><div style='font-size:0.8rem'>kwa kila kilo</div></div>", unsafe_allow_html=True)

with c2:
    color = "#16a34a" if trend_val > 0 else "#dc2626"
    st.markdown(f"<div class='kpi-card'><div style='color:#6b7280'>Bei ya Baadaye<br>(Expected Soon)</div><div class='price-val' style='color:{color}'>KES {future_price:.1f}</div><div style='font-size:0.8rem'>mwezi ujao</div></div>", unsafe_allow_html=True)

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
fig.update_layout(xaxis_title="Weeks", yaxis_title="KES/kg", height=400, margin=dict(l=0,r=0,b=0,t=40))
st.plotly_chart(fig, use_container_width=True)

# WhatsApp Share Placeholder
st.markdown("--- ")
st.button("ሀ Shiriki na kikundi cha WhatsApp / Share with Group")

# County Tip with Specific Peak Month Data
peak_info = PEAK_MONTHS.get(selected_county, "N/A")
st.info(f"ፄ **Insight:** In {selected_county}, prices typically peak in **{peak_info}**. Plan your harvest and storage accordingly to maximize profit.")
