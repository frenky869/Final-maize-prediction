
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime

st.set_page_config(page_title="Kenya Maize Price Forecast", layout="wide")

# Define standard colors
C_NAVY = '#0C447C'
C_GREEN = '#1D9E75'

# Improved helper for Plotly compatibility - ensures hex alpha is handled correctly
def hex_to_rgba(hex_code, opacity=0.2):
    hex_code = hex_code.lstrip('#')
    # If an 8-digit hex was passed, override the opacity with the hex alpha channel
    if len(hex_code) == 8:
        opacity = int(hex_code[6:8], 16) / 255.0
        hex_code = hex_code[:6]
    r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {opacity})'

@st.cache_data
def load_data():
    df = pd.read_csv('panel_clean.csv', parse_dates=['week_start'])
    forecast = pd.read_csv('forecast.csv', parse_dates=['week_start'])
    return df, forecast

try:
    df, forecast = load_data()

    st.title("🌽 Kenya Maize Price Forecasting Dashboard")
    st.markdown("--- updates based on latest Colab analysis ---")

    # 1. Price Ranking
    st.subheader("📊 Average Maize Prices by County (Ranked)")
    county_avg = df.groupby('county')['kamis_smooth'].mean().sort_values()
    fig_rank = px.bar(county_avg, orientation='h', 
                      labels={'value': 'Average Price (KES)', 'county': 'County'},
                      color=county_avg.index,
                      color_discrete_sequence=px.colors.qualitative.Prism)
    fig_rank.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_rank, use_container_width=True)

    # 2. Forecasting Visuals
    st.subheader("🔮 36-Week Price Forecast")
    selected_county = st.selectbox("Select County to View Forecast", df['county'].unique())

    hist_c = df[df['county'] == selected_county]
    fore_c = forecast[forecast['county'] == selected_county]

    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(x=hist_c['week_start'], y=hist_c['agri_price'], 
                                  name="Actual Historical", line=dict(color=C_NAVY, width=2)))
    fig_fore.add_trace(go.Scatter(x=fore_c['week_start'], y=fore_c['predicted_price'], 
                                  name="36-Week Forecast", line=dict(color=C_GREEN, dash='dash', width=2)))
    
    # Fixed fillcolor using the helper instead of string concatenation
    fig_fore.add_trace(go.Scatter(x=fore_c['week_start'], y=fore_c['predicted_price'], 
                                  fill='tozeroy', fillcolor=hex_to_rgba(C_GREEN, 0.1), 
                                  showlegend=False, line=dict(width=0)))

    fig_fore.update_layout(title=f"{selected_county} Price Projection", xaxis_title="Date", yaxis_title="Price (KES)", hovermode="x unified")
    st.plotly_chart(fig_fore, use_container_width=True)

    # 3. Monthly Peaks
    st.subheader("📅 Seasonal Monthly Patterns")
    df['month_name'] = df['week_start'].dt.strftime('%b')
    seasonal = df.groupby(['county', 'month_name'])['kamis_smooth'].mean().reset_index()
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig_season = px.line(seasonal, x='month_name', y='kamis_smooth', color='county',
                         category_orders={'month_name': month_order}, markers=True)
    fig_season.update_layout(yaxis_title="Avg Price (KES)", xaxis_title="Month")
    st.plotly_chart(fig_season, use_container_width=True)

except Exception as e:
    st.error(f"Error loading dashboard data: {e}")
