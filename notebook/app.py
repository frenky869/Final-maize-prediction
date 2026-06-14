import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import timedelta

# --- Page Config ---
st.set_page_config(page_title="Maize Price Forecast", page_icon="ፀ", layout="wide")

# --- Load Data Helper Functions ---
@st.cache_data
def load_raw_data():
    # Adjusted to point to the dashboard directory where artifacts are saved
    dashboard_dir = 'dashboard'
    panel_clean = pd.read_csv(os.path.join(dashboard_dir, 'panel_clean.csv'), parse_dates=['week_start'])
    forecast_df = pd.read_csv(os.path.join(dashboard_dir, 'forecast.csv'), parse_dates=['week_start'])
    test_results = pd.read_csv(os.path.join(dashboard_dir, 'test_results.csv'), parse_dates=['week_start'])
    return panel_clean, forecast_df, test_results

# --- Sidebar Navigation ---
st.sidebar.title("ፀ Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Weekly Prices", "Forecasts", "Model Performance", "Downloads"])

# Load datasets
try:
    panel_data, forecast_data, test_data = load_raw_data()
    counties = panel_data['county'].unique()
except Exception as e:
    st.error(f"Artifacts not found. Please ensure dashboard/ folder contains the CSV files. Error: {e}")
    st.stop()

# --- Section 1: Data Overview ---
if page == "Data Overview":
    st.title("ፀ Data Overview")
    st.write("Upload new data or preview the current processed dataset.")

    uploaded_file = st.file_uploader("Upload a new maize price CSV", type="csv")
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df_upload.head())

    st.subheader("Integrated Panel Data (Cleaned)")
    st.dataframe(panel_data.head(10))
    st.metric("Total Records", len(panel_data))

# --- Section 2: Weekly Prices ---
elif page == "Weekly Prices":
    st.title("ፂ Weekly Price Trends")
    selected_county = st.selectbox("Select County", counties)

    county_df = panel_data[panel_data['county'] == selected_county].sort_values('week_start')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=county_df['week_start'], y=county_df['kamis_price'], name='KAMIS Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=county_df['week_start'], y=county_df['agri_price'], name='AgriBORA Price', line=dict(color='orange')))

    fig.update_layout(title=f"KAMIS vs AgriBORA Weekly Prices in {selected_county}", xaxis_title="Date", yaxis_title="Price (KES)")
    st.plotly_chart(fig, use_container_width=True)

# --- Section 3: Forecasts ---
elif page == "Forecasts":
    st.title("ፁ Price Forecasts")
    selected_county = st.selectbox("Select County", counties)
    horizon = st.slider("Forecast Horizon (Weeks)", 4, 36, 12)

    hist_c = panel_data[panel_data['county'] == selected_county].tail(20)
    fore_c = forecast_data[forecast_data['county'] == selected_county].head(horizon)

    fig = px.line(title=f"Forecast for {selected_county}")
    fig.add_scatter(x=hist_c['week_start'], y=hist_c['agri_price'], name="Historical (AgriBORA)")
    fig.add_scatter(x=fore_c['week_start'], y=fore_c['predicted_price'], name="Forecast", line=dict(dash='dash', color='green'))

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Forecast Data Table")
    st.dataframe(fore_c[['week_start', 'predicted_price']].rename(columns={'predicted_price': 'Predicted Price (KES)'}))

# --- Section 4: Model Performance ---
elif page == "Model Performance":
    st.title("ፃ Model Performance Analysis")

    # In a real scenario, you'd load the results_df saved from the notebook
    st.write("Summary of model performance across test splits.")
    metrics = test_data.groupby('county').agg({
        'agri_price': 'mean',
        'prediction': 'mean',
        'abs_error': 'mean'
    }).rename(columns={'abs_error': 'MAE'})

    st.table(metrics)

    fig = px.bar(metrics.reset_index(), x='county', y='MAE', title="Mean Absolute Error by County", color='county')
    st.plotly_chart(fig, use_container_width=True)

# --- Section 5: Downloads ---
elif page == "Downloads":
    st.title("ፄ Export Data")

    csv_forecast = forecast_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Forecast as CSV", data=csv_forecast, file_name="maize_forecast.csv", mime="text/csv")

    csv_panel = panel_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cleaned Panel Data", data=csv_panel, file_name="panel_clean.csv", mime="text/csv")
