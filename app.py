import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import json

# --- Page Config ---
st.set_page_config(page_title="Maize Price Forecast", page_icon="🌾", layout="wide")

# --- Path Configuration ---
# Ensure we look in the correct local directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')

@st.cache_resource
def load_model_artifacts():
    files = {
        'model': 'model.pkl',
        'le': 'label_encoder.pkl',
        'features': 'feature_columns.pkl',
        'panel': 'panel_clean.csv',
        'forecast': 'forecast.csv',
        'test': 'test_results.csv',
        'meta': 'metadata.json'
    }
    
    loaded = {}
    for key, filename in files.items():
        path = os.path.join(DASHBOARD_DIR, filename)
        if not os.path.exists(path):
            st.error(f"Missing required file: {path}")
            st.stop()
        
        if filename.endswith('.pkl'):
            loaded[key] = joblib.load(path)
        elif filename.endswith('.csv'):
            loaded[key] = pd.read_csv(path, parse_dates=['week_start'] if 'week_start' in pd.read_csv(path, nrows=1).columns else [])
        elif filename.endswith('.json'):
            with open(path, 'r') as f:
                loaded[key] = json.load(f)
    return loaded

# Load everything
data_bundle = load_model_artifacts()
panel_data = data_bundle['panel']
forecast_data = data_bundle['forecast']
test_data = data_bundle['test']
counties = panel_data['county'].unique()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Forecasts", "Data Overview", "Model Performance"])

if page == "Forecasts":
    st.title("Maize Price Forecasts")
    selected_county = st.selectbox("Select County", counties)
    
    hist_c = panel_data[panel_data['county'] == selected_county].tail(20)
    fore_c = forecast_data[forecast_data['county'] == selected_county]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_c['week_start'], y=hist_c['agri_price'], name="Historical (AgriBORA)"))
    fig.add_trace(go.Scatter(x=fore_c['week_start'], y=fore_c['predicted_price'], name="Forecast", line=dict(dash='dash', color='green')))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Forecast Table")
    st.dataframe(fore_c[['week_start', 'predicted_price']])

elif page == "Data Overview":
    st.title("Data Overview")
    st.dataframe(panel_data.head(50))

elif page == "Model Performance":
    st.title("Model Performance")
    metrics = test_data.groupby('county').agg({'abs_error': 'mean'}).rename(columns={'abs_error': 'MAE'})
    st.table(metrics)
