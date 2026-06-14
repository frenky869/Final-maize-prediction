import os, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(page_title="Kenya Maize Price Predictor", page_icon="🌽", layout="wide")

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')

@st.cache_resource
def load_artifacts():
    files = {'panel': 'panel_clean.csv', 'forecast': 'forecast.csv', 'test': 'test_results.csv', 'results_df': 'results_df.csv', 'importance_df': 'importance_df.csv', 'meta': 'metadata.json'}
    loaded = {}
    for key, fn in files.items():
        path = os.path.join(DASHBOARD_DIR, fn)
        if not os.path.exists(path): continue
        if fn.endswith('.csv'):
            df = pd.read_csv(path)
            if 'week_start' in df.columns: df['week_start'] = pd.to_datetime(df['week_start'])
            loaded[key] = df
        elif fn.endswith('.json'):
            with open(path, 'r') as f: loaded[key] = json.load(f)
    return loaded

art = load_artifacts()
panel_data, forecast_data, test_data, metadata = art['panel'], art['forecast'], art['test'], art['meta']
counties = sorted(panel_data['county'].unique())

# --- UI Layout ---
st.title("🌽 Kenya Maize Price Forecasting")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Forecasts", "Model Performance", "Data Explorer"])

if page == "Forecasts":
    st.header("📈 36-Week Price Forecast")
    c = st.selectbox("Select County", counties)
    hist = panel_data[panel_data['county'] == c].tail(20)
    fore = forecast_data[forecast_data['county'] == c]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['week_start'], y=hist['agri_price'], name="Historical", line=dict(color='#0C447C')))
    fig.add_trace(go.Scatter(x=fore['week_start'], y=fore['predicted_price'], name="Forecast", line=dict(dash='dash', color='#1D9E75')))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.header("⚙️ Model Performance Metrics")
    best_model = metadata.get('best_model', 'N/A')
    res_df = art.get('results_df')
    if res_df is not None:
        best_metrics = res_df[(res_df['Model'] == best_model) & (res_df['Split'] == 'Test')].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Model", best_model)
        c2.metric("MAE", f"KES {best_metrics['MAE']:.2f}")
        c3.metric("R² Score", f"{best_metrics['R²']:.4f}")
    perf = test_data.groupby('county').agg({'agri_price': 'mean', 'prediction': 'mean', 'abs_error': 'mean'}).rename(columns={'abs_error': 'MAE'}).round(2)
    st.table(perf)

elif page == "Data Explorer":
    st.header("📋 Cleaned Dataset")
    st.dataframe(panel_data)
