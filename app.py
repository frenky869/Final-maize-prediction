import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import json

# --- Page Config ---
st.set_page_config(page_title="Kenya Maize Price Forecasting", page_icon="🌾", layout="wide")

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')

@st.cache_resource
def load_model_artifacts():
    files = {
        'panel': 'panel_clean.csv',
        'forecast': 'forecast.csv',
        'test': 'test_results.csv',
        'results_df': 'results_df.csv',
        'importance_df': 'importance_df.csv',
        'meta': 'metadata.json'
    }
    
    loaded = {}
    for key, filename in files.items():
        path = os.path.join(DASHBOARD_DIR, filename)
        if not os.path.exists(path):
            if key in ['importance_df', 'results_df']: continue
            st.error(f"Missing required file: {path}")
            st.stop()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(path)
            if 'week_start' in df.columns: df['week_start'] = pd.to_datetime(df['week_start'])
            loaded[key] = df
        elif filename.endswith('.json'):
            with open(path, 'r') as f: loaded[key] = json.load(f)
    return loaded

bundle = load_model_artifacts()
panel_data = bundle['panel']
forecast_data = bundle['forecast']
test_data = bundle['test']
metadata = bundle['meta']
counties = sorted(panel_data['county'].unique())

# --- Sidebar ---
st.sidebar.title("🌾 Maize Forecast App")
st.sidebar.info(f"**Best Model:** {metadata.get('best_model')}\n**Trained:** {metadata.get('trained_date')}")
page = st.sidebar.radio("Navigation", ["Forecast Dashboard", "Historical Analysis", "Model Performance", "Data Explorer"])

if page == "Forecast Dashboard":
    st.title("📈 36-Week Price Forecasts")
    selected_county = st.selectbox("Select County", counties)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        hist_c = panel_data[panel_data['county'] == selected_county].tail(26)
        fore_c = forecast_data[forecast_data['county'] == selected_county]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_c['week_start'], y=hist_c['agri_price'], name="Historical", line=dict(color='#0C447C', width=3)))
        fig.add_trace(go.Scatter(x=fore_c['week_start'], y=fore_c['predicted_price'], name="Forecast", line=dict(dash='dash', color='#1D9E75', width=3)))
        fig.update_layout(title=f"Maize Price Forecast: {selected_county}", xaxis_title="Date", yaxis_title="Price (KES)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Forecast Table")
        st.dataframe(fore_c[['week_start', 'predicted_price']].rename(columns={'week_start': 'Date', 'predicted_price': 'Price'}), hide_index=True)

elif page == "Historical Analysis":
    st.title("📊 Historical Trends & Distribution")
    selected_county = st.selectbox("Select County", counties)
    c_data = panel_data[panel_data['county'] == selected_county]
    
    fig1 = px.line(c_data, x='week_start', y=['kamis_price', 'agri_price'], title=f"Price Source Comparison: {selected_county}")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.box(panel_data, x='county', y='agri_price', color='county', title="Price Distribution by County")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Model Performance":
    st.title("⚙️ Model Evaluation & Insights")
    if 'results_df' in bundle:
        fig3 = px.bar(bundle['results_df'][bundle['results_df']['Split'] == 'Test'], x='Model', y='MAE', color='Model', title="Model Comparison (Lower MAE is better)")
        st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("County-Level Accuracy")
    metrics = test_data.groupby('county').agg({'abs_error': 'mean', 'pct_error': 'mean'}).rename(columns={'abs_error': 'MAE', 'pct_error': 'MAPE %'}).round(2)
    st.table(metrics)
    
    if 'importance_df' in bundle:
        st.subheader("Feature Importance")
        fig4 = px.bar(bundle['importance_df'].head(10), x='Importance', y='Feature', orientation='h', title="Top 10 Predictors")
        st.plotly_chart(fig4, use_container_width=True)

elif page == "Data Explorer":
    st.title("📝 Data Explorer")
    st.dataframe(panel_data)
    st.download_button("Download CSV", panel_data.to_csv(index=False), "maize_data.csv")
