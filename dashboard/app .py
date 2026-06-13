import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

DASHBOARD_DIR = "/content/drive/MyDrive/MaizePrediction/dashboard"

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Kenya Maize Price Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌽 Kenya Maize Price Forecasting Dashboard")

# --- Load Artifacts ---
@st.cache_data
def load_artifacts():
    model = joblib.load(os.path.join(DASHBOARD_DIR, "model.pkl"))
    label_encoder = joblib.load(os.path.join(DASHBOARD_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(DASHBOARD_DIR, "feature_cols.pkl"))
    panel_clean = pd.read_csv(os.path.join(DASHBOARD_DIR, "panel_clean.csv"))
    test_results = pd.read_csv(os.path.join(DASHBOARD_DIR, "test_results.csv"))
    forecast_df = pd.read_csv(os.path.join(DASHBOARD_DIR, "forecast.csv"))
    with open(os.path.join(DASHBOARD_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    with open(os.path.join(DASHBOARD_DIR, "model_name.txt"), "r") as f:
        model_name = f.read().strip()

    panel_clean['week_start'] = pd.to_datetime(panel_clean['week_start'])
    test_results['week_start'] = pd.to_datetime(test_results['week_start'])
    forecast_df['week_start'] = pd.to_datetime(forecast_df['week_start'])

    return model, label_encoder, feature_cols, panel_clean, test_results, forecast_df, metadata, model_name

model, le, feature_cols, panel_clean, test_results, forecast_df, metadata, model_name = load_artifacts()

# --- Sidebar ---
st.sidebar.header("Dashboard Controls")
selected_county = st.sidebar.selectbox(
    "Select County for Detailed View:",
    options=panel_clean['county'].unique()
)

# --- Key Metrics ---
st.header("1. Model Performance Summary")
cols = st.columns(4)
for i, (metric, value) in enumerate(metadata['metrics'].items()):
    with cols[i]:
        st.metric(label=metric, value=value)

st.markdown(f"**Best Model:** {metadata['model']} (Trained on {metadata['trained_on']})")
st.info("These metrics reflect the model's performance on the unseen test data.")


# --- Feature Importance ---
st.header("2. Feature Importance")
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax_fi)
    ax_fi.set_title(f'Top 15 Feature Importances — {model_name}')
    ax_fi.set_xlabel('Importance Score')
    ax_fi.set_ylabel('')
    st.pyplot(fig_fi)
else:
    st.write("Feature importance not available for the selected model type.")


# --- Historical vs Predicted Prices for Selected County ---
st.header(f"3. Price Trends & Forecast for {selected_county}")

county_historical = panel_clean[panel_clean['county'] == selected_county].copy()
county_test = test_results[test_results['county'] == selected_county].copy()
county_forecast = forecast_df[forecast_df['county'] == selected_county].copy()

fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
ax_pred.plot(county_historical['week_start'], county_historical['agri_price'], label='Historical Actual (Training)', color='skyblue')
ax_pred.plot(county_test['week_start'], county_test['agri_price'], label='Historical Actual (Test)', color='steelblue', linestyle='-')
ax_pred.plot(county_test['week_start'], county_test['prediction'], label='Predicted (Test)', color='orange', linestyle='--')
ax_pred.plot(county_forecast['week_start'], county_forecast['predicted_price'], label='Future Forecast', color='green', linestyle='-.')

ax_pred.set_title(f'{selected_county} Maize Price: Historical, Test Predictions & Future Forecast')
ax_pred.set_xlabel('Date')
ax_pred.set_ylabel('Price (KES)')
ax_pred.legend()
ax_pred.grid(True)
st.pyplot(fig_pred)


# --- Forecast Table ---
st.subheader(f"Future Price Forecast for {selected_county}")
st.dataframe(county_forecast[['week_start', 'predicted_price']].rename(columns={'week_start': 'Week Start', 'predicted_price': 'Forecasted Price (KES)'}).set_index('Week Start'))


st.header("4. All Counties Forecast")
fig_all_forecast, ax_all_forecast = plt.subplots(figsize=(14, 8))
for county in panel_clean['county'].unique():
    county_df = forecast_df[forecast_df['county'] == county]
    ax_all_forecast.plot(county_df['week_start'], county_df['predicted_price'], label=county)
ax_all_forecast.set_title('Maize Price Forecast Across All Counties')
ax_all_forecast.set_xlabel('Date')
ax_all_forecast.set_ylabel('Predicted Price (KES)')
ax_all_forecast.legend()
ax_all_forecast.grid(True)
st.pyplot(fig_all_forecast)

st.markdown("---<br>_This dashboard was generated based on the maize price forecasting project._")
