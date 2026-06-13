
import streamlit as st
import os
import joblib
import json
import pandas as pd
import plotly.express as px

st.set_page_config(layout='wide', page_title='Maize Price Forecast')

st.title('Kenya Maize Price Forecasting Dashboard')
st.write('This interactive dashboard presents the maize price forecasts and model insights.')

# --- Define paths to artifacts --- #
# We use a relative path for deployment compatibility
dashboard_dir = 'dashboard'

# --- Load Model Artifacts --- #
@st.cache_resource
def load_model_artifacts():
    try:
        # Verify directory exists
        if not os.path.exists(dashboard_dir):
            st.error(f"Dashboard directory '{dashboard_dir}' not found.")
            st.stop()
            
        model = joblib.load(os.path.join(dashboard_dir, 'model.pkl'))
        label_encoder = joblib.load(os.path.join(dashboard_dir, 'label_encoder.pkl'))
        feature_columns = joblib.load(os.path.join(dashboard_dir, 'feature_columns.pkl'))
        panel_clean = pd.read_csv(os.path.join(dashboard_dir, 'panel_clean.csv'), parse_dates=['week_start'])
        test_results = pd.read_csv(os.path.join(dashboard_dir, 'test_results.csv'), parse_dates=['week_start'])
        forecast_data = pd.read_csv(os.path.join(dashboard_dir, 'forecast.csv'), parse_dates=['week_start'])
        with open(os.path.join(dashboard_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        return model, label_encoder, feature_columns, panel_clean, test_results, forecast_data, metadata
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

model, le, feature_cols, panel_clean_data, test_out, forecast_df, metadata = load_model_artifacts()

best_model_name = metadata.get('best_model', 'N/A')
forecast_end_date = metadata.get('forecast_end_date', 'N/A')

# --- Sidebar --- #
st.sidebar.header('Settings')
selected_county = st.sidebar.selectbox('Select County', panel_clean_data['county'].unique())

# --- Main Content --- #
col1, col2 = st.columns(2)
with col1:
    st.metric("Best Model", best_model_name)
with col2:
    st.metric("Forecast End Date", forecast_end_date)

st.subheader(f'Price Trend and Forecast: {selected_county}')

historical_county = panel_clean_data[panel_clean_data['county'] == selected_county].sort_values('week_start')
forecast_county = forecast_df[forecast_df['county'] == selected_county].sort_values('week_start')

# Prepare Plotly Data
hist_plot = historical_county[['week_start', 'agri_price']].copy()
hist_plot['Type'] = 'Actual'
hist_plot.columns = ['Date', 'Price (KES)', 'Type']

fore_plot = forecast_county[['week_start', 'predicted_price']].copy()
fore_plot['Type'] = 'Forecast'
fore_plot.columns = ['Date', 'Price (KES)', 'Type']

plot_df = pd.concat([hist_plot, fore_plot])

fig = px.line(plot_df, x='Date', y='Price (KES)', color='Type', 
              title=f'Weekly Maize Prices in {selected_county}')
fig.update_layout(width=None) # Use container width handles this
st.plotly_chart(fig, use_container_width=True)

st.subheader('Model Evaluation (Test Data)')
county_metrics = test_out.groupby('county').agg({
    'agri_price': 'mean',
    'prediction': 'mean',
    'abs_error': 'mean',
    'pct_error': 'mean'
}).rename(columns={'abs_error': 'MAE', 'pct_error': 'MAPE%'}).round(2)

st.dataframe(county_metrics, use_container_width=True)
