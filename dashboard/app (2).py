"""
Kenya Maize Price Forecasting Dashboard
Streamlit app — dashboard subfolder
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import os
import joblib
from datetime import timedelta

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Maize Price Forecasting",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #0C447C;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f7ff; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #1D9E75;
    }
    .section-title {
        font-size: 1.25rem; font-weight: 600; color: #0C447C;
        border-bottom: 2px solid #1D9E75; padding-bottom: 4px; margin-bottom: 1rem;
    }
    .stSelectbox label, .stMultiSelect label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
COUNTY_RENAME = {
    "Uasin Gishu": "Uasin-Gishu", "Uasingishu": "Uasin-Gishu",
    "Nairobi City": "Nairobi", "Kiambu County": "Kiambu",
}
COLORS = {
    "actual": "#0C447C", "forecast": "#1D9E75",
    "kamis": "#E07B39", "agri": "#9B59B6",
}

# ── Data paths (relative to this file's location) ─────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ── Helper: load & cache data ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_raw_data():
    kamis_path   = os.path.join(DATA_DIR, "kamis_maize_prices.csv")
    agri_path    = os.path.join(DATA_DIR, "agriBORA_maize_prices.csv")
    weather_path = os.path.join(DATA_DIR, "weather_kenya_all_counties_2021_2025 (2).csv")
    kamis_df   = pd.read_csv(kamis_path)
    agri_df    = pd.read_csv(agri_path)
    weather_df = pd.read_csv(weather_path)
    return kamis_df, agri_df, weather_df


def clean_kamis(df):
    d = df[df["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
    d.columns = d.columns.str.strip().str.lower().str.replace(" ", "_")
    d["county"] = d["county"].str.strip().str.title().replace(COUNTY_RENAME)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    d["wholesale"] = pd.to_numeric(d["wholesale"], errors="coerce")
    d = d[d["wholesale"].between(10, 500, inclusive="both")]
    d["wholesale"] = d.groupby("county")["wholesale"].transform(lambda x: x.ffill().bfill())
    d["wholesale"] = d["wholesale"].fillna(d["wholesale"].median())
    dup_key = ["county", "date", "market"] if "market" in d.columns else ["county", "date"]
    d = d.drop_duplicates(subset=dup_key, keep="first")
    return d


def clean_agri(df):
    d = df[df["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
    d.columns = d.columns.str.strip().str.lower().str.replace(" ", "_")
    d["county"] = d["county"].str.strip().str.title().replace(COUNTY_RENAME)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    d["wholesale"] = pd.to_numeric(d["wholesale"], errors="coerce")
    d = d[d["wholesale"].between(10, 500, inclusive="both")]
    d["wholesale"] = d.groupby("county")["wholesale"].transform(lambda x: x.ffill().bfill())
    d["wholesale"] = d["wholesale"].fillna(d["wholesale"].median())
    d = d.drop_duplicates(subset=["county", "date"], keep="first")
    return d


def clean_weather(df):
    d = df.copy()
    d.columns = d.columns.str.strip().str.lower().str.replace(" ", "_")
    d["county"] = d["county"].str.strip().str.title().replace(COUNTY_RENAME)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    for col in ["temp_max_c", "temp_min_c", "temp_avg_c"]:
        if col in d.columns:
            d[col] = d.groupby("county")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )
            d[col] = d[col].fillna(d[col].mean())
    for col in ["rain_mm", "precipitation_mm", "precipitation_hours"]:
        if col in d.columns:
            d[col] = d[col].fillna(0)
    if "temp_max_c" in d.columns: d["temp_max_c"] = d["temp_max_c"].clip(10, 40)
    if "temp_min_c" in d.columns: d["temp_min_c"] = d["temp_min_c"].clip(5, 30)
    if "temp_avg_c" in d.columns: d["temp_avg_c"] = d["temp_avg_c"].clip(8, 35)
    if "rain_mm"    in d.columns: d["rain_mm"]    = d["rain_mm"].clip(upper=200)
    d = d.drop_duplicates(subset=["county", "date"], keep="first")
    return d


@st.cache_data(show_spinner="Building panel…")
def build_panel(kamis_df, agri_df, weather_df):
    kamis_c   = clean_kamis(kamis_df)
    agri_c    = clean_agri(agri_df)
    weather_c = clean_weather(weather_df)

    kf = kamis_c[kamis_c["county"].isin(TARGET_COUNTIES)].copy()
    af = agri_c[agri_c["county"].isin(TARGET_COUNTIES)].copy()
    wf = weather_c[weather_c["county"].isin(TARGET_COUNTIES)].copy()

    for df_ in [kf, af, wf]:
        df_["week_start"] = df_["date"].dt.to_period("W").apply(lambda p: p.start_time)

    kamis_w = kf.groupby(["county", "week_start"])["wholesale"].mean().reset_index()
    kamis_w.columns = ["county", "week_start", "kamis_price"]

    agri_w = af.groupby(["county", "week_start"])["wholesale"].mean().reset_index()
    agri_w.columns = ["county", "week_start", "agri_price"]

    weather_agg_cols = {c: ("mean" if "temp" in c else "sum" if c == "rain_mm" else "mean")
                        for c in ["temp_avg_c", "temp_max_c", "temp_min_c", "rain_mm"]
                        if c in wf.columns}
    weather_w = wf.groupby(["county", "week_start"]).agg(weather_agg_cols).reset_index()

    panel = kamis_w.merge(agri_w, on=["county", "week_start"], how="outer")
    panel = panel.merge(weather_w, on=["county", "week_start"], how="left")
    panel = panel.sort_values(["county", "week_start"])

    for county in TARGET_COUNTIES:
        mask = panel["county"] == county
        panel.loc[mask, "kamis_price"] = panel.loc[mask, "kamis_price"].ffill().bfill()
        panel.loc[mask, "agri_price"]  = panel.loc[mask, "agri_price"].ffill().bfill()
        for col in ["temp_avg_c", "temp_max_c", "temp_min_c", "rain_mm"]:
            if col in panel.columns:
                panel.loc[mask, col] = panel.loc[mask, col].ffill().bfill()

    panel["price"] = panel["agri_price"].fillna(panel["kamis_price"])

    # Smoothing
    panel["kamis_smooth"] = panel.groupby("county")["kamis_price"].transform(
        lambda x: x.rolling(4, min_periods=1, center=True).mean()
    )
    panel["agri_smooth"] = panel.groupby("county")["agri_price"].transform(
        lambda x: x.rolling(4, min_periods=1, center=True).mean()
    )

    # Lag & rolling features
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    for lag in [1, 2, 3, 4, 8]:
        panel[f"price_lag_{lag}"] = panel.groupby("county")["price"].shift(lag)
    for win in [4, 8, 12]:
        panel[f"price_rolling_mean_{win}"] = panel.groupby("county")["price"].transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        panel[f"price_rolling_std_{win}"] = panel.groupby("county")["price"].transform(
            lambda x: x.rolling(win, min_periods=1).std()
        )

    panel["month"]     = panel["week_start"].dt.month
    panel["week"]      = panel["week_start"].dt.isocalendar().week.astype(int)
    panel["year"]      = panel["week_start"].dt.year
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)
    panel["week_sin"]  = np.sin(2 * np.pi * panel["week"] / 52)
    panel["week_cos"]  = np.cos(2 * np.pi * panel["week"] / 52)

    panel_clean = panel.dropna()
    return panel, panel_clean


# ── Model training ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model…")
def train_model(panel_clean):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import GradientBoostingRegressor
    import xgboost as xgb

    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    TARGET = "agri_price"
    data = panel_clean.copy()
    le = LabelEncoder()
    data["county_encoded"] = le.fit_transform(data["county"])

    EXCLUDE = ["county", "week_start", TARGET, "agri_std", "kamis_std",
               "price", "kamis_smooth", "agri_smooth", "kamis_price"]
    feature_cols = [c for c in data.columns
                    if c not in EXCLUDE and not c.startswith("county_") and c != "county_encoded"] \
                   + ["county_encoded"]

    data = data.sort_values("week_start").reset_index(drop=True)
    split_idx = int(len(data) * 0.80)
    train_df = data.iloc[:split_idx]
    test_df  = data.iloc[split_idx:]

    X_train = train_df[feature_cols].fillna(0)
    X_test  = test_df[feature_cols].fillna(0)
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    candidates = {}

    # XGBoost
    m_xgb = xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.8,
                               min_child_weight=3, random_state=42,
                               n_jobs=-1, verbosity=0)
    m_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = m_xgb.predict(X_test)
    candidates["XGBoost"] = (m_xgb, np.mean(np.abs(y_test.values - preds)))

    # LightGBM
    if HAS_LGB:
        m_lgb = lgb.LGBMRegressor(n_estimators=400, max_depth=8, learning_rate=0.04,
                                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                    min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)
        m_lgb.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        preds_lgb = m_lgb.predict(X_test)
        candidates["LightGBM"] = (m_lgb, np.mean(np.abs(y_test.values - preds_lgb)))

    # Gradient Boosting
    m_gb = GradientBoostingRegressor(n_estimators=300, max_depth=6,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
    m_gb.fit(X_train, y_train)
    preds_gb = m_gb.predict(X_test)
    candidates["Gradient Boosting"] = (m_gb, np.mean(np.abs(y_test.values - preds_gb)))

    best_name = min(candidates, key=lambda k: candidates[k][1])
    best_model = candidates[best_name][0]

    return best_model, best_name, le, feature_cols, TARGET, test_df, X_test, y_test


# ── Forecasting ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating forecasts…")
def generate_forecast(_model, _le, feature_cols, panel_clean, forecast_weeks, _target="agri_price"):
    TARGET = _target
    data = panel_clean.copy()
    last_date = data["week_start"].max()

    future_rows = []
    for i in range(forecast_weeks):
        nw = last_date + timedelta(weeks=i + 1)
        for county in TARGET_COUNTIES:
            future_rows.append({"week_start": nw, "county": county})
    future_df = pd.DataFrame(future_rows)
    for col in feature_cols:
        if col != "county_encoded":
            future_df[col] = np.nan
    future_df["county_encoded"] = _le.transform(future_df["county"])

    full = pd.concat([data, future_df], ignore_index=True)
    full["week_start"] = pd.to_datetime(full["week_start"])
    full = full.sort_values(["county", "week_start"]).reset_index(drop=True)
    full[TARGET] = pd.to_numeric(full[TARGET], errors="coerce")

    future_predictions = []
    for i in range(forecast_weeks):
        cur_date = last_date + timedelta(weeks=i + 1)
        for county in TARGET_COUNTIES:
            mask = (full["county"] == county) & (full["week_start"] == cur_date)
            idx  = full[mask].index[0]

            full.loc[idx, "month"]     = cur_date.month
            full.loc[idx, "week"]      = cur_date.isocalendar().week
            full.loc[idx, "year"]      = cur_date.year
            full.loc[idx, "month_sin"] = np.sin(2 * np.pi * cur_date.month / 12)
            full.loc[idx, "month_cos"] = np.cos(2 * np.pi * cur_date.month / 12)
            full.loc[idx, "week_sin"]  = np.sin(2 * np.pi * cur_date.isocalendar().week / 52)
            full.loc[idx, "week_cos"]  = np.cos(2 * np.pi * cur_date.isocalendar().week / 52)

            for wc in ["temp_avg_c", "temp_max_c", "temp_min_c", "rain_mm"]:
                if wc in full.columns:
                    full.loc[idx, wc] = panel_clean[panel_clean["county"] == county][wc].mean() \
                        if wc in panel_clean.columns else 0.0

            for lag in [1, 2, 3, 4, 8]:
                lag_date = cur_date - timedelta(weeks=lag)
                lag_val = full.loc[(full["county"] == county) &
                                   (full["week_start"] == lag_date), TARGET]
                full.loc[idx, f"price_lag_{lag}"] = lag_val.iloc[0] if not lag_val.empty else 0.0

            cdf = full[(full["county"] == county) &
                       (full["week_start"] <= cur_date)][TARGET].astype(float)
            for win in [4, 8, 12]:
                full.loc[idx, f"price_rolling_mean_{win}"] = cdf.rolling(win, min_periods=1).mean().iloc[-1]
                full.loc[idx, f"price_rolling_std_{win}"]  = cdf.rolling(win, min_periods=1).std().fillna(0).iloc[-1]

            X_row = full.loc[idx, feature_cols].to_frame().T
            X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(0)
            pred  = _model.predict(X_row)[0]

            full.loc[idx, TARGET] = pred
            future_predictions.append({
                "county": county, "week_start": cur_date, "predicted_price": pred
            })

    return pd.DataFrame(future_predictions)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown('<div class="main-header">🌽 Kenya Maize Price Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">White Maize Wholesale Price Intelligence — Kiambu · Kirinyaga · Mombasa · Nairobi · Uasin-Gishu</div>', unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/200px-Flag_of_Kenya.svg.png", width=120)
        st.markdown("### ⚙️ Controls")

        selected_counties = st.multiselect(
            "Counties to display",
            options=TARGET_COUNTIES,
            default=TARGET_COUNTIES,
        )
        forecast_weeks = st.slider("Forecast horizon (weeks)", min_value=4, max_value=26, value=12, step=4)
        show_ci = st.checkbox("Show ±10% confidence band", value=True)
        st.markdown("---")
        st.markdown("**Data sources**")
        st.markdown("- KAMIS wholesale prices\n- AgriBORA transaction prices\n- Open-Meteo weather (2021–2025)")
        st.markdown("---")
        st.caption("Maize Prediction Project · 2025")

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        kamis_df, agri_df, weather_df = load_raw_data()
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}\n\nMake sure the `data/` folder contains:\n- `kamis_maize_prices.csv`\n- `agriBORA_maize_prices.csv`\n- `weather_kenya_all_counties_2021_2025 (2).csv`")
        return

    panel, panel_clean = build_panel(kamis_df, agri_df, weather_df)
    model, model_name, le, feature_cols, TARGET, test_df, X_test, y_test = train_model(panel_clean)

    # ── Top KPIs ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Current Price Snapshot</div>', unsafe_allow_html=True)

    latest_prices = (
        panel_clean.sort_values("week_start")
        .groupby("county")
        .last()
        .reset_index()[["county", "agri_price", "kamis_price"]]
    )
    cols = st.columns(len(TARGET_COUNTIES))
    for i, county in enumerate(TARGET_COUNTIES):
        row = latest_prices[latest_prices["county"] == county]
        if not row.empty:
            price = row["agri_price"].values[0]
            with cols[i]:
                st.metric(label=county, value=f"KES {price:.0f}", delta=None)

    st.markdown("---")

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Price Trends", "🔮 Forecast", "🏆 Model Performance", "🌤️ Weather Impact"]
    )

    # ═══════════════════════════════════
    # TAB 1 — PRICE TRENDS
    # ═══════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Historical Wholesale Prices</div>', unsafe_allow_html=True)

        view_option = st.radio("Price source", ["AgriBORA", "KAMIS", "Both"], horizontal=True)

        fig, axes = plt.subplots(
            len(selected_counties), 1,
            figsize=(12, 3.5 * len(selected_counties)),
            sharex=False,
        )
        if len(selected_counties) == 1:
            axes = [axes]

        for ax, county in zip(axes, selected_counties):
            cdf = panel[panel["county"] == county].sort_values("week_start")
            if view_option in ("AgriBORA", "Both"):
                ax.plot(cdf["week_start"], cdf["agri_price"],
                        color=COLORS["agri"], alpha=0.35, linewidth=1)
                ax.plot(cdf["week_start"], cdf["agri_smooth"],
                        color=COLORS["agri"], linewidth=2, label="AgriBORA (smoothed)")
            if view_option in ("KAMIS", "Both"):
                ax.plot(cdf["week_start"], cdf["kamis_price"],
                        color=COLORS["kamis"], alpha=0.35, linewidth=1)
                ax.plot(cdf["week_start"], cdf["kamis_smooth"],
                        color=COLORS["kamis"], linewidth=2, label="KAMIS (smoothed)")
            ax.set_title(county, fontsize=12, fontweight="bold")
            ax.set_ylabel("Price (KES)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        fig.suptitle("Kenya White Maize Wholesale Prices by County", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Seasonality
        st.markdown('<div class="section-title">Monthly Seasonality</div>', unsafe_allow_html=True)
        monthly = (
            panel_clean[panel_clean["county"].isin(selected_counties)]
            .groupby(["county", "month"])["agri_price"]
            .mean()
            .reset_index()
        )
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        palette = sns.color_palette("tab10", len(selected_counties))
        for idx, county in enumerate(selected_counties):
            cdf = monthly[monthly["county"] == county].sort_values("month")
            ax2.plot(cdf["month"], cdf["agri_price"], marker="o",
                     label=county, color=palette[idx], linewidth=2)
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(month_labels)
        ax2.set_ylabel("Average Price (KES)")
        ax2.set_title("Average Price by Month", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ═══════════════════════════════════
    # TAB 2 — FORECAST
    # ═══════════════════════════════════
    with tab2:
        st.markdown(f'<div class="section-title">🔮 {forecast_weeks}-Week Price Forecast  ·  Model: {model_name}</div>', unsafe_allow_html=True)

        forecast_df = generate_forecast(model, le, feature_cols, panel_clean, forecast_weeks, TARGET)

        fig3, axes3 = plt.subplots(
            len(selected_counties), 1,
            figsize=(12, 3.8 * len(selected_counties)),
        )
        if len(selected_counties) == 1:
            axes3 = [axes3]

        for ax, county in zip(axes3, selected_counties):
            hist = panel_clean[panel_clean["county"] == county].sort_values("week_start").tail(52)
            fore = forecast_df[forecast_df["county"] == county].sort_values("week_start")

            ax.plot(hist["week_start"], hist["agri_price"],
                    color=COLORS["actual"], linewidth=2, label="Historical (AgriBORA)")
            ax.plot(fore["week_start"], fore["predicted_price"],
                    color=COLORS["forecast"], linewidth=2.5, linestyle="--", label="Forecast")

            if show_ci:
                ax.fill_between(
                    fore["week_start"],
                    fore["predicted_price"] * 0.90,
                    fore["predicted_price"] * 1.10,
                    alpha=0.18, color=COLORS["forecast"], label="±10% band",
                )

            # Vertical line at forecast start
            ax.axvline(hist["week_start"].max(), color="gray",
                       linestyle=":", linewidth=1.2, alpha=0.7)

            ax.set_title(county, fontsize=12, fontweight="bold")
            ax.set_ylabel("Price (KES)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        fig3.suptitle(f"{forecast_weeks}-Week Maize Price Forecast by County", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Forecast table
        st.markdown("#### Forecast Data Table")
        pivot_fc = forecast_df[forecast_df["county"].isin(selected_counties)].copy()
        pivot_fc["week_start"] = pivot_fc["week_start"].dt.strftime("%Y-%m-%d")
        pivot_fc["predicted_price"] = pivot_fc["predicted_price"].round(2)
        pivot_table = pivot_fc.pivot(index="week_start", columns="county", values="predicted_price")
        st.dataframe(pivot_table, use_container_width=True)

        # Download button
        csv = pivot_table.reset_index().to_csv(index=False)
        st.download_button(
            label="⬇️ Download forecast CSV",
            data=csv,
            file_name=f"maize_forecast_{forecast_weeks}wk.csv",
            mime="text/csv",
        )

    # ═══════════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ═══════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

        test_preds = model.predict(X_test.fillna(0))
        test_out = test_df.copy().reset_index(drop=True)
        test_out["prediction"]  = test_preds
        test_out["abs_error"]   = np.abs(test_out[TARGET] - test_preds)
        test_out["pct_error"]   = test_out["abs_error"] / test_out[TARGET] * 100

        mae  = test_out["abs_error"].mean()
        rmse = np.sqrt((test_out["abs_error"] ** 2).mean())
        mape = test_out["pct_error"].mean()
        r2   = 1 - np.sum((test_out[TARGET] - test_out["prediction"])**2) / \
                   np.sum((test_out[TARGET] - test_out[TARGET].mean())**2)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE",  f"KES {mae:.2f}")
        k2.metric("RMSE", f"KES {rmse:.2f}")
        k3.metric("MAPE", f"{mape:.1f}%")
        k4.metric("R²",   f"{r2:.4f}")

        # Per-county metrics
        st.markdown("#### Per-County Test Performance")
        county_metrics = test_out[test_out["county"].isin(selected_counties)].groupby("county").agg(
            Actual_Mean  =(TARGET, "mean"),
            Predicted_Mean=("prediction", "mean"),
            MAE=("abs_error", "mean"),
            MAPE=("pct_error", "mean"),
            Weeks=("abs_error", "count"),
        ).round(2)
        county_metrics["Bias_KES"] = (county_metrics["Predicted_Mean"] - county_metrics["Actual_Mean"]).round(2)
        st.dataframe(county_metrics, use_container_width=True)

        # Actual vs predicted chart
        st.markdown("#### Actual vs Predicted (Test Set)")
        fig4, axes4 = plt.subplots(
            len(selected_counties), 1,
            figsize=(12, 3.5 * len(selected_counties)),
        )
        if len(selected_counties) == 1:
            axes4 = [axes4]

        for ax, county in zip(axes4, selected_counties):
            cdf = test_out[test_out["county"] == county].sort_values("week_start")
            ax.plot(cdf["week_start"], cdf[TARGET],
                    color=COLORS["actual"], linewidth=2, label="Actual")
            ax.plot(cdf["week_start"], cdf["prediction"],
                    color=COLORS["forecast"], linewidth=2, linestyle="--", label="Predicted")
            mae_c  = cdf["abs_error"].mean()
            mape_c = cdf["pct_error"].mean()
            ax.set_title(f"{county}  |  MAE={mae_c:.2f} KES  MAPE={mape_c:.1f}%",
                         fontsize=11, fontweight="bold")
            ax.set_ylabel("Price (KES)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        fig4.suptitle(f"Actual vs Predicted — {model_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        # Feature importance
        if hasattr(model, "feature_importances_"):
            st.markdown("#### Top 15 Feature Importances")
            fi_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False).head(15)

            fig5, ax5 = plt.subplots(figsize=(10, 5))
            colors_fi = ["#1D9E75" if i < 3 else "#B4B2A9" for i in range(len(fi_df))]
            ax5.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
                     color=colors_fi[::-1], edgecolor="black", linewidth=0.5)
            ax5.set_title(f"Feature Importance — {model_name}", fontsize=12, fontweight="bold")
            ax5.set_xlabel("Importance Score")
            ax5.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()

    # ═══════════════════════════════════
    # TAB 4 — WEATHER IMPACT
    # ═══════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Weather & Price Relationship</div>', unsafe_allow_html=True)

        weather_cols_avail = [c for c in ["temp_avg_c", "rain_mm"] if c in panel_clean.columns]
        if not weather_cols_avail:
            st.info("Weather columns not found in the processed panel.")
        else:
            selected_county_w = st.selectbox("Select county", TARGET_COUNTIES, key="weather_county")
            cdf_w = panel_clean[panel_clean["county"] == selected_county_w].sort_values("week_start")

            fig6, axes6 = plt.subplots(len(weather_cols_avail) + 1, 1,
                                        figsize=(12, 4 * (len(weather_cols_avail) + 1)))
            if not isinstance(axes6, np.ndarray):
                axes6 = [axes6]

            # Price panel
            axes6[0].plot(cdf_w["week_start"], cdf_w["agri_price"],
                          color=COLORS["actual"], linewidth=2)
            axes6[0].set_title(f"{selected_county_w} — AgriBORA Price", fontsize=11, fontweight="bold")
            axes6[0].set_ylabel("Price (KES)")
            axes6[0].grid(True, alpha=0.3)

            labels = {"temp_avg_c": "Avg Temp (°C)", "rain_mm": "Weekly Rainfall (mm)"}
            colors_w = {"temp_avg_c": "#E07B39", "rain_mm": "#3498DB"}
            for ax, col in zip(axes6[1:], weather_cols_avail):
                ax.plot(cdf_w["week_start"], cdf_w[col],
                        color=colors_w.get(col, "gray"), linewidth=1.5)
                ax.set_title(labels.get(col, col), fontsize=11, fontweight="bold")
                ax.set_ylabel(labels.get(col, col))
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

            plt.tight_layout()
            st.pyplot(fig6)
            plt.close()

            # Correlation table
            st.markdown("#### Price–Weather Correlation (all counties)")
            corr_rows = []
            for county in TARGET_COUNTIES:
                cdf_c = panel_clean[panel_clean["county"] == county]
                row = {"County": county}
                for wc in weather_cols_avail:
                    row[f"Price vs {labels.get(wc, wc)}"] = round(
                        cdf_c["agri_price"].corr(cdf_c[wc]), 3
                    )
                corr_rows.append(row)
            st.dataframe(pd.DataFrame(corr_rows).set_index("County"), use_container_width=True)

            # Price correlation heatmap
            st.markdown("#### County Price Correlation Heatmap")
            price_pivot = panel_clean.pivot_table(
                index="week_start", columns="county", values="agri_price"
            )
            corr_mat = price_pivot.corr()
            fig7, ax7 = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr_mat, annot=True, cmap="coolwarm", center=0,
                        fmt=".3f", linewidths=0.5, ax=ax7)
            ax7.set_title("County Price Correlation Matrix", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig7)
            plt.close()


if __name__ == "__main__":
    main()
