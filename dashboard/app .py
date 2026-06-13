"""
Kenya Maize Price Forecasting — Streamlit Dashboard
----------------------------------------------------
Loads pre-exported artifacts from the dashboard/ folder:
  dashboard/panel_clean.csv
  dashboard/test_results.csv
  dashboard/forecast.csv
  dashboard/metadata.json

Run with:  streamlit run app.py
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Maize Price Forecasting",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1D9E75; }
    h2 { color: #0C447C; border-bottom: 2px solid #1D9E75; padding-bottom: 4px; }
    h3 { color: #0C447C; }
    .metric-card {
        background: #f0f8f4;
        border-left: 4px solid #1D9E75;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_DIR = Path("dashboard")

@st.cache_data
def load_data():
    panel     = pd.read_csv(DASHBOARD_DIR / "panel_clean.csv",  parse_dates=["week_start"])
    test_res  = pd.read_csv(DASHBOARD_DIR / "test_results.csv", parse_dates=["week_start"])
    forecast  = pd.read_csv(DASHBOARD_DIR / "forecast.csv",     parse_dates=["week_start"])
    with open(DASHBOARD_DIR / "metadata.json") as f:
        meta = json.load(f)
    return panel, test_res, forecast, meta

try:
    panel, test_res, forecast, meta = load_data()
except FileNotFoundError as e:
    st.error(
        f"**Missing artifact file:** `{e.filename}`\n\n"
        "Make sure the `dashboard/` folder (exported from your Colab notebook) "
        "sits next to `app.py` in your GitHub repo."
    )
    st.stop()

TARGET_COUNTIES = sorted(panel["county"].unique().tolist())
PALETTE = ["#1D9E75", "#0C447C", "#E8A838", "#C0392B", "#8E44AD"]
COUNTY_COLOR = dict(zip(TARGET_COUNTIES, PALETTE))

# derived columns expected
if "kamis_smooth" not in panel.columns and "kamis_price" in panel.columns:
    panel["kamis_smooth"] = (
        panel.groupby("county")["kamis_price"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

panel["month"] = panel["week_start"].dt.month

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/320px-Flag_of_Kenya.svg.png",
        width=120,
    )
    st.title("🌽 Kenya Maize")
    st.caption("Price Forecasting Dashboard")

    st.markdown("---")
    st.subheader("Filters")
    selected_counties = st.multiselect(
        "Counties", TARGET_COUNTIES, default=TARGET_COUNTIES
    )
    if not selected_counties:
        selected_counties = TARGET_COUNTIES

    st.markdown("---")
    st.subheader("Model Info")
    st.success(f"**Model:** {meta['model']}")
    m = meta["metrics"]
    st.metric("MAE",  m["MAE"])
    st.metric("RMSE", m["RMSE"])
    st.metric("MAPE", m["MAPE"])
    st.metric("R²",   m["R2"])
    st.caption(f"Trained: {meta['trained_on']}")
    st.caption(f"Samples: {meta['n_samples']:,}")
    st.caption(f"Range: {meta['date_range']}")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌽 Kenya Maize Price Forecasting Dashboard")
st.caption(
    "Weekly wholesale white maize price analysis across 5 Kenyan counties · "
    "Data sources: KAMIS & AgriBORA"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Descriptive Analysis",
    "🔍 Diagnostic Analysis",
    "🤖 Model Performance",
    "📈 Price Forecast",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Descriptive Analysis")

    panel_f = panel[panel["county"].isin(selected_counties)]

    # ── Q1: Average prices ────────────────────────────────────────────────────
    st.subheader("Q1 · Average Wholesale Maize Prices by County")

    avg_prices = (
        panel_f.groupby("county")["kamis_smooth"]
        .agg(Mean="mean", Median="median", Std="std", Weeks="count")
        .round(2)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [COUNTY_COLOR.get(c, "#888") for c in avg_prices.index]
        bars = ax.bar(avg_prices.index, avg_prices["Mean"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title("Average Wholesale Price (KES)", fontweight="bold")
        ax.set_ylabel("KES / bag")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, avg_prices["Mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        plt.xticks(rotation=20, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        panel_f.boxplot(column="kamis_smooth", by="county", ax=ax, showfliers=False,
                        patch_artist=True,
                        boxprops=dict(facecolor="#d6eaf8"),
                        medianprops=dict(color="#1D9E75", linewidth=2))
        ax.set_title("Price Distribution by County (IQR, no outliers)", fontweight="bold")
        plt.suptitle("")
        ax.set_xlabel("County")
        ax.set_ylabel("KES / bag")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=20, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.dataframe(avg_prices.style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")

    # ── Q2: Highest / lowest ──────────────────────────────────────────────────
    st.subheader("Q2 · Highest & Lowest Average Prices")

    county_avg = panel_f.groupby("county")["kamis_smooth"].mean()
    highest = county_avg.idxmax()
    lowest  = county_avg.idxmin()

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Highest", highest, f"KES {county_avg[highest]:.2f}")
    c2.metric("📉 Lowest",  lowest,  f"KES {county_avg[lowest]:.2f}")
    c3.metric("💰 Gap", "", f"KES {county_avg[highest] - county_avg[lowest]:.2f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sorted_avg = county_avg.sort_values()
    colors_q2 = [
        "#e74c3c" if c == highest else "#2ecc71" if c == lowest else "#95a5a6"
        for c in sorted_avg.index
    ]
    sorted_avg.plot(kind="barh", color=colors_q2, edgecolor="black", ax=ax)
    ax.set_xlabel("Average Price (KES)")
    ax.set_title("Average Maize Prices by County (Ranked)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    for i, (_, v) in enumerate(sorted_avg.items()):
        ax.text(v + 0.3, i, f"KES {v:.1f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Q3: Monthly peaks ─────────────────────────────────────────────────────
    st.subheader("Q3 · Monthly Price Patterns & Peak Months")

    monthly = panel_f.groupby(["county", "month"])["kamis_smooth"].mean().reset_index()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    n_counties = len(selected_counties)
    ncols = 2
    nrows = (n_counties + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, county in enumerate(selected_counties):
        ax = axes[idx]
        cdf = monthly[monthly["county"] == county]
        ax.plot(cdf["month"], cdf["kamis_smooth"], marker="o", linewidth=2,
                color=COUNTY_COLOR.get(county, "#333"), markersize=7)
        ax.fill_between(cdf["month"], cdf["kamis_smooth"], alpha=0.15,
                        color=COUNTY_COLOR.get(county, "#333"))
        peak = int(cdf.loc[cdf["kamis_smooth"].idxmax(), "month"])
        ax.axvline(x=peak, color="red", linestyle="--", alpha=0.6,
                   label=f"Peak: {month_labels[peak-1]}")
        ax.set_title(county, fontweight="bold")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels, rotation=45, fontsize=8)
        ax.set_ylabel("KES / bag")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(selected_counties), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Monthly Price Patterns by County", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Q4 & Q5: Range + Records ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Q4 · Price Range by County")
        price_range = panel_f.groupby("county")["kamis_smooth"].agg(["min","max","std"]).round(2)
        price_range["range"] = price_range["max"] - price_range["min"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        price_range["range"].sort_values().plot(kind="barh", color="#1D9E75", edgecolor="black", ax=ax)
        ax.set_xlabel("Min-Max Range (KES)")
        ax.set_title("Price Volatility (Range)", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Q5 · Weekly Records per County")
        weekly_records = panel_f.groupby("county")["kamis_smooth"].count().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(weekly_records.index, weekly_records.values,
                      color=[COUNTY_COLOR.get(c, "#888") for c in weekly_records.index],
                      edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Number of Weeks")
        ax.set_title("Data Availability", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, weekly_records.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=20, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Diagnostic Analysis")
    panel_f = panel[panel["county"].isin(selected_counties)]

    # ── D1: Mombasa vs Uasin-Gishu ────────────────────────────────────────────
    st.subheader("D1 · Why are Mombasa Prices Higher than Uasin-Gishu?")

    d1_counties = ["Mombasa", "Uasin-Gishu"]
    available_d1 = [c for c in d1_counties if c in panel["county"].unique()]

    if len(available_d1) == 2:
        d1_data = {c: panel[panel["county"] == c] for c in available_d1}

        col1, col2, col3 = st.columns(3)
        mom_avg  = d1_data["Mombasa"]["kamis_smooth"].mean()
        uasin_avg = d1_data["Uasin-Gishu"]["kamis_smooth"].mean()
        col1.metric("Mombasa Avg", f"KES {mom_avg:.2f}")
        col2.metric("Uasin-Gishu Avg", f"KES {uasin_avg:.2f}")
        col3.metric("Price Premium", f"KES {mom_avg - uasin_avg:.2f}",
                    f"+{(mom_avg - uasin_avg) / uasin_avg * 100:.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        for c, color in zip(available_d1, ["#e74c3c", "#2ecc71"]):
            df_c = d1_data[c].sort_values("week_start")
            ax.plot(df_c["week_start"], df_c["kamis_smooth"], label=c, color=color, linewidth=2)
        ax.set_title("Price Trends Over Time", fontweight="bold")
        ax.set_ylabel("KES / bag")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        temp_data = panel[panel["county"].isin(available_d1)].groupby("county")["temp_avg_c"].mean()
        bars = ax.bar(temp_data.index, temp_data.values,
                      color=["#e74c3c", "#2ecc71"], edgecolor="black")
        ax.set_title("Avg Temperature (°C)", fontweight="bold")
        ax.set_ylabel("°C")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, temp_data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.1f}°C", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "**Key drivers:** Mombasa is a consumption hub with high demand and limited local production. "
            "Transportation costs, export market exposure, and higher storage/handling costs due to heat "
            "all contribute to the price premium over Uasin-Gishu (a major surplus-producing region)."
        )
    else:
        st.warning("Select both Mombasa and Uasin-Gishu in the sidebar to see this comparison.")

    st.markdown("---")

    # ── D2: Price spike analysis (Nairobi) ────────────────────────────────────
    st.subheader("D2 · Price Spike Analysis — Nairobi")

    if "Nairobi" in panel["county"].unique():
        nairobi = panel[panel["county"] == "Nairobi"].sort_values("week_start").copy()
        nairobi["price_change"] = nairobi["kamis_smooth"].pct_change() * 100
        spikes = nairobi[nairobi["price_change"] > 20]

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(nairobi["week_start"], nairobi["kamis_smooth"],
                    linewidth=2, color="#3498db", label="Price")
            if not spikes.empty:
                ax.scatter(spikes["week_start"], spikes["kamis_smooth"],
                           color="red", s=80, zorder=5, label="Spike >20%")
            ax.set_title("Nairobi: Price Trend & Spikes", fontweight="bold")
            ax.set_ylabel("KES / bag")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            monthly_avg = nairobi.groupby("month")["kamis_smooth"].mean()
            ax.plot(monthly_avg.index, monthly_avg.values, marker="o",
                    linewidth=2, color="#e74c3c")
            ax.fill_between(monthly_avg.index, monthly_avg.values, alpha=0.2, color="#e74c3c")
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(
                ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                rotation=45, fontsize=8
            )
            ax.set_title("Nairobi: Seasonal Pattern", fontweight="bold")
            ax.set_ylabel("Avg KES / bag")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.info(
            "**Spike drivers:** supply shortages during harvest gaps (Jan–Mar), "
            "increased demand in festive seasons (Nov–Dec), transport disruptions, "
            "speculative hoarding, and maize import/export policy changes."
        )

    st.markdown("---")

    # ── D3: Volatility producing vs consuming ─────────────────────────────────
    st.subheader("D3 · Price Volatility — Producing vs Consuming Counties")

    producing  = [c for c in ["Uasin-Gishu", "Kirinyaga"] if c in panel["county"].unique()]
    consuming  = [c for c in ["Nairobi", "Mombasa"]       if c in panel["county"].unique()]

    vol = (
        panel.groupby("county")["kamis_smooth"]
        .agg(std="std", mean="mean")
        .assign(cv=lambda d: d["std"] / d["mean"] * 100)
        .round(2)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_v = [
            "#2ecc71" if c in producing else "#e74c3c" if c in consuming else "#95a5a6"
            for c in vol.index
        ]
        ax.bar(vol.index, vol["cv"], color=colors_v, edgecolor="black", linewidth=0.5)
        if producing:
            ax.axhline(vol.loc[producing, "cv"].mean(), color="green", linestyle="--",
                       label=f"Producing avg {vol.loc[producing,'cv'].mean():.1f}%")
        if consuming:
            ax.axhline(vol.loc[consuming, "cv"].mean(), color="red", linestyle="--",
                       label=f"Consuming avg {vol.loc[consuming,'cv'].mean():.1f}%")
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_title("Price Volatility by County", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        prod_prices = panel[panel["county"].isin(producing)]["kamis_smooth"]
        cons_prices = panel[panel["county"].isin(consuming)]["kamis_smooth"]
        ax.hist(prod_prices, bins=30, alpha=0.6, label="Producing", color="#2ecc71", edgecolor="black")
        ax.hist(cons_prices, bins=30, alpha=0.6, label="Consuming", color="#e74c3c", edgecolor="black")
        ax.set_xlabel("Price (KES)")
        ax.set_ylabel("Frequency")
        ax.set_title("Price Distribution Comparison", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── D4: Correlation heatmap ────────────────────────────────────────────────
    st.subheader("D4 · Price Correlation Between Counties")

    price_pivot = panel.pivot_table(index="week_start", columns="county", values="kamis_smooth")
    corr_matrix = price_pivot.corr()

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0,
                    fmt=".3f", ax=ax, linewidths=0.5)
        ax.set_title("Price Correlation Heatmap", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        if "Kirinyaga" in price_pivot.columns and "Nairobi" in price_pivot.columns:
            merged = price_pivot[["Kirinyaga", "Nairobi"]].dropna()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(merged["Kirinyaga"], merged["Nairobi"], alpha=0.5, color="#0C447C")
            z = np.polyfit(merged["Kirinyaga"], merged["Nairobi"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(merged["Kirinyaga"].min(), merged["Kirinyaga"].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=1.5)
            r = merged["Kirinyaga"].corr(merged["Nairobi"])
            ax.set_title(f"Kirinyaga vs Nairobi  (r = {r:.3f})", fontweight="bold")
            ax.set_xlabel("Kirinyaga Price (KES)")
            ax.set_ylabel("Nairobi Price (KES)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.info(
        "**Kirinyaga–Nairobi correlation:** Geographic proximity and efficient transport routes "
        "mean Kirinyaga supplies directly to Nairobi markets. Shared seasonal weather patterns "
        "and price transmission through integrated supply chains keep prices tightly coupled."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Performance")

    # ── Overall metrics ───────────────────────────────────────────────────────
    st.subheader("Overall Test-Set Metrics")
    m = meta["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",  m["MAE"],  help="Mean Absolute Error — lower is better")
    c2.metric("RMSE", m["RMSE"], help="Root Mean Squared Error — lower is better")
    c3.metric("MAPE", m["MAPE"], help="Mean Absolute Percentage Error — lower is better")
    c4.metric("R²",   m["R2"],   help="Coefficient of determination — higher is better")

    st.markdown("---")

    # ── Actual vs Predicted per county ────────────────────────────────────────
    st.subheader("Actual vs Predicted — Per County")

    test_f = test_res[test_res["county"].isin(selected_counties)]

    price_col = "agri_price" if "agri_price" in test_res.columns else "kamis_price"

    n = len(selected_counties)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, county in enumerate(selected_counties):
        ax = axes[idx]
        cdf = test_f[test_f["county"] == county].sort_values("week_start")
        ax.plot(cdf["week_start"], cdf[price_col],
                label="Actual", linewidth=2, color="#0C447C")
        ax.plot(cdf["week_start"], cdf["prediction"],
                label="Predicted", linewidth=2, color="#1D9E75", linestyle="--")
        mae_c  = cdf["abs_error"].mean()
        mape_c = cdf["pct_error"].mean()
        ax.set_title(f"{county}  |  MAE={mae_c:.2f} KES  MAPE={mape_c:.1f}%", fontweight="bold")
        ax.set_ylabel("KES / bag")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    for idx in range(len(selected_counties), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Actual vs Predicted — {meta['model']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Per-county error table ─────────────────────────────────────────────────
    st.subheader("Per-County Error Summary")

    county_stats = (
        test_f.groupby("county")
        .agg(
            Actual_Mean   =(price_col,    "mean"),
            Predicted_Mean=("prediction", "mean"),
            MAE           =("abs_error",  "mean"),
            MAPE          =("pct_error",  "mean"),
            N_Weeks       =("abs_error",  "count"),
        )
        .round(3)
    )
    county_stats["Bias_KES"] = (county_stats["Predicted_Mean"] - county_stats["Actual_Mean"]).round(3)
    st.dataframe(county_stats.style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")

    # ── Residuals distribution ─────────────────────────────────────────────────
    st.subheader("Residuals Distribution")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        residuals = test_f[price_col] - test_f["prediction"]
        ax.hist(residuals, bins=40, color="#1D9E75", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title("Residuals Distribution", fontweight="bold")
        ax.set_xlabel("Actual − Predicted (KES)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(test_f["prediction"], residuals, alpha=0.4, color="#0C447C", s=20)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Predicted Price (KES)")
        ax.set_ylabel("Residual (KES)")
        ax.set_title("Residuals vs Fitted", fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRICE FORECAST
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Price Forecast")

    forecast_f = forecast[forecast["county"].isin(selected_counties)]

    n_weeks   = forecast_f["week_start"].nunique()
    date_min  = forecast_f["week_start"].min().strftime("%d %b %Y")
    date_max  = forecast_f["week_start"].max().strftime("%d %b %Y")

    st.info(f"**{n_weeks}-week forecast** · {date_min} → {date_max}  |  Model: {meta['model']}")

    price_col = "agri_price" if "agri_price" in panel.columns else "kamis_smooth"

    n = len(selected_counties)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n))
    if n == 1:
        axes = [axes]

    for idx, county in enumerate(selected_counties):
        ax = axes[idx]

        hist = panel[panel["county"] == county].sort_values("week_start")
        fc   = forecast_f[forecast_f["county"] == county].sort_values("week_start")

        # last 52 weeks of history for context
        hist_tail = hist.tail(52)

        ax.plot(hist_tail["week_start"], hist_tail[price_col],
                label="Historical (last 52 wks)", color="#0C447C", linewidth=2)
        ax.plot(fc["week_start"], fc["predicted_price"],
                label="Forecast", color="#1D9E75", linestyle="--", linewidth=2.5)

        # Shaded CI (±5% simple band for visual guidance)
        ax.fill_between(
            fc["week_start"],
            fc["predicted_price"] * 0.95,
            fc["predicted_price"] * 1.05,
            alpha=0.15, color="#1D9E75", label="±5% band"
        )

        # Connect history to forecast
        if not hist_tail.empty and not fc.empty:
            ax.plot(
                [hist_tail["week_start"].iloc[-1], fc["week_start"].iloc[0]],
                [hist_tail[price_col].iloc[-1], fc["predicted_price"].iloc[0]],
                color="#1D9E75", linestyle="--", linewidth=2
            )

        ax.set_title(f"{county}", fontweight="bold", fontsize=12)
        ax.set_ylabel("KES / bag")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    fig.suptitle(f"Maize Price Forecast — {meta['model']}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Forecast table ────────────────────────────────────────────────────────
    st.subheader("Forecast Data Table")

    pivot = forecast_f.pivot_table(
        index="week_start", columns="county", values="predicted_price"
    ).round(2)
    pivot.index = pivot.index.strftime("%Y-%m-%d")
    pivot.index.name = "Week"
    st.dataframe(pivot.style.format("KES {:.2f}"), use_container_width=True)

    csv = forecast_f.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download forecast CSV",
        data=csv,
        file_name="maize_price_forecast.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Kenya Maize Price Forecasting · Data: KAMIS & AgriBORA · "
    f"Model trained: {meta['trained_on']} · Built with Streamlit 🌽"
)
