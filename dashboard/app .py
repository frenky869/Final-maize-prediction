"""
app.py  –  Kenya Maize Price Forecasting Dashboard
===================================================
Pure UI layer.  All data loading and computation lives in notebook_pipeline.py.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
def hex_to_rgba(hex_code, opacity=0.2):
    """Converts hex to rgba string for Plotly compatibility."""
    hex_code = hex_code.lstrip('#')
    # If hex includes alpha (8 chars), separate it
    if len(hex_code) == 8:
        alpha = int(hex_code[6:8], 16) / 255.0
        hex_code = hex_code[:6]
    else:
        alpha = opacity
    r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'

import streamlit as st
from plotly.subplots import make_subplots

# notebook_pipeline.py lives in the same directory as this file (dashboard/)
import notebook_pipeline as pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Page config & global style
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kenya Maize Price Forecasting",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour tokens (aligned with Kenyan agricultural / earth palette)
C_GREEN   = "#2D6A4F"   # deep maize-field green  – primary accent
C_GOLD    = "#D4A017"   # dried-maize gold         – highlight
C_RUST    = "#C0392B"   # harvest rust             – alert / loss
C_SKY     = "#2980B9"   # open-sky blue            – neutral series
C_LIGHT   = "#F4F1EA"   # parchment background
C_MID     = "#E8E4DA"   # card surface
C_DARK    = "#1B2631"   # near-black text
C_MUTED   = "#7F8C8D"   # secondary text

COUNTY_COLOURS = {
    "Kiambu":     C_GREEN,
    "Kirinyaga":  "#1ABC9C",
    "Mombasa":    C_RUST,
    "Nairobi":    C_SKY,
    "Uasin-Gishu": C_GOLD,
}

st.markdown(
    f"""
    <style>
        /* ── Global resets ── */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {C_LIGHT};
            color: {C_DARK};
            font-family: 'Inter', sans-serif;
        }}
        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: {C_DARK};
        }}
        [data-testid="stSidebar"] * {{
            color: {C_LIGHT} !important;
        }}
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label {{
            color: {C_MUTED} !important;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        /* ── Metric cards ── */
        div[data-testid="metric-container"] {{
            background: {C_MID};
            border-radius: 8px;
            padding: 1rem 1.25rem;
            border-left: 4px solid {C_GREEN};
        }}
        /* ── Section headers ── */
        h2 {{
            border-bottom: 2px solid {C_GOLD};
            padding-bottom: 0.35rem;
            margin-top: 2rem;
        }}
        /* ── Plotly chart background ── */
        .js-plotly-plot .plotly {{
            background: transparent !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (cached so it runs only once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model & data …")
def get_data() -> dict:
    return pipeline.load_all()


def safe_load() -> dict | None:
    try:
        return get_data()
    except FileNotFoundError as exc:
        st.error(
            f"**Missing dashboard artefact.**\n\n{exc}\n\n"
            "**Quick fix:** run the notebook export cell, commit the `dashboard/` "
            "folder to [JKUAT-PROJECT](https://github.com/frenky869/JKUAT-PROJECT), "
            "then redeploy / restart this app."
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared chart helpers
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=C_DARK),
    margin=dict(l=16, r=16, t=40, b=16),
)

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def _apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**{**PLOTLY_LAYOUT, **kwargs})
    fig.update_xaxes(gridcolor=C_MID, zeroline=False)
    fig.update_yaxes(gridcolor=C_MID, zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

PAGES = [
    "🌽  Overview",
    "📊  EDA",
    "🔍  Diagnostics",
    "🤖  Model Performance",
    "🔮  Forecast",
    "🎯  Feature Importance",
]

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{C_GOLD};margin-bottom:0.25rem;'>🌽 Maize Watch</h2>"
        f"<p style='color:{C_MUTED};font-size:0.8rem;margin-top:0;'>Kenya · Weekly wholesale prices</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()

    # County filter (shared across pages)
    selected_counties = st.multiselect(
        "Counties",
        pipeline.TARGET_COUNTIES,
        default=pipeline.TARGET_COUNTIES,
    )
    if not selected_counties:
        selected_counties = pipeline.TARGET_COUNTIES

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

data = safe_load()
if data is None:
    st.stop()

panel        = data["panel"]
test_results = data["test_results"]
forecast     = data["forecast"]
metadata     = data["metadata"]
model_name   = data["model_name"]
county_stats = data["county_stats"]
monthly_avg  = data["monthly_avg"]
corr_matrix  = data["corr_matrix"]
county_met   = data["county_metrics"]
feat_imp     = data["feature_importance"]

# Apply county filter
panel_f    = panel[panel["county"].isin(selected_counties)]
test_f     = test_results[test_results["county"].isin(selected_counties)]
forecast_f = forecast[forecast["county"].isin(selected_counties)]


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

if page == PAGES[0]:
    st.title("Kenya White Maize Price Forecasting")
    st.caption(
        f"Data: KAMIS + AgriBORA · Model: **{model_name}** · "
        f"Last trained: {metadata.get('trained_on', 'N/A')} · "
        f"Data range: {metadata.get('date_range', 'N/A')}"
    )

    # ── KPI row ──
    m = metadata.get("metrics", {})
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Model", model_name)
    k2.metric("MAE",  m.get("MAE",  "—"))
    k3.metric("RMSE", m.get("RMSE", "—"))
    k4.metric("MAPE", m.get("MAPE", "—"))
    k5.metric("R²",   m.get("R2",   "—"))

    st.markdown("## All-county price history")

    fig = go.Figure()
    for county in selected_counties:
        cdf = panel_f[panel_f["county"] == county].sort_values("week_start")
        fig.add_trace(
            go.Scatter(
                x=cdf["week_start"], y=cdf["agri_price"],
                name=county, mode="lines",
                line=dict(color=COUNTY_COLOURS.get(county, C_SKY), width=1.8),
            )
        )
    _apply_layout(fig, title="Weekly wholesale price (KES / 90 kg bag)",
                  legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ──
    st.markdown("## County snapshot")
    display_stats = (
        county_stats
        .loc[county_stats.index.isin(selected_counties)]
        .rename(columns={
            "mean": "Mean (KES)", "median": "Median (KES)",
            "std": "Std (KES)", "min": "Min (KES)", "max": "Max (KES)",
            "cv_pct": "CV %", "n_weeks": "Weeks",
        })
    )
    st.dataframe(display_stats.style.format("{:.2f}"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ═════════════════════════════════════════════════════════════════════════════

elif page == PAGES[1]:
    st.title("Exploratory Data Analysis")

    # ── Monthly seasonality ──
    st.markdown("## Seasonal price patterns")
    cols = st.columns(min(len(selected_counties), 3))
    for i, county in enumerate(selected_counties):
        ax = cols[i % 3]
        cdf = monthly_avg[monthly_avg["county"] == county]
        peak_month = int(cdf.loc[cdf["avg_price"].idxmax(), "month"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cdf["month"], y=cdf["avg_price"],
                fill="tozeroy", mode="lines+markers",
                line=dict(color=COUNTY_COLOURS.get(county, C_SKY), width=2),
                fillcolor=hex_to_rgba(COUNTY_COLOURS.get(county, C_SKY), 0.2),
            )
        )
        fig.add_vline(x=peak_month, line_dash="dash",
                      line_color=C_RUST, annotation_text=f"Peak: {MONTH_LABELS[peak_month-1]}")
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                         ticktext=MONTH_LABELS)
        _apply_layout(fig, title=county, height=280)
        ax.plotly_chart(fig, use_container_width=True)

    # ── Price distributions ──
    st.markdown("## Price distributions")
    fig = go.Figure()
    for county in selected_counties:
        prices = panel_f[panel_f["county"] == county]["agri_price"].dropna()
        fig.add_trace(
            go.Violin(
                y=prices, name=county, box_visible=True, meanline_visible=True,
                fillcolor=COUNTY_COLOURS.get(county, C_SKY) + "80",
                line_color=COUNTY_COLOURS.get(county, C_SKY),
            )
        )
    _apply_layout(fig, title="Distribution of weekly prices (KES)", yaxis_title="Price (KES)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap ──
    st.markdown("## Inter-county price correlation")
    corr_sub = corr_matrix.loc[
        corr_matrix.index.isin(selected_counties),
        corr_matrix.columns.isin(selected_counties),
    ]
    fig = px.imshow(
        corr_sub.round(3), text_auto=True, color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
    )
    _apply_layout(fig, title="Pearson correlation of weekly AgriBORA prices")
    st.plotly_chart(fig, use_container_width=True)

    # ── Weather scatter ──
    st.markdown("## Temperature vs price")
    weather_panel = panel_f.dropna(subset=["temp_avg_c", "agri_price"])
    if not weather_panel.empty:
        fig = px.scatter(
            weather_panel, x="temp_avg_c", y="agri_price",
            color="county", color_discrete_map=COUNTY_COLOURS,
            trendline="ols", opacity=0.6,
            labels={"temp_avg_c": "Avg temp (°C)", "agri_price": "Price (KES)"},
        )
        _apply_layout(fig, title="Temperature vs wholesale price by county")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Temperature data not available for the selected counties.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════

elif page == PAGES[2]:
    st.title("Diagnostic Analysis")

    # ── Producing vs consuming volatility ──
    producing  = [c for c in ["Uasin-Gishu", "Kirinyaga"] if c in selected_counties]
    consuming  = [c for c in ["Nairobi", "Mombasa", "Kiambu"] if c in selected_counties]

    st.markdown("## Price volatility: producing vs consuming counties")
    vol_data = county_stats.loc[county_stats.index.isin(selected_counties), "cv_pct"].reset_index()
    vol_data.columns = ["county", "cv_pct"]
    vol_data["role"] = vol_data["county"].apply(
        lambda c: "Producing" if c in producing else "Consuming"
    )
    fig = px.bar(
        vol_data.sort_values("cv_pct"),
        x="cv_pct", y="county", orientation="h",
        color="role",
        color_discrete_map={"Producing": C_GREEN, "Consuming": C_RUST},
        labels={"cv_pct": "Coefficient of variation (%)", "county": ""},
        text_auto=".1f",
    )
    _apply_layout(fig, title="Price volatility (coefficient of variation, %)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Price spike detector ──
    st.markdown("## Price spike detector  (week-on-week change > 10 %)")
    spike_threshold = st.slider("Spike threshold (%)", 5, 30, 10, 1)

    spike_rows = []
    for county in selected_counties:
        cdf = panel_f[panel_f["county"] == county].sort_values("week_start").copy()
        cdf["pct_change"] = cdf["agri_price"].pct_change() * 100
        spikes = cdf[cdf["pct_change"] > spike_threshold]
        for _, row in spikes.iterrows():
            spike_rows.append({
                "county": county,
                "week": row["week_start"].strftime("%Y-%m-%d"),
                "price (KES)": round(row["agri_price"], 2),
                "change (%)": round(row["pct_change"], 1),
            })

    if spike_rows:
        spike_df = pd.DataFrame(spike_rows).sort_values("change (%)", ascending=False)
        st.dataframe(spike_df, use_container_width=True)
    else:
        st.success("No spikes above the selected threshold.")

    # ── Mombasa vs Uasin-Gishu gap ──
    st.markdown("## Consumption–production price gap over time")
    if "Mombasa" in selected_counties and "Uasin-Gishu" in selected_counties:
        mom = panel[panel["county"] == "Mombasa"][["week_start", "agri_price"]].rename(
            columns={"agri_price": "Mombasa"})
        uasin = panel[panel["county"] == "Uasin-Gishu"][["week_start", "agri_price"]].rename(
            columns={"agri_price": "Uasin-Gishu"})
        gap_df = pd.merge(mom, uasin, on="week_start", how="inner")
        gap_df["gap"] = gap_df["Mombasa"] - gap_df["Uasin-Gishu"]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Weekly prices", "Mombasa − Uasin-Gishu gap (KES)"))
        for county, colour in [("Mombasa", C_RUST), ("Uasin-Gishu", C_GOLD)]:
            col_name = county
            if col_name in gap_df.columns:
                fig.add_trace(
                    go.Scatter(x=gap_df["week_start"], y=gap_df[col_name],
                               name=county, line=dict(color=colour, width=1.8)),
                    row=1, col=1,
                )
        fig.add_trace(
            go.Bar(x=gap_df["week_start"], y=gap_df["gap"],
                   name="Gap", marker_color=C_SKY, opacity=0.7),
            row=2, col=1,
        )
        _apply_layout(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select both **Mombasa** and **Uasin-Gishu** to see the gap chart.")

    # ── Missing data heatmap ──
    st.markdown("## Historical data completeness")
    panel_all = panel[panel["county"].isin(selected_counties)].copy()
    panel_all["year_week"] = panel_all["week_start"].dt.to_period("W").astype(str)
    availability = panel_all.pivot_table(
        index="county", columns="year_week",
        values="agri_price", aggfunc="count"
    ).notna().astype(int)

    # Show only every 8th column label to avoid crowding
    n_cols = availability.shape[1]
    shown_cols = availability.columns[::max(1, n_cols // 30)]

    fig = px.imshow(
        availability,
        color_continuous_scale=[[0, C_RUST], [1, C_GREEN]],
        labels={"color": "Has data"},
        aspect="auto",
    )
    _apply_layout(fig, title="Data availability by county and week (green = present)")
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 – MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

elif page == PAGES[3]:
    st.title(f"Model Performance — {model_name}")

    # ── Per-county metrics table ──
    st.markdown("## Test-set metrics by county")
    met_sub = county_met.loc[county_met.index.isin(selected_counties)].copy()
    met_sub.columns = [c.replace("_", " ").title() for c in met_sub.columns]
    st.dataframe(
        met_sub.style.format("{:.2f}")
            .background_gradient(subset=["Mae", "Mape"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    # ── Actual vs predicted per county ──
    st.markdown("## Actual vs predicted prices")
    county_sel = st.selectbox("County", selected_counties, key="avp_county")
    cdf = test_f[test_f["county"] == county_sel].sort_values("week_start")

    if cdf.empty:
        st.info("No test data for this county in the selected filter.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cdf["week_start"], y=cdf["agri_price"],
            name="Actual", mode="lines",
            line=dict(color=C_DARK, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=cdf["week_start"], y=cdf["prediction"],
            name="Predicted", mode="lines",
            line=dict(color=COUNTY_COLOURS.get(county_sel, C_GREEN), width=2, dash="dash"),
        ))
        mae_c  = cdf["abs_error"].mean()
        mape_c = cdf["pct_error"].mean()
        _apply_layout(
            fig,
            title=f"{county_sel}  |  MAE = KES {mae_c:.2f}  |  MAPE = {mape_c:.1f}%",
            yaxis_title="Price (KES)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Residuals ──
    st.markdown("## Residual distribution")
    residuals = test_f["agri_price"] - test_f["prediction"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=60,
        marker_color=hex_to_rgba(C_GREEN, 0.6), marker_line_color=C_GREEN, marker_line_width=0.5,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=C_RUST)
    _apply_layout(fig, title="Prediction residuals (actual − predicted, all counties)",
                  xaxis_title="Residual (KES)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: actual vs predicted ──
    st.markdown("## Predicted vs actual scatter")
    fig = px.scatter(
        test_f, x="agri_price", y="prediction",
        color="county", color_discrete_map=COUNTY_COLOURS,
        opacity=0.65,
        labels={"agri_price": "Actual (KES)", "prediction": "Predicted (KES)"},
    )
    min_v = min(test_f["agri_price"].min(), test_f["prediction"].min())
    max_v = max(test_f["agri_price"].max(), test_f["prediction"].max())
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                             mode="lines", line=dict(color=C_RUST, dash="dot"),
                             name="Perfect prediction"))
    _apply_layout(fig, title="Predicted vs actual (dots on the red line = perfect)")
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 – FORECAST
# ═════════════════════════════════════════════════════════════════════════════

elif page == PAGES[4]:
    st.title("Price Forecast")
    n_weeks = forecast_f["week_start"].nunique()
    st.caption(
        f"**{model_name}** · Horizon: {n_weeks} weeks "
        f"({forecast_f['week_start'].min().date()} → {forecast_f['week_start'].max().date()})"
    )

    # ── Full history + forecast per county ──
    for county in selected_counties:
        hist = panel[panel["county"] == county].sort_values("week_start")
        fut  = forecast_f[forecast_f["county"] == county].sort_values("week_start")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["week_start"], y=hist["agri_price"],
            name="Historical", mode="lines",
            line=dict(color=COUNTY_COLOURS.get(county, C_SKY), width=1.8),
        ))
        fig.add_trace(go.Scatter(
            x=fut["week_start"], y=fut["predicted_price"],
            name="Forecast", mode="lines+markers",
            line=dict(color=C_GOLD, width=2, dash="dash"),
            marker=dict(size=5),
        ))
        # Shade the forecast region
        fig.add_vrect(
            x0=fut["week_start"].min(), x1=fut["week_start"].max(),
            fillcolor=C_GOLD, opacity=0.06, line_width=0,
        )
        _apply_layout(fig, title=f"{county} — historical + {n_weeks}-week forecast",
                      yaxis_title="Price (KES)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ──
    st.markdown("## Forecast table")
    pivot = forecast_f.pivot_table(
        index="week_start", columns="county", values="predicted_price"
    ).round(2)
    pivot.index = pivot.index.strftime("%Y-%m-%d")
    st.dataframe(pivot, use_container_width=True)

    # ── Download ──
    csv = forecast_f.to_csv(index=False).encode()
    st.download_button(
        "⬇  Download forecast CSV",
        data=csv,
        file_name="maize_price_forecast.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 – FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

elif page == PAGES[5]:
    st.title("Feature Importance")

    if feat_imp is None:
        st.warning("Feature importance is not available for this model type.")
        st.stop()

    st.caption(f"Model: **{model_name}** — top features driving the price prediction")

    top_n = st.slider("Show top N features", 5, min(40, len(feat_imp)), 20)
    fi = feat_imp.head(top_n).copy()

    # Normalise to percentage share for clarity
    fi["pct"] = fi["importance"] / fi["importance"].sum() * 100

    # Colour by feature category
    def _category(f: str) -> str:
        if "lag"     in f: return "Lag features"
        if "rolling" in f: return "Rolling stats"
        if "month"   in f or "week" in f or "year" in f: return "Calendar"
        if any(w in f for w in ["temp","rain","wind","heat","heavy","strong"]): return "Weather"
        if "county"  in f: return "County"
        return "Price level"

    fi["category"] = fi["feature"].apply(_category)
    CAT_COLOURS = {
        "Lag features": C_GREEN,
        "Rolling stats": C_GOLD,
        "Calendar": C_SKY,
        "Weather": "#8E44AD",
        "County": C_RUST,
        "Price level": C_MUTED,
    }

    fig = px.bar(
        fi.sort_values("pct"),
        x="pct", y="feature", orientation="h",
        color="category", color_discrete_map=CAT_COLOURS,
        text=fi.sort_values("pct")["pct"].map("{:.1f}%".format),
        labels={"pct": "Share of total importance (%)", "feature": "", "category": "Category"},
    )
    fig.update_traces(textposition="outside")
    _apply_layout(
        fig,
        title=f"Top {top_n} features — {model_name}",
        height=max(400, top_n * 28),
        xaxis_title="Share of total importance (%)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Category breakdown ──
    st.markdown("## Importance by category")
    cat_sum = fi.groupby("category")["pct"].sum().reset_index().sort_values("pct", ascending=False)
    fig2 = px.pie(
        cat_sum, names="category", values="pct",
        color="category", color_discrete_map=CAT_COLOURS,
        hole=0.45,
    )
    fig2.update_traces(textinfo="label+percent")
    _apply_layout(fig2, title="Feature category contribution", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Raw table ──
    with st.expander("Raw importance scores"):
        st.dataframe(feat_imp.style.format({"importance": "{:.6f}", "pct": "{:.2f}%"}),
                     use_container_width=True)
