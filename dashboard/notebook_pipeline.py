"""
notebook_pipeline.py
--------------------
Pure-logic layer for the Kenya Maize Price Forecasting project.
Loads pre-exported dashboard artefacts and exposes clean Python
objects / DataFrames that app.py can consume without knowing anything
about training, file paths, or sklearn internals.

Repo layout (https://github.com/frenky869/JKUAT-PROJECT):
    JKUAT-PROJECT/
        dashboard/              ← ALL artefacts + app live here
            app.py              ← Streamlit entry point
            notebook_pipeline.py← this file
            model.pkl
            label_encoder.pkl
            feature_cols.pkl
            model_name.txt
            panel_clean.csv
            test_results.csv
            forecast.csv
            metadata.json
            config.toml
            requirements.txt
            README.md
        data/                   ← raw CSVs (not read here)
        notebook/               ← .ipynb source

Path resolution (in order of priority):
    1. DASHBOARD_DIR env var  – set this to override everything
    2. Same folder as this file  – works automatically because app.py,
       notebook_pipeline.py, and all artefacts share the same directory
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All artefacts live in the same directory as this file
# (i.e. JKUAT-PROJECT/dashboard/).
# Override with the DASHBOARD_DIR env var if needed.
DASHBOARD_DIR = Path(
    os.environ.get("DASHBOARD_DIR", str(Path(__file__).resolve().parent))
)

TARGET_COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _p(filename: str) -> Path:
    """Return a full path inside DASHBOARD_DIR."""
    return DASHBOARD_DIR / filename


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artefact not found:\n  {path}\n\n"
            "Fix options:\n"
            "  1. Run the notebook export cell, then commit/push the\n"
            "     contents of dashboard/ to the repo.\n"
            "  2. Override the search path with the DASHBOARD_DIR env var.\n"
            f"  Current DASHBOARD_DIR = {DASHBOARD_DIR}"
        )


# ---------------------------------------------------------------------------
# Public loader functions (each returns a clean Python object)
# ---------------------------------------------------------------------------


def load_model():
    """Return the trained sklearn/XGBoost model."""
    _require(_p("model.pkl"))
    return joblib.load(_p("model.pkl"))


def load_label_encoder():
    """Return the fitted LabelEncoder for county names."""
    _require(_p("label_encoder.pkl"))
    return joblib.load(_p("label_encoder.pkl"))


def load_feature_cols() -> list[str]:
    """Return the ordered list of feature column names used at training time."""
    _require(_p("feature_cols.pkl"))
    return joblib.load(_p("feature_cols.pkl"))


def load_model_name() -> str:
    """Return the name of the best model (e.g. 'XGBoost')."""
    _require(_p("model_name.txt"))
    return _p("model_name.txt").read_text().strip()


def load_panel() -> pd.DataFrame:
    """
    Return the cleaned historical panel.
    Columns guaranteed: county, week_start, agri_price, kamis_price,
                        temp_avg_c, rain_mm  (plus engineered features).
    """
    _require(_p("panel_clean.csv"))
    df = pd.read_csv(_p("panel_clean.csv"), parse_dates=["week_start"])
    df["county"] = df["county"].str.strip().str.title()
    return df


def load_test_results() -> pd.DataFrame:
    """
    Return test-set actuals + predictions.
    Columns: county, week_start, agri_price, kamis_price,
             prediction, abs_error, pct_error
    """
    _require(_p("test_results.csv"))
    df = pd.read_csv(_p("test_results.csv"), parse_dates=["week_start"])
    df["county"] = df["county"].str.strip().str.title()
    return df


def load_forecast() -> pd.DataFrame:
    """
    Return the pre-computed future price forecast.
    Columns: county, week_start, predicted_price
    """
    _require(_p("forecast.csv"))
    df = pd.read_csv(_p("forecast.csv"), parse_dates=["week_start"])
    df["county"] = df["county"].str.strip().str.title()
    return df


def load_metadata() -> dict:
    """Return training metadata dict (model name, metrics, date range, etc.)."""
    _require(_p("metadata.json"))
    with open(_p("metadata.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Derived / computed helpers used by multiple dashboard pages
# ---------------------------------------------------------------------------


def county_price_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-county summary stats for the historical panel.
    Index = county.  Columns: mean, median, std, min, max, cv_pct, n_weeks
    """
    g = panel.groupby("county")["agri_price"]
    stats = g.agg(["mean", "median", "std", "min", "max", "count"]).rename(
        columns={"count": "n_weeks"}
    )
    stats["cv_pct"] = (stats["std"] / stats["mean"] * 100).round(2)
    return stats.round(2)


def monthly_avg_prices(panel: pd.DataFrame) -> pd.DataFrame:
    """Return (county, month) → mean agri_price for seasonal analysis."""
    df = panel.copy()
    df["month"] = df["week_start"].dt.month
    return (
        df.groupby(["county", "month"])["agri_price"]
        .mean()
        .reset_index()
        .rename(columns={"agri_price": "avg_price"})
    )


def price_correlation_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a county × county Pearson correlation matrix of weekly prices."""
    pivot = panel.pivot_table(
        index="week_start", columns="county", values="agri_price"
    )
    return pivot.corr()


def county_test_metrics(test_results: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-county MAE / MAPE / bias on the held-out test set.
    """
    grp = test_results.groupby("county")
    metrics = grp.agg(
        actual_mean=("agri_price", "mean"),
        predicted_mean=("prediction", "mean"),
        MAE=("abs_error", "mean"),
        MAPE=("pct_error", "mean"),
        n_weeks=("abs_error", "count"),
    ).round(3)
    metrics["bias_kes"] = (metrics["predicted_mean"] - metrics["actual_mean"]).round(3)
    return metrics


def feature_importance_df(model, feature_cols: list[str]) -> Optional[pd.DataFrame]:
    """
    Return a tidy DataFrame of feature importances (tree models) or
    absolute coefficients (linear models).  Returns None if unavailable.
    """
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_)
    else:
        return None

    df = (
        pd.DataFrame({"feature": feature_cols, "importance": scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# One-shot "load everything" convenience function
# ---------------------------------------------------------------------------


def load_all() -> dict:
    """
    Load every artefact and return a single dict.
    Keys: model, label_encoder, feature_cols, model_name,
          panel, test_results, forecast, metadata,
          county_stats, monthly_avg, corr_matrix,
          county_metrics, feature_importance
    Raises FileNotFoundError with a clear message if any artefact is missing.
    """
    model = load_model()
    le = load_label_encoder()
    feature_cols = load_feature_cols()
    model_name = load_model_name()
    panel = load_panel()
    test_results = load_test_results()
    forecast = load_forecast()
    metadata = load_metadata()

    return {
        # raw artefacts
        "model": model,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "model_name": model_name,
        "panel": panel,
        "test_results": test_results,
        "forecast": forecast,
        "metadata": metadata,
        # derived tables (computed once, reused everywhere)
        "county_stats": county_price_stats(panel),
        "monthly_avg": monthly_avg_prices(panel),
        "corr_matrix": price_correlation_matrix(panel),
        "county_metrics": county_test_metrics(test_results),
        "feature_importance": feature_importance_df(model, feature_cols),
    }
