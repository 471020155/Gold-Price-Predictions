"""Streamlit frontend — calls FastAPI POST /predict, exact feature order.

Feature order: [SPX, GLD, USO, SLV, EUR/USD, Year] : features.py:10
Model: LinearRegression (R² 0.9583, RMSE 0.99) : model_info.json
"""
import json
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Local fallback if API is not running
try:
    from project.predict import predict_next_day
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from project.predict import predict_next_day

# --- Config ---
API_URL = "http://localhost:8000"
FEATURE_ORDER = ["SPX", "GLD", "USO", "SLV", "EUR/USD", "Year"]
MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_PATH = Path(__file__).resolve().parent / "data" / "gld_price_data.csv"

st.set_page_config(page_title="Gold Price Prediction", layout="wide")
st.title("Gold Price Prediction")
st.caption(
    "Next-day GLD forecast via **Linear Regression** (chronological 80/20, R² 0.9583) — "
    "FastAPI + Streamlit · Feature order: SPX, GLD, USO, SLV, EUR/USD, Year"
)

tab_predict, tab_info = st.tabs(["Predict", "Model Info"])

# --- Helper: call API with fallback ---
def call_predict(payload: dict) -> tuple[float | None, str]:
    """Try API first, fallback to local model."""
    try:
        r = httpx.post(f"{API_URL}/predict", json=payload, timeout=5.0)
        if r.status_code == 200:
            return r.json()["prediction"], "api"
        else:
            return None, f"api_error: {r.text[:200]}"
    except Exception as e:
        # Fallback to local inference
        try:
            pred = predict_next_day(
                payload["SPX"],
                payload["GLD"],
                payload["USO"],
                payload["SLV"],
                payload["EUR/USD"],
                payload["Year"],
            )
            return pred, f"local_fallback ({e.__class__.__name__})"
        except Exception as le:
            return None, f"local_error: {le}"

# --- Tab 1: Predict ---
with tab_predict:
    st.subheader("Predict next-day GLD")
    st.write("Enter today's market values. The model predicts **tomorrow's GLD**.")

    # Defaults from describe() : Gold_prices_predoctions_(1).ipynb:662
    # SPX mean 1654, GLD 122.7, USO 31.8, SLV 20.08, EUR/USD 1.28, Year 2016
    c1, c2, c3 = st.columns(3)
    with c1:
        spx = st.number_input("SPX (S&P 500)", min_value=600.0, max_value=3000.0, value=1654.3, step=1.0)
        gld = st.number_input("GLD (today)", min_value=60.0, max_value=200.0, value=122.73, step=0.5)
    with c2:
        uso = st.number_input("USO (oil ETF)", min_value=5.0, max_value=120.0, value=31.84, step=0.5)
        slv = st.number_input("SLV (silver)", min_value=8.0, max_value=50.0, value=20.08, step=0.2)
    with c3:
        eur_usd = st.number_input("EUR/USD", min_value=1.0, max_value=1.7, value=1.2836, step=0.01, format="%.4f")
        year = st.number_input("Year", min_value=2008, max_value=2030, value=2016, step=1)

    if st.button("Predict next-day GLD", type="primary"):
        payload = {
            "SPX": float(spx),
            "GLD": float(gld),
            "USO": float(uso),
            "SLV": float(slv),
            "EUR/USD": float(eur_usd),
            "Year": int(year),
        }
        with st.spinner("Calling API ..."):
            pred, source = call_predict(payload)

        if pred is None:
            st.error(f"Prediction failed: {source}")
        else:
            # Show clearly
            delta = pred - gld
            st.metric(
                label="Predicted next-day GLD",
                value=f"{pred:.2f}",
                delta=f"{delta:+.2f} vs today ({gld:.2f})",
            )
            st.success(f"Prediction: **{pred:.2f} USD** (via {source})")
            if abs(delta) < 0.5:
                st.info("Model expects little movement from today.")
            elif delta > 0:
                st.info("Model predicts an increase vs today.")
            else:
                st.info("Model predicts a decrease vs today.")

            with st.expander("Request payload (exact feature order)"):
                st.json(payload)
                st.code(f"Feature order = {FEATURE_ORDER}", language="text")

    st.divider()
    st.caption("API: `POST /predict` at `http://localhost:8000/predict` — run `uvicorn api.main:app` first. Auto-fallback to local model if API is down.")

# --- Tab 2: Model Info + most useful visuals ---
with tab_info:
    st.subheader("Model & Evaluation")
    # Load metrics
    metrics_path = MODELS_DIR / "metrics.json"
    info_path = MODELS_DIR / "model_info.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        df_metrics = pd.DataFrame(
            [
                {"Model": k, "R²": v["r2"], "RMSE": v["rmse"], "MAE": v["mae"]}
                for k, v in metrics.items()
            ]
        )
        st.dataframe(df_metrics.style.format({"R²": "{:.4f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}), use_container_width=True)
        # Bar chart R² — reuse notebook barplot :2473
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.barplot(data=df_metrics, x="Model", y="R²", hue="Model", legend=False, palette="Blues_d", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("R² — higher is better")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("metrics.json not found. Run training first.")

    if info_path.exists():
        info = json.loads(info_path.read_text())
        st.json(info)

    st.divider()
    st.subheader("Most useful EDA (from notebook)")
    if not DATA_PATH.exists():
        st.info("Place gld_price_data.csv at data/gld_price_data.csv to see visuals.")
    else:
        df = pd.read_csv(DATA_PATH)
        # 1. Correlation heatmap — reuse :1178
        st.write("**Correlation heatmap** (note SLV-GLD 0.86, USO negative)")
        # Compute Year for correlation
        try:
            df["Year"] = pd.to_datetime(df["Date"]).dt.year
        except Exception:
            pass
        corr = df[["SPX", "GLD", "USO", "SLV", "EUR/USD", "Year"]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig, clear_figure=True)

        # 2. GLD distribution — reuse histplot :1333
        st.write("**GLD distribution**")
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.histplot(df["GLD"], color="steelblue", kde=True, ax=ax)
        ax.set_xlabel("GLD")
        st.pyplot(fig, clear_figure=True)

        st.caption("Full EDA in notebook: boxplot outliers :1408, SLV vs GLD scatter :1449, trend lineplots :1489.")
