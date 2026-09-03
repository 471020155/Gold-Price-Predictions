# Gold Price Prediction — Next-Day GLD Forecasting

> Chronological time-series regression that predicts **tomorrow's Gold ETF (GLD)** from today's market data. Built with **FastAPI + Streamlit + UV** on the exact notebook logic.

**Deployed model:** `LinearRegression` — **R² 0.9583, RMSE 0.99, MAE 0.73** on chronological 80/20 (better than Random Forest 0.9207 and Gradient Boosting 0.9173). Feature order is strict: `[SPX, GLD, USO, SLV, EUR/USD, Year]`.

## Features
- **Prediction:** Single next-day GLD via `POST /predict` (FastAPI validates with Pydantic, auto docs at `/docs`)
- **Streamlit UI:** Clean inputs for the 6 exact features, `st.metric` with delta vs today, auto-fallback to local model if API is down
- **Honest evaluation:** Chronological `iloc[:80%] / iloc[80%:]` split — fixes notebook `train_test_split(shuffle=True)` leakage
- **Reuse of notebook visuals:** Correlation heatmap (`SLV-GLD 0.86`), GLD histogram, R² bar — kept minimal
- **UV workspace:** `uv sync` / `uv run` for reproducible env (Python >=3.11)

## Technologies
Python 3.11, pandas, scikit-learn, matplotlib, seaborn, FastAPI, Pydantic, Uvicorn, Streamlit, Joblib, Httpx, UV

## Architecture
```
project/
├── data/gld_price_data.csv        # Kaggle source, 2290 rows (2008-01-02 … ~2018)
├── models/
│   ├── gold_lr.pkl                # Primary — LinearRegression (primary)
│   ├── gold_rf.pkl                # Comparison — Random Forest
│   ├── metrics.json               # R²/RMSE/MAE for 3 models
│   └── model_info.json            # feature_order, split, sizes
├── src/project/
│   ├── data_loader.py             # Date -> Year (notebook :893)
│   ├── features.py                # shift(-1) target, FEATURE_COLUMNS
│   ├── train.py                   # chronological split, 3-model compare
│   ├── predict.py                 # load_model + predict_next_day (exact order)
│   └── schemas.py                 # GoldFeatures (alias EUR/USD)
├── api/main.py                    # FastAPI: GET /, /health, /model/info, POST /predict
└── app.py                         # Streamlit: calls POST /predict, fallback local
```

## Installation (UV)

```bash
# from repo root (where uv.lock lives) or from project/
uv sync

# Or if uv binary not on PATH, use the venv directly:
# D:\NTI_ML\final project\.venv\Scripts\python.exe -m pip install -e project
```

Requires Python >=3.11 (`.python-version` = 3.11). Dependencies are pinned in `project/pyproject.toml:7` and `pyproject.toml:8`.

## Dataset

- **Source:** [Kaggle Gold Price Data](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data)
- **File:** `data/gld_price_data.csv` (also `project/data/gld_price_data.csv`) — `2290 × 6` (`Date, SPX, GLD, USO, SLV, EUR/USD`)
- **Stats (real):** SPX mean 1654.3, GLD 122.73, USO 31.84, SLV 20.08, EUR/USD 1.28 (`gold_data.describe() :662`)
- **Preprocessing preserved:** `pd.to_datetime(Date, "%m/%d/%Y") -> Year` (`:893`), `Month/DayOfWeek` dropped, `Target = GLD.shift(-1)` then `dropna` (`:1556`)

## Model Information

- **Primary:** `LinearRegression` — no hyperparams, fit on chronological train `1831` rows, test `458` rows (`2016-2018`)
- **Comparison:** `RandomForestRegressor(n_estimators=100, random_state=2)` and `GradientBoostingRegressor(random_state=2)` (`:2082`)
- **Metrics (chronological):**
  | Model | R² | RMSE | MAE |
  |-------|----|------|-----|
  | Linear Regression | 0.9583 | 0.99 | 0.73 |
  | Random Forest | 0.9207 | 1.36 | 1.07 |
  | Gradient Boosting | 0.9173 | 1.39 | 1.13 |
- **Why Linear wins:** GLD is highly autocorrelated (next-day ≈ today); linear extrapolation generalizes better than trees beyond 2016 range.
- **Feature order (must match):** `["SPX","GLD","USO","SLV","EUR/USD","Year"]` (`features.py:10`)

## Running

### 1. Train (already done — rerun if you replace CSV)
```bash
uv run python -m project.train
# from project/ dir:
# ..\.venv\Scripts\python.exe -m project.train
```

### 2. FastAPI (terminal 1)
```bash
uv run uvicorn api.main:app --reload --app-dir project
# or: ..\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --app-dir project
# Open http://localhost:8000/docs
```

### 3. Streamlit (terminal 2, from project/)
```bash
uv run streamlit run app.py
# or: ..\.venv\Scripts\streamlit.exe run app.py
# Calls http://localhost:8000/predict, falls back to local model if API down
```

## API Usage

**POST /predict**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SPX":1654.3,"GLD":122.73,"USO":31.84,"SLV":20.08,"EUR/USD":1.2836,"Year":2016}'
```
Response:
```json
{"prediction":122.88,"model":"LinearRegression","features_used":["SPX","GLD","USO","SLV","EUR/USD","Year"]}
```

**Python**
```python
import httpx
r = httpx.post("http://localhost:8000/predict", json={"SPX":1654.3,"GLD":122.73,"USO":31.84,"SLV":20.08,"EUR/USD":1.2836,"Year":2016})
print(r.json()["prediction"])
# Fallback (no API):
from project.predict import predict_next_day
print(predict_next_day(1654.3,122.73,31.84,20.08,1.2836,2016))
```

**Other endpoints:** `GET /`, `GET /health`, `GET /model/info` (returns primary + comparison metrics + feature_order).

## Example Usage (Streamlit)

1. Enter today's values (defaults are dataset means).
2. Click **Predict next-day GLD**.
3. See `st.metric` with predicted value and delta vs today; payload shown in expander for audit.

## Future Improvements

- Add lag/rolling features (`GLD lag 5, pct_change`) — notebook dropped Month/DayOfWeek without ablation
- TimeSeriesSplit cross-validation + `GridSearchCV` already scaffolded in `train.py` but not required for course
- SHAP/feature importance for explainability
- Plot actual vs predicted regplot (`:2176`) in UI when test set is available

## License

For course use. Data © Kaggle Gold Price Data. No trading advice — model is educational and not calibrated for risk.
