# 📈 Stock Forecast

Time-series forecasting for stocks, commodities, FX and crypto: pretrained **foundation models**, classic baselines,
**automatic ensembles** and a **web app** to run everything from the browser.

## Pipeline

1. **Backtest** – every model is evaluated on *N* rolling-origin windows of *H* observations at the end of the series
2. **Selection** – choose the final forecaster with a strategy:
   - `best` – the model with the lowest validation error
   - `ensemble` – a combination of the top-*k* models (weighted by inverse error, mean or median)
   - `auto` – the ensemble only if it beats the best single model in backtest
3. **Retrain & Forecast** – the winner is retrained on the full series and produces the final forecast

Metrics: MAE, RMSE, MAPE, **MAE / Naive** (below 1 = the model beats the random walk "tomorrow = today" on the same
windows – the real bar on price series) and **coverage** of the 80% prediction interval (≈80% = well calibrated).

## Models

| Model | Family | Notes |
|---|---|---|
| **Chronos-2** (Amazon, 2025) | Foundation, zero-shot | 120M params, Apache-2.0 – default choice |
| **Chronos-Bolt** (Amazon, 2024) | Foundation, zero-shot | Fast direct-quantile Chronos, Apache-2.0 |
| **Chronos** (Amazon, 2024) | Foundation, zero-shot | Original T5 sampling model, kept for comparison |
| **Timestral** (Mistral AI, 2026) | Foundation, zero-shot | Extra `timestral` – modified-MIT weights, CPU friendly |
| **TimesFM-3** (Google, 2026) | Foundation, zero-shot | Extra `timesfm` – ⚠️ weights for **non-commercial** use only |
| **TiRex** (NX-AI, 2025) | Foundation, zero-shot | Extra `tirex` – NXAI community license (commercial terms above €100M revenue) |
| **N-HiTS**, **N-BEATS** | Deep learning | NeuralForecast, trained from scratch on every run (slow) |
| **XGBoost** | Machine learning | Lag features, recursive forecast; models the changes by default so it can extrapolate |
| **Auto-ARIMA** | Statistical | ADF-based differencing + AIC grid search |
| **AutoETS**, **Theta** | Statistical | statsmodels, fast |
| **Naive**, **Drift** | Baseline | Random walk (with drift) – the benchmark to beat |

Foundation models are downloaded from Hugging Face on first use and cached in memory, so later runs only pay the
inference cost (well under a second per forecast on CPU for Chronos-2 / Bolt / TimesFM-3 / TiRex).

## Quick start

```bash
# 1. Install dependencies (Python 3.11 – 3.13)
poetry install

# Optional extra foundation models
poetry install --extras "timesfm tirex timestral"

# 2a. Launch the web app (from the repository root)
poetry run streamlit run app/streamlit_app.py

# 2b. …or the notebook
poetry run jupyter lab notebooks/forecast.ipynb
```

## Web app

- **Sidebar** – asset picker by category (indices, commodities such as gold and oil, US / Italian stocks, crypto, FX,
  rates), by **ISIN code** or any Yahoo Finance ticker, history length, frequency and horizon.
- **📈 Previsione** – latest price, forecast and 80% interval, interactive chart (ensemble members can be overlaid),
  forecast table with CSV download.
- **🧪 Backtest** – metrics of every model, best first, and their out-of-sample forecasts against the actual prices.
- **⚙️ Impostazioni** – strategy (automatic / best model / ensemble / single model), models grouped by family,
  per-model parameters, number of backtest windows, selection metric, ensemble method and size.
- **🗃️ Dati** – downloaded OHLCV data with CSV download.

## Python API

```python
from stock_forecast import download_series, plot_forecast, run_pipeline

df = download_series("GC=F", period="5y")
val_results, forecast = run_pipeline(
    df,
    horizon=15,
    models=["Chronos-2", "TimesFM-3", "Theta", "Naive"],
    selection_metric="mae",
    n_windows=3,
    strategy="auto",
    model_params={"Chronos-Bolt": {"model_size": "base"}},
)
plot_forecast(forecast, ticker="Gold", show_members=True).show()
```

Every model can also be called directly, e.g. `MODEL_REGISTRY["Chronos-2"](df, horizon=15)`.

## ISIN codes

`download_series`, `get_ticker_info` and the web app accept an ISIN instead of a ticker:

- **Yahoo Finance search** resolves the ISIN to the instrument's main listing;
- **[OpenFIGI](https://www.openfigi.com/api)** – Bloomberg's free mapping API, no key required – adds the same
  instrument on the other main exchanges, keeping only the symbols Yahoo Finance has prices for. E.g. the iShares Core
  MSCI World ETF `IE00B4L5Y983` is listed as `IWDA.L` (London, USD) but also as `SWDA.MI` (Borsa Italiana, EUR).

```python
from stock_forecast.data import download_series, lookup_isin

lookup_isin("IE00B4L5Y983")           # IWDA.L, SWDA.MI, IWDA.AS, …
df = download_series("IT0003132476")  # ENI → ENI.MI
```

Stocks, ETFs and ETCs are covered; bonds (e.g. BTPs) and unlisted funds are not on Yahoo Finance.

## Project structure

```
Time-Serie-Forecasting/
├── app/
│   └── streamlit_app.py      # web app
├── src/
│   └── stock_forecast/
│       ├── __init__.py       # public API
│       ├── data.py           # yfinance download, ticker catalog, ISIN lookup
│       ├── metrics.py        # MAE, RMSE, MAPE, MAE/Naive, coverage
│       ├── backtest.py       # rolling-origin windows, validate_models
│       ├── ensemble.py       # weights and forecast combination
│       ├── pipeline.py       # model selection: best / ensemble / auto
│       ├── plot.py           # Plotly charts + text summaries
│       └── models/
│           ├── base.py       # ForecastResult, registry, date helpers
│           ├── foundation.py # Chronos-2 / Bolt / T5, Timestral, TimesFM-3, TiRex
│           ├── neural.py     # N-HiTS, N-BEATS
│           ├── ml.py         # XGBoost
│           └── statistical.py# Naive, Drift, Auto-ARIMA, AutoETS, Theta
├── notebooks/
│   └── forecast.ipynb
├── tests/
└── pyproject.toml
```

## Adding a model

Write a runner in the module of its family and register it – the pipeline and the web app (model list and settings
widgets) pick it up automatically:

```python
@register(
    "MyModel",
    family=STATISTICAL,
    description="One line shown in the web app.",
    params=(Param("window", "Window", 20, min_value=2, max_value=200, step=1),),
)
def run_my_model(df: pd.DataFrame, horizon: int, window: int = 20) -> ForecastResult:
    ...
    return make_result("MyModel", df, horizon, yhat, yhat_lo, yhat_hi)
```

Models that need an optional package declare it with `requires=("module",)` and `extra="name"`: when it is missing
the model is hidden in the app and reported with the install command.

## Tests

```bash
poetry run pytest            # fast suite, includes headless tests of the web app
poetry run pytest -m slow    # foundation & neural models (downloads weights)
```

## Notes

- On liquid price series a random walk is very hard to beat: check **MAE / Naive** in the backtest before trusting a
  forecast. These are statistical forecasts, not investment advice.
- All models output **80% prediction intervals**, shown as shaded bands.
- Crypto series (trading on weekends) get calendar-day forecast dates, other markets business days.
