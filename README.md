# 📈 Stock Forecast

Time-series forecasting for stocks, commodities and crypto using a **model selection pipeline**.

## Pipeline

1. **Train/Validation split** – the last *H* observations are held out as a validation set
2. **Training & Evaluation** – each model is trained on the train set and evaluated on validation
3. **Selection** – the best model (lowest MAE/RMSE/MAPE) is selected
4. **Retrain & Forecast** – the winner is retrained on the full series and produces the final forecast

## Models

| Model | Type | Notes |
|---|---|---|
| **Chronos** (Amazon) | Zero-shot transformer | No training needed, great out-of-the-box accuracy |
| **N-HiTS** (NeuralForecast) | Deep learning | Trained on-the-fly, excels on longer horizons |
| **N-BEATS** (NeuralForecast) | Deep learning | Interpretable architecture, strong on univariate series |
| **Auto-ARIMA** (statsmodels) | Statistical | Automatic order selection via ADF + AIC grid search |
| **XGBoost** | Gradient boosting | ML with lag features, fast and robust |

## Project structure

```
stock-forecast/
├── src/
│   └── stock_forecast/
│       ├── __init__.py   # public API
│       ├── data.py       # yfinance download helpers
│       ├── models.py     # model wrappers + pipeline
│       └── plot.py       # Plotly interactive charts
├── notebooks/
│   └── forecast.ipynb   # main notebook
├── outputs/             # saved HTML charts
└── pyproject.toml
```

## Quick start

```bash
# 1. Install dependencies
poetry install

# 2. Launch notebook
poetry run jupyter lab notebooks/forecast.ipynb
```

## Configuration (in the notebook)

```python
TICKER   = "CL=F"    # any Yahoo Finance ticker
PERIOD   = "max"     # 6mo | 1y | 2y | 5y | 10y | max
INTERVAL = "1d"      # 1d  | 1wk | 1mo
HORIZON  = 15        # steps to forecast

CHRONOS_SIZE     = "large"  # tiny | mini | small | base | large
MAX_STEPS        = 500      # training iterations for N-HiTS / N-BEATS
MODELS           = ["N-HiTS", "N-BEATS", "Chronos", "Auto-ARIMA", "XGBoost"]
SELECTION_METRIC = "mae"    # mae | rmse | mape
```

## Useful tickers

| Asset | Ticker |
|---|---|
| S&P 500 | `^GSPC` |
| NASDAQ | `^IXIC` |
| Gold (futures) | `GC=F` |
| Brent Oil | `BZ=F` |
| WTI Oil | `CL=F` |
| Bitcoin | `BTC-USD` |
| EUR/USD | `EURUSD=X` |
| Apple | `AAPL` |
| ENI | `ENI.MI` |

## Notes

- Chronos uses **zero-shot inference** – no training, fast, good baseline.
- N-HiTS and N-BEATS are trained **from scratch on each run** – slower but can adapt to the specific asset.
- Auto-ARIMA automatically determines differencing order (ADF test) and searches for the best (p, d, q) via AIC.
- XGBoost builds a supervised dataset from lag features and forecasts recursively.
- All models output **80% prediction intervals** shown as shaded bands.
- The interactive chart includes a **candlestick**, volume subplot, range selector, and forecast bands.
