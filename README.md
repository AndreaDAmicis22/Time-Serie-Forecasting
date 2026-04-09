# 📈 Stock Forecast

Time-series forecasting for stocks, bonds and commodities using state-of-the-art models.

## Models

| Model | Type | Notes |
|---|---|---|
| **Chronos** (Amazon) | Zero-shot transformer | No training needed, great out-of-the-box accuracy |
| **N-HiTS** (NeuralForecast) | Deep learning | Trained on-the-fly, excels on longer horizons |

## Project structure

```
stock-forecast/
├── src/
│   └── stock_forecast/
│       ├── __init__.py   # public API
│       ├── data.py       # yfinance download helpers
│       ├── models.py     # Chronos + N-HiTS wrappers
│       └── plot.py       # Plotly interactive charts
├── notebooks/
│   └── forecast.ipynb   # main notebook
├── outputs/             # saved HTML charts
├── data/                # optional: cached CSVs
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
TICKER   = "GC=F"    # any Yahoo Finance ticker
PERIOD   = "5y"      # 6mo | 1y | 2y | 5y | 10y | max
INTERVAL = "1d"      # 1d  | 1wk | 1mo
HORIZON  = 30        # steps to forecast

CHRONOS_SIZE = "small"   # tiny | mini | small | base | large
NHITS_STEPS  = 300       # training iterations
RUN_BOTH     = True      # False → Chronos only (faster)
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
- N-HiTS is trained **from scratch on each run** – slower but can adapt to the specific asset.
- Both models output **80% prediction intervals** shown as shaded bands.
- The interactive chart includes a **candlestick**, volume subplot, range selector, and forecast bands.
