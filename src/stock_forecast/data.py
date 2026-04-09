"""Data acquisition – download OHLCV time series via yfinance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

# Map of friendly names → yfinance ticker
PRESET_TICKERS: dict[str, str] = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Gold": "GC=F",
    "Brent Oil": "BZ=F",
    "WTI Oil": "CL=F",
    "BTP 10Y (Italy)": "BTP10Y=F",
    "EUR/USD": "EURUSD=X",
    "Bitcoin": "BTC-USD",
}

VALID_INTERVALS = ["1d", "1wk", "1mo"]
VALID_PERIODS = ["6mo", "1y", "2y", "5y", "10y", "max"]


def download_series(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    price_col: str = "Close",
) -> pd.DataFrame:
    """Download a price series from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol (e.g. ``"AAPL"``, ``"GC=F"``).
    period:
        History length accepted by yfinance (``"1y"``, ``"5y"``, …).
    interval:
        Bar granularity: ``"1d"``, ``"1wk"``, or ``"1mo"``.
    price_col:
        Which OHLCV column to use as the target series.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["ds", "y"]`` ready for NeuralForecast,
        plus the full OHLCV columns for plotting.
    """
    if interval not in VALID_INTERVALS:
        msg = f"interval must be one of {VALID_INTERVALS}"
        raise ValueError(msg)
    if period not in VALID_PERIODS:
        msg = f"period must be one of {VALID_PERIODS}"
        raise ValueError(msg)

    raw: pd.DataFrame = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        msg = f"No data returned for ticker '{ticker}'. Check the symbol or your internet connection."
        raise ValueError(msg)

    # yfinance ≥ 0.2 may return MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename_axis("ds").reset_index()
    raw["ds"] = pd.to_datetime(raw["ds"]).dt.tz_localize(None)

    if price_col not in raw.columns:
        msg = f"Column '{price_col}' not found. Available: {list(raw.columns)}"
        raise ValueError(msg)

    df = raw.copy()
    df["y"] = df[price_col]
    df = df.dropna(subset=["y"])
    return df.sort_values("ds").reset_index(drop=True)


def get_ticker_info(ticker: str) -> dict:
    """Return basic metadata for a ticker (name, currency, sector, …)."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "currency": info.get("currency", "USD"),
            "sector": info.get("sector", "—"),
            "industry": info.get("industry", "—"),
            "exchange": info.get("exchange", "—"),
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return {"ticker": ticker, "name": ticker, "currency": "USD"}
