"""Data acquisition – download OHLCV time series via yfinance, by ticker or ISIN."""

from __future__ import annotations

import logging
import re

import pandas as pd
import requests
import yfinance as yf

# Category → friendly name → yfinance ticker (drives the asset dropdown of the web app)
TICKER_CATALOG: dict[str, dict[str, str]] = {
    "Materie prime": {
        "Oro": "GC=F",
        "Argento": "SI=F",
        "Petrolio WTI": "CL=F",
        "Petrolio Brent": "BZ=F",
        "Gas naturale": "NG=F",
        "Rame": "HG=F",
        "Platino": "PL=F",
        "Grano": "ZW=F",
        "Mais": "ZC=F",
    },
    "Indici": {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "FTSE MIB": "FTSEMIB.MI",
        "DAX": "^GDAXI",
        "Euro Stoxx 50": "^STOXX50E",
        "Nikkei 225": "^N225",
    },
    "Azioni USA": {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Amazon": "AMZN",
        "Alphabet": "GOOGL",
        "Meta": "META",
        "Tesla": "TSLA",
    },
    "Azioni Italia": {
        "ENI": "ENI.MI",
        "Enel": "ENEL.MI",
        "Intesa Sanpaolo": "ISP.MI",
        "UniCredit": "UCG.MI",
        "Generali": "G.MI",
        "Ferrari": "RACE.MI",
        "Stellantis": "STLAM.MI",
        "Leonardo": "LDO.MI",
    },
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
    },
    "Valute": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "EUR/CHF": "EURCHF=X",
    },
    "Tassi": {
        "Rendimento Treasury USA 10Y": "^TNX",
        "Rendimento Treasury USA 30Y": "^TYX",
    },
}

# Flat map of friendly names → yfinance ticker
PRESET_TICKERS: dict[str, str] = {name: ticker for group in TICKER_CATALOG.values() for name, ticker in group.items()}

VALID_INTERVALS = ["1d", "1wk", "1mo"]
VALID_PERIODS = ["6mo", "1y", "2y", "5y", "10y", "max"]

OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

# Bloomberg exchange code (as returned by OpenFIGI) → (Yahoo Finance suffix, venue as Yahoo names it)
_EXCHANGES: dict[str, tuple[str, str]] = {
    "US": ("", "USA"),
    "UN": ("", "NYSE"),
    "UW": ("", "NASDAQ"),
    "UQ": ("", "NASDAQ"),
    "UP": ("", "NYSE Arca"),
    "IM": (".MI", "Milan"),
    "GY": (".DE", "XETRA"),
    "GF": (".F", "Frankfurt"),
    "LN": (".L", "London"),
    "NA": (".AS", "Amsterdam"),
    "FP": (".PA", "Paris"),
    "BB": (".BR", "Brussels"),
    "SM": (".MC", "Madrid"),
    "SW": (".SW", "Swiss"),
    "SE": (".SW", "Swiss"),
    "AV": (".VI", "Vienna"),
    "ID": (".IR", "Dublin"),
    "PL": (".LS", "Lisbon"),
    "SS": (".ST", "Stockholm"),
    "DC": (".CO", "Copenhagen"),
    "NO": (".OL", "Oslo"),
    "FH": (".HE", "Helsinki"),
    "CN": (".TO", "Toronto"),
    "JT": (".T", "Tokyo"),
    "HK": (".HK", "Hong Kong"),
    "AU": (".AX", "ASX"),
}

_ISIN_PATTERN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")


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
        Yahoo Finance ticker symbol (e.g. ``"AAPL"``, ``"GC=F"``) or ISIN code
        (e.g. ``"IT0003132476"``), resolved to its main listing.
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

    ticker = resolve_ticker(ticker)
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
    """Return basic metadata for a ticker or ISIN (name, currency, sector, …)."""
    try:
        ticker = resolve_ticker(ticker)
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


# ──────────────────────────────────────────────────────────────────────────────
# ISIN lookup
# ──────────────────────────────────────────────────────────────────────────────


def is_valid_isin(code: str) -> bool:
    """Check format and check digit of an ISIN code (e.g. ``IT0003132476``)."""
    code = code.strip().upper()
    if not _ISIN_PATTERN.fullmatch(code):
        return False
    digits = "".join(str(int(char, 36)) for char in code)  # letters become 10…35
    total = 0
    for i, digit in enumerate(reversed(digits)):  # Luhn algorithm
        n = int(digit) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def lookup_isin(isin: str, *, other_exchanges: bool = True) -> list[dict[str, str]]:
    """Find the Yahoo Finance listings of the instrument identified by *isin*.

    Yahoo Finance search returns the main listing; OpenFIGI (Bloomberg's free
    mapping API, no key needed) adds the same instrument on the other main
    exchanges – e.g. the Borsa Italiana listing of an ETF – keeping only the
    symbols Yahoo Finance has prices for. Bonds and unlisted funds are not on
    Yahoo Finance and give no listing.

    Returns
    -------
    list of dicts with keys ``symbol`` (Yahoo ticker), ``name``, ``exchange``, ``type``.
    """
    isin = isin.strip().upper()
    if not is_valid_isin(isin):
        msg = f"'{isin}' is not a valid ISIN code."
        raise ValueError(msg)

    listings = _yahoo_listings(isin)
    if other_exchanges or not listings:
        known = {listing["symbol"] for listing in listings}
        others = _with_prices([listing for listing in _openfigi_listings(isin) if listing["symbol"] not in known])
        if listings:  # same instrument: Yahoo's name reads better than OpenFIGI's upper case
            others = [{**listing, "name": listings[0]["name"]} for listing in others]
        listings += others
    return listings


def resolve_ticker(code: str) -> str:
    """Return *code* unchanged, or the main Yahoo Finance ticker when *code* is an ISIN."""
    code = code.strip()
    if not is_valid_isin(code):
        return code
    listings = lookup_isin(code, other_exchanges=False)
    if not listings:
        msg = f"No Yahoo Finance listing found for ISIN {code.upper()} (bonds and unlisted funds are not covered)."
        raise ValueError(msg)
    return listings[0]["symbol"]


def _yahoo_listings(isin: str) -> list[dict[str, str]]:
    try:
        quotes = yf.Search(isin, max_results=10, news_count=0).quotes
    except Exception:
        return []
    return [
        {
            "symbol": quote["symbol"],
            "name": " ".join((quote.get("longname") or quote.get("shortname") or quote["symbol"]).split()),
            "exchange": quote.get("exchDisp") or quote.get("exchange", ""),
            "type": quote.get("quoteType", ""),
        }
        for quote in quotes
        if quote.get("symbol")
    ]


def _openfigi_listings(isin: str) -> list[dict[str, str]]:
    try:
        response = requests.post(OPENFIGI_URL, json=[{"idType": "ID_ISIN", "idValue": isin}], timeout=15)
        response.raise_for_status()
        rows = response.json()[0].get("data", [])
    except (requests.RequestException, ValueError, LookupError):
        return []

    listings: dict[str, dict[str, str]] = {}
    for row in rows:
        venue = _EXCHANGES.get(row.get("exchCode", ""))
        ticker = row.get("ticker") or ""
        if venue is None or not ticker or " " in ticker:  # bonds come as descriptions, e.g. "BTPS 4.2 03/01/34"
            continue
        suffix, exchange = venue
        symbol = ticker.replace("/", "-") + suffix  # BRK/B → BRK-B
        listings.setdefault(
            symbol,
            {"symbol": symbol, "name": row.get("name") or symbol, "exchange": exchange, "type": row.get("securityType", "")},
        )
    # Borsa Italiana first: prices in EUR
    return sorted(listings.values(), key=lambda listing: not listing["symbol"].endswith(".MI"))


def _with_prices(listings: list[dict[str, str]], max_listings: int = 12) -> list[dict[str, str]]:
    """Keep the listings for which Yahoo Finance has recent prices."""
    listings = listings[:max_listings]
    if not listings:
        return []
    symbols = [listing["symbol"] for listing in listings]
    logger = logging.getLogger("yfinance")
    level = logger.level
    logger.setLevel(logging.CRITICAL)  # candidates without data are expected: don't log them as errors
    try:
        close = yf.download(symbols, period="5d", progress=False, auto_adjust=True)["Close"]
    except Exception:
        return []
    finally:
        logger.setLevel(level)
    traded = {symbol for symbol in close.columns if close[symbol].notna().any()}
    return [listing for listing in listings if listing["symbol"] in traded]
