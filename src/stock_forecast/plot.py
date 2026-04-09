"""Plotting utilities – Plotly interactive charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import ForecastResult

# ── Palette ───────────────────────────────────────────────────────────────────
_COLORS = {
    "history": "#636EFA",
    "Chronos-tiny": "#EF553B",
    "Chronos-mini": "#EF553B",
    "Chronos-small": "#EF553B",
    "Chronos-base": "#EF553B",
    "Chronos-large": "#EF553B",
    "N-HiTS": "#00CC96",
    "N-BEATS": "#AB63FA",
    "default": "#FFA15A",
}


def _model_color(name: str) -> str:
    for k, v in _COLORS.items():
        if k in name:
            return v
    return _COLORS["default"]


# ── Main chart ────────────────────────────────────────────────────────────────


def plot_forecast(
    results: dict[str, ForecastResult] | ForecastResult,
    ticker: str = "",
    title: str | None = None,
    show_volume: bool = True,
    last_n_days: int | None = 365,
) -> go.Figure:
    """Build an interactive Plotly figure with history + forecasts.

    Parameters
    ----------
    results:
        Single ``ForecastResult`` or a dict ``{name: ForecastResult}``.
    ticker:
        Used in the chart title.
    title:
        Override auto title.
    show_volume:
        If ``True`` and volume data is present, add a secondary volume subplot.
    last_n_days:
        Trim history to the last N calendar days for readability.
        ``None`` = show all.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if isinstance(results, ForecastResult):
        results = {results.model_name: results}

    # Pick any result to get the training series (they share the same history)
    any_result = next(iter(results.values()))
    hist = any_result.train_df.copy()

    if last_n_days is not None:
        cutoff = hist["ds"].max() - pd.Timedelta(days=last_n_days)
        hist = hist[hist["ds"] >= cutoff]

    has_volume = "Volume" in hist.columns and hist["Volume"].notna().any()
    rows = 2 if (show_volume and has_volume) else 1
    row_heights = [0.75, 0.25] if rows == 2 else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # ── Candlestick (if OHLC available) or line ───────────────────────────────
    ohlc_cols = {"Open", "High", "Low", "Close"}
    if ohlc_cols.issubset(hist.columns):
        fig.add_trace(
            go.Candlestick(
                x=hist["ds"],
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=hist["ds"],
                y=hist["y"],
                mode="lines",
                name="History",
                line={"color": _COLORS["history"], "width": 1.5},
            ),
            row=1,
            col=1,
        )

    # ── Volume bars ───────────────────────────────────────────────────────────
    if rows == 2:
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(hist.get("Close", hist["y"]), hist.get("Open", hist["y"]), strict=False)
        ]
        fig.add_trace(
            go.Bar(
                x=hist["ds"],
                y=hist["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.6,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    # ── Forecast traces ───────────────────────────────────────────────────────
    for model_name, result in results.items():
        fc = result.forecast
        color = _model_color(model_name)

        # Confidence band
        if "yhat_lo" in fc.columns and "yhat_hi" in fc.columns:
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                    y=pd.concat([fc["yhat_hi"], fc["yhat_lo"].iloc[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({_hex_to_rgb(color)},0.15)",
                    line={"color": "rgba(0,0,0,0)"},
                    name=f"{model_name} 80% CI",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Point forecast
        fig.add_trace(
            go.Scatter(
                x=fc["ds"],
                y=fc["yhat"],
                mode="lines",
                name=f"{model_name} forecast",
                line={"color": color, "width": 2, "dash": "dash"},
            ),
            row=1,
            col=1,
        )

    # ── Vertical line at forecast start ───────────────────────────────────────
    split_date = any_result.train_df["ds"].max()
    fig.add_vline(
        x=split_date.timestamp() * 1000,
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast start",
        annotation_position="top left",
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    auto_title = title or f"{ticker} – Price Forecast" if ticker else "Price Forecast"
    fig.update_layout(
        title={"text": auto_title, "font": {"size": 20}},
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600 if rows == 1 else 750,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 60, "r": 40, "t": 80, "b": 40},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
        row=1,
        col=1,
    )

    return fig


# ── Summary table ─────────────────────────────────────────────────────────────


def print_summary(
    results: dict[str, ForecastResult],
    ticker_info: dict | None = None,
) -> None:
    """Print a text summary of forecasts to stdout."""
    info = ticker_info or {}
    if info:
        print(f"\n{'═' * 55}")
        print(f"  {info.get('name', info.get('ticker', ''))}  ({info.get('ticker', '')})")
        print(f"  Currency : {info.get('currency', '—')}")
        print(f"  Sector   : {info.get('sector', '—')}")
        print(f"{'═' * 55}\n")

    for name, result in results.items():
        fc = result.forecast

        # .item() is the safest way to extract a single scalar from
        # numpy arrays or pandas objects regardless of dimensionality.
        last_price = result.train_df["y"].iloc[-1]
        if hasattr(last_price, "item"):
            last_price = last_price.item()

        last_forecast = fc["yhat"].values[-1]
        if hasattr(last_forecast, "item"):
            last_forecast = last_forecast.item()

        pct_change = (last_forecast / last_price - 1) * 100

        print(f"  ┌─ {name}")
        print(f"  │  Horizon      : {result.horizon} steps")
        print(f"  │  Last price   : {last_price:,.4f}")
        print(f"  │  Forecast end : {last_forecast:,.4f}  ({pct_change:+.2f}%)")

        if "yhat_lo" in fc.columns:
            lo_raw = fc["yhat_lo"].values.ravel()[-1]
            hi_raw = fc["yhat_hi"].values.ravel()[-1]

            lo = float(lo_raw)
            hi = float(hi_raw)

            print(f"  │  80% CI       : [{lo:,.4f} – {hi:,.4f}]")
        print(f"  └{'─' * 40}\n")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string for rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
