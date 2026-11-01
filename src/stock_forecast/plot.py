"""Plotting utilities – Plotly interactive charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .ensemble import EnsembleResult
from .models import MODEL_REGISTRY, ForecastResult

# ── Palette ───────────────────────────────────────────────────────────────────
# Categorical hues (dark-mode steps) in a fixed, colour-blind-checked order.
_SERIES_COLORS = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
_HISTORY_COLOR = "#c3c2b7"
_INCREASING_COLOR = "#26a69a"
_DECREASING_COLOR = "#ef5350"
# Horizontal legend under the plot: with many models it would collide with the title / range buttons on top.
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.1, "xanchor": "left", "x": 0}


def _series_style(name: str) -> dict:
    """Line style that follows the model across every chart (never its rank).

    The ensemble owns the first hue; registered models take the other seven in
    registry order, then repeat them with a dotted line so no two models look alike.
    """
    if name.startswith("Ensemble"):
        return {"color": _SERIES_COLORS[0], "width": 3, "dash": "solid"}

    # Results carry descriptive names ("Chronos-Bolt-small"): match the longest registry prefix.
    names = list(MODEL_REGISTRY)
    matches = [n for n in names if name.startswith(n)]
    i = names.index(max(matches, key=len)) if matches else len(names)
    n_hues = len(_SERIES_COLORS) - 1
    return {"color": _SERIES_COLORS[1 + i % n_hues], "width": 2, "dash": "dash" if i < n_hues else "dot"}


# ── Main chart ────────────────────────────────────────────────────────────────


def plot_forecast(
    results: dict[str, ForecastResult] | ForecastResult,
    ticker: str = "",
    title: str | None = None,
    show_volume: bool = True,
    last_n_days: int | None = 365,
    show_members: bool = False,
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
    show_members:
        For ensemble results, also draw the forecast of each member model.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if isinstance(results, ForecastResult):
        results = {results.model_name: results}

    if not results:
        msg = "No forecast results to plot. All models may have failed."
        raise ValueError(msg)

    # Pick any result to get the training series (they share the same history)
    any_result = next(iter(results.values()))
    hist = any_result.train_df.copy()

    if last_n_days is not None:
        cutoff = hist["ds"].max() - pd.Timedelta(days=last_n_days)
        hist = hist[hist["ds"] >= cutoff]

    has_volume = "Volume" in hist.columns and hist["Volume"].notna().any() and hist["Volume"].gt(0).any()
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
                name="Prezzo",
                increasing_line_color=_INCREASING_COLOR,
                decreasing_line_color=_DECREASING_COLOR,
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
                name="Storico",
                line={"color": _HISTORY_COLOR, "width": 1.5},
            ),
            row=1,
            col=1,
        )

    # ── Volume bars ───────────────────────────────────────────────────────────
    if rows == 2:
        colors = [
            _INCREASING_COLOR if c >= o else _DECREASING_COLOR
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
        style = _series_style(model_name)

        # Confidence band
        if "yhat_lo" in fc.columns and "yhat_hi" in fc.columns:
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                    y=pd.concat([fc["yhat_hi"], fc["yhat_lo"].iloc[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({_hex_to_rgb(style['color'])},0.15)",
                    line={"color": "rgba(0,0,0,0)"},
                    name=f"{model_name} intervallo 80%",
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Ensemble members: thin lines, no band
        if show_members and isinstance(result, EnsembleResult):
            for member_name, member in result.members.items():
                member_style = {**_series_style(member_name), "width": 1.5}
                weight = result.weights.get(member_name, 0.0)
                fig.add_trace(
                    go.Scatter(
                        x=member.forecast["ds"],
                        y=member.forecast["yhat"],
                        mode="lines",
                        name=f"{member_name} (peso {weight:.0%})",
                        line=member_style,
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
                name=f"{model_name} previsione",
                line=style,
            ),
            row=1,
            col=1,
        )

    # ── Vertical line at forecast start ───────────────────────────────────────
    split_date = any_result.train_df["ds"].max()
    fig.add_vline(
        x=split_date,
        line_dash="dot",
        line_color="gray",
    )
    fig.add_annotation(
        x=split_date,
        y=1,
        yref="paper",
        text="Inizio previsione",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={"color": "gray", "size": 11},
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    auto_title = title or (f"{ticker} – Previsione" if ticker else "Previsione")
    fig.update_layout(
        title={"text": auto_title, "font": {"size": 20}},
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600 if rows == 1 else 750,
        hovermode="x unified",
        legend=_LEGEND_BELOW,
        margin={"l": 60, "r": 40, "t": 80, "b": 90},
    )
    fig.update_yaxes(title_text="Prezzo", row=1, col=1)
    fig.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ],
            # plotly_dark leaves the buttons light grey with white text: make them readable
            "bgcolor": "#2c2c2a",
            "activecolor": _SERIES_COLORS[0],
            "font": {"color": "#f0efec"},
        },
        row=1,
        col=1,
    )

    return fig


# ── Backtest chart ────────────────────────────────────────────────────────────


def plot_backtest(
    validation_results: dict[str, dict],
    df: pd.DataFrame,
    title: str | None = None,
    context_windows: int = 2,
) -> go.Figure:
    """Actual series over the backtest period with each model's out-of-sample forecasts.

    Parameters
    ----------
    validation_results:
        Output of ``validate_models`` / ``run_pipeline``.
    df:
        The full series the backtest was run on.
    context_windows:
        How many window lengths of history to show before the first window.
    """
    succeeded = {name: data for name, data in validation_results.items() if data.get("windows")}
    if not succeeded:
        msg = "No successful model to plot."
        raise ValueError(msg)

    df = df.sort_values("ds").reset_index(drop=True)
    windows = next(iter(succeeded.values()))["windows"]
    horizon = windows[0].horizon
    first_cut = len(windows[0].train_df)
    hist = df.iloc[max(first_cut - context_windows * horizon, 0) :]

    fig = go.Figure()

    # ── Shade the validation windows ──────────────────────────────────────────
    for w, result in enumerate(windows):
        test_ds = df["ds"].iloc[len(result.train_df) : len(result.train_df) + horizon]
        fig.add_vrect(
            x0=test_ds.iloc[0],
            x1=test_ds.iloc[-1],
            fillcolor="white",
            opacity=0.04 if w % 2 == 0 else 0.08,
            line_width=0,
        )

    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            mode="lines",
            name="Reale",
            line={"color": _HISTORY_COLOR, "width": 1.5},
        )
    )

    # ── Out-of-sample forecasts, plotted on the actual dates they were scored against ──
    for name, data in succeeded.items():
        style = _series_style(name)
        for w, result in enumerate(data["windows"]):
            n_train = len(result.train_df)
            x = df["ds"].iloc[n_train : n_train + len(result.forecast)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=result.forecast["yhat"].to_numpy()[: len(x)],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=w == 0,
                    line=style,
                )
            )

    fig.update_layout(
        title={"text": title or f"Backtest su {len(windows)} finestre da {horizon} barre", "font": {"size": 18}},
        template="plotly_dark",
        height=560,
        hovermode="x unified",
        legend=_LEGEND_BELOW,
        margin={"l": 60, "r": 40, "t": 80, "b": 90},
    )
    fig.update_yaxes(title_text="Prezzo")
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
        last_price = float(result.train_df["y"].iloc[-1])
        last_forecast = float(fc["yhat"].iloc[-1])
        pct_change = (last_forecast / last_price - 1) * 100

        print(f"  ┌─ {name}")
        print(f"  │  Horizon      : {result.horizon} steps")
        print(f"  │  Last price   : {last_price:,.4f}")
        print(f"  │  Forecast end : {last_forecast:,.4f}  ({pct_change:+.2f}%)")

        if "yhat_lo" in fc.columns:
            lo = float(fc["yhat_lo"].iloc[-1])
            hi = float(fc["yhat_hi"].iloc[-1])
            print(f"  │  80% CI       : [{lo:,.4f} – {hi:,.4f}]")
        if isinstance(result, EnsembleResult):
            weights = ", ".join(f"{member} {weight:.0%}" for member, weight in result.weights.items())
            print(f"  │  Members      : {weights}")
        if result.metrics:
            print(f"  │  Val MAPE     : {result.metrics.get('mape', 0):.2f}%")
            if "rel_mae" in result.metrics:
                print(f"  │  Val MAE/Naive: {result.metrics['rel_mae']:.3f}")
        print(f"  └{'─' * 40}\n")


def print_validation_table(validation_results: dict[str, dict], best_name: str | None = None) -> None:
    """Print a comparison table of all validated models."""
    print(f"\n{'═' * 78}")
    print(f"  {'Model':<15} {'MAE':>10} {'RMSE':>10} {'MAPE':>9} {'MAE/Naive':>10} {'Cov.80':>8}")
    print(f"  {'─' * 73}")
    for name, data in validation_results.items():
        m = data["metrics"]
        marker = " 🏆" if name == best_name else ""
        if m["mae"] == float("inf"):
            print(f"  {name:<15} {'FAILED':>10}{marker}")
        else:
            rel_mae = f"{m['rel_mae']:>10.3f}" if "rel_mae" in m else f"{'—':>10}"
            coverage = f"{m['coverage']:>7.0f}%" if "coverage" in m else f"{'—':>8}"
            print(f"  {name:<15} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['mape']:>8.2f}% {rel_mae} {coverage}{marker}")
    print(f"{'═' * 78}\n")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string for rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
