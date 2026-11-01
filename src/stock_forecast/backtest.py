"""Backtesting: train/validation splits and rolling-origin evaluation of models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .metrics import compute_metrics, failed_metrics
from .models import available_models, get_model

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    # progress(step, total, message): lets callers (e.g. the web app) follow long runs.
    ProgressCallback = Callable[[int, int, str], None]

MIN_TRAIN_SIZE = 30  # observations required before the first validation window


# ──────────────────────────────────────────────────────────────────────────────
# Train / Validation split
# ──────────────────────────────────────────────────────────────────────────────


def train_val_split(df: pd.DataFrame, val_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into train and validation sets.

    Parameters
    ----------
    df:
        DataFrame with at least columns ``ds`` and ``y``.
    val_size:
        Number of most-recent observations to hold out for validation.

    Returns
    -------
    (train_df, val_df)
    """
    df = df.sort_values("ds").reset_index(drop=True)
    split_idx = len(df) - val_size
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def backtest_windows(df: pd.DataFrame, horizon: int, n_windows: int = 1) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling-origin splits: *n_windows* consecutive validation blocks of *horizon* observations.

    Windows are ordered oldest → newest. The last one holds out the final
    *horizon* observations, so ``n_windows=1`` is the same as ``train_val_split``.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    min_len = horizon * n_windows + MIN_TRAIN_SIZE
    if len(df) < min_len:
        msg = (
            f"Series too short for {n_windows} backtest window(s) of {horizon} steps: "
            f"{len(df)} observations, at least {min_len} needed."
        )
        raise ValueError(msg)

    splits = []
    for w in range(n_windows, 0, -1):
        cutoff = len(df) - w * horizon
        splits.append((df.iloc[:cutoff].copy(), df.iloc[cutoff : cutoff + horizon].copy()))
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    """Score a forecast against the held-out actuals."""
    n = len(test_df)
    return compute_metrics(
        test_df["y"].to_numpy(),
        forecast["yhat"].to_numpy()[:n],
        y_train=train_df["y"].to_numpy(),
        y_lo=forecast["yhat_lo"].to_numpy()[:n],
        y_hi=forecast["yhat_hi"].to_numpy()[:n],
    )


def average_metrics(per_window: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric over the backtest windows."""
    return {key: float(np.mean([m[key] for m in per_window])) for key in per_window[0]}


def resolve_model_params(
    models: list[str],
    model_params: dict[str, dict] | None = None,
    chronos_size: str | None = None,
    max_steps: int | None = None,
) -> dict[str, dict]:
    """Per-model keyword arguments, merging the legacy ``chronos_size`` / ``max_steps`` shortcuts."""
    params = {name: dict((model_params or {}).get(name, {})) for name in models}
    if chronos_size is not None and "Chronos" in params:
        params["Chronos"].setdefault("model_size", chronos_size)
    if max_steps is not None:
        for name in ("N-HiTS", "N-BEATS"):
            if name in params:
                params[name].setdefault("max_steps", max_steps)
    return params


# ──────────────────────────────────────────────────────────────────────────────
# Backtest of the candidate models
# ──────────────────────────────────────────────────────────────────────────────


def validate_models(
    df: pd.DataFrame,
    horizon: int,
    models: list[str] | None = None,
    chronos_size: str | None = None,
    max_steps: int | None = None,
    *,
    n_windows: int = 1,
    model_params: dict[str, dict] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, dict]:
    """Train each model before every validation window and evaluate it on the window.

    Parameters
    ----------
    df:
        Full time series with columns ``ds``, ``y``.
    horizon:
        Forecast horizon (also the size of each validation window).
    models:
        List of model names to evaluate. Default: all available.
    chronos_size:
        Shortcut for the size of the original "Chronos" model.
    max_steps:
        Shortcut for the training steps of the neural models.
    n_windows:
        Number of rolling-origin validation windows; metrics are averaged over them.
    model_params:
        Extra keyword arguments per model, e.g. ``{"XGBoost": {"max_depth": 4}}``.
    progress:
        Optional callback ``progress(step, total, message)``.

    Returns
    -------
    dict mapping model_name → {
        "metrics": metrics averaged over the windows,
        "result": ForecastResult of the most recent window,
        "windows": list of ForecastResult (oldest → newest),
        "error": None, or why the model failed,
    }
    """
    if models is None:
        models = available_models()

    windows = backtest_windows(df, horizon, n_windows)
    params = resolve_model_params(models, model_params, chronos_size, max_steps)
    total = len(models) * n_windows

    results: dict[str, dict] = {}
    print(f"──────────── Models to validate: {', '.join(models)} ────────────")
    for i, name in enumerate(models):
        print(f"\n── Validating: {name} ─────────────────────────────────")
        spec = get_model(name)
        if not spec.available:
            error = f"missing optional dependency, install it with: poetry install --extras {spec.extra}"
            print(f"  ⚠️  {name} skipped: {error}")
            results[name] = {"metrics": failed_metrics(), "result": None, "windows": [], "error": error}
            continue

        window_results, window_metrics = [], []
        try:
            for w, (train_df, test_df) in enumerate(windows):
                if progress is not None:
                    progress(i * n_windows + w, total, f"Backtest {name} – finestra {w + 1}/{n_windows}")
                result = spec(train_df, horizon, **params[name])
                window_results.append(result)
                window_metrics.append(evaluate_forecast(train_df, test_df, result.forecast))

        except Exception as e:
            print(f"  ⚠️  {name} failed: {e}")
            results[name] = {"metrics": failed_metrics(), "result": None, "windows": [], "error": str(e)}
            continue

        metrics = average_metrics(window_metrics)
        window_results[-1].metrics = metrics
        results[name] = {"metrics": metrics, "result": window_results[-1], "windows": window_results, "error": None}
        print(
            f"  ✅ MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAPE={metrics['mape']:.2f}%  MAE/Naive={metrics['rel_mae']:.3f}"
        )

    return results
