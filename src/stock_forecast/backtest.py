"""Backtesting: train/validation split and evaluation of models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .metrics import compute_metrics, failed_metrics
from .models import available_models, get_model

if TYPE_CHECKING:
    import pandas as pd


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


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_forecast(test_df: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    """Score a forecast against the held-out actuals."""
    n = len(test_df)
    return compute_metrics(test_df["y"].to_numpy(), forecast["yhat"].to_numpy()[:n])


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
# Validation of the candidate models
# ──────────────────────────────────────────────────────────────────────────────


def validate_models(
    df: pd.DataFrame,
    horizon: int,
    models: list[str] | None = None,
    chronos_size: str | None = None,
    max_steps: int | None = None,
    *,
    model_params: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Train each model on train set, evaluate on validation set.

    Parameters
    ----------
    df:
        Full time series with columns ``ds``, ``y``.
    horizon:
        Forecast horizon (also used as validation set size).
    models:
        List of model names to evaluate. Default: all available.
    chronos_size:
        Shortcut for the size of the "Chronos" model.
    max_steps:
        Shortcut for the training steps of the neural models.
    model_params:
        Extra keyword arguments per model, e.g. ``{"XGBoost": {"max_depth": 4}}``.

    Returns
    -------
    dict mapping model_name → {"metrics": {...}, "result": ForecastResult, "error": None or message}
    """
    if models is None:
        models = available_models()

    train_df, val_df = train_val_split(df, val_size=horizon)
    params = resolve_model_params(models, model_params, chronos_size, max_steps)

    results: dict[str, dict] = {}
    print(f"──────────── Models to validate: {', '.join(models)} ────────────")
    for name in models:
        print(f"\n── Validating: {name} ─────────────────────────────────")
        spec = get_model(name)
        try:
            result = spec(train_df, horizon, **params[name])
            metrics = evaluate_forecast(val_df, result.forecast)
        except Exception as e:
            print(f"  ⚠️  {name} failed: {e}")
            results[name] = {"metrics": failed_metrics(), "result": None, "error": str(e)}
            continue

        result.metrics = metrics
        results[name] = {"metrics": metrics, "result": result, "error": None}
        print(f"  ✅ MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}%")

    return results
