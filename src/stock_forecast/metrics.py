"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np

# Metrics where lower is better – valid choices for model selection.
SELECTION_METRICS = ("mae", "rmse", "mape", "rel_mae")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train: np.ndarray | None = None,
    y_lo: np.ndarray | None = None,
    y_hi: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE between actual and predicted values.

    Optional extras:

    - ``y_train`` → **rel_mae**: MAE divided by the MAE of the naive random walk
      (last training value repeated) on the same window. Below 1 means the model
      beats the random walk – the real bar to clear on price series.
    - ``y_lo`` / ``y_hi`` → **coverage**: % of actuals inside the prediction
      interval (≈80 for a well-calibrated 80% interval).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("inf")
    metrics = {"mae": mae, "rmse": rmse, "mape": mape}

    if y_train is not None:
        naive_mae = np.mean(np.abs(y_true - np.asarray(y_train, dtype=float).ravel()[-1]))
        metrics["rel_mae"] = float(mae / naive_mae) if naive_mae > 0 else float("inf")

    if y_lo is not None and y_hi is not None:
        inside = (y_true >= np.asarray(y_lo, dtype=float).ravel()) & (y_true <= np.asarray(y_hi, dtype=float).ravel())
        metrics["coverage"] = float(inside.mean() * 100)

    return metrics


def failed_metrics() -> dict[str, float]:
    """Placeholder metrics for a model that failed (always ranks last)."""
    inf = float("inf")
    return {"mae": inf, "rmse": inf, "mape": inf, "rel_mae": inf, "coverage": float("nan")}
