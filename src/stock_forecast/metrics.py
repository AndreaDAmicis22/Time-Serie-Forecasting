"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np

# Metrics where lower is better – valid choices for model selection.
SELECTION_METRICS = ("mae", "rmse", "mape")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE between actual and predicted values."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("inf")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def failed_metrics() -> dict[str, float]:
    """Placeholder metrics for a model that failed (always ranks last)."""
    inf = float("inf")
    return {"mae": inf, "rmse": inf, "mape": inf}
