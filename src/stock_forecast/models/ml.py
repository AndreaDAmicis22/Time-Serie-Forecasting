"""Machine-learning models: gradient boosting on lag features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .base import ML, Z_80, ForecastResult, Param, make_result, register

if TYPE_CHECKING:
    import pandas as pd


def _create_lag_features(y: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised learning dataset from a 1-D time series.

    Returns (X, Y) where each row of X contains [y(t-n_lags), ..., y(t-1)]
    and Y contains y(t).
    """
    windows = np.lib.stride_tricks.sliding_window_view(y, n_lags + 1)
    return windows[:, :-1], windows[:, -1]


@register(
    "XGBoost",
    family=ML,
    description="Gradient boosting su feature di lag, previsione ricorsiva passo dopo passo.",
    params=(
        Param("n_estimators", "Numero di alberi", 500, min_value=50, max_value=5000, step=50),
        Param("max_depth", "Profondità massima", 6, min_value=2, max_value=12, step=1),
        Param("learning_rate", "Learning rate", 0.05, min_value=0.005, max_value=0.5, step=0.005),
        Param(
            "target",
            "Target",
            "diff",
            choices=("diff", "level"),
            help="diff = modella le variazioni (consigliato: gli alberi non estrapolano oltre i valori visti); "
            "level = modella direttamente il prezzo.",
        ),
    ),
)
def run_xgboost(
    df: pd.DataFrame,
    horizon: int,
    n_lags: int | None = None,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    target: Literal["diff", "level"] = "diff",
) -> ForecastResult:
    """Train XGBoost regressor with lag features for time series forecasting.

    Uses a recursive multi-step strategy: predict one step, feed prediction
    back as a feature for the next step. With ``target="diff"`` the model
    learns the period-over-period changes, which are then cumulated.
    """
    from xgboost import XGBRegressor

    y = df["y"].to_numpy(dtype=float)
    series = np.diff(y) if target == "diff" else y

    if n_lags is None:
        n_lags = min(max(horizon * 3, 30), len(series) // 3)

    X, Y = _create_lag_features(series, n_lags)

    print(f"  🌲 Training XGBoost (lags={n_lags}, estimators={n_estimators}, target={target}) …")
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective="reg:squarederror",
        verbosity=0,
        random_state=42,
    )
    model.fit(X, Y)

    # ── Recursive forecasting ─────────────────────────────────────────────────
    print(f"  🔮 Forecasting {horizon} steps (recursive) …")
    last_window = series[-n_lags:].copy()
    predictions = []

    for _ in range(horizon):
        yhat_step = model.predict(last_window.reshape(1, -1))[0]
        predictions.append(yhat_step)
        last_window = np.append(last_window[1:], yhat_step)

    predictions = np.array(predictions)
    yhat = y[-1] + np.cumsum(predictions) if target == "diff" else predictions

    # ── Confidence interval via residual std on training set ───────────────────
    # In-sample residuals of a boosted model are overfit (≈0): floor them at the random-walk volatility.
    residual_std = max(np.std(Y - model.predict(X)), np.std(np.diff(y)))
    # Widen intervals further into the future
    interval_width = Z_80 * residual_std * np.sqrt(np.arange(1, horizon + 1))

    return make_result("XGBoost", df, horizon, yhat, yhat - interval_width, yhat + interval_width)
