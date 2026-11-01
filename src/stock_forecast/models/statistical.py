"""Statistical models.

- **Auto-ARIMA** – statsmodels SARIMAX with ADF-based differencing + AIC grid search.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import STATISTICAL, ForecastResult, Param, make_result, register

# ──────────────────────────────────────────────────────────────────────────────
# Stationarity helpers
# ──────────────────────────────────────────────────────────────────────────────


def _check_stationarity(y: np.ndarray, significance: float = 0.05) -> tuple[bool, float]:
    """Run Augmented Dickey-Fuller test.

    Returns (is_stationary, p_value).
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(y, autolag="AIC")
    p_value = result[1]
    return p_value < significance, p_value


def _make_stationary(y: np.ndarray, max_d: int = 2) -> tuple[np.ndarray, int]:
    """Difference the series until it passes the ADF test.

    Returns (stationary_series, d) where d is the number of differencings applied.
    """
    d = 0
    current = y.copy()
    for _ in range(max_d):
        is_stat, _p = _check_stationarity(current)
        if is_stat:
            break
        current = np.diff(current)
        d += 1
    return current, d


# ──────────────────────────────────────────────────────────────────────────────
# Auto-ARIMA (statsmodels)
# ──────────────────────────────────────────────────────────────────────────────


@register(
    "Auto-ARIMA",
    family=STATISTICAL,
    description="ARIMA con differenziazione scelta dal test ADF e ordini (p, q) cercati per AIC. Lento su serie lunghe.",
    params=(
        Param("max_p", "p massimo", 5, min_value=0, max_value=8, step=1, help="Ordine autoregressivo massimo."),
        Param("max_q", "q massimo", 5, min_value=0, max_value=8, step=1, help="Ordine media mobile massimo."),
    ),
)
def run_arima(
    df: pd.DataFrame,
    horizon: int,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
) -> ForecastResult:
    """Fit ARIMA with automatic stationarity enforcement and order selection.

    The function:
    1. Tests stationarity via ADF and determines the differencing order d.
    2. Searches (p, q) combinations via AIC to pick the best ARIMA(p, d, q).
    3. Forecasts *horizon* steps ahead with confidence intervals.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = df["y"].to_numpy(dtype=float)

    # ── Step 1: Determine d via ADF test ──────────────────────────────────────
    is_stationary, p_value = _check_stationarity(y)
    print(f"  📊 ADF test p-value: {p_value:.6f} → {'stationary' if is_stationary else 'non-stationary'}")

    _, d = _make_stationary(y, max_d=max_d)
    print(f"  📐 Differencing order d={d}")

    # ── Step 2: Grid search for best (p, q) by AIC ───────────────────────────
    print(f"  🔍 Searching ARIMA orders (p=0..{max_p}, d={d}, q=0..{max_q}) …")
    best_aic = float("inf")
    best_order = (1, d, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings are expected on poor candidate orders
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = SARIMAX(y, order=(p, d, q), enforce_stationarity=True, enforce_invertibility=True)
                    fit = model.fit(disp=False, maxiter=200)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

        print(f"  ✅ Best order: ARIMA{best_order} (AIC={best_aic:.2f})")

        # ── Step 3: Fit final model and forecast ──────────────────────────────
        final_model = SARIMAX(y, order=best_order, enforce_stationarity=True, enforce_invertibility=True)
        final_fit = final_model.fit(disp=False, maxiter=500)

    forecast_obj = final_fit.get_forecast(steps=horizon)
    yhat = np.asarray(forecast_obj.predicted_mean)
    conf_int = np.asarray(forecast_obj.conf_int(alpha=0.2))  # 80% CI

    return make_result("Auto-ARIMA", df, horizon, yhat, conf_int[:, 0], conf_int[:, 1])
