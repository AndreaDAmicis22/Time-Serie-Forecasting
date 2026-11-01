"""Building blocks shared by every forecasting model.

- ``ForecastResult`` – common return type of all model runners.
- ``ModelSpec`` / ``Param`` – model metadata: family, description and tunable
  parameters (used by the pipeline and to build the web-app settings).
- ``register`` – decorator that adds a runner to ``MODEL_REGISTRY``.
- Date / result helpers so each runner only has to produce the numbers.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# Lower / median / upper quantiles: every model reports an 80% prediction interval.
QUANTILES = (0.1, 0.5, 0.9)
Z_80 = 1.2816  # standard-normal quantile for a two-sided 80% interval

# Model families, in the order the web app shows them.
FOUNDATION = "Foundation (zero-shot)"
NEURAL = "Deep learning"
ML = "Machine learning"
STATISTICAL = "Statistici"
BASELINE = "Baseline"
FAMILIES = (FOUNDATION, NEURAL, ML, STATISTICAL, BASELINE)


# ──────────────────────────────────────────────────────────────────────────────
# Return type
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ForecastResult:
    """Container for a forecast produced by any model."""

    model_name: str
    horizon: int
    forecast: pd.DataFrame  # columns: ds, yhat, yhat_lo, yhat_hi
    train_df: pd.DataFrame  # the series used for fitting
    metrics: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ForecastResult(model={self.model_name!r}, horizon={self.horizon}, rows={len(self.forecast)})"


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Param:
    """A tunable model parameter, rendered as a widget in the web-app settings.

    ``choices`` → dropdown; otherwise a number input bounded by ``min_value`` / ``max_value``.
    """

    name: str
    label: str
    default: Any
    choices: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    help: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry: how to run a model plus the metadata shown in the UI."""

    name: str
    runner: Callable[..., ForecastResult]
    family: str
    description: str
    params: tuple[Param, ...] = ()
    requires: tuple[str, ...] = ()  # importable modules the model needs (optional extras)
    extra: str | None = None  # pyproject extra that installs them
    slow: bool = False  # trains from scratch on every call

    @property
    def available(self) -> bool:
        """``True`` when the optional dependencies of the model are installed."""
        return all(importlib.util.find_spec(module) is not None for module in self.requires)

    @property
    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def __call__(self, df: pd.DataFrame, horizon: int, **params: Any) -> ForecastResult:
        return self.runner(df, horizon, **params)


# Registry of model runners: name → ModelSpec (callable as runner(df, horizon, **params)).
MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register(
    name: str,
    *,
    family: str,
    description: str,
    params: tuple[Param, ...] = (),
    requires: tuple[str, ...] = (),
    extra: str | None = None,
    slow: bool = False,
) -> Callable[[Callable[..., ForecastResult]], Callable[..., ForecastResult]]:
    """Decorator adding a runner ``fn(df, horizon, **params) -> ForecastResult`` to the registry."""

    def decorator(runner: Callable[..., ForecastResult]) -> Callable[..., ForecastResult]:
        MODEL_REGISTRY[name] = ModelSpec(
            name=name,
            runner=runner,
            family=family,
            description=description,
            params=params,
            requires=requires,
            extra=extra,
            slow=slow,
        )
        return runner

    return decorator


def get_model(name: str) -> ModelSpec:
    """Look up a registered model, with a helpful error for unknown names."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        msg = f"Unknown model {name!r}. Available: {', '.join(MODEL_REGISTRY)}"
        raise KeyError(msg) from None


def available_models() -> list[str]:
    """Names of the registered models whose dependencies are installed."""
    return [name for name, spec in MODEL_REGISTRY.items() if spec.available]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def infer_freq(ds: pd.Series) -> str:
    """Infer pandas frequency string from a datetime series."""
    ds = ds.sort_values()
    median_days = ds.diff().dropna().dt.days.median()
    if median_days <= 1.5:
        # Markets trading on weekends (crypto) need calendar days, not business days.
        weekend_share = (ds.dt.dayofweek >= 5).mean()
        return "D" if weekend_share > 0.1 else "B"
    if median_days <= 8:
        # Anchor on the weekday of the bars (yfinance weekly bars start on Monday).
        return f"W-{ds.iloc[-1].day_name()[:3].upper()}"
    if median_days <= 32:
        return "MS"
    return "QS"


def future_dates(last_date: pd.Timestamp, horizon: int, freq: str) -> pd.DatetimeIndex:
    """Generate *horizon* future business/calendar dates after *last_date*."""
    if freq == "B":
        return pd.bdate_range(start=last_date, periods=horizon + 1, freq="B")[1:]
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]


def make_result(
    model_name: str,
    df: pd.DataFrame,
    horizon: int,
    yhat: np.ndarray,
    yhat_lo: np.ndarray,
    yhat_hi: np.ndarray,
) -> ForecastResult:
    """Wrap raw forecast arrays into a ``ForecastResult`` with future dates."""
    yhat = np.asarray(yhat, dtype=float).ravel()[:horizon]
    # Guard against quantile crossing: the band must always contain the point forecast.
    yhat_lo = np.minimum(np.asarray(yhat_lo, dtype=float).ravel()[:horizon], yhat)
    yhat_hi = np.maximum(np.asarray(yhat_hi, dtype=float).ravel()[:horizon], yhat)

    future_ds = future_dates(df["ds"].iloc[-1], horizon, infer_freq(df["ds"]))
    forecast_df = pd.DataFrame({"ds": future_ds, "yhat": yhat, "yhat_lo": yhat_lo, "yhat_hi": yhat_hi})
    return ForecastResult(model_name=model_name, horizon=horizon, forecast=forecast_df, train_df=df)
