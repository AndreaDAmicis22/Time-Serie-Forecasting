"""Forecasting models.

Every model is a function ``run_xxx(df, horizon, **params) -> ForecastResult``
registered in ``MODEL_REGISTRY`` with its metadata (family, description,
tunable parameters, optional dependencies).

To add a model, write the runner in the module of its family and decorate it
with ``@register(...)``: the pipeline and the web app pick it up automatically.
"""

from .base import FAMILIES, MODEL_REGISTRY, ForecastResult, ModelSpec, Param, available_models, get_model, register
from .foundation import run_chronos
from .ml import run_xgboost
from .neural import run_nbeats, run_nhits
from .statistical import run_arima, run_drift, run_ets, run_naive, run_theta

__all__ = [
    "FAMILIES",
    "MODEL_REGISTRY",
    "ForecastResult",
    "ModelSpec",
    "Param",
    "available_models",
    "get_model",
    "register",
    "run_arima",
    "run_chronos",
    "run_drift",
    "run_ets",
    "run_naive",
    "run_nbeats",
    "run_nhits",
    "run_theta",
    "run_xgboost",
]
