"""Pretrained foundation models: zero-shot forecasting, no training needed.

- **Chronos** (Amazon, 2024) – T5-based, sampling.

Loaded models are cached in memory, so repeated calls (backtest windows,
web-app reruns) only pay the loading cost once.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

from .base import FOUNDATION, QUANTILES, ForecastResult, Param, make_result, register

if TYPE_CHECKING:
    import pandas as pd


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Chronos family (Amazon) – chronos-forecasting
# ──────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=2)
def _load_chronos(model_id: str) -> Any:
    import torch
    from chronos import BaseChronosPipeline

    device = _device()
    print(f"  ⏳ Loading {model_id} on {device} …")
    return BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )


def _chronos_forecast(
    df: pd.DataFrame,
    horizon: int,
    model_id: str,
    model_name: str,
    **predict_kwargs: Any,
) -> ForecastResult:
    import torch

    pipe = _load_chronos(model_id)
    context = torch.tensor(df["y"].to_numpy(), dtype=torch.float32)

    print(f"  🔮 Forecasting {horizon} steps ({model_name}) …")
    quantiles, _mean = pipe.predict_quantiles(
        [context],
        prediction_length=horizon,
        quantile_levels=list(QUANTILES),
        **predict_kwargs,
    )
    q = quantiles[0].reshape(horizon, len(QUANTILES)).float().numpy()
    return make_result(model_name, df, horizon, q[:, 1], q[:, 0], q[:, 2])


@register(
    "Chronos",
    family=FOUNDATION,
    description="Amazon, 2024 · Apache-2.0. Transformer T5 zero-shot basato su campionamento.",
    params=(
        Param("model_size", "Dimensione", "small", choices=("tiny", "mini", "small", "base", "large")),
        Param("num_samples", "Campioni", 20, min_value=5, max_value=200, step=5, help="Traiettorie campionate."),
    ),
)
def run_chronos(
    df: pd.DataFrame,
    horizon: int,
    model_size: Literal["tiny", "mini", "small", "base", "large"] = "small",
    num_samples: int = 20,
) -> ForecastResult:
    """Run the original Amazon Chronos (T5, zero-shot, sampling-based).

    Parameters
    ----------
    df:
        DataFrame with columns ``ds`` (datetime) and ``y`` (float).
    horizon:
        Number of future steps to predict.
    model_size:
        Chronos variant – ``"tiny"`` is fastest, ``"large"`` most accurate.
    num_samples:
        Posterior samples used to build prediction intervals.
    """
    return _chronos_forecast(
        df,
        horizon,
        f"amazon/chronos-t5-{model_size}",
        f"Chronos-{model_size}",
        num_samples=num_samples,
    )
