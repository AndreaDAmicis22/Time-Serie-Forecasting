"""Pretrained foundation models: zero-shot forecasting, no training needed.

- **Chronos-2** (Amazon, Oct 2025) – 120M params, Apache-2.0. Default choice.
- **Chronos-Bolt** (Amazon, Nov 2024) – fast direct-quantile Chronos, Apache-2.0.
- **Chronos** (Amazon, 2024) – original T5 sampling model, kept for comparison.
- **Timestral** (Mistral AI, Oct 2026) – optional extra ``timestral``, modified-MIT weights.
- **TimesFM-3** (Google, Aug 2026) – optional extra ``timesfm``, non-commercial weights.
- **TiRex** (NX-AI, 2025) – optional extra ``tirex``, NXAI community license.

Loaded models are cached in memory, so repeated calls (backtest windows,
web-app reruns) only pay the loading cost once.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .base import FOUNDATION, QUANTILES, ForecastResult, Param, make_result, register

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_quantiles(quantiles: np.ndarray, levels: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the (lower, median, upper) columns of a ``(horizon, n_levels)`` quantile array."""
    levels = np.asarray(levels, dtype=float)
    lo, med, hi = (quantiles[:, int(np.argmin(np.abs(levels - q)))] for q in QUANTILES)
    return lo, med, hi


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
    # T5/Bolt return (batch, horizon, q); Chronos-2 a list of (n_variates, horizon, q).
    q = quantiles[0].reshape(horizon, len(QUANTILES)).float().numpy()
    return make_result(model_name, df, horizon, q[:, 1], q[:, 0], q[:, 2])


@register(
    "Chronos-2",
    family=FOUNDATION,
    description="Amazon, ott. 2025 · 120M parametri · Apache-2.0. Zero-shot, tra i migliori su GIFT-Eval/fev-bench.",
)
def run_chronos2(df: pd.DataFrame, horizon: int) -> ForecastResult:
    """Run Amazon Chronos-2 (zero-shot, encoder-only, up to 8192 context points)."""
    return _chronos_forecast(df, horizon, "amazon/chronos-2", "Chronos-2")


@register(
    "Chronos-Bolt",
    family=FOUNDATION,
    description="Amazon, nov. 2024 · Apache-2.0. Variante veloce di Chronos (fino a 250× più rapida).",
    params=(Param("model_size", "Dimensione", "small", choices=("tiny", "mini", "small", "base")),),
)
def run_chronos_bolt(
    df: pd.DataFrame,
    horizon: int,
    model_size: Literal["tiny", "mini", "small", "base"] = "small",
) -> ForecastResult:
    """Run Amazon Chronos-Bolt (zero-shot, direct multi-step quantile forecast)."""
    return _chronos_forecast(df, horizon, f"amazon/chronos-bolt-{model_size}", f"Chronos-Bolt-{model_size}")


@register(
    "Chronos",
    family=FOUNDATION,
    description="Amazon, 2024 · Apache-2.0. Chronos originale (T5) a campionamento: lento, tenuto per confronto.",
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


# ──────────────────────────────────────────────────────────────────────────────
# Timestral (Mistral AI) – optional extra
# ──────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _load_timestral(repo_id: str) -> Any:
    from timestral import TimestralForecaster

    device = _device()
    print(f"  ⏳ Loading {repo_id} on {device} …")
    return TimestralForecaster.from_pretrained(repo_id, device=device)


@register(
    "Timestral",
    family=FOUNDATION,
    description="Mistral AI, ott. 2026 · pesi modified-MIT. Zero-shot, contesto fino a 32k punti, veloce anche su CPU.",
    params=(Param("model_size", "Dimensione", "209M", choices=("5M", "28M", "123M", "209M")),),
    requires=("timestral",),
    extra="timestral",
)
def run_timestral(
    df: pd.DataFrame,
    horizon: int,
    model_size: Literal["5M", "28M", "123M", "209M"] = "209M",
) -> ForecastResult:
    """Run Mistral Timestral (zero-shot, 21 quantile levels)."""
    model = _load_timestral(f"mistralai/Timestral-{model_size}")

    print(f"  🔮 Forecasting {horizon} steps (Timestral-{model_size}) …")
    fc = model.forecast(df["y"].to_numpy(dtype=np.float32), prediction_length=horizon)
    quantiles = fc.quantiles[0, 0].float().cpu().numpy()  # (B, V, H, Q) → (H, Q)
    lo, med, hi = _pick_quantiles(quantiles, fc.quantile_levels)
    return make_result(f"Timestral-{model_size}", df, horizon, med, lo, hi)


# ──────────────────────────────────────────────────────────────────────────────
# TimesFM 3 (Google) – optional extra
# ──────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _load_timesfm(repo_id: str) -> Any:
    from timesfm3 import TimesFM3Forecaster

    device = _device()
    print(f"  ⏳ Loading {repo_id} on {device} …")
    return TimesFM3Forecaster.from_pretrained(repo_id, device=device)


@register(
    "TimesFM-3",
    family=FOUNDATION,
    description="Google, ago. 2026 · 330M parametri. Zero-shot. Pesi con licenza NON commerciale.",
    requires=("timesfm3",),
    extra="timesfm",
)
def run_timesfm(df: pd.DataFrame, horizon: int) -> ForecastResult:
    """Run Google TimesFM 3.0 (zero-shot, decoder-only, quantile head 0.1…0.9)."""
    model = _load_timesfm("google/timesfm-3.0-pytorch")

    print(f"  🔮 Forecasting {horizon} steps (TimesFM-3) …")
    out = model.predict(df["y"].to_numpy(dtype=np.float32), horizon=horizon, return_quantiles=True)
    quantiles = np.asarray(out.quantiles, dtype=float).reshape(horizon, -1)
    lo, med, hi = _pick_quantiles(quantiles, model.config.quantiles)
    return make_result("TimesFM-3", df, horizon, med, lo, hi)


# ──────────────────────────────────────────────────────────────────────────────
# TiRex (NX-AI) – optional extra
# ──────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _load_tirex(repo_id: str) -> Any:
    from tirex import load_model

    device = _device()
    print(f"  ⏳ Loading {repo_id} on {device} …")
    return load_model(repo_id, device=device, backend="torch")


@register(
    "TiRex",
    family=FOUNDATION,
    description="NX-AI, 2025 · 35M parametri (xLSTM) · licenza NXAI community. Leggero e accurato.",
    requires=("tirex",),
    extra="tirex",
)
def run_tirex(df: pd.DataFrame, horizon: int) -> ForecastResult:
    """Run NX-AI TiRex (zero-shot xLSTM forecaster)."""
    import torch

    model = _load_tirex("NX-AI/TiRex")

    print(f"  🔮 Forecasting {horizon} steps (TiRex) …")
    context = torch.tensor(df["y"].to_numpy(), dtype=torch.float32).unsqueeze(0)
    quantiles, _mean = model.forecast(context=context, prediction_length=horizon, output_type="numpy")
    lo, med, hi = _pick_quantiles(np.asarray(quantiles[0], dtype=float), model.config.quantiles)
    return make_result("TiRex", df, horizon, med, lo, hi)
