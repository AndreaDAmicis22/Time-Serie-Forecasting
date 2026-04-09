"""Forecasting models.

Two complementary approaches:
- **Chronos** (Amazon, zero-shot transformer): no training needed, fast inference.
- **N-HiTS** (NeuralForecast): trained on the fly on the downloaded series.

Both return a ``ForecastResult`` dataframe with columns
``["ds", "yhat", "yhat_lo", "yhat_hi"]``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _infer_freq(ds: pd.Series) -> str:
    """Infer pandas frequency string from a datetime series."""
    diffs = ds.sort_values().diff().dropna()
    median_days = diffs.dt.days.median()
    if median_days <= 1.5:
        return "B"  # business day
    if median_days <= 8:
        return "W"
    if median_days <= 32:
        return "MS"
    return "QS"


def _future_dates(last_date: pd.Timestamp, horizon: int, freq: str) -> pd.DatetimeIndex:
    """Generate *horizon* future business/calendar dates after *last_date*."""
    if freq == "B":
        # business days
        return pd.bdate_range(start=last_date, periods=horizon + 1, freq="B")[1:]
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]


# ──────────────────────────────────────────────────────────────────────────────
# Chronos (Amazon, zero-shot)
# ──────────────────────────────────────────────────────────────────────────────


def run_chronos(
    df: pd.DataFrame,
    horizon: int,
    model_size: Literal["tiny", "mini", "small", "base", "large"] = "small",
    num_samples: int = 20,
) -> ForecastResult:
    """Run Amazon Chronos (zero-shot transformer).

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

    Returns
    -------
    ForecastResult
    """
    from transformers import pipeline  # type: ignore

    model_id = f"amazon/chronos-t5-{model_size}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  ⏳ Loading Chronos [{model_size}] on {device} …")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    context = torch.tensor(df["y"].values, dtype=torch.float32).unsqueeze(0)

    print(f"  🔮 Forecasting {horizon} steps …")
    output = pipe(
        context,
        prediction_length=horizon,
        num_samples=num_samples,
    )

    # output[0] shape: (num_samples, horizon)
    samples = np.array(output[0])  # (S, H)
    yhat = samples.mean(axis=0)
    yhat_lo = np.percentile(samples, 10, axis=0)
    yhat_hi = np.percentile(samples, 90, axis=0)

    freq = _infer_freq(df["ds"])
    future_ds = _future_dates(df["ds"].iloc[-1], horizon, freq)

    forecast_df = pd.DataFrame({"ds": future_ds, "yhat": yhat, "yhat_lo": yhat_lo, "yhat_hi": yhat_hi})

    return ForecastResult(
        model_name=f"Chronos-{model_size}",
        horizon=horizon,
        forecast=forecast_df,
        train_df=df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# N-HiTS (NeuralForecast, trained on-the-fly)
# ──────────────────────────────────────────────────────────────────────────────


def run_nhits(
    df: pd.DataFrame,
    horizon: int,
    input_size_multiplier: int = 5,
    max_steps: int = 500,
    val_check_steps: int = 50,
    early_stop_patience: int = 5,
) -> ForecastResult:
    """Train and run N-HiTS on the fly.

    Parameters
    ----------
    df:
        DataFrame with columns ``ds`` (datetime) and ``y`` (float).
    horizon:
        Steps to predict.
    input_size_multiplier:
        ``input_size = horizon * multiplier`` – context window.
    max_steps:
        Training iterations.
    val_check_steps / early_stop_patience:
        Early stopping configuration.

    Returns
    -------
    ForecastResult
    """
    from neuralforecast import NeuralForecast  # type: ignore
    from neuralforecast.losses.pytorch import MQLoss  # type: ignore
    from neuralforecast.models import NHITS  # type: ignore

    # NeuralForecast expects a "unique_id" column
    nf_df = df[["ds", "y"]].copy()
    nf_df["unique_id"] = "series_1"
    nf_df = nf_df[["unique_id", "ds", "y"]]

    input_size = horizon * input_size_multiplier

    # Quantile loss → prediction intervals
    loss = MQLoss(level=[80, 90])

    model = NHITS(
        h=horizon,
        input_size=input_size,
        loss=loss,
        max_steps=max_steps,
        val_check_steps=val_check_steps,
        early_stop_patience_steps=early_stop_patience,
        enable_progress_bar=True,
    )

    print(f"  🏋️  Training N-HiTS (max_steps={max_steps}) …")
    nf = NeuralForecast(models=[model], freq=_infer_freq(df["ds"]))
    nf.fit(df=nf_df)

    print(f"  🔮 Forecasting {horizon} steps …")
    pred = nf.predict().reset_index()

    # Rename NeuralForecast columns
    col_map = {}
    for c in pred.columns:
        cl = c.lower()
        if "median" in cl or cl.endswith("-median"):
            col_map[c] = "yhat"
        elif "lo-80" in cl or "lo80" in cl:
            col_map[c] = "yhat_lo"
        elif "hi-80" in cl or "hi80" in cl:
            col_map[c] = "yhat_hi"

    # Fallback: use first NHITS column as point forecast
    nhits_cols = [c for c in pred.columns if "NHITS" in c]
    if "yhat" not in col_map and nhits_cols:
        col_map[nhits_cols[0]] = "yhat"
    if "yhat_lo" not in col_map and len(nhits_cols) > 1:
        col_map[nhits_cols[1]] = "yhat_lo"
    if "yhat_hi" not in col_map and len(nhits_cols) > 2:
        col_map[nhits_cols[2]] = "yhat_hi"

    pred = pred.rename(columns=col_map)

    needed = [c for c in ["yhat", "yhat_lo", "yhat_hi"] if c not in pred.columns]
    if needed:
        # Construct missing interval columns from available ones
        if "yhat" in pred.columns:
            if "yhat_lo" not in pred.columns:
                pred["yhat_lo"] = pred["yhat"] * 0.97
            if "yhat_hi" not in pred.columns:
                pred["yhat_hi"] = pred["yhat"] * 1.03

    forecast_df = pred[["ds", "yhat", "yhat_lo", "yhat_hi"]].copy()
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    return ForecastResult(
        model_name="N-HiTS",
        horizon=horizon,
        forecast=forecast_df,
        train_df=df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: run both and return dict
# ──────────────────────────────────────────────────────────────────────────────


def run_all(
    df: pd.DataFrame,
    horizon: int,
    chronos_size: str = "small",
    nhits_steps: int = 300,
) -> dict[str, ForecastResult]:
    """Run Chronos + N-HiTS and return results keyed by model name."""
    results: dict[str, ForecastResult] = {}

    print("\n── Chronos ─────────────────────────────────────────────")
    try:
        results["chronos"] = run_chronos(df, horizon, model_size=chronos_size)
    except Exception as e:
        print(f"  ⚠️  Chronos failed: {e}")

    print("\n── N-HiTS ──────────────────────────────────────────────")
    try:
        results["nhits"] = run_nhits(df, horizon, max_steps=nhits_steps)
    except Exception as e:
        print(f"  ⚠️  N-HiTS failed: {e}")

    return results
