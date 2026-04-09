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
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS

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

    Uses the official ``chronos-forecasting`` pipeline from the
    ``autogluon.timeseries`` / ``chronos`` package when available,
    otherwise falls back to loading the model directly via ``transformers``.

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
    model_id = f"amazon/chronos-t5-{model_size}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"  ⏳ Loading Chronos [{model_size}] on {device} …")

    # ── Try the official Chronos pipeline (installed via pip install chronos-forecasting)
    try:
        from chronos import ChronosPipeline

        pipe = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
        )
        context = torch.tensor(df["y"].values, dtype=torch.float32)

        print(f"  🔮 Forecasting {horizon} steps (ChronosPipeline) …")

        # Capture the tuple (forecast, scale)
        forecast_data = pipe.predict_quantiles(
            context.unsqueeze(0),
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            num_samples=num_samples,
        )

        # forecast_data[0] is a tensor of shape (batch, horizon, num_quantiles)
        forecast_tensor = forecast_data[0]

        # Unpack the quantiles from the last dimension
        # 0 = 0.1, 1 = 0.5 (median), 2 = 0.9
        yhat = forecast_tensor[0, :, 1].numpy()
        yhat_lo = forecast_tensor[0, :, 0].numpy()
        yhat_hi = forecast_tensor[0, :, 2].numpy()

    except ImportError:
        # ── Fallback: load the AutoRegressive model directly via transformers
        #    (works without the chronos-forecasting extra package)
        print("  ℹ️  chronos-forecasting not found – using transformers fallback …")
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        # Chronos uses T5 under the hood; we load it and run sampling manually
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, config=config, torch_dtype=dtype, trust_remote_code=True).to(
            device
        )

        # The Chronos T5 models expose a `generate` method that accepts raw
        # float context via their custom tokenizer.
        context_np = df["y"].values.astype(np.float32)

        # Tokenise context (Chronos tokenizer returns input_ids & attention_mask)
        inputs = tokenizer(
            context_np,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"  🔮 Forecasting {horizon} steps (transformers fallback) …")
        with torch.no_grad():
            sample_ids = model.generate(
                **inputs,
                min_new_tokens=horizon,
                max_new_tokens=horizon,
                do_sample=True,
                num_return_sequences=num_samples,
            )

        # Decode back to float values
        samples = tokenizer.decode(sample_ids, target_length=horizon)  # (S, H)
        samples = np.array(samples)

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
    """Train and run N-HiTS on the fly."""

    nf_df = df[["ds", "y"]].copy()
    nf_df["unique_id"] = "series_1"
    nf_df = nf_df[["unique_id", "ds", "y"]]

    input_size = horizon * input_size_multiplier
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
    nf.fit(df=nf_df, val_size=horizon)

    print(f"  🔮 Forecasting {horizon} steps …")
    pred = nf.predict().reset_index()

    # ── Rename NeuralForecast output columns ─────────────────────────────────
    col_map: dict[str, str] = {}
    for c in pred.columns:
        cl = c.lower()
        if "median" in cl or cl.endswith("-median"):
            col_map[c] = "yhat"
        elif "lo-80" in cl or "lo80" in cl:
            col_map[c] = "yhat_lo"
        elif "hi-80" in cl or "hi80" in cl:
            col_map[c] = "yhat_hi"

    nhits_cols = [c for c in pred.columns if "NHITS" in c]
    if "yhat" not in col_map and nhits_cols:
        col_map[nhits_cols[0]] = "yhat"
    if "yhat_lo" not in col_map and len(nhits_cols) > 1:
        col_map[nhits_cols[1]] = "yhat_lo"
    if "yhat_hi" not in col_map and len(nhits_cols) > 2:
        col_map[nhits_cols[2]] = "yhat_hi"

    pred = pred.rename(columns=col_map)

    if "yhat" in pred.columns:
        if "yhat_lo" not in pred.columns:
            pred["yhat_lo"] = pred["yhat"] * 0.97
        if "yhat_hi" not in pred.columns:
            pred["yhat_hi"] = pred["yhat"] * 1.03

    forecast_df = pred[["ds", "yhat", "yhat_lo", "yhat_hi"]].copy()
    forecast_df["yhat"] = forecast_df["yhat"].astype(float)
    forecast_df["yhat_lo"] = forecast_df["yhat_lo"].astype(float)
    forecast_df["yhat_hi"] = forecast_df["yhat_hi"].astype(float)
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
