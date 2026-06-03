"""Forecasting models with train/validation model selection.

Pipeline:
1. Split series into train / validation sets.
2. Train multiple models on the train set.
3. Evaluate each on the validation set (MAE, RMSE, MAPE).
4. Select the best model based on validation MAPE.
5. Retrain the winner on the **full** series.
6. Produce the final n-step-ahead forecast.

Models available:
- **Chronos** (Amazon, zero-shot transformer)
- **N-HiTS** (NeuralForecast)
- **N-BEATS** (NeuralForecast)
- **Auto-ARIMA** (statsmodels, with stationarity enforcement)
- **XGBoost** (gradient boosting with lag features)
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
from neuralforecast.models import NBEATS, NHITS
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor

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
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# Train / Validation split
# ──────────────────────────────────────────────────────────────────────────────


def train_val_split(df: pd.DataFrame, val_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into train and validation sets.

    Parameters
    ----------
    df:
        DataFrame with at least columns ``ds`` and ``y``.
    val_size:
        Number of most-recent observations to hold out for validation.

    Returns
    -------
    (train_df, val_df)
    """
    df = df.sort_values("ds").reset_index(drop=True)
    split_idx = len(df) - val_size
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


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
# Stationarity helpers
# ──────────────────────────────────────────────────────────────────────────────


def _check_stationarity(y: np.ndarray, significance: float = 0.05) -> tuple[bool, float]:
    """Run Augmented Dickey-Fuller test.

    Returns (is_stationary, p_value).
    """
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


def run_arima(
    df: pd.DataFrame,
    horizon: int,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    seasonal: bool = False,
) -> ForecastResult:
    """Fit ARIMA with automatic stationarity enforcement and order selection.

    The function:
    1. Tests stationarity via ADF and determines the differencing order d.
    2. Searches (p, q) combinations via AIC to pick the best ARIMA(p, d, q).
    3. Forecasts *horizon* steps ahead with confidence intervals.
    """
    y = df["y"].values.astype(float)

    # ── Step 1: Determine d via ADF test ──────────────────────────────────────
    is_stationary, p_value = _check_stationarity(y)
    print(f"  📊 ADF test p-value: {p_value:.6f} → {'stationary' if is_stationary else 'non-stationary'}")

    _, d = _make_stationary(y, max_d=max_d)
    print(f"  📐 Differencing order d={d}")

    # ── Step 2: Grid search for best (p, q) by AIC ───────────────────────────
    print(f"  🔍 Searching ARIMA orders (p=0..{max_p}, d={d}, q=0..{max_q}) …")
    best_aic = float("inf")
    best_order = (1, d, 1)

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

    # ── Step 3: Fit final model and forecast ──────────────────────────────────
    final_model = SARIMAX(y, order=best_order, enforce_stationarity=True, enforce_invertibility=True)
    final_fit = final_model.fit(disp=False, maxiter=500)

    forecast_obj = final_fit.get_forecast(steps=horizon)
    yhat = np.asarray(forecast_obj.predicted_mean)
    conf_int = np.asarray(forecast_obj.conf_int(alpha=0.2))  # 80% CI

    freq = _infer_freq(df["ds"])
    future_ds = _future_dates(df["ds"].iloc[-1], horizon, freq)

    forecast_df = pd.DataFrame(
        {
            "ds": future_ds,
            "yhat": yhat,
            "yhat_lo": conf_int[:, 0],
            "yhat_hi": conf_int[:, 1],
        }
    )

    return ForecastResult(
        model_name="Auto-ARIMA",
        horizon=horizon,
        forecast=forecast_df,
        train_df=df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost (ML with lag features)
# ──────────────────────────────────────────────────────────────────────────────


def _create_lag_features(y: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised learning dataset from a 1-D time series.

    Returns (X, Y) where each row of X contains [y(t-n_lags), ..., y(t-1)]
    and Y contains y(t).
    """
    X, Y = [], []
    for i in range(n_lags, len(y)):
        X.append(y[i - n_lags : i])
        Y.append(y[i])
    return np.array(X), np.array(Y)


def run_xgboost(
    df: pd.DataFrame,
    horizon: int,
    n_lags: int | None = None,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> ForecastResult:
    """Train XGBoost regressor with lag features for time series forecasting.

    Uses a recursive multi-step strategy: predict one step, feed prediction
    back as a feature for the next step.
    """
    y = df["y"].values.astype(float)

    if n_lags is None:
        n_lags = min(max(horizon * 3, 30), len(y) // 3)

    X, Y = _create_lag_features(y, n_lags)

    print(f"  🌲 Training XGBoost (lags={n_lags}, estimators={n_estimators}) …")
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
    last_window = y[-n_lags:].copy()
    predictions = []

    for _ in range(horizon):
        x_input = last_window.reshape(1, -1)
        yhat_step = model.predict(x_input)[0]
        predictions.append(yhat_step)
        last_window = np.append(last_window[1:], yhat_step)

    yhat = np.array(predictions)

    # ── Confidence interval via residual std on training set ───────────────────
    train_preds = model.predict(X)
    residual_std = np.std(Y - train_preds)
    # Widen intervals further into the future
    steps = np.arange(1, horizon + 1)
    interval_width = 1.28 * residual_std * np.sqrt(steps)  # ~80% CI
    yhat_lo = yhat - interval_width
    yhat_hi = yhat + interval_width

    freq = _infer_freq(df["ds"])
    future_ds = _future_dates(df["ds"].iloc[-1], horizon, freq)

    forecast_df = pd.DataFrame(
        {
            "ds": future_ds,
            "yhat": yhat,
            "yhat_lo": yhat_lo,
            "yhat_hi": yhat_hi,
        }
    )

    return ForecastResult(
        model_name="XGBoost",
        horizon=horizon,
        forecast=forecast_df,
        train_df=df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# NeuralForecast helpers
# ──────────────────────────────────────────────────────────────────────────────


def _nf_predict_to_forecast_df(pred: pd.DataFrame, model_prefix: str) -> pd.DataFrame:
    """Normalize NeuralForecast predict output to standard columns."""
    col_map: dict[str, str] = {}
    for c in pred.columns:
        cl = c.lower()
        if "median" in cl or cl.endswith("-median"):
            col_map[c] = "yhat"
        elif "lo-80" in cl or "lo80" in cl:
            col_map[c] = "yhat_lo"
        elif "hi-80" in cl or "hi80" in cl:
            col_map[c] = "yhat_hi"

    model_cols = [c for c in pred.columns if model_prefix in c]
    if "yhat" not in col_map.values() and model_cols:
        col_map[model_cols[0]] = "yhat"
    if "yhat_lo" not in col_map.values() and len(model_cols) > 1:
        col_map[model_cols[1]] = "yhat_lo"
    if "yhat_hi" not in col_map.values() and len(model_cols) > 2:
        col_map[model_cols[2]] = "yhat_hi"

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
    return forecast_df


# ──────────────────────────────────────────────────────────────────────────────
# N-HiTS (NeuralForecast)
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
    loss = MQLoss(level=[80])

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
    forecast_df = _nf_predict_to_forecast_df(pred, "NHITS")

    return ForecastResult(
        model_name="N-HiTS",
        horizon=horizon,
        forecast=forecast_df,
        train_df=df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# N-BEATS (NeuralForecast)
# ──────────────────────────────────────────────────────────────────────────────


def run_nbeats(
    df: pd.DataFrame,
    horizon: int,
    input_size_multiplier: int = 5,
    max_steps: int = 500,
    val_check_steps: int = 50,
    early_stop_patience: int = 5,
) -> ForecastResult:
    """Train and run N-BEATS on the fly."""
    nf_df = df[["ds", "y"]].copy()
    nf_df["unique_id"] = "series_1"
    nf_df = nf_df[["unique_id", "ds", "y"]]

    input_size = horizon * input_size_multiplier
    loss = MQLoss(level=[80])

    model = NBEATS(
        h=horizon,
        input_size=input_size,
        loss=loss,
        max_steps=max_steps,
        val_check_steps=val_check_steps,
        early_stop_patience_steps=early_stop_patience,
        enable_progress_bar=True,
    )

    print(f"  🏋️  Training N-BEATS (max_steps={max_steps}) …")
    nf = NeuralForecast(models=[model], freq=_infer_freq(df["ds"]))
    nf.fit(df=nf_df, val_size=horizon)

    print(f"  🔮 Forecasting {horizon} steps …")
    pred = nf.predict().reset_index()
    forecast_df = _nf_predict_to_forecast_df(pred, "NBEATS")

    return ForecastResult(
        model_name="N-BEATS",
        horizon=horizon,
        forecast=forecast_df,
        train_df=df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model Selection Pipeline
# ──────────────────────────────────────────────────────────────────────────────

# Registry of model runners: name → callable(df, horizon, **kwargs) → ForecastResult
MODEL_REGISTRY: dict[str, callable] = {
    "N-HiTS": run_nhits,
    "N-BEATS": run_nbeats,
    "Chronos": run_chronos,
    "Auto-ARIMA": run_arima,
    "XGBoost": run_xgboost,
}


def validate_models(
    df: pd.DataFrame,
    horizon: int,
    models: list[str] | None = None,
    chronos_size: str = "small",
    max_steps: int = 500,
) -> dict[str, dict]:
    """Train each model on train set, evaluate on validation set.

    Parameters
    ----------
    df:
        Full time series with columns ``ds``, ``y``.
    horizon:
        Forecast horizon (also used as validation set size).
    models:
        List of model names to evaluate. Default: all registered.
    chronos_size:
        Chronos variant size.
    max_steps:
        Max training steps for neural models.

    Returns
    -------
    dict mapping model_name → {"metrics": {...}, "forecast": ForecastResult}
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())

    train_df, val_df = train_val_split(df, val_size=horizon)
    val_y = val_df["y"].values

    results: dict[str, dict] = {}
    print(f"──────────── Models to validate: {', '.join(models)} ────────────")
    for name in models:
        print(f"\n── Validating: {name} ─────────────────────────────────")
        try:
            if name == "Chronos":
                result = run_chronos(train_df, horizon, model_size=chronos_size)
            elif name == "N-HiTS":
                result = run_nhits(train_df, horizon, max_steps=max_steps)
            elif name == "N-BEATS":
                result = run_nbeats(train_df, horizon, max_steps=max_steps)
            elif name == "Auto-ARIMA":
                result = run_arima(train_df, horizon)
            elif name == "XGBoost":
                result = run_xgboost(train_df, horizon)
            else:
                runner = MODEL_REGISTRY[name]
                result = runner(train_df, horizon)

            pred_y = result.forecast["yhat"].values[: len(val_y)]
            metrics = compute_metrics(val_y, pred_y)
            result.metrics = metrics

            results[name] = {"metrics": metrics, "result": result}
            print(f"  ✅ MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}%")

        except Exception as e:
            print(f"  ⚠️  {name} failed: {e}")
            results[name] = {"metrics": {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}, "result": None}

    return results


def select_best_model(
    validation_results: dict[str, dict],
    metric: str = "mape",
) -> str:
    """Select the best model name based on a validation metric (lower is better)."""
    best_name = None
    best_score = float("inf")
    for name, data in validation_results.items():
        score = data["metrics"].get(metric, float("inf"))
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def run_best_model(
    df: pd.DataFrame,
    horizon: int,
    best_model_name: str,
    chronos_size: str = "small",
    max_steps: int = 500,
) -> ForecastResult:
    """Retrain the best model on the **full** series and produce the final forecast."""
    print(f"\n{'═' * 55}")
    print(f"  🏆 Best model: {best_model_name}")
    print(f"  🔄 Retraining on full series ({len(df)} obs) …")
    print(f"{'═' * 55}\n")

    if best_model_name == "Chronos":
        return run_chronos(df, horizon, model_size=chronos_size)
    if best_model_name == "N-HiTS":
        return run_nhits(df, horizon, max_steps=max_steps)
    if best_model_name == "N-BEATS":
        return run_nbeats(df, horizon, max_steps=max_steps)
    if best_model_name == "Auto-ARIMA":
        return run_arima(df, horizon)
    if best_model_name == "XGBoost":
        return run_xgboost(df, horizon)
    runner = MODEL_REGISTRY[best_model_name]
    return runner(df, horizon)


def run_pipeline(
    df: pd.DataFrame,
    horizon: int,
    models: list[str] | None = None,
    chronos_size: str = "small",
    max_steps: int = 500,
    selection_metric: str = "mape",
) -> tuple[dict[str, dict], ForecastResult]:
    """Full pipeline: validate → select best → retrain on full data → forecast.

    Parameters
    ----------
    df:
        Full time series.
    horizon:
        Number of steps to forecast.
    models:
        Which models to compare. Default: all registered.
    chronos_size:
        Chronos variant.
    max_steps:
        Training budget for neural models.
    selection_metric:
        Metric to use for model selection ("mae", "rmse", or "mape").

    Returns
    -------
    (validation_results, final_forecast)
    """
    print("╔═══════════════════════════════════════════════════════╗")
    print("║          MODEL SELECTION PIPELINE                    ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print(f"\n  Series length : {len(df)} observations")
    print(f"  Horizon       : {horizon} steps")
    print(f"  Val set size  : {horizon} (last {horizon} obs)")
    print(f"  Metric        : {selection_metric}\n")

    # Step 1: Validate all models
    val_results = validate_models(df, horizon, models=models, chronos_size=chronos_size, max_steps=max_steps)

    # Step 2: Select best
    best_name = select_best_model(val_results, metric=selection_metric)

    if best_name is None:
        msg = "All models failed during validation."
        raise RuntimeError(msg)

    # Step 3: Retrain on full data and forecast
    final_result = run_best_model(df, horizon, best_name, chronos_size=chronos_size, max_steps=max_steps)
    final_result.metrics = val_results[best_name]["metrics"]

    return val_results, final_result


# ──────────────────────────────────────────────────────────────────────────────
# Legacy convenience function (backward-compatible)
# ──────────────────────────────────────────────────────────────────────────────


def run_all(
    df: pd.DataFrame,
    horizon: int,
    chronos_size: str = "small",
    nhits_steps: int = 300,
) -> dict[str, ForecastResult]:
    """Run all models and return results keyed by model name (no validation)."""
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

    print("\n── N-BEATS ─────────────────────────────────────────────")
    try:
        results["nbeats"] = run_nbeats(df, horizon, max_steps=nhits_steps)
    except Exception as e:
        print(f"  ⚠️  N-BEATS failed: {e}")

    print("\n── Auto-ARIMA ──────────────────────────────────────────")
    try:
        results["arima"] = run_arima(df, horizon)
    except Exception as e:
        print(f"  ⚠️  Auto-ARIMA failed: {e}")

    print("\n── XGBoost ─────────────────────────────────────────────")
    try:
        results["xgboost"] = run_xgboost(df, horizon)
    except Exception as e:
        print(f"  ⚠️  XGBoost failed: {e}")

    return results
