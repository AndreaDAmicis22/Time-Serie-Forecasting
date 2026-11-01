"""Deep-learning models trained on the fly with NeuralForecast (N-HiTS, N-BEATS)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import NEURAL, ForecastResult, Param, infer_freq, make_result, register

if TYPE_CHECKING:
    import pandas as pd

_NEURAL_PARAMS = (
    Param(
        "max_steps",
        "Passi di training",
        500,
        min_value=50,
        max_value=5000,
        step=50,
        help="Iterazioni massime di training (con early stopping).",
    ),
    Param(
        "input_size_multiplier",
        "Finestra di input (× orizzonte)",
        5,
        min_value=1,
        max_value=30,
        step=1,
        help="Quante barre di storico vede la rete, in multipli dell'orizzonte.",
    ),
)


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

    return pred[["ds", "yhat", "yhat_lo", "yhat_hi"]].astype({"yhat": float, "yhat_lo": float, "yhat_hi": float})


def _run_neuralforecast(
    model_cls: Any,
    display_name: str,
    df: pd.DataFrame,
    horizon: int,
    **model_kwargs: Any,
) -> ForecastResult:
    """Train a NeuralForecast model on *df* and forecast *horizon* steps."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss

    nf_df = df[["ds", "y"]].copy()
    nf_df["unique_id"] = "series_1"
    nf_df = nf_df[["unique_id", "ds", "y"]]

    input_size = horizon * model_kwargs.pop("input_size_multiplier")
    early_stop_patience = model_kwargs.pop("early_stop_patience")

    model = model_cls(
        h=horizon,
        input_size=input_size,
        loss=MQLoss(level=[80]),
        early_stop_patience_steps=early_stop_patience,
        enable_progress_bar=True,
        logger=False,  # don't litter the working directory with lightning_logs/
        **model_kwargs,
    )

    print(f"  🏋️  Training {display_name} (max_steps={model_kwargs['max_steps']}) …")
    nf = NeuralForecast(models=[model], freq=infer_freq(df["ds"]))
    nf.fit(df=nf_df, val_size=horizon)

    print(f"  🔮 Forecasting {horizon} steps …")
    pred = nf.predict().reset_index()
    fc = _nf_predict_to_forecast_df(pred, model_cls.__name__)
    return make_result(display_name, df, horizon, fc["yhat"], fc["yhat_lo"], fc["yhat_hi"])


@register(
    "N-HiTS",
    family=NEURAL,
    description="Rete neurale gerarchica multi-scala (NeuralForecast), addestrata da zero ad ogni esecuzione.",
    params=_NEURAL_PARAMS,
    slow=True,
)
def run_nhits(
    df: pd.DataFrame,
    horizon: int,
    input_size_multiplier: int = 5,
    max_steps: int = 500,
    val_check_steps: int = 50,
    early_stop_patience: int = 5,
) -> ForecastResult:
    """Train and run N-HiTS on the fly."""
    from neuralforecast.models import NHITS

    return _run_neuralforecast(
        NHITS,
        "N-HiTS",
        df,
        horizon,
        input_size_multiplier=input_size_multiplier,
        max_steps=max_steps,
        val_check_steps=val_check_steps,
        early_stop_patience=early_stop_patience,
    )


@register(
    "N-BEATS",
    family=NEURAL,
    description="Rete neurale interpretabile (NeuralForecast), addestrata da zero ad ogni esecuzione.",
    params=_NEURAL_PARAMS,
    slow=True,
)
def run_nbeats(
    df: pd.DataFrame,
    horizon: int,
    input_size_multiplier: int = 5,
    max_steps: int = 500,
    val_check_steps: int = 50,
    early_stop_patience: int = 5,
) -> ForecastResult:
    """Train and run N-BEATS on the fly."""
    from neuralforecast.models import NBEATS

    return _run_neuralforecast(
        NBEATS,
        "N-BEATS",
        df,
        horizon,
        input_size_multiplier=input_size_multiplier,
        max_steps=max_steps,
        val_check_steps=val_check_steps,
        early_stop_patience=early_stop_patience,
    )
