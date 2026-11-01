"""Model selection pipeline.

1. Backtest the candidate models on rolling-origin validation windows (MAE, RMSE, MAPE, MAE vs naive).
2. Choose the final forecaster with a strategy:
   - ``"best"``     – the model with the lowest validation metric;
   - ``"ensemble"`` – a combination of the top-k models;
   - ``"auto"``     – the ensemble if it beats the best single model in backtest, else the best model.
3. Retrain the chosen model(s) on the **full** series and produce the final forecast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .backtest import resolve_model_params, validate_models
from .ensemble import EnsembleMethod, backtest_ensemble, combine_forecasts, ensemble_weights, select_members
from .models import available_models, get_model

if TYPE_CHECKING:
    import pandas as pd

    from .backtest import ProgressCallback
    from .models import ForecastResult

Strategy = Literal["best", "ensemble", "auto"]
STRATEGIES: tuple[Strategy, ...] = ("auto", "best", "ensemble")
ENSEMBLE_NAME = "Ensemble"


def select_best_model(
    validation_results: dict[str, dict],
    metric: str = "mape",
) -> str | None:
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
    chronos_size: str | None = None,
    max_steps: int | None = None,
    *,
    model_params: dict[str, dict] | None = None,
) -> ForecastResult:
    """Retrain the best model on the **full** series and produce the final forecast."""
    print(f"\n{'═' * 55}")
    print(f"  🏆 Best model: {best_model_name}")
    print(f"  🔄 Retraining on full series ({len(df)} obs) …")
    print(f"{'═' * 55}\n")

    params = resolve_model_params([best_model_name], model_params, chronos_size, max_steps)
    return get_model(best_model_name)(df, horizon, **params[best_model_name])


def run_pipeline(
    df: pd.DataFrame,
    horizon: int,
    models: list[str] | None = None,
    chronos_size: str | None = None,
    max_steps: int | None = None,
    selection_metric: str = "mape",
    *,
    n_windows: int = 1,
    model_params: dict[str, dict] | None = None,
    strategy: Strategy = "best",
    ensemble_method: EnsembleMethod = "weighted",
    top_k: int = 3,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, dict], ForecastResult]:
    """Full pipeline: backtest → select best model or ensemble → retrain on full data → forecast.

    Parameters
    ----------
    df:
        Full time series.
    horizon:
        Number of steps to forecast.
    models:
        Which models to compare. Default: all available.
    chronos_size:
        Shortcut for the size of the original "Chronos" model.
    max_steps:
        Shortcut for the training budget of the neural models.
    selection_metric:
        Metric to use for model selection ("mae", "rmse", "mape" or "rel_mae").
    n_windows:
        Number of rolling-origin backtest windows (each of ``horizon`` observations).
    model_params:
        Extra keyword arguments per model, e.g. ``{"Chronos-Bolt": {"model_size": "base"}}``.
    strategy:
        ``"best"``, ``"ensemble"`` or ``"auto"`` (see module docstring).
    ensemble_method:
        ``"weighted"`` (inverse validation error), ``"mean"`` or ``"median"``.
    top_k:
        Number of best models combined in the ensemble.
    progress:
        Optional callback ``progress(step, total, message)``.

    Returns
    -------
    (validation_results, final_forecast)
        When an ensemble is evaluated, ``validation_results`` also contains an
        ``"Ensemble"`` entry and ``final_forecast`` may be an ``EnsembleResult``.
    """
    if models is None:
        models = available_models()

    print("╔═══════════════════════════════════════════════════════╗")
    print("║          MODEL SELECTION PIPELINE                    ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print(f"\n  Series length : {len(df)} observations")
    print(f"  Horizon       : {horizon} steps")
    print(f"  Backtest      : {n_windows} window(s) of {horizon} obs")
    print(f"  Metric        : {selection_metric}")
    print(f"  Strategy      : {strategy}\n")

    backtest_steps = len(models) * n_windows
    total_steps = backtest_steps + (1 if strategy == "best" else top_k)

    def report(step: int, message: str) -> None:
        if progress is not None:
            progress(step, total_steps, message)

    # Step 1: Backtest all models
    val_results = validate_models(
        df,
        horizon,
        models=models,
        chronos_size=chronos_size,
        max_steps=max_steps,
        n_windows=n_windows,
        model_params=model_params,
        progress=lambda step, _total, message: report(step, message),
    )

    # Step 2: Select best model, and possibly an ensemble
    best_name = select_best_model(val_results, metric=selection_metric)
    if best_name is None:
        msg = "All models failed during validation."
        raise RuntimeError(msg)

    weights: dict[str, float] = {}
    if strategy in ("ensemble", "auto"):
        scores = select_members(val_results, metric=selection_metric, top_k=top_k)
        if len(scores) >= 2:
            weights = ensemble_weights(scores, ensemble_method)
            val_results[ENSEMBLE_NAME] = backtest_ensemble(df, horizon, val_results, weights, ensemble_method)
            ensemble_score = val_results[ENSEMBLE_NAME]["metrics"][selection_metric]
            best_score = val_results[best_name]["metrics"][selection_metric]
            print(f"\n  🧩 Ensemble ({ensemble_method}) of {', '.join(weights)}: {selection_metric}={ensemble_score:.4f}")
            print(f"     Best single model {best_name}: {selection_metric}={best_score:.4f}")
            if strategy == "auto" and ensemble_score >= best_score:
                print("     → keeping the best single model")
                weights = {}
        else:
            print("\n  ℹ️  Fewer than 2 models succeeded: using the best single model.")

    # Step 3: Retrain on full data and forecast
    params = resolve_model_params(models, model_params, chronos_size, max_steps)
    if weights:
        members = {}
        for i, name in enumerate(weights):
            report(backtest_steps + i, f"Training finale {name}")
            print(f"\n  🔄 Retraining ensemble member {name} on full series ({len(df)} obs) …")
            members[name] = get_model(name)(df, horizon, **params[name])
        final_result = combine_forecasts(members, ensemble_method, weights, name=ENSEMBLE_NAME)
        final_result.metrics = val_results[ENSEMBLE_NAME]["metrics"]
    else:
        report(backtest_steps, f"Training finale {best_name}")
        final_result = run_best_model(df, horizon, best_name, model_params=params)
        final_result.metrics = val_results[best_name]["metrics"]

    report(total_steps, "Completato")
    return val_results, final_result


# ──────────────────────────────────────────────────────────────────────────────
# Legacy convenience function (backward-compatible)
# ──────────────────────────────────────────────────────────────────────────────


def run_all(
    df: pd.DataFrame,
    horizon: int,
    chronos_size: str | None = None,
    nhits_steps: int | None = None,
    *,
    models: list[str] | None = None,
    model_params: dict[str, dict] | None = None,
) -> dict[str, ForecastResult]:
    """Run several models on the full series and return results keyed by model name (no validation)."""
    if models is None:
        models = available_models()

    params = resolve_model_params(models, model_params, chronos_size, nhits_steps)
    results: dict[str, ForecastResult] = {}
    for name in models:
        print(f"\n── {name} {'─' * (52 - len(name))}")
        try:
            results[name] = get_model(name)(df, horizon, **params[name])
        except Exception as e:
            print(f"  ⚠️  {name} failed: {e}")
    return results
