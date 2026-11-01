"""Model selection pipeline.

1. Split series into train / validation sets.
2. Train every candidate model on the train set and evaluate it on the validation set.
3. Select the best model based on the validation metric.
4. Retrain the winner on the **full** series and produce the final forecast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backtest import resolve_model_params, validate_models
from .models import available_models, get_model

if TYPE_CHECKING:
    import pandas as pd

    from .models import ForecastResult


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
    model_params: dict[str, dict] | None = None,
) -> tuple[dict[str, dict], ForecastResult]:
    """Full pipeline: validate → select best → retrain on full data → forecast.

    Parameters
    ----------
    df:
        Full time series.
    horizon:
        Number of steps to forecast.
    models:
        Which models to compare. Default: all available.
    chronos_size:
        Shortcut for the size of the "Chronos" model.
    max_steps:
        Shortcut for the training budget of the neural models.
    selection_metric:
        Metric to use for model selection ("mae", "rmse", or "mape").
    model_params:
        Extra keyword arguments per model, e.g. ``{"XGBoost": {"max_depth": 4}}``.

    Returns
    -------
    (validation_results, final_forecast)
    """
    if models is None:
        models = available_models()

    print("╔═══════════════════════════════════════════════════════╗")
    print("║          MODEL SELECTION PIPELINE                    ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print(f"\n  Series length : {len(df)} observations")
    print(f"  Horizon       : {horizon} steps")
    print(f"  Val set size  : {horizon} (last {horizon} obs)")
    print(f"  Metric        : {selection_metric}\n")

    # Step 1: Validate all models
    val_results = validate_models(
        df,
        horizon,
        models=models,
        chronos_size=chronos_size,
        max_steps=max_steps,
        model_params=model_params,
    )

    # Step 2: Select best
    best_name = select_best_model(val_results, metric=selection_metric)
    if best_name is None:
        msg = "All models failed during validation."
        raise RuntimeError(msg)

    # Step 3: Retrain on full data and forecast
    final_result = run_best_model(df, horizon, best_name, chronos_size, max_steps, model_params=model_params)
    final_result.metrics = val_results[best_name]["metrics"]

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
