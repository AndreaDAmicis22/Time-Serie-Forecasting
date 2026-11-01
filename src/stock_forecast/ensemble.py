"""Model ensembles: combine the forecasts of several models into one.

Forecast combination is one of the most reliable ways to improve accuracy:
individual errors partly cancel out. Point forecasts and interval bounds are
combined with the same rule (quantile averaging).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .backtest import average_metrics, backtest_windows, evaluate_forecast
from .models import ForecastResult

EnsembleMethod = Literal["weighted", "mean", "median"]
ENSEMBLE_METHODS: tuple[EnsembleMethod, ...] = ("weighted", "mean", "median")


@dataclass
class EnsembleResult(ForecastResult):
    """Forecast obtained by combining several models."""

    weights: dict[str, float] = field(default_factory=dict)  # member → weight (sum to 1)
    members: dict[str, ForecastResult] = field(default_factory=dict)


def select_members(validation_results: dict[str, dict], metric: str = "mape", top_k: int = 3) -> dict[str, float]:
    """The *top_k* successful models ranked by validation *metric* → their scores."""
    scores = {
        name: data["metrics"][metric]
        for name, data in validation_results.items()
        if data.get("result") is not None and np.isfinite(data["metrics"].get(metric, np.inf))
    }
    return dict(sorted(scores.items(), key=lambda item: item[1])[:top_k])


def ensemble_weights(scores: dict[str, float], method: EnsembleMethod = "weighted") -> dict[str, float]:
    """Member weights: inverse validation error for ``"weighted"``, equal otherwise.

    ``"median"`` ignores weights when combining; equal weights are reported for display.
    """
    if method == "weighted":
        raw = {name: 1.0 / max(score, 1e-12) for name, score in scores.items()}
    else:
        raw = dict.fromkeys(scores, 1.0)
    total = sum(raw.values())
    return {name: w / total for name, w in raw.items()}


def combine_forecasts(
    results: dict[str, ForecastResult],
    method: EnsembleMethod = "weighted",
    weights: dict[str, float] | None = None,
    name: str = "Ensemble",
) -> EnsembleResult:
    """Combine forecasts that share the same dates (same training series and horizon)."""
    if not results:
        msg = "Cannot build an ensemble from zero forecasts."
        raise ValueError(msg)

    weights = weights or ensemble_weights(dict.fromkeys(results, 1.0), "mean")
    w = np.array([weights[member] for member in results])
    first = next(iter(results.values()))

    combined = {}
    for col in ("yhat", "yhat_lo", "yhat_hi"):
        stacked = np.vstack([r.forecast[col].to_numpy(dtype=float) for r in results.values()])
        combined[col] = np.median(stacked, axis=0) if method == "median" else np.average(stacked, axis=0, weights=w)

    forecast = pd.DataFrame({"ds": first.forecast["ds"].to_numpy(), **combined})
    return EnsembleResult(
        model_name=name,
        horizon=first.horizon,
        forecast=forecast,
        train_df=first.train_df,
        weights=dict(weights),
        members=dict(results),
    )


def backtest_ensemble(
    df: pd.DataFrame,
    horizon: int,
    validation_results: dict[str, dict],
    weights: dict[str, float],
    method: EnsembleMethod = "weighted",
    name: str = "Ensemble",
) -> dict:
    """Score the ensemble on the same backtest windows as its members.

    Uses the members' stored window forecasts, so nothing is retrained. Members
    and weights come from these same windows, so – like picking the single best
    model – the score is slightly optimistic.

    Returns the same structure as one entry of ``validate_models``.
    """
    n_windows = len(validation_results[next(iter(weights))]["windows"])
    window_results, window_metrics = [], []
    for w, (train_df, test_df) in enumerate(backtest_windows(df, horizon, n_windows)):
        members = {member: validation_results[member]["windows"][w] for member in weights}
        ensemble = combine_forecasts(members, method, weights, name=name)
        window_results.append(ensemble)
        window_metrics.append(evaluate_forecast(train_df, test_df, ensemble.forecast))

    metrics = average_metrics(window_metrics)
    window_results[-1].metrics = metrics
    return {"metrics": metrics, "result": window_results[-1], "windows": window_results, "error": None}
