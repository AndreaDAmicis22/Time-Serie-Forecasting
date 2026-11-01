import pytest
from conftest import assert_valid_forecast

from stock_forecast.ensemble import EnsembleResult
from stock_forecast.models import MODEL_REGISTRY, ModelSpec
from stock_forecast.models.base import BASELINE
from stock_forecast.pipeline import ENSEMBLE_NAME, run_all, run_pipeline
from stock_forecast.plot import _series_style, plot_backtest, plot_forecast

MODELS = ["Naive", "Drift", "Theta"]


def test_best_strategy(series):
    val, final = run_pipeline(series, 10, models=MODELS, selection_metric="mae", n_windows=2)

    assert ENSEMBLE_NAME not in val
    assert final.model_name in MODELS
    assert final.metrics == val[final.model_name]["metrics"]
    assert_valid_forecast(final, series, 10)


def test_ensemble_strategy(series):
    val, final = run_pipeline(series, 10, models=MODELS, n_windows=2, strategy="ensemble", ensemble_method="mean")

    assert isinstance(final, EnsembleResult)
    assert set(final.weights) == set(MODELS)
    assert sum(final.weights.values()) == pytest.approx(1.0)
    assert len(val[ENSEMBLE_NAME]["windows"]) == 2
    assert_valid_forecast(final, series, 10)


def test_auto_strategy_uses_the_ensemble_only_when_it_wins(series):
    val, final = run_pipeline(series, 10, models=MODELS, selection_metric="mae", n_windows=2, strategy="auto")

    ensemble_wins = val[ENSEMBLE_NAME]["metrics"]["mae"] < min(val[m]["metrics"]["mae"] for m in MODELS)
    assert isinstance(final, EnsembleResult) == ensemble_wins


def test_ensemble_needs_two_successful_models(series, monkeypatch):
    def broken(df, horizon):
        raise RuntimeError

    monkeypatch.setitem(MODEL_REGISTRY, "Broken", ModelSpec("Broken", broken, BASELINE, "always fails"))
    val, final = run_pipeline(series, 10, models=["Naive", "Broken"], strategy="ensemble")

    assert ENSEMBLE_NAME not in val
    assert final.model_name == "Naive"


def test_all_models_failing_raises(series, monkeypatch):
    def broken(df, horizon):
        raise RuntimeError

    monkeypatch.setitem(MODEL_REGISTRY, "Broken", ModelSpec("Broken", broken, BASELINE, "always fails"))
    with pytest.raises(RuntimeError, match="All models failed"):
        run_pipeline(series, 10, models=["Broken"])


def test_progress_is_monotonic_and_completes(series):
    calls = []
    run_pipeline(series, 10, models=MODELS, n_windows=2, progress=lambda step, total, _msg: calls.append((step, total)))

    steps = [step for step, _ in calls]
    assert steps == sorted(steps)
    assert {total for _, total in calls} == {len(MODELS) * 2 + 1}
    assert calls[-1] == (len(MODELS) * 2 + 1, len(MODELS) * 2 + 1)


def test_run_all(series):
    results = run_all(series, 10, models=["Naive", "Drift"])
    assert set(results) == {"Naive", "Drift"}


def test_charts(series):
    val, final = run_pipeline(series, 10, models=MODELS, n_windows=2, strategy="ensemble")

    names = [trace.name for trace in plot_forecast(final, show_members=True).data]
    assert any(name.startswith("Naive (peso") for name in names)
    assert "Ensemble previsione" in names

    backtest = plot_backtest(val, series)
    assert sum(trace.showlegend is not False for trace in backtest.data) == 1 + len(val)


def test_each_model_keeps_its_own_style():
    assert _series_style("Chronos-Bolt-small") == _series_style("Chronos-Bolt")
    assert _series_style("Chronos-large") == _series_style("Chronos")
    styles = {tuple(sorted(_series_style(name).items())) for name in [*MODEL_REGISTRY, ENSEMBLE_NAME]}
    assert len(styles) == len(MODEL_REGISTRY) + 1
