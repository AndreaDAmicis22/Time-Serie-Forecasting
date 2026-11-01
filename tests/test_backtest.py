import numpy as np
import pandas as pd
import pytest

from stock_forecast.backtest import backtest_windows, resolve_model_params, train_val_split, validate_models
from stock_forecast.models import MODEL_REGISTRY, ModelSpec, available_models
from stock_forecast.models.base import BASELINE, FOUNDATION
from stock_forecast.pipeline import select_best_model


def test_windows_are_consecutive_and_end_at_the_last_observation(series):
    windows = backtest_windows(series, horizon=10, n_windows=3)

    assert [len(train) for train, _ in windows] == [270, 280, 290]
    for train, test in windows:
        assert len(test) == 10
        assert train["ds"].max() < test["ds"].min()
    assert windows[-1][1]["ds"].iloc[-1] == series["ds"].iloc[-1]


def test_single_window_matches_train_val_split(series):
    ((train, test),) = backtest_windows(series, horizon=10, n_windows=1)
    expected_train, expected_test = train_val_split(series, val_size=10)
    pd.testing.assert_frame_equal(train, expected_train)
    pd.testing.assert_frame_equal(test, expected_test)


def test_too_short_series_raises(series):
    with pytest.raises(ValueError, match="too short"):
        backtest_windows(series.head(50), horizon=10, n_windows=3)


def test_validate_models_structure(series):
    results = validate_models(series, 10, models=["Naive", "Drift"], n_windows=2)

    for name in ("Naive", "Drift"):
        assert results[name]["error"] is None
        assert len(results[name]["windows"]) == 2
        assert results[name]["result"] is results[name]["windows"][-1]
        assert {"mae", "rmse", "mape", "rel_mae", "coverage"} <= set(results[name]["metrics"])
    assert results["Naive"]["metrics"]["rel_mae"] == pytest.approx(1.0)


def test_failing_model_is_reported_not_raised(series, monkeypatch):
    def broken(df, horizon):
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setitem(MODEL_REGISTRY, "Broken", ModelSpec("Broken", broken, BASELINE, "always fails"))
    results = validate_models(series, 10, models=["Naive", "Broken"])

    assert results["Broken"]["error"] == "boom"
    assert results["Broken"]["metrics"]["mae"] == np.inf
    assert select_best_model(results, "mae") == "Naive"


def test_missing_optional_dependency_gives_install_hint(series, monkeypatch):
    spec = ModelSpec("Ghost", lambda df, horizon: None, FOUNDATION, "", requires=("no_such_module_xyz",), extra="ghost")
    monkeypatch.setitem(MODEL_REGISTRY, "Ghost", spec)

    assert "Ghost" not in available_models()
    results = validate_models(series, 10, models=["Ghost"])
    assert "poetry install --extras ghost" in results["Ghost"]["error"]


def test_legacy_shortcuts_are_merged_into_model_params():
    params = resolve_model_params(["Chronos", "N-HiTS", "Naive"], chronos_size="large", max_steps=123)
    assert params == {"Chronos": {"model_size": "large"}, "N-HiTS": {"max_steps": 123}, "Naive": {}}

    # explicit per-model parameters win over the shortcuts
    params = resolve_model_params(["Chronos"], {"Chronos": {"model_size": "tiny"}}, chronos_size="large")
    assert params["Chronos"]["model_size"] == "tiny"
