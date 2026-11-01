import inspect

import numpy as np
import pandas as pd
import pytest
from conftest import assert_valid_forecast

from stock_forecast.models import FAMILIES, MODEL_REGISTRY, run_xgboost
from stock_forecast.models.base import future_dates, infer_freq

FAST_MODELS = {
    "Naive": {},
    "Drift": {},
    "AutoETS": {},
    "Theta": {},
    "Auto-ARIMA": {"max_p": 1, "max_q": 1},
    "XGBoost": {"n_estimators": 50},
}

# Pretrained weights are downloaded on first use; neural nets train from scratch.
SLOW_MODELS = {
    "Chronos-2": {},
    "Chronos-Bolt": {"model_size": "tiny"},
    "Chronos": {"model_size": "tiny", "num_samples": 5},
    "Timestral": {"model_size": "5M"},
    "TimesFM-3": {},
    "TiRex": {},
    "N-HiTS": {"max_steps": 20},
    "N-BEATS": {"max_steps": 20},
}


@pytest.mark.parametrize(("name", "params"), FAST_MODELS.items())
def test_fast_models(series, name, params):
    assert_valid_forecast(MODEL_REGISTRY[name](series, 10, **params), series, 10)


@pytest.mark.slow
@pytest.mark.parametrize(("name", "params"), SLOW_MODELS.items())
def test_slow_models(series, name, params):
    spec = MODEL_REGISTRY[name]
    if not spec.available:
        pytest.skip(f"{name} is not installed (extra '{spec.extra}')")
    assert_valid_forecast(spec(series, 10, **params), series, 10)


def test_every_registered_model_is_covered_by_a_test():
    assert set(FAST_MODELS) | set(SLOW_MODELS) == set(MODEL_REGISTRY)


def test_registry_params_match_runner_signatures():
    for spec in MODEL_REGISTRY.values():
        signature = inspect.signature(spec.runner).parameters
        assert list(signature)[:2] == ["df", "horizon"], spec.name
        for param in spec.params:
            assert param.name in signature, (spec.name, param.name)
            assert signature[param.name].default == param.default, (spec.name, param.name)
            if param.choices is not None:
                assert param.default in param.choices, (spec.name, param.name)


def test_registry_metadata():
    for spec in MODEL_REGISTRY.values():
        assert spec.family in FAMILIES, spec.name
        assert spec.description, spec.name
        if spec.requires:
            assert spec.extra, spec.name


def test_infer_freq():
    assert infer_freq(pd.Series(pd.bdate_range("2024-01-01", periods=50))) == "B"
    assert infer_freq(pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))) == "D"  # crypto trades on weekends
    assert infer_freq(pd.Series(pd.date_range("2024-01-01", periods=50, freq="W-MON"))) == "W-MON"
    assert infer_freq(pd.Series(pd.date_range("2024-01-01", periods=24, freq="MS"))) == "MS"


def test_weekly_future_dates_keep_the_weekday():
    dates = future_dates(pd.Timestamp("2024-06-03"), 3, "W-MON")  # a Monday
    assert list(dates) == list(pd.to_datetime(["2024-06-10", "2024-06-17", "2024-06-24"]))


def test_xgboost_diff_target_extrapolates_a_trend():
    y = np.linspace(100, 300, 200)
    df = pd.DataFrame({"ds": pd.bdate_range("2022-01-03", periods=200), "y": y})

    on_diff = run_xgboost(df, 10, n_estimators=50)
    on_level = run_xgboost(df, 10, n_estimators=50, target="level")

    assert on_diff.forecast["yhat"].iloc[-1] > y.max()
    assert on_level.forecast["yhat"].iloc[-1] <= y.max() + 1e-6  # trees cannot leave the training range
