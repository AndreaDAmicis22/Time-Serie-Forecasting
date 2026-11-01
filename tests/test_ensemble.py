import numpy as np
import pytest
from conftest import make_result

from stock_forecast.ensemble import EnsembleResult, combine_forecasts, ensemble_weights, select_members


@pytest.fixture
def members():
    return {"a": make_result([1, 1, 1], "a"), "b": make_result([3, 3, 3], "b"), "c": make_result([8, 8, 8], "c")}


def test_mean(members):
    ens = combine_forecasts(members, "mean")
    assert isinstance(ens, EnsembleResult)
    np.testing.assert_allclose(ens.forecast["yhat"], 4.0)
    np.testing.assert_allclose(ens.forecast["yhat_lo"], 3.0)
    assert ens.weights == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_median(members):
    np.testing.assert_allclose(combine_forecasts(members, "median").forecast["yhat"], 3.0)


def test_weighted(members):
    ens = combine_forecasts(members, "weighted", {"a": 0.5, "b": 0.5, "c": 0.0})
    np.testing.assert_allclose(ens.forecast["yhat"], 2.0)
    assert set(ens.members) == {"a", "b", "c"}


def test_inverse_error_weights():
    assert ensemble_weights({"a": 1.0, "b": 3.0}, "weighted") == pytest.approx({"a": 0.75, "b": 0.25})
    assert ensemble_weights({"a": 1.0, "b": 3.0}, "mean") == pytest.approx({"a": 0.5, "b": 0.5})


def test_select_members_ranks_and_skips_failures():
    ok = make_result([1], "x")
    results = {
        "a": {"metrics": {"mae": 2.0}, "result": ok},
        "b": {"metrics": {"mae": np.inf}, "result": None},
        "c": {"metrics": {"mae": 1.0}, "result": ok},
        "d": {"metrics": {"mae": 5.0}, "result": ok},
    }
    assert select_members(results, "mae", top_k=2) == {"c": 1.0, "a": 2.0}


def test_empty_ensemble_raises():
    with pytest.raises(ValueError, match="zero forecasts"):
        combine_forecasts({})
