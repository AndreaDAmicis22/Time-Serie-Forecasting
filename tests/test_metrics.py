import numpy as np
import pytest

from stock_forecast.metrics import compute_metrics, failed_metrics


def test_basic_metrics():
    m = compute_metrics([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["mape"] == pytest.approx(100 / 9)
    assert "rel_mae" not in m
    assert "coverage" not in m


def test_mape_ignores_zero_actuals():
    m = compute_metrics([0, 2], [1, 3])
    assert m["mape"] == pytest.approx(50.0)


def test_rel_mae_is_one_for_the_naive_forecast():
    y_train = np.array([10.0, 11.0, 12.0])
    m = compute_metrics([13, 11, 15], np.full(3, 12.0), y_train=y_train)
    assert m["rel_mae"] == pytest.approx(1.0)


def test_rel_mae_below_one_when_beating_the_naive_forecast():
    m = compute_metrics([13, 14, 15], [13, 14, 15.5], y_train=[10, 11, 12])
    assert m["rel_mae"] < 1


def test_coverage():
    m = compute_metrics([1, 2, 3, 4], [1, 2, 3, 4], y_lo=[0, 0, 3.5, 0], y_hi=[2, 2, 4, 5])
    assert m["coverage"] == pytest.approx(75.0)


def test_failed_metrics_rank_last():
    m = failed_metrics()
    assert m["mae"] == np.inf
    assert m["rel_mae"] == np.inf
