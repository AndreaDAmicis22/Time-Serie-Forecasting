import numpy as np
import pandas as pd
import pytest

from stock_forecast.models import ForecastResult


def make_series(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily OHLCV prices (geometric random walk on business days)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "ds": pd.bdate_range("2022-01-03", periods=n),
            "Open": open_,
            "High": np.maximum(open_, close) * 1.005,
            "Low": np.minimum(open_, close) * 0.995,
            "Close": close,
            "Volume": rng.integers(1_000, 10_000, n),
            "y": close,
        }
    )


def make_result(values: list[float], name: str = "m", train_df: pd.DataFrame | None = None) -> ForecastResult:
    """A ForecastResult with a ±1 band around *values*."""
    values = np.asarray(values, dtype=float)
    forecast = pd.DataFrame(
        {
            "ds": pd.bdate_range("2023-01-02", periods=len(values)),
            "yhat": values,
            "yhat_lo": values - 1,
            "yhat_hi": values + 1,
        }
    )
    return ForecastResult(model_name=name, horizon=len(values), forecast=forecast, train_df=train_df)


def assert_valid_forecast(result: ForecastResult, df: pd.DataFrame, horizon: int) -> None:
    fc = result.forecast
    assert list(fc.columns) == ["ds", "yhat", "yhat_lo", "yhat_hi"]
    assert len(fc) == horizon
    assert fc["ds"].iloc[0] > df["ds"].iloc[-1]
    assert fc["ds"].is_monotonic_increasing
    assert np.isfinite(fc[["yhat", "yhat_lo", "yhat_hi"]].to_numpy()).all()
    assert (fc["yhat_lo"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["yhat_hi"]).all()


@pytest.fixture
def series() -> pd.DataFrame:
    return make_series()
