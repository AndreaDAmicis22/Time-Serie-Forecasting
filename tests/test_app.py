"""Headless tests of the Streamlit app (no browser, no network)."""

from pathlib import Path

import pytest
from conftest import make_series
from streamlit.testing.v1 import AppTest

import stock_forecast.data

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
FAST = ("Naive", "Drift", "Theta")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(stock_forecast.data, "download_series", lambda ticker, period, interval: make_series())
    monkeypatch.setattr(stock_forecast.data, "get_ticker_info", lambda ticker: {"ticker": ticker, "name": "Test"})


def start_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def select_fast_models(at: AppTest) -> None:
    for multiselect in at.multiselect:
        multiselect.set_value([name for name in multiselect.options if name in FAST])
    at.run()


def test_initial_render():
    at = start_app()
    assert len(at.tabs) == 4
    assert "run" not in at.session_state


@pytest.mark.parametrize("strategy", ["auto", "best", "ensemble"])
def test_full_run(strategy):
    at = start_app()
    at.radio[0].set_value(strategy).run()
    select_fast_models(at)
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.error
    run = at.session_state["run"]
    assert len(run["final"].forecast) == 15
    assert set(FAST) <= set(run["val_results"])
    if strategy == "ensemble":
        assert run["final"].model_name == "Ensemble"


def test_single_model_run():
    at = start_app()
    at.radio[0].set_value("single").run()
    model_select = next(s for s in at.selectbox if s.label == "Modello")
    model_select.set_value("Theta").run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.session_state["run"]["final"].model_name == "Theta"


def test_no_model_selected_shows_an_error():
    at = start_app()
    for multiselect in at.multiselect:
        multiselect.set_value([])
    at.run()
    at.sidebar.button[0].click().run()

    assert at.error
    assert "run" not in at.session_state
