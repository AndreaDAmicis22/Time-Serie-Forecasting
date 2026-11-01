import pytest

from stock_forecast import data
from stock_forecast.data import is_valid_isin, lookup_isin, resolve_ticker

ETF = "IE00B4L5Y983"  # iShares Core MSCI World


def listing(symbol: str, exchange: str = "", name: str = "") -> dict[str, str]:
    return {"symbol": symbol, "name": name or symbol, "exchange": exchange, "type": "ETF"}


class FakeResponse:
    def __init__(self, payload: list) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> list:
        return self.payload


@pytest.mark.parametrize("isin", ["US0378331005", "IT0003132476", ETF, "DE0007164600", "nl0011585146 "])
def test_valid_isin(isin):
    assert is_valid_isin(isin)


@pytest.mark.parametrize("code", ["US0378331006", "AAPL", "IT000313247", "GC=F", "", "1T0003132476"])
def test_invalid_isin(code):
    assert not is_valid_isin(code)


def test_lookup_puts_yahoo_first_and_adds_other_exchanges_with_prices(monkeypatch):
    monkeypatch.setattr(data, "_yahoo_listings", lambda isin: [listing("IWDA.L", "London", "iShares Core MSCI World")])
    monkeypatch.setattr(
        data,
        "_openfigi_listings",
        lambda isin: [listing("SWDA.MI", "Milan"), listing("IWDA.L", "London"), listing("EUNL.DE", "XETRA")],
    )
    monkeypatch.setattr(data, "_with_prices", lambda listings: [item for item in listings if item["symbol"] != "EUNL.DE"])

    listings = lookup_isin(ETF)

    assert [item["symbol"] for item in listings] == ["IWDA.L", "SWDA.MI"]
    assert listings[1]["name"] == "iShares Core MSCI World"


def test_lookup_rejects_invalid_isin():
    with pytest.raises(ValueError, match="not a valid ISIN"):
        lookup_isin("XX123")


def test_openfigi_rows_are_mapped_to_yahoo_symbols(monkeypatch):
    rows = [
        {"ticker": "IWDA", "exchCode": "NA", "name": "ISHARES CORE MSCI WORLD", "securityType": "ETP"},
        {"ticker": "SWDA", "exchCode": "IM", "name": "ISHARES CORE MSCI WORLD", "securityType": "ETP"},
        {"ticker": "SWDA", "exchCode": "IM", "name": "ISHARES CORE MSCI WORLD", "securityType": "ETP"},
        {"ticker": "EUNL", "exchCode": "XO", "name": "ISHARES CORE MSCI WORLD", "securityType": "ETP"},
        {"ticker": "BRK/B", "exchCode": "US", "name": "BERKSHIRE HATHAWAY", "securityType": "Common Stock"},
        {"ticker": "BTPS 4.2 03/01/34 10Y", "exchCode": "MOT", "name": "BTP", "securityType": "EURO-ZONE"},
    ]
    monkeypatch.setattr(data.requests, "post", lambda *args, **kwargs: FakeResponse([{"data": rows}]))
    monkeypatch.setattr(data, "_yahoo_listings", lambda isin: [])
    monkeypatch.setattr(data, "_with_prices", lambda listings: listings)

    listings = lookup_isin(ETF)

    assert [item["symbol"] for item in listings] == ["SWDA.MI", "IWDA.AS", "BRK-B"]  # Milan first, unknown venues dropped
    assert listings[0]["exchange"] == "Milan"


def test_resolve_ticker(monkeypatch):
    monkeypatch.setattr(data, "_yahoo_listings", lambda isin: [listing("ENI.MI", "Milan")])
    assert resolve_ticker("GC=F") == "GC=F"
    assert resolve_ticker("IT0003132476") == "ENI.MI"


def test_resolve_ticker_without_listing(monkeypatch):
    monkeypatch.setattr(data, "_yahoo_listings", lambda isin: [])
    monkeypatch.setattr(data, "_openfigi_listings", lambda isin: [])
    with pytest.raises(ValueError, match="No Yahoo Finance listing"):
        resolve_ticker("IT0005560948")  # a BTP: bonds are not on Yahoo Finance


@pytest.mark.slow
def test_lookup_isin_online():
    listings = lookup_isin(ETF)
    symbols = [item["symbol"] for item in listings]
    assert "SWDA.MI" in symbols
    assert lookup_isin("IT0003132476", other_exchanges=False)[0]["symbol"] == "ENI.MI"
