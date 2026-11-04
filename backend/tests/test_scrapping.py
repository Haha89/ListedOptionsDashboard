""" Option chain parsing from Yahoo, with yfinance replaced by a fake """
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from src.option_vol import scrapping
from src.option_vol.models import Call, Environment, Put

TODAY = date.today()
EXPIRIES = [TODAY, TODAY + timedelta(days=10), TODAY + timedelta(days=40)]


class FakeTicker:
    options = tuple(e.isoformat() for e in EXPIRIES)

    def __init__(self, symbol):
        assert symbol == "TEST"

    def option_chain(self, expiry):
        quotes = pd.DataFrame({  # two-sided quote, no quote but a last trade, nothing at all
            "strike": [90.0, 100.0, 110.0],
            "bid": [10.0, 0.0, float("nan")],
            "ask": [12.0, 0.0, float("nan")],
            "lastPrice": [11.5, 3.0, 0.0],
        })
        return SimpleNamespace(calls=quotes, puts=quotes)


@pytest.fixture(autouse=True)
def fake_yahoo(monkeypatch):
    monkeypatch.setattr(scrapping, "Ticker", FakeTicker)
    monkeypatch.setitem(Environment.spots, "TEST", 100.0)


def test_prices_use_mid_then_last_trade_and_skip_empty_quotes():
    options = scrapping.Scrapping.parse_underlying_until("TEST", EXPIRIES[1])
    assert [(type(o), o.strike, o.price) for o in options] == [
        (Call, 90.0, 11.0), (Call, 100.0, 3.0), (Put, 90.0, 11.0), (Put, 100.0, 3.0)]


def test_expiries_after_today_up_to_max_maturity():
    options = scrapping.Scrapping.parse_underlying_until("TEST", EXPIRIES[2])
    assert sorted({o.maturity for o in options}) == EXPIRIES[1:]


def test_nearest_expiry_when_none_before_max_maturity():
    assert {o.maturity for o in scrapping.Scrapping().parse_option("TEST")} == {EXPIRIES[1]}
