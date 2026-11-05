"""Option chain parsing from Yahoo, with yfinance replaced by a fake"""

import math
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from src.option_vol import scrapping
from src.option_vol.models import Call, Environment, Put
from src.option_vol.models.base_option import CLOSE, OPEN

TODAY = date.today()
EXPIRIES = [TODAY, TODAY + timedelta(days=10), TODAY + timedelta(days=40)]


class FakeTicker:
    options = tuple(e.isoformat() for e in EXPIRIES)
    with_am_root = False

    def __init__(self, symbol):
        assert symbol == "TEST"

    def option_chain(self, expiry):
        quotes = pd.DataFrame(
            {  # two-sided quote, no quote but a last trade, nothing at all
                "contractSymbol": [f"TEST{expiry.replace('-', '')[2:]}C000{k}000" for k in (90, 100, 110)],
                "strike": [90.0, 100.0, 110.0],
                "bid": [10.0, 0.0, float("nan")],
                "ask": [12.0, 0.0, float("nan")],
                "lastPrice": [11.5, 3.0, 0.0],
                "volume": [5.0, float("nan"), float("nan")],
                "openInterest": [100.0, 7.0, 0.0],
            }
        )
        if self.with_am_root:  # the same strikes again under SPX-style AM and SPXW-style PM roots
            am, pm = quotes.copy(), quotes.copy()
            am["contractSymbol"] = "SPX" + am.contractSymbol.str[4:]
            pm["contractSymbol"] = "SPXW" + pm.contractSymbol.str[4:]
            pm["bid"] += 1  # tell them apart
            quotes = pd.concat([am, pm], ignore_index=True)
        return SimpleNamespace(calls=quotes, puts=quotes)


@pytest.fixture(autouse=True)
def fake_yahoo(monkeypatch):
    monkeypatch.setattr(scrapping, "Ticker", FakeTicker)
    monkeypatch.setitem(Environment.spots, "TEST", 100.0)
    monkeypatch.setattr(Environment, "risk_free_rate", 0.05)


def test_prices_use_mid_then_last_trade_and_skip_empty_quotes():
    options = scrapping.Scrapping.parse_underlying_until("TEST", EXPIRIES[1])
    assert [(type(o), o.strike, o.price) for o in options] == [
        (Call, 90.0, 11.0),
        (Call, 100.0, 3.0),
        (Put, 90.0, 11.0),
        (Put, 100.0, 3.0),
    ]


def test_expiries_after_today_up_to_max_maturity():
    options = scrapping.Scrapping.parse_underlying_until("TEST", EXPIRIES[2])
    assert sorted({o.maturity for o in options}) == EXPIRIES[1:]


def test_nearest_expiry_when_none_before_max_maturity():
    assert {o.maturity for o in scrapping.Scrapping().parse_option("TEST")} == {EXPIRIES[1]}


def test_keeps_bid_ask_volume_and_open_interest():
    call_90, call_100 = scrapping.Scrapping.parse_underlying_until("TEST", EXPIRIES[1])[:2]
    assert (call_90.bid, call_90.ask, call_90.volume, call_90.open_interest) == (10.0, 12.0, 5, 100)
    assert math.isnan(call_100.bid) and math.isnan(call_100.ask) and call_100.volume == 0


def test_pm_root_wins_over_am_root_on_the_same_date(monkeypatch):
    monkeypatch.setattr(FakeTicker, "with_am_root", True)
    options = scrapping.Scrapping.parse_underlying_until("TEST", EXPIRIES[1])
    assert [o.bid for o in options if o.type == "Call" and o.bid == o.bid] == [11.0]  # SPXW's bid 10 + 1
    assert all(o.expires_at == CLOSE for o in options)


def test_am_root_alone_expires_at_the_open(monkeypatch):
    option_chain = FakeTicker.option_chain

    def am_only(self, expiry):
        chain = option_chain(self, expiry)
        chain.calls["contractSymbol"] = "SPX" + chain.calls.contractSymbol.str[4:]
        return chain

    monkeypatch.setattr(FakeTicker, "option_chain", am_only)
    expected = OPEN if scrapping.settles_at_open("SPX", EXPIRIES[1]) else CLOSE
    assert {o.expires_at for o in scrapping.Scrapping.parse_underlying_until("TEST", EXPIRIES[1])} == {expected}


@pytest.mark.parametrize(
    "root, expiry, at_open",
    [
        ("SPX", date(2026, 11, 20), True),  # third Friday monthly
        ("SPX", date(2026, 12, 31), False),  # quarterly, listed under SPX by Yahoo but PM-settled
        ("SPXW", date(2026, 11, 20), False),
        ("SPY", date(2026, 11, 20), False),
    ],
)
def test_only_third_friday_index_monthlies_settle_at_the_open(root, expiry, at_open):
    assert scrapping.settles_at_open(root, expiry) is at_open
