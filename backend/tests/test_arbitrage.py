from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.option_vol import arbitrage
from src.option_vol import black_scholes as bs

SPOT, RATE = 100.0, 0.05
EXPIRIES = {date(2027, 1, 15): 0.25, date(2027, 4, 16): 0.5}


def chain(vol=0.3, spread=0.01) -> pd.DataFrame:
    """Black-Scholes quotes, no dividend, bid/ask a relative spread around the price"""
    rows = [
        dict(type=kind, maturity=m, strike=float(k), T=T)
        for m, T in EXPIRIES.items()
        for kind in ("Call", "Put")
        for k in range(80, 121, 5)
    ]
    q = pd.DataFrame(rows).assign(spot=SPOT, r=RATE, div_yield=0.0)
    q["forward"] = SPOT * np.exp(RATE * q["T"])
    price = bs.price(SPOT, q.strike, q["T"], RATE, vol, q.type == "Call")
    return q.assign(bid=price * (1 - spread / 2), ask=price * (1 + spread / 2))


def at(q, kind, strike, maturity=date(2027, 1, 15)):
    return (q.type == kind) & (q.strike == strike) & (q.maturity == maturity)


def test_a_clean_chain_has_no_arbitrage():
    assert arbitrage.scan(chain()).empty


def test_wide_quotes_hide_nothing_real():
    assert arbitrage.scan(chain(spread=0.5)).empty


def test_lower_bound():
    q = chain()
    q.loc[at(q, "Call", 100), ["bid", "ask"]] = [0.9, 1.0]  # in the money by about 1.26, within the live band
    found = arbitrage.scan(q)
    row = found[found.check == "lower bound"].iloc[0]
    intrinsic = np.exp(-RATE * 0.25) * (SPOT * np.exp(RATE * 0.25) - 100)
    assert row.strikes == "100" and row.profit == pytest.approx(intrinsic - 1.0)


def test_stale_deep_in_the_money_quotes_are_not_scanned():
    q = chain()
    q.loc[at(q, "Call", 80), ["bid", "ask"]] = [10.0, 10.5]  # far below intrinsic, but 80 is 21% in the money
    assert arbitrage.scan(q).empty


def test_upper_bound():
    q = chain()
    q.loc[at(q, "Put", 90), ["bid", "ask"]] = [91.0, 92.0]  # a put is worth at most its strike
    found = arbitrage.scan(q)
    assert found[found.check == "upper bound"].profit.tolist() == [pytest.approx(1.0)]


def test_vertical_out_of_order():
    q = chain()
    ask_105 = float(q.loc[at(q, "Call", 105), "ask"].iloc[0])
    q.loc[at(q, "Call", 110), ["bid", "ask"]] = [ask_105 + 0.5, ask_105 + 0.6]  # the 110 call dearer than the 105
    found = arbitrage.scan(q)
    vertical = found[(found.check == "vertical") & (found.strikes == "105/110")]
    assert vertical.profit.tolist() == [pytest.approx(0.5)]


def test_butterfly_with_unequal_gaps():
    q = chain()
    q = q[~(q.strike == 110)].copy()  # out-of-the-money calls 105 / 115 / 120: weights 1/3 and 2/3 around 115
    low, high = (float(q.loc[at(q, "Call", k), "ask"].iloc[0]) for k in (105, 120))
    body = low / 3 + 2 * high / 3 + 0.2  # the body above the wings: selling it pays more than the wings cost
    q.loc[at(q, "Call", 115), ["bid", "ask"]] = [body, body + 0.01]
    found = arbitrage.scan(q)
    fly = found[(found.check == "butterfly") & (found.strikes == "105/115/120")]
    assert fly.profit.tolist() == [pytest.approx(0.2)]


def test_box(monkeypatch):
    monkeypatch.setattr(arbitrage, "LIVE_BAND", 0.1)  # strikes 5 apart: widen the band to have several near spot
    q = chain()
    q.loc[at(q, "Call", 110), ["bid", "ask"]] += 1.0  # call 110 rich by 1: sell it, buy the put, reverse elsewhere
    found = arbitrage.scan(q)
    box = found[found.check == "box"]
    assert len(box) == 1 and box.strikes.iloc[0].startswith("110/")
    assert 0.8 < box.profit.iloc[0] < 1.0  # the 1.0 less the half-spreads of the four legs


def test_calendar():
    q = chain()
    later = q.maturity == date(2027, 4, 16)
    cheap = bs.price(SPOT, q.strike, q["T"], RATE, 0.15, q.type == "Call")  # 0.15^2 * 0.5 < 0.3^2 * 0.25
    q.loc[later & (q.strike == 110), ["bid", "ask"]] = np.column_stack(
        [cheap[later & (q.strike == 110)] * 0.99, cheap[later & (q.strike == 110)] * 1.01]
    )
    found = arbitrage.scan(q)
    calendar = found[found.check == "calendar"]
    assert calendar.strikes.tolist() == ["110"] and (calendar.profit > 0).all()
    assert calendar.maturity.iloc[0] == date(2027, 4, 16)
