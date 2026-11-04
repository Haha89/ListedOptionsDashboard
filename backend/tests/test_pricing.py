"""Black-Scholes, implied vol and binomial checks against textbook values and no-arbitrage identities"""

import math
from datetime import datetime, timedelta

import pytest

from src.option_vol.binomial.pricer import price_call
from src.option_vol.models import Call, Environment, Put
from src.option_vol.models.base_option import NEW_YORK

SPOT, RATE = 100.0, 0.05


@pytest.fixture(autouse=True)
def market(monkeypatch):
    """Fake market so options never hit Yahoo: spot 100 for 'TEST', 5% rate"""
    monkeypatch.setitem(Environment.spots, "TEST", SPOT)
    monkeypatch.setattr(Environment, "risk_free_rate", RATE)


def make(cls, strike=100.0, days=365, price=None, vol=None):
    """Option on TEST expiring in `days` calendar days (365 days = 1 year), optionally with a set implied vol"""
    option = cls(strike, datetime.now(NEW_YORK).date() + timedelta(days=days), "TEST", price)
    option.implied_vol = vol
    return option


def test_one_year_is_one_year():
    assert make(Call).T == 1.0


# Hull, S=K=100, r=5%, vol=20%, T=1y: call 10.4506, put 5.5735
@pytest.mark.parametrize("cls, expected", [(Call, 10.4506), (Put, 5.5735)])
def test_black_scholes_textbook_price(cls, expected):
    assert make(cls).price_with_black_scholes(0.2) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("strike", [60, 90, 100, 110, 150])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.8])
def test_put_call_parity(strike, vol):
    call, put = make(Call, strike), make(Put, strike)
    lhs = call.price_with_black_scholes(vol) - put.price_with_black_scholes(vol)
    assert lhs == pytest.approx(SPOT - strike * math.exp(-RATE), abs=1e-9)


@pytest.mark.parametrize("cls", [Call, Put])
@pytest.mark.parametrize("strike", [50, 80, 100, 120, 200])
@pytest.mark.parametrize("vol", [0.05, 0.3, 1.5])
@pytest.mark.parametrize("days", [7, 90, 730])
def test_implied_vol_round_trip(cls, strike, vol, days):
    option = make(cls, strike, days)
    price = option.price_with_black_scholes(vol)
    solved = make(cls, strike, days, price).set_implied_volatility()
    assert option.price_with_black_scholes(solved) == pytest.approx(price, abs=1e-8)
    # deep in/out of the money the price barely moves with vol (tiny vega), so many vols reproduce it
    if option.get_vega(vol) > 1e-2:
        assert solved == pytest.approx(vol, abs=1e-5)


@pytest.mark.parametrize(
    "cls, strike, price",
    [
        (Call, 80, 15.0),  # below the ~23.9 a zero-vol call is worth
        (Put, 120, 10.0),  # below the ~14.1 a zero-vol put is worth
        (Call, 100, 150.0),  # above the spot: no vol can produce it
    ],
)
def test_implied_vol_is_nan_when_no_vol_matches(cls, strike, price):
    assert math.isnan(make(cls, strike, price=price).set_implied_volatility())


# Same textbook option: theta per calendar day (annual / 365), rho per 1% rate move, vega per 1.00 of vol
@pytest.mark.parametrize(
    "cls, delta, theta, rho",
    [
        (Call, 0.636831, -6.414028 / 365, 0.532325),
        (Put, -0.363169, -1.657880 / 365, -0.418905),
    ],
)
def test_greeks_textbook_values(cls, delta, theta, rho):
    option = make(cls, vol=0.2)
    option.set_greeks()
    assert option.delta == pytest.approx(delta, abs=1e-6)
    assert option.gamma == pytest.approx(0.018762, abs=1e-6)
    assert option.vega == pytest.approx(37.524035, abs=1e-6)
    assert option.theta == pytest.approx(theta, abs=1e-6)
    assert option.rho == pytest.approx(rho, abs=1e-6)


@pytest.mark.parametrize("cls", [Call, Put])
def test_greeks_match_finite_differences(cls, monkeypatch):
    vol, h = 0.25, 1e-3
    option = make(cls, 110, 200, vol=vol)
    option.set_greeks()

    def price_at(spot=SPOT, sigma=vol):
        monkeypatch.setitem(Environment.spots, "TEST", spot)
        return make(cls, 110, 200).price_with_black_scholes(sigma)

    assert option.delta == pytest.approx((price_at(SPOT + h) - price_at(SPOT - h)) / (2 * h), abs=1e-6)
    assert option.gamma == pytest.approx((price_at(SPOT + h) - 2 * price_at() + price_at(SPOT - h)) / h**2, abs=1e-4)
    assert option.vega == pytest.approx((price_at(sigma=vol + h) - price_at(sigma=vol - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("strike", [80, 100, 120])
def test_binomial_converges_to_black_scholes(strike):
    call = make(Call, strike)
    assert price_call(call, vol=0.2, periods=2000) == pytest.approx(call.price_with_black_scholes(0.2), abs=5e-3)


def test_binomial_one_step_by_hand():
    # u = e^0.2, d = 1/u, p = (e^0.05 - d) / (u - d); payoff only in the up state; discount e^-0.05
    u = math.exp(0.2)
    p = (math.exp(RATE) - 1 / u) / (u - 1 / u)
    expected = math.exp(-RATE) * p * (SPOT * u - 100)
    assert price_call(make(Call), vol=0.2, periods=1) == pytest.approx(expected, abs=1e-12)
