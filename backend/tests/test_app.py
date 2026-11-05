"""Dashboard data path: pricing a chain, filtering it and building the surface, with Yahoo replaced by a fake"""

import numpy as np
import pytest

import app
from src.option_vol.models import Call, Environment, Put
from tests.test_pricing import make, market  # noqa: F401  (autouse fixture: spot 100, 5% rate)

VOL = 0.3


def vol_of(strike):  # flat by default; tests swap it for a skew
    return VOL + 0 * strike


DIV_YIELD = 0.0  # tests swap it to check the implied yield


@pytest.fixture(autouse=True)
def fake_chain(monkeypatch):
    """Calls and puts at strikes 50..150 and 3 expiries, priced at vol_of(strike)"""

    def chain(underlying, expiry_date):
        monkeypatch.setitem(Environment.spots, "TEST", 100.0)  # load_options dropped it to force a fresh spot
        options = [make(cls, k, days) for cls in (Call, Put) for k in range(50, 151, 10) for days in (30, 90, 180)]
        for o in options:
            o.div_yield = globals()["DIV_YIELD"]
            o.price = o.price_with_black_scholes(globals()["vol_of"](o.strike))
        return options

    monkeypatch.setattr(app.Scrapping, "parse_underlying_until", staticmethod(chain))
    monkeypatch.setattr(app, "_cache", {})


def test_load_options_recovers_vol_and_greeks():
    df = app.load_options("TEST", 6)
    assert len(df) == 66
    liquid = df[df.vega > 1e-2]  # elsewhere vol is not identifiable from the price
    assert np.allclose(liquid.implied_vol, VOL, atol=1e-6)
    atm_call = make(Call, 100, 90, vol=VOL)
    atm_call.set_greeks()
    row = df[(df.type == "Call") & (df.strike == 100) & (df["T"] == atm_call.T)].iloc[0]
    assert row.delta == pytest.approx(atm_call.delta) and row.theta == pytest.approx(atm_call.theta)


def test_one_snapshot_per_ticker(monkeypatch):
    fetches = []
    fetch = app.Scrapping.parse_underlying_until
    monkeypatch.setattr(
        app.Scrapping, "parse_underlying_until", staticmethod(lambda u, e: fetches.append(e) or fetch(u, e))
    )
    six_months = app.load_options("TEST", 6)
    one_month = app.load_options("TEST", 1)  # filtered from the same snapshot
    assert len(fetches) == 1 and one_month.maturity.nunique() == 1 and len(six_months) == 66
    assert one_month.equals(six_months[six_months.maturity == six_months.maturity.min()])
    app.load_options("TEST", 12)  # beyond the snapshot: fetch
    app.load_options("TEST", 12, refresh=True)
    assert len(fetches) == 3


def test_select_and_surface():
    view = app.select(app.load_options("TEST", 6), "Put", 15)
    assert set(view.type) == {"Put"} and set(view.strike) == {90, 100, 110}
    fig = app.surface(view, "implied_vol")
    assert np.asarray(fig.data[0].z).shape == (3, 3)  # strikes x expiries


def test_by_delta_flat_vol():
    grid = app.by_delta(app.load_options("TEST", 6))
    assert list(grid.index) == [
        "5Δ put",
        "10Δ put",
        "25Δ put",
        "40Δ put",
        "ATM (fwd)",
        "40Δ call",
        "25Δ call",
        "10Δ call",
        "5Δ call",
    ]
    assert grid.shape == (9, 3) and np.allclose(grid, VOL)


def test_by_delta_follows_the_skew(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.3 + 0.002 * (100 - strike))  # vol falls with strike
    grid = app.by_delta(app.load_options("TEST", 6))
    assert grid.notna().all().all()
    assert (grid.diff().iloc[1:] < 0).all().all()  # rows run by increasing strike, so vol decreases down each column


def test_smile_is_otm_puts_then_otm_calls():
    df = app.load_options("TEST", 6)
    puts, calls, atm, fit = app.smile(app.in_range(df, 50), app.expiries(df)[0], 100.0).data
    forward = 100 * np.exp(0.05 * df["T"].min())  # nearest expiry, no dividend
    assert atm.x[0] == pytest.approx(forward) and atm.y[0] == pytest.approx(VOL)
    assert puts.name == "OTM puts" and max(puts.x) < forward <= min(calls.x) and calls.name == "OTM calls"
    assert np.allclose(np.concatenate([puts.y, calls.y]), VOL)
    assert fit.name.startswith("eSSVI fit") and np.allclose(fit.y, VOL, atol=1e-3)


def test_ssvi_fit_follows_a_put_skew_without_calendar_arbitrage(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.3 + 0.002 * (100 - strike))
    df = app.load_options("TEST", 6)
    fitted = app.ssvi_fit(df)
    assert len(fitted) == 3 and (fitted.rho < -0.5).all() and fitted.theta.is_monotonic_increasing
    for earlier, later in zip(fitted.to_numpy(), fitted.to_numpy()[1:]):
        assert app.ssvi.calendar_margin(earlier, later).min() >= -1e-9
    smiles = app.otm(df).join(fitted.add_prefix("ssvi_"), on="maturity")  # the chain has greeks theta and rho
    k = np.log(smiles.strike / smiles.forward)
    errors = app.ssvi.vol(k, smiles["T"], smiles.ssvi_theta, smiles.ssvi_rho, smiles.ssvi_psi) - smiles.implied_vol
    near = (smiles.strike / 100 - 1).abs() <= 0.2  # a linear skew is not an SSVI smile; it should match near the money
    assert errors[near].abs().max() < 0.01


def test_implied_yield_recovers_dividend_and_aligns_call_and_put_vols(monkeypatch):
    monkeypatch.setitem(globals(), "DIV_YIELD", 0.02)
    df = app.load_options("TEST", 6)  # the fake options themselves carry q = 0; load_options must find it
    assert np.allclose(df.div_yield, 0.02)
    liquid = df[df.vega > 1e-2]
    assert np.allclose(liquid.implied_vol, VOL, atol=1e-6)


def test_atm_vol_interpolates_at_the_forward(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.3 + 0.002 * (100 - strike))
    df = app.load_options("TEST", 6)
    df = df[df.strike != 100]  # forward falls between the 90 and 110 strikes, where the vol is linear in strike
    forwards = (100 * np.exp(0.05 * df.groupby("maturity")["T"].first())).to_numpy()
    assert np.allclose(app.atm_vol(df), 0.3 - 0.002 * (forwards - 100), atol=1e-6)


def test_rr_bf_from_a_skewed_smile(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.3 + 0.002 * (100 - strike))
    grid = app.by_delta(app.load_options("TEST", 6))
    spreads = app.rr_bf(grid)
    assert list(spreads.index) == ["RR 25Δ", "RR 10Δ", "BF 25Δ", "BF 10Δ"]
    assert np.allclose(spreads.loc["RR 25Δ"], grid.loc["25Δ call"] - grid.loc["25Δ put"])
    assert np.allclose(spreads.loc["BF 10Δ"], (grid.loc["10Δ call"] + grid.loc["10Δ put"]) / 2 - grid.loc["ATM (fwd)"])
    assert (spreads.loc[["RR 25Δ", "RR 10Δ"]] < 0).all().all()  # vol falling with strike: puts above calls
    assert (spreads.loc["RR 10Δ"] < spreads.loc["RR 25Δ"]).all()  # wider apart in strike, larger skew


def test_flat_smile_has_zero_rr_and_bf():
    assert np.allclose(app.rr_bf(app.by_delta(app.load_options("TEST", 6))), 0, atol=1e-6)


def test_vol_view_takes_the_otm_side_whatever_the_quotes_in_the_money(monkeypatch):
    df = app.load_options("TEST", 6).copy()
    itm_call = (df.type == "Call") & (df.strike == 90)
    df.loc[itm_call, "implied_vol"] = 2.8  # a stale deep in-the-money quote
    view = app.vol_view(df, 15)
    assert set(view.strike) == {90, 100, 110} and not view.duplicated(["maturity", "strike"]).any()
    assert np.allclose(view.implied_vol, VOL)  # the 90 strike comes from the put
    assert set(view[view.strike == 90].type) == {"Put"}


def with_k_w(smiles):
    return smiles.assign(k=np.log(smiles.strike / smiles.forward), w=smiles.implied_vol**2 * smiles["T"])


def test_calendar_violations_flags_a_later_expiry_below_an_earlier_one():
    df = app.load_options("TEST", 6)
    assert app.calendar_violations(with_k_w(app.otm(df))).empty  # flat vol: w grows with T at every k
    last = (df.maturity == df.maturity.max()) & (df.strike == 120) & (df.type == "Call")
    df.loc[last, "implied_vol"] = 0.1  # 0.1^2 * 180/365 is below 0.3^2 * 90/365
    bad = app.calendar_violations(with_k_w(app.otm(df)))
    assert list(bad.strike) == [120] and bad.maturity.iloc[0] == df.maturity.max()
    assert bad.shortfall.iloc[0] == pytest.approx(0.3**2 * 90 / 365 - 0.1**2 * 180 / 365, rel=1e-3)


def test_total_variance_and_fitted_surface_figures(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.3 + 0.002 * (100 - strike))
    df = app.in_range(app.load_options("TEST", 6), 50)
    for source in ("market", "fit"):
        *lines, crosses = app.total_variance(df, source).data
        assert len(lines) == 3 and len(crosses.x) == 0
    fitted, dots = app.surface(app.vol_view(df, 50), "essvi_vol").data
    assert np.isfinite(fitted.z).all() and len(dots.z) == len(app.otm(df))
