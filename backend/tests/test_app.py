"""Dashboard data path: pricing a chain, filtering it and building the surface, with Yahoo replaced by a fake"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

import app
from src.option_vol import strategy
from src.option_vol.models import Call, Environment, Put
from tests.test_pricing import make, market  # noqa: F401  (autouse fixture: spot 100, 5% rate)

VOL = 0.3


def vol_of(strike):  # flat by default; tests swap it for a skew
    return VOL + 0 * strike


DIV_YIELD = 0.0  # tests swap it to check the implied yield
STALE = set()  # (type, strike, days) quoted by a last trade 20% off instead of a two-sided quote
OFF = set()  # (type, strike, days) with a two-sided quote 20% off: a stale snapshot that breaks put-call parity


@pytest.fixture(autouse=True)
def fake_chain(monkeypatch):
    """Calls and puts at strikes 50..150 and 3 expiries, priced at vol_of(strike)"""

    def chain(underlying, expiry_date):
        monkeypatch.setitem(Environment.spots, "TEST", 100.0)  # load_options dropped it to force a fresh spot
        options = [make(cls, k, days) for cls in (Call, Put) for k in range(50, 151, 10) for days in (30, 90, 180)]
        for o in options:
            o.div_yield = globals()["DIV_YIELD"]
            o.price = o.price_with_black_scholes(globals()["vol_of"](o.strike))
            if (o.type, o.strike, round(o.T * 365)) in globals()["STALE"]:
                o.price *= 1.2
            else:
                o.price *= 1.2 if (o.type, o.strike, round(o.T * 365)) in globals()["OFF"] else 1
                o.bid, o.ask = o.price * 0.99, o.price * 1.01
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


def test_total_variance_and_fitted_surface_figures(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.3 + 0.002 * (100 - strike))
    df = app.in_range(app.load_options("TEST", 6), 50)
    for source in ("market", "fit"):
        *lines, crosses = app.total_variance(df, source).data
        assert len(lines) == 3 and len(crosses.x) == 0
    fitted, dots = app.surface(app.vol_view(df, 50), "essvi_vol").data
    assert np.isfinite(fitted.z).all() and len(dots.z) == len(app.otm(df))


def test_usable_quotes_are_two_sided_and_tight():
    quotes = pd.DataFrame({"bid": [1.0, 0.01, np.nan], "ask": [1.2, 0.03, np.nan], "price": [1.1, 0.02, 0.5]})
    assert list(app.usable(quotes)) == [True, False, False]  # spread 18% of mid, 100%, one-sided
    closed = quotes.assign(bid=np.nan, ask=np.nan)
    assert app.usable(closed).all()  # nothing two-sided: market closed, fall back on last trades


def test_a_stale_last_trade_moves_neither_forward_nor_fit(monkeypatch):
    monkeypatch.setitem(globals(), "STALE", {("Call", 100, 30), ("Call", 110, 30)})
    df = app.load_options("TEST", 6)
    assert np.allclose(df.div_yield, 0.0)  # 2 of the 3 strikes nearest spot: the median would not hide them
    assert df.usable.sum() == len(df) - 2
    front = app.expiries(df)[0]
    traces = app.smile(app.in_range(df, 50), front, 100.0).data
    excluded = next(t for t in traces if t.name == "Excluded quotes")
    assert list(excluded.x) == [110.0]  # the call at 100 is in the money (forward 100.4), off the smile
    fit = next(t for t in traces if t.name.startswith("eSSVI"))
    assert np.allclose(fit.y, VOL, atol=1e-3)


def test_a_vol_spike_shows_as_arbitrage_in_the_table_and_on_the_smile(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.9 if strike == 110 else 0.3)  # 110 call dearer than 100
    rows, title = app.update_arbitrage({"ticker": "TEST", "months": 6, "range": 50})
    front = str(app.expiries(app.load_options("TEST", 6))[0])
    assert {"check": "vertical", "maturity": front, "strikes": "100/110"}.items() <= next(
        r for r in rows if r["strikes"] == "100/110"
    ).items()
    assert f"{len(rows)} to check" in title and all(r["dollars"] > 0 for r in rows)
    df = app.load_options("TEST", 6)
    marks = next(t for t in app.smile(df, app.expiries(df)[0], 100.0).data if t.name == "In an arbitrage")
    assert 110.0 in marks.x


def test_one_fit_per_snapshot(monkeypatch):
    monkeypatch.setattr(app, "_fits", {})
    fits = []
    fit = app.ssvi_fit
    monkeypatch.setattr(app, "ssvi_fit", lambda df: fits.append(1) or fit(df))
    first = app.fit_for("TEST", 6, 50)
    assert app.fit_for("TEST", 6, 50) is first and len(fits) == 1
    app.load_options("TEST", 6, refresh=True)  # new snapshot: refit, and the old fit goes
    app.fit_for("TEST", 6, 50)
    assert len(fits) == 2 and len(app._fits) == 1


def test_fitted_heatmap_by_delta_matches_the_market_on_a_skew(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.3 + 0.002 * (100 - strike))
    df = app.load_options("TEST", 6)
    fitted = app.fit_for("TEST", 6, 20)
    quoted, fit = app.by_delta(df), app.by_delta(app.fitted_smiles(df, fitted))
    inner = ["25Δ put", "40Δ put", app.ATM, "40Δ call", "25Δ call"]
    assert np.abs(fit.loc[inner] - quoted.loc[inner]).max().max() < 0.01


def test_residuals_and_bid_ask_bars():
    df = app.load_options("TEST", 6)
    found = app.residuals(df, app.ssvi_fit(df))
    assert len(found) == len(app.otm(df)) and found.error.abs().max() < 1e-3  # flat 30% is an SSVI smile
    assert found.spreads.abs().is_monotonic_decreasing
    df.loc[app.otm(df).index[0], "ask"] = df.loc[app.otm(df).index[0], "bid"]  # bid = ask: no ratio, sorted last
    assert np.isnan(app.residuals(df, app.ssvi_fit(df)).spreads.iloc[-1])
    puts = app.smile(df, app.expiries(df)[0], 100.0).data[0]
    assert (np.asarray(puts.error_y.array) > 0).all() and (np.asarray(puts.error_y.arrayminus) > 0).all()


def test_term_structure_at_a_strike(monkeypatch):
    monkeypatch.setitem(globals(), "vol_of", lambda strike: 0.3 + 0.002 * (100 - strike))
    df = app.load_options("TEST", 6)
    puts, calls, fit = app.term_structure(df, 90.0, app.ssvi_fit(df)).data
    assert len(puts.x) == 3 and len(calls.x) == 0  # 90 is below every forward: the put side, all 3 expiries
    assert np.allclose(puts.y, 0.32)
    assert len(fit.x) == 3 and np.abs(np.asarray(fit.y) - 0.32).max() < 0.01
    options, strike = app.update_strikes({"ticker": "TEST", "months": 6, "range": 50}, None)
    assert strike == 100.0 and len(options) == 11  # nearest spot by default; strikes 50..150
    assert app.update_strikes({"ticker": "TEST", "months": 6, "range": 50}, 120.0)[1] == 120.0  # the choice is kept
    weekly = df.iloc[[0]].assign(strike=100.5)  # an odd strike nearer spot, on one expiry only
    assert app.default_strike(pd.concat([df, weekly])) == 100.0


def test_a_surface_click_picks_the_nearest_expiry_and_listed_strike():
    df = app.load_options("TEST", 6)
    dates = app.expiries(df)
    clicked = (dates[1] + timedelta(days=3)).isoformat() + " 00:00"  # a grid point between expiries
    assert app.surface_pick(df, 50, clicked, 103.7) == (1, 100.0)
    assert app.surface_pick(df, 15, dates[2].isoformat(), 140.0) == (2, 110.0)  # 140 is beyond the range
    weekly = df[df.maturity == dates[0]].iloc[[0]].assign(strike=104.0)  # listed on the first expiry only
    assert app.surface_pick(pd.concat([df, weekly]), 50, dates[1].isoformat(), 103.7) == (1, 100.0)
    assert app.surface_pick(pd.concat([df, weekly]), 50, dates[0].isoformat(), 103.7) == (0, 104.0)
    fig = app.surface(app.vol_view(df, 50), "implied_vol")
    assert "Click" in fig.layout.title.subtitle.text


def legs_of(preset, width=0.1):
    chain = app.load_options("TEST", 6)
    expiry = min(chain.maturity)
    return strategy.price(strategy.build(preset, chain, expiry, width), chain)


def test_strategy_legs_snap_and_fill_at_bid_ask():
    priced = legs_of("Covered call")
    stock, call = priced.iloc[0], priced.iloc[1]
    assert (stock.units, stock.fill) == (100, 100.0)
    assert (call.strike, call.units, call.instrument) == (110.0, -100, "Call")
    assert call.fill == pytest.approx(call.mid * 0.99)  # sold at the bid
    assert strategy.greeks(priced)["delta"] == pytest.approx(100 - 100 * call.delta)


def test_strategy_today_is_worth_the_mids():
    priced = legs_of("Iron condor")
    assert strategy.value(priced, [100.0])[0] == pytest.approx(strategy.cost(priced, "mid"))


def test_covered_call_at_expiry():
    priced = legs_of("Covered call")
    spots = np.linspace(0.1, 400, 4000)
    pnl = strategy.value(priced, spots, priced["T"].max()) - strategy.cost(priced)
    stats = strategy.payoff_stats(spots, pnl)
    premium = priced.fill.iloc[1] * 100
    assert stats["max_profit"] == pytest.approx(10 * 100 + premium)
    assert stats["max_loss"] == pytest.approx(-(100 - 0.1) * 100 + premium, rel=1e-3)
    assert stats["breakevens"] == [pytest.approx(100 - premium / 100, abs=0.01)]


def test_straddle_profit_is_unlimited_and_conversion_is_flat():
    straddle = legs_of("Straddle")
    spots = np.linspace(1, 400, 4000)
    expired = strategy.value(straddle, spots, straddle["T"].max()) - strategy.cost(straddle)
    assert strategy.payoff_stats(spots, expired)["max_profit"] is None
    conversion = legs_of("Conversion")
    flat = strategy.value(conversion, spots, conversion["T"].max())
    assert np.ptp(flat) == pytest.approx(0, abs=1e-6)  # stock + put - call pays the strike whatever the spot


def test_calendar_takes_a_later_expiry_and_values_its_far_leg():
    priced = legs_of("Call calendar")
    near, far = priced.iloc[0], priced.iloc[1]
    assert far.expiry > near.expiry and far.strike == near.strike
    at_near = strategy.value(priced, [100.0], near["T"])[0]
    assert at_near > 0  # the long far call still has time value


def test_chance_of_profit_of_long_stock():
    spots = np.linspace(1, 400, 8000)
    chance = strategy.chance_of_profit(spots, spots - 100, 100, 1.0, 0.3, 0.0)
    assert chance == pytest.approx(ndtr(-0.15), abs=2e-3)  # P(S_1 > S_0) = N(-vol/2) without drift


def test_a_pair_off_parity_is_dropped_and_not_flagged_as_arbitrage(monkeypatch):
    monkeypatch.setitem(globals(), "OFF", {("Call", 100, 30)})
    df = app.load_options("TEST", 6)
    front = df[df.maturity == df.maturity.min()]
    dropped = front[~front.usable]
    assert set(zip(dropped.type, dropped.strike)) == {("Call", 100.0), ("Put", 100.0)}
    assert front.div_yield.iloc[0] == pytest.approx(0, abs=1e-3)
    rows, _ = app.update_arbitrage({"ticker": "TEST", "months": 6, "range": 50})
    assert not [r for r in rows if "100" in r["strikes"].split("/")]


def test_parity_by_expiry(monkeypatch):
    monkeypatch.setitem(globals(), "OFF", {("Call", 100, 30)})
    df = app.load_options("TEST", 6)
    table = app.parity_by_expiry(df, dividend_yield=0.01, band=0.15)  # 5% holds only the stale strike 100 here
    assert table.implied_yield.to_numpy() == pytest.approx(0, abs=1e-3)
    assert (table.low_yield <= table.implied_yield + 1e-9).all() and (
        table.implied_yield <= table.high_yield + 1e-9
    ).all()
    assert table.dividends.to_numpy() == pytest.approx(0, abs=0.05)
    assert table.borrow.to_numpy() == pytest.approx(-0.01, abs=1e-3)
    assert table.stale.tolist() == [1, 0, 0] and (table.pairs == 11).all()
    monkeypatch.setitem(Environment.div_yield, "TEST", 0.01)  # no Yahoo
    _figure, _rows, stale, title = app.update_parity({"ticker": "TEST", "months": 6, "range": 50})
    assert [(r["strike"], round(r["miss"], 2) > 0) for r in stale] == [(100.0, True)] and "1 of 33" in title


def test_strategy_on_essvi_vols_matches_own_vols_on_a_flat_smile():
    priced = legs_of("Iron condor")
    fitted = app.fit_for("TEST", 6, 50)
    own, fit = strategy.value(priced, [90.0, 100.0, 110.0]), strategy.value(priced, [90.0, 100.0, 110.0], fitted=fitted)
    assert fit == pytest.approx(own, abs=0.5)  # $ on a position of 4 contracts
