"""Positions of listed calls, puts and the underlying: legs priced from a chain, valued at a horizon, with greeks.
A leg is a dict: side ("Buy" / "Sell"), quantity (contracts, or shares for "Stock"), instrument ("Call", "Put",
"Stock"), expiry (ISO date) and strike. Values are in dollars."""

from datetime import date

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.option_vol import black_scholes as bs
from src.option_vol import ssvi

MULTIPLIER = 100  # shares per listed US option contract
GREEKS = ["delta", "gamma", "vega", "theta"]
# (side, quantity, instrument, strike in widths from spot, expiry: 0 the chosen one, 1 about a month later)
PRESETS = {
    "Long call": [("Buy", 1, "Call", 0, 0)],
    "Long put": [("Buy", 1, "Put", 0, 0)],
    "Covered call": [("Buy", 100, "Stock", 0, 0), ("Sell", 1, "Call", 1, 0)],
    "Cash-secured put": [("Sell", 1, "Put", -1, 0)],
    "Protective put": [("Buy", 100, "Stock", 0, 0), ("Buy", 1, "Put", -1, 0)],
    "Collar": [("Buy", 100, "Stock", 0, 0), ("Buy", 1, "Put", -1, 0), ("Sell", 1, "Call", 1, 0)],
    "Conversion": [("Buy", 100, "Stock", 0, 0), ("Buy", 1, "Put", 0, 0), ("Sell", 1, "Call", 0, 0)],
    "Bull call spread": [("Buy", 1, "Call", 0, 0), ("Sell", 1, "Call", 1, 0)],
    "Bear put spread": [("Buy", 1, "Put", 0, 0), ("Sell", 1, "Put", -1, 0)],
    "Risk reversal": [("Sell", 1, "Put", -1, 0), ("Buy", 1, "Call", 1, 0)],
    "Straddle": [("Buy", 1, "Call", 0, 0), ("Buy", 1, "Put", 0, 0)],
    "Strangle": [("Buy", 1, "Put", -1, 0), ("Buy", 1, "Call", 1, 0)],
    "Call butterfly": [("Buy", 1, "Call", -1, 0), ("Sell", 2, "Call", 0, 0), ("Buy", 1, "Call", 1, 0)],
    "Iron condor": [
        ("Buy", 1, "Put", -2, 0),
        ("Sell", 1, "Put", -1, 0),
        ("Sell", 1, "Call", 1, 0),
        ("Buy", 1, "Call", 2, 0),
    ],
    "Call calendar": [("Sell", 1, "Call", 0, 0), ("Buy", 1, "Call", 0, 1)],
}


def build(preset: str, chain: pd.DataFrame, expiry, width: float) -> list[dict]:
    """The preset's legs on chain at expiry, strikes width (fraction of spot) apart, snapped to listed strikes"""
    dates = sorted(chain.maturity.unique())
    expiry = min(dates, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(expiry)))
    later = [d for d in dates if pd.Timestamp(d) >= pd.Timestamp(expiry) + pd.Timedelta(days=25)]
    expiries = [expiry, later[0] if later else dates[-1]]
    spot = chain.spot.iloc[0]
    legs = [
        dict(
            side=side, quantity=quantity, instrument=instrument, expiry=str(expiries[e]), strike=spot * (1 + n * width)
        )
        for side, quantity, instrument, n, e in PRESETS[preset]
    ]
    return price(legs, chain).pipe(as_rows)


def price(legs: list[dict], chain: pd.DataFrame) -> pd.DataFrame:
    """Legs matched to the chain: expiry and strike snapped to the nearest listed, with the executable fill (ask to
    buy, bid to sell, the last price without a two-sided quote), the mid, the vol and per-unit greeks. units is
    the signed number of shares the leg stands for."""
    spot = chain.spot.iloc[0]
    rows = []
    for leg in legs:
        sign = 1 if leg["side"] == "Buy" else -1
        quantity = abs(float(leg.get("quantity") or 0))
        if leg["instrument"] == "Stock":
            rows.append(
                dict(leg, expiry=None, strike=None, units=sign * quantity, fill=spot, mid=spot, vol=np.nan, T=0.0)
                | dict(delta=1.0, gamma=0.0, vega=0.0, theta=0.0, r=0.0, q=0.0)
            )
            continue
        dates = chain.maturity.unique()
        wanted = pd.Timestamp(leg.get("expiry") or min(dates))
        expiry = min(dates, key=lambda d: abs(pd.Timestamp(d) - wanted))
        listed = chain[(chain.maturity == expiry) & (chain.type == leg["instrument"])]
        quote = listed.iloc[(listed.strike - float(leg.get("strike") or spot)).abs().argmin()]
        two_sided = quote.bid > 0 and quote.ask > 0
        fill = (quote.ask if sign > 0 else quote.bid) if two_sided else quote.price
        rows.append(
            dict(leg, expiry=str(expiry), strike=float(quote.strike), units=sign * quantity * MULTIPLIER)
            | dict(fill=fill, mid=quote.price, vol=_vol(quote, chain), T=quote["T"], r=quote.r, q=quote.div_yield)
            | {g: quote[g] for g in GREEKS}
        )
    priced = pd.DataFrame(rows)
    return priced.assign(position_delta=priced.units * priced.delta)


def _vol(quote: pd.Series, chain: pd.DataFrame) -> float:
    """The quote's implied vol; without one (a price below intrinsic) the other side's at that strike, by put-call
    parity, else the expiry's median vol"""
    if pd.notna(quote.implied_vol):
        return float(quote.implied_vol)
    expiry = chain[chain.maturity == quote.maturity]
    other = expiry[(expiry.strike == quote.strike) & (expiry.type != quote.type)].implied_vol.dropna()
    return float(other.iloc[0] if len(other) else expiry.implied_vol.median())


def as_rows(priced: pd.DataFrame) -> list[dict]:
    """Priced legs as rows of the legs table, numbered"""
    return [
        {k: (None if pd.isna(v) else v) for k, v in row.items()} | {"id": i}
        for i, row in enumerate(priced.drop(columns=["id"], errors="ignore").to_dict("records"))
    ]


def cost(priced: pd.DataFrame, at: str = "fill") -> float:
    """Cash paid to open the position (negative: received), at the fills or at the mids"""
    return float((priced.units * priced[at]).sum())


def value(
    priced: pd.DataFrame, spots, elapsed: float = 0.0, vol_shift: float = 0.0, fitted: pd.DataFrame | None = None
) -> np.ndarray:
    """Position value at each spot after `elapsed` trading years: expired options at intrinsic, live ones by
    Black-Scholes at their own vol plus vol_shift (sticky strike), the stock at spot. With fitted (eSSVI theta, rho,
    psi by maturity) an option of a fitted expiry takes the fit's vol at its moneyness ln(K / F) instead, F moving
    with spot (sticky moneyness)."""
    spots = np.asarray(spots, dtype=float)
    total = np.zeros_like(spots)
    for leg in priced.itertuples():
        if leg.instrument == "Stock":
            total += leg.units * spots
            continue
        left = leg.T - elapsed
        is_call = leg.instrument == "Call"
        if left <= 1e-9:
            payoff = np.maximum(spots - leg.strike, 0) if is_call else np.maximum(leg.strike - spots, 0)
        else:
            vol = np.maximum(_vol_at(leg, spots, fitted) + vol_shift, 1e-4)
            payoff = bs.price(spots, leg.strike, left, leg.r, vol, is_call, leg.q)
        total += leg.units * payoff
    return total


def _vol_at(leg, spots: np.ndarray, fitted: pd.DataFrame | None):
    expiry = date.fromisoformat(leg.expiry)
    if fitted is None or expiry not in fitted.index:
        return leg.vol
    # shortcut: vols keep their expiry's level as time passes, no roll down the term structure; fine for short horizons
    forward = spots * np.exp((leg.r - leg.q) * leg.T)
    return ssvi.vol(np.log(leg.strike / forward), leg.T, *fitted.loc[expiry])


def greeks(priced: pd.DataFrame) -> dict:
    """Position greeks: delta in shares, gamma in shares per $1, vega in $ per vol point, theta in $ per day"""
    totals = {g: float((priced.units * priced[g]).sum()) for g in GREEKS}
    return totals | {"vega": totals["vega"] / 100}


def payoff_stats(spots: np.ndarray, pnl: np.ndarray) -> dict:
    """Max profit and loss over spots (None when still growing at the top of the grid: unbounded), and breakevens"""
    rising = pnl[-1] - pnl[-2]
    tol = 1e-6 * max(1.0, np.abs(pnl).max())
    crossings = np.flatnonzero(np.sign(pnl[:-1]) * np.sign(pnl[1:]) < 0)
    breakevens = [float(spots[i] - pnl[i] * (spots[i + 1] - spots[i]) / (pnl[i + 1] - pnl[i])) for i in crossings]
    return dict(
        max_profit=None if rising > tol else float(pnl.max()),
        max_loss=None if rising < -tol else float(pnl.min()),
        breakevens=breakevens,
    )


def chance_of_profit(spots: np.ndarray, pnl: np.ndarray, spot: float, t: float, vol: float, drift: float) -> float:
    """Probability that pnl > 0 at time t when log spot is normal with the given vol and drift (r - q): each grid
    cell between spot midpoints weighs its lognormal probability"""
    if t <= 0 or not vol > 0:
        return float(pnl[np.abs(spots - spot).argmin()] > 0)
    edges = np.concatenate([[0], (spots[1:] + spots[:-1]) / 2, [np.inf]])
    with np.errstate(divide="ignore"):
        z = (np.log(edges / spot) - (drift - vol**2 / 2) * t) / (vol * np.sqrt(t))
    return float(np.diff(ndtr(z))[pnl > 0].sum())
