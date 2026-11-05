"""Static arbitrage on executable prices: every trade buys at the ask and sells at the bid, so a violation is money
today with no loss at expiry, before fees, not an effect of using mids.

scan() takes two-sided quotes, one row per option, with columns type ("Call"/"Put"), maturity, strike, bid, ask,
spot, r, T, forward and div_yield, and returns one row per violation: check, maturity, strikes, profit (per unit of
underlying, after crossing the spread) and trade.

Listed US options on stocks and ETFs are American. Bounds, verticals and calendars below hold for American options
too; butterflies and boxes assume European exercise (an early assignment on the short leg can break them), and
bounds against the forward rely on the forward implied from put-call parity.

Free quotes (Yahoo) are snapshots of each contract taken at different times, and deep in-the-money ones are often
hours stale: on SPX and SPY they made most of the violations, worth thousands of dollars that no one can trade.
So scan() only checks the live part of the chain: out-of-the-money options and strikes within LIVE_BAND of the
forward, and boxes only within LIVE_BAND. Even there a violation is a quote to check, not a guaranteed trade."""

import numpy as np
import pandas as pd

from src.option_vol import black_scholes as bs

COLUMNS = ["check", "maturity", "strikes", "profit", "trade"]
LIVE_BAND = 0.02  # strikes within 2% of the forward count as live on both sides


def scan(quotes: pd.DataFrame) -> pd.DataFrame:
    quotes = quotes[(quotes.bid > 0) & (quotes.ask > 0)].sort_values(["maturity", "type", "strike"])
    otm = ((quotes.type == "Put") & (quotes.strike < quotes.forward)) | (
        (quotes.type == "Call") & (quotes.strike >= quotes.forward)
    )
    near = (quotes.strike / quotes.forward - 1).abs() <= LIVE_BAND
    live = quotes[otm | near]
    found = [bounds(live), verticals(live), butterflies(live), boxes(quotes[near]), calendars(quotes)]
    return pd.concat([f for f in found if not f.empty] or [pd.DataFrame(columns=COLUMNS)], ignore_index=True)


def _rows(check, frame, strikes, profit, trade) -> pd.DataFrame:
    hit = np.asarray(profit > 1e-9)
    return pd.DataFrame(
        {
            "check": check,
            "maturity": np.asarray(frame.maturity)[hit],
            "strikes": np.asarray(strikes)[hit],
            "profit": np.asarray(profit)[hit],
            "trade": trade,
        },
        columns=COLUMNS,
    )


def bounds(q: pd.DataFrame) -> pd.DataFrame:
    discount = np.exp(-q.r * q["T"])
    call = q.type == "Call"
    intrinsic = discount * np.where(call, (q.forward - q.strike).clip(lower=0), (q.strike - q.forward).clip(lower=0))
    ceiling = np.where(call, q.spot, q.strike)  # an American call is worth at most the stock, a put its strike
    strikes = q.strike.map("{:g}".format)
    return pd.concat(
        [
            _rows("lower bound", q, strikes, intrinsic - q.ask, "buy the option below its discounted intrinsic value"),
            _rows("upper bound", q, strikes, q.bid - ceiling, "sell the option above spot (call) or strike (put)"),
        ]
    )


def _neighbours(q: pd.DataFrame, n: int):
    """Groups of n consecutive strikes of one expiry and type, as lists of aligned Series"""
    groups = q.groupby(["maturity", "type"], sort=False)
    return [groups.shift(-i).assign(maturity=q.maturity, type=q.type) if i else q for i in range(n)]


def verticals(q: pd.DataFrame) -> pd.DataFrame:
    low, high = _neighbours(q, 2)  # low strike, next strike up
    call = q.type == "Call"
    gap = high.strike - low.strike
    # the option that should be dearer: the low strike for calls, the high one for puts
    rich_ask, cheap_bid = np.where(call, low.ask, high.ask), np.where(call, high.bid, low.bid)
    rich_bid, cheap_ask = np.where(call, low.bid, high.bid), np.where(call, high.ask, low.ask)
    strikes = low.strike.map("{:g}".format) + "/" + high.strike.map("{:g}".format)
    return pd.concat(
        [
            _rows("vertical", q, strikes, cheap_bid - rich_ask, "buy the dearer strike for less than the cheaper one"),
            _rows("vertical", q, strikes, rich_bid - cheap_ask - gap, "sell a spread above the strike gap"),
        ]
    )


def butterflies(q: pd.DataFrame) -> pd.DataFrame:
    low, mid, high = _neighbours(q, 3)
    weight = (high.strike - mid.strike) / (high.strike - low.strike)  # unequal gaps: the middle strike's weights
    profit = mid.bid - weight * low.ask - (1 - weight) * high.ask
    strikes = low.strike.map("{:g}".format) + "/" + mid.strike.map("{:g}".format) + "/" + high.strike.map("{:g}".format)
    return _rows("butterfly", q, strikes, profit, "buy the wings, sell the body: negative price for a payoff >= 0")


def boxes(q: pd.DataFrame) -> pd.DataFrame:
    """Each strike with a call and a put brackets the forward: a conversion (buy put, sell call) locks in
    F >= K + (C_bid - P_ask) / D, a reversal F <= K + (C_ask - P_bid) / D. Two strikes whose brackets don't overlap
    make a box: per expiry, the widest such pair."""
    calls = q[q.type == "Call"].set_index(["maturity", "strike"])
    puts = q[q.type == "Put"].set_index(["maturity", "strike"])
    pairs = calls.join(puts[["bid", "ask"]].add_suffix("_put"), how="inner").reset_index()
    if pairs.empty:
        return pd.DataFrame(columns=COLUMNS)
    discount = np.exp(-pairs.r * pairs["T"])
    pairs["low"] = pairs.strike + (pairs.bid - pairs.ask_put) / discount
    pairs["high"] = pairs.strike + (pairs.ask - pairs.bid_put) / discount
    rows = []
    for maturity, p in pairs.groupby("maturity"):
        a, b = p.low.idxmax(), p.high.idxmin()
        rows.append(
            dict(
                check="box",
                maturity=maturity,
                strikes=f"{p.strike[a]:g}/{p.strike[b]:g}",
                profit=discount[a] * (p.low[a] - p.high[b]),
                trade="buy the put and sell the call at the first strike, the reverse at the second",
            )
        )
    found = pd.DataFrame(rows, columns=COLUMNS)
    return found[found.profit > 1e-9]


def calendars(q: pd.DataFrame) -> pd.DataFrame:
    """At the same log-moneyness k = ln(K/F), total variance vol^2 T must not fall with maturity. Violation: a later
    expiry's ask, as total variance, below an earlier expiry's bid at that k (interpolated between its strikes, never
    extrapolated). Out-of-the-money side only. Profit: the later option's price at the earlier's bid variance minus
    its ask, the value of buying it."""
    otm = q[((q.type == "Put") & (q.strike < q.forward)) | ((q.type == "Call") & (q.strike >= q.forward))]
    otm = otm.sort_values(["maturity", "strike"])
    market = dict(S=otm.spot, K=otm.strike, T=otm["T"], r=otm.r, is_call=otm.type == "Call", q=otm.div_yield)
    otm = otm.assign(
        k=np.log(otm.strike / otm.forward),
        w_bid=bs.implied_vol(otm.bid, **market) ** 2 * otm["T"],
        w_ask=bs.implied_vol(otm.ask, **market) ** 2 * otm["T"],
    )
    slices = [s for _, s in otm.groupby("maturity")]
    found = []
    for j, later in enumerate(slices):
        floor = np.full(len(later), -np.inf)
        for earlier in slices[:j]:  # k rises with strike within an expiry, as np.interp needs
            bid_w = earlier.dropna(subset=["w_bid"])
            floor = np.fmax(floor, np.interp(later.k, bid_w.k, bid_w.w_bid, left=np.nan, right=np.nan))
        below = later.w_ask.to_numpy() < floor
        if below.any():
            hit = later[below]
            fair = bs.price(
                hit.spot,
                hit.strike,
                hit["T"],
                hit.r,
                np.sqrt(floor[below] / hit["T"]),
                hit.type == "Call",
                hit.div_yield,
            )
            found.append(
                pd.DataFrame(
                    {
                        "check": "calendar",
                        "maturity": hit.maturity,
                        "strikes": hit.strike.map("{:g}".format),
                        "profit": fair - hit.ask,
                        "trade": "buy the later expiry, sell the earlier one at the same moneyness",
                        "k": hit.k,
                        "w": hit.w_ask,
                    }
                )
            )
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=[*COLUMNS, "k", "w"])
