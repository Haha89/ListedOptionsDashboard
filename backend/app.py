"""Dash dashboard of listed options: greek surface, smile per maturity, implied vol heatmap and a table.
Run with `uv run python app.py` and open http://127.0.0.1:8050 (HOST, PORT and DASH_DEBUG env vars apply)."""

import math
import time
from datetime import date

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dateutil.relativedelta import relativedelta
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from src.option_vol import arbitrage, ssvi, strategy
from src.option_vol import black_scholes as bs
from src.option_vol.models import Environment
from src.option_vol.scrapping import Scrapping
from src.option_vol.utils import to_title

FIELDS = ["implied_vol", "price", "delta", "gamma", "vega", "theta", "rho"]
COLUMNS = ["name", "strike", "maturity", *FIELDS, "bid", "ask", "volume", "open_interest", "usable"]
DELTAS = (5, 10, 25, 40)
TABLE_FORMATS = {"implied_vol": ".2%", "price": ".2f", "strike": ".2f", "bid": ".2f", "ask": ".2f"} | dict.fromkeys(
    FIELDS[2:], ".4f"
)
PERCENT = {"function": "params.value == null ? '' : d3.format('.2%')(params.value)"}
DOLLARS = {"function": "params.value == null ? '' : d3.format(',.2f')(params.value)"}
WIDTHS = (0.025, 0.05, 0.1, 0.2)  # strategy strikes apart, fraction of spot
MAX_SPREAD = 0.5  # widest usable quote, (ask - bid) / mid
PARITY_TOL = 0.0025  # a call/put pair whose bid/ask forward misses the expiry's by more is stale, fraction of F
PARITY_PAIRS = 9  # pairs nearest spot whose median forward is the expiry's
ATM = "ATM (fwd)"
ESSVI = "Implied vol, eSSVI fit"
SOURCES = [{"label": "Market", "value": "market"}, {"label": "eSSVI fit", "value": "fit"}]
VOL_MODES = [{"label": "Own vols", "value": "own"}, {"label": "eSSVI vols", "value": "fit"}]
TICKERS = {  # symbols as Scrapping takes them; indices go through YAHOO_MAPPING
    "SPX": "S&P 500 index",
    "SPY": "S&P 500 ETF",
    "NDX": "Nasdaq-100 index",
    "QQQ": "Nasdaq-100 ETF",
    "IWM": "Russell 2000 ETF",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "TSLA": "Tesla",
    "AVGO": "Broadcom",
    "AMD": "AMD",
    "NFLX": "Netflix",
    "JPM": "JPMorgan",
    "XOM": "Exxon Mobil",
    "KO": "Coca-Cola",
}
CACHE_SECONDS = 15 * 60
MULTIPLIER = 100  # underlying units per listed US option contract
MARGIN = dict(l=50, r=10, b=40, t=50)
INK, MUTED, RULE, PUT, CALL, ERROR = "#18212e", "#5d6878", "#dce1e7", "#2f6db5", "#c7870f", "#b3261e"
FONT = "IBM Plex Sans, system-ui, sans-serif"
GRID_THEME = {  # AG Grid's Balham theme in the page's palette; built in, unlike dag.themes.BALHAM, a CDN stylesheet
    "function": f"themeBalham.withParams({{fontFamily: '{FONT}', foregroundColor: '{INK}', accentColor: '{PUT}', "
    f"borderColor: '{RULE}', headerBackgroundColor: '#f3f5f7'}})"
}
VOL_SCALE = "Cividis"  # perceptually uniform and colorblind-safe, unlike a rainbow scale
pio.templates["desk"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, color=INK, size=12),
        title=dict(font=dict(size=15), x=0, xref="paper"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[PUT, CALL, INK],
        xaxis=dict(gridcolor=RULE, linecolor=RULE, zeroline=False),
        yaxis=dict(gridcolor=RULE, linecolor=RULE, zeroline=False),
        hoverlabel=dict(font_family=FONT),
    )
)
pio.templates.default = "plotly_white+desk"
# shortcut: per-process, never evicted; fine for one user, use flask-caching to share between gunicorn workers
_cache: dict = {}
_fits: dict = {}  # eSSVI fits by (ticker, months, strike range, snapshot time)


def load_options(ticker: str, months: int, refresh=False) -> pd.DataFrame:
    """Listed options of ticker expiring within `months`, with implied vol and greeks, one row per option.
    One snapshot per ticker: a shorter max maturity filters the cached chain instead of fetching new quotes, so every
    view compares the same quotes. A longer one, Reload or a 15-minute-old snapshot fetches anew."""
    max_expiry = date.today() + relativedelta(months=months)
    hit = _cache.get(ticker)
    if refresh or hit is None or time.monotonic() - hit[0] > CACHE_SECONDS or hit[1] < max_expiry:
        Environment.spots.pop(ticker, None)  # new quotes need the spot of the same moment
        options = Scrapping.parse_underlying_until(ticker, max_expiry)
        if not options:
            raise ValueError(f"no listed options found for {ticker}")
        df = pd.DataFrame([vars(o) for o in options])
        df["usable"] = usable(df)
        for _ in range(2):  # the second pass reads the forward without the pairs the first one found stale
            df["div_yield"] = implied_yield(df)
            df["forward"] = df.spot * np.exp((df.r - df.div_yield) * df["T"])
            df["usable"] &= ~parity_breaks(df)
        market = dict(S=df.spot, K=df.strike, T=df["T"], r=df.r, is_call=df.type == "Call", q=df.div_yield)
        df["implied_vol"] = bs.implied_vol(df.price, **market)
        df["bid_vol"] = bs.implied_vol(df.bid, **market)  # nan without a bid, or a bid below intrinsic
        df["ask_vol"] = bs.implied_vol(df.ask, **market)
        df = df.assign(**bs.greeks(vol=df.implied_vol, **market))
        hit = _cache[ticker] = (time.monotonic(), max_expiry, df)
    df = hit[2]
    return df[df.maturity <= max_expiry]


def usable(df: pd.DataFrame) -> pd.Series:
    """Quotes the forward, the fit and the arbitrage checks rely on: two-sided, with a spread of at most MAX_SPREAD
    of the mid. A last trade can be hours or days old, and a minimum-tick quote (0.01 / 0.03) has a mid that says
    little about the price. When nothing is two-sided (market closed), every quote, so the dashboard still shows
    the last trades."""
    good = (df.bid > 0) & (df.ask - df.bid <= MAX_SPREAD * df.price)
    return good if good.any() else pd.Series(True, index=df.index)


def implied_yield(df: pd.DataFrame) -> pd.Series:
    """Per expiry, the continuous yield q making the model forward S e^((r-q)T) equal the market's, read from
    put-call parity F = K + (C - P) e^(rT) at the PARITY_PAIRS strikes nearest spot (median: a few stale pairs among
    them don't move it). Without it calls and puts of one strike give different vols; it absorbs dividends, borrow
    costs and a spot taken at another time than the quotes. Usable quotes only; 0 for expiries with no usable
    call/put pair at a strike."""
    quotes = df[df.usable]
    calls = quotes[quotes.type == "Call"].set_index(["maturity", "strike"])
    puts = quotes[quotes.type == "Put"].set_index(["maturity", "strike"]).price
    pairs = calls.assign(put=puts).dropna(subset=["put"])
    strikes = pairs.index.get_level_values("strike")
    pairs["forward"] = strikes + (pairs.price - pairs.put) * np.exp(pairs.r * pairs["T"])
    pairs["distance"] = abs(strikes / pairs.spot - 1)
    nearest = pairs.sort_values("distance").groupby(level="maturity").head(PARITY_PAIRS).groupby(level="maturity")
    first = nearest[["spot", "r", "T"]].first()
    q = first.r - np.log(nearest.forward.median() / first.spot) / first["T"]
    return df.maturity.map(q).fillna(0.0)


def parity_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Two-sided call/put pairs by (maturity, strike), with the forward range their bid/ask allows: a reversal (buy
    the call, sell the put, short the forward) pays if F < low, a conversion if F > high"""
    quotes = df[(df.bid > 0) & (df.ask > 0)]
    calls = quotes[quotes.type == "Call"].set_index(["maturity", "strike"])
    puts = quotes[quotes.type == "Put"].set_index(["maturity", "strike"])[["bid", "ask"]].add_suffix("_put")
    pairs = calls[["bid", "ask", "spot", "r", "T", "forward", "usable"]].join(puts, how="inner")
    strikes, growth = pairs.index.get_level_values("strike"), np.exp(pairs.r * pairs["T"])
    pairs["low"] = strikes + (pairs.bid - pairs.ask_put) * growth
    pairs["high"] = strikes + (pairs.ask - pairs.bid_put) * growth
    pairs["miss"] = np.maximum(pairs.low - pairs.forward, pairs.forward - pairs.high).clip(lower=0) / pairs.forward
    return pairs


def parity_breaks(df: pd.DataFrame) -> pd.Series:
    """Both quotes of each call/put pair whose bid/ask can't reach the expiry's forward within PARITY_TOL. On a free
    feed that is one stale quote, not a conversion to trade: calls and puts of one strike quoted vols 40 points
    apart made the $359k NDX boxes. American early exercise moves an in-the-money pair's forward by far less."""
    pairs = parity_pairs(df)
    stale = pairs.index[pairs.miss > PARITY_TOL]
    return pd.Series(pd.MultiIndex.from_frame(df[["maturity", "strike"]]).isin(stale), index=df.index)


def in_range(df: pd.DataFrame, strike_range: float) -> pd.DataFrame:
    """Options whose strike is within +/- strike_range % of spot"""
    return df[(df.strike / df.spot - 1).abs() <= strike_range / 100].sort_values(["maturity", "strike"])


def select(df: pd.DataFrame, option_type: str, strike_range: float) -> pd.DataFrame:
    near_spot = in_range(df, strike_range)
    return near_spot[near_spot.type == option_type]


def vol_view(df: pd.DataFrame, strike_range: float) -> pd.DataFrame:
    """For vol plots: one vol per strike from the out-of-the-money side, whatever the Call/Put choice. By put-call
    parity a call and a put of one strike have the same vol, and only the OTM one has live quotes: deep in-the-money
    quotes are often stale and give absurd vols."""
    return otm(in_range(df, strike_range))


def expiries(df: pd.DataFrame) -> list:
    return sorted(df.maturity.unique())


def by_strike(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """field as a strike x maturity grid, gaps between listed strikes filled linearly, never extrapolated"""
    pivot = df.pivot_table(index="strike", columns="maturity", values=field, aggfunc="first")
    return pivot.interpolate(method="index", limit_area="inside")


def otm(df: pd.DataFrame, usable_only=True) -> pd.DataFrame:
    """One implied vol per strike from the liquid side: puts below the forward, calls at and above it. Usable quotes
    only unless usable_only is False."""
    side = ((df.type == "Put") & (df.strike < df.forward)) | ((df.type == "Call") & (df.strike >= df.forward))
    keep = side & df.implied_vol.notna() & (df.usable if usable_only else True)
    return df[keep].sort_values(["maturity", "strike"])


def atm_vol(df: pd.DataFrame) -> pd.Series:
    """ATM vol per maturity: implied vol at strike = forward, interpolated linearly on the out-of-the-money smile"""
    smiles = otm(df)
    if smiles.empty:  # groupby.apply would return an empty DataFrame, not a Series
        return pd.Series(dtype=float)
    return (
        smiles.groupby("maturity")
        .apply(lambda s: np.interp(s.forward.iloc[0], s.strike, s.implied_vol, left=np.nan, right=np.nan))
        .astype(float)
    )


def fit_for(ticker: str, months: int, strike_range: float) -> pd.DataFrame:
    """ssvi_fit of the chain within strike_range, computed once per snapshot: every chart shares it"""
    df = in_range(load_options(ticker, months), strike_range)
    key = (ticker, months, strike_range, _cache[ticker][0])
    if key not in _fits:
        for old in [k for k in _fits if k[0] == ticker and k[3] != key[3]]:  # fits of an older snapshot
            del _fits[old]
        _fits[key] = ssvi_fit(df)
    return _fits[key]


def ssvi_fit(df: pd.DataFrame) -> pd.DataFrame:
    """eSSVI slices (theta, rho, psi) per maturity, fitted to the out-of-the-money vols one expiry at a time from the
    shortest, each free of calendar arbitrage against the one before. theta is the market ATM total variance, raised
    where needed so it never falls with maturity. Expiries with no ATM vol or fewer than 3 vols are left out.
    Errors are counted in half-spreads, vega x vol error / half-spread, so a tight ATM quote counts more than a wide
    wing quote; by vega alone when there are no spreads (market closed)."""
    smiles = otm(df)
    atm = atm_vol(df).dropna()
    T = smiles.groupby("maturity")["T"].first().reindex(atm.index)
    thetas = pd.Series(np.maximum.accumulate((atm**2 * T).to_numpy()), index=atm.index)
    slices, earlier = {}, None
    for maturity, theta in thetas[thetas > 0].items():
        smile_ = smiles[smiles.maturity == maturity]
        if len(smile_) < 3:
            continue
        k = np.log(smile_.strike / smile_.forward)
        half_spread = (smile_.ask - smile_.bid) / 2
        weight = (smile_.vega / half_spread) ** 2 if (half_spread > 0).all() else smile_.vega**2
        rho, psi = ssvi.fit_slice(k, smile_["T"].iloc[0], theta, smile_.implied_vol, weight, earlier)
        slices[maturity] = earlier = (theta, rho, psi)
    return pd.DataFrame.from_dict(slices, orient="index", columns=["theta", "rho", "psi"])


def by_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Implied vol at fixed deltas per maturity. Each strike's call delta is taken at its out-of-the-money vol, which
    gives one delta axis for the whole smile; an X-delta put is the strike whose call delta is e^-qT - X (put-call
    parity). Rows run from the 5-delta put through ATM (strike = forward) to the 5-delta call, i.e. by increasing
    strike; vols are interpolated linearly between listed strikes, nan where the strikes don't reach that delta."""
    puts, calls = [f"{d}Δ put" for d in DELTAS], [f"{d}Δ call" for d in reversed(DELTAS)]
    rows = puts + [ATM] + calls
    grid = pd.DataFrame(np.nan, index=rows, columns=expiries(df))
    for maturity, smile_ in otm(df).groupby("maturity"):
        call_delta = bs.greeks(
            smile_.spot, smile_.strike, smile_["T"], smile_.r, smile_.implied_vol, True, smile_.div_yield
        )["delta"].to_numpy()
        parity = np.exp(-smile_.div_yield.iloc[0] * smile_["T"].iloc[0])
        targets = np.array([parity - d / 100 for d in DELTAS] + [d / 100 for d in reversed(DELTAS)])
        # call delta falls as strike rises; np.interp needs increasing x, hence the minus signs
        grid.loc[puts + calls, maturity] = np.interp(
            -targets, -call_delta, smile_.implied_vol, left=np.nan, right=np.nan
        )
    grid.loc[ATM] = atm_vol(df)
    return grid


def fitted_vols(df: pd.DataFrame, fitted: pd.DataFrame | None = None, points=60) -> pd.DataFrame:
    """eSSVI vols on a strike x maturity grid spanning df's strikes, for the expiries the fit covers (fitted, or
    fitted to df when None)"""
    fitted = ssvi_fit(df) if fitted is None else fitted
    fitted = fitted[fitted.index.isin(df.maturity)]
    strikes = np.linspace(df.strike.min(), df.strike.max(), points)
    first = df.groupby("maturity")[["T", "forward"]].first()
    return pd.DataFrame(
        {m: ssvi.vol(np.log(strikes / first.forward[m]), first["T"][m], *fitted.loc[m]) for m in fitted.index},
        index=strikes,
    )


def fitted_smiles(df: pd.DataFrame, fitted: pd.DataFrame, points=81) -> pd.DataFrame:
    """Out-of-the-money quotes made from the eSSVI fit, on a strike grid spanning df's strikes for each expiry the fit
    covers, shaped like df so by_delta and atm_vol read them as they read the market"""
    first = df.groupby("maturity")[["T", "forward", "spot", "r", "div_yield"]].first()
    strikes = np.linspace(df.strike.min(), df.strike.max(), points)
    rows = [
        pd.DataFrame({"maturity": m, "strike": strikes, **first.loc[m]}).assign(
            implied_vol=ssvi.vol(np.log(strikes / first.forward[m]), first["T"][m], *fitted.loc[m])
        )
        for m in fitted.index
        if m in first.index
    ]
    if not rows:
        return df.iloc[:0]
    smiles = pd.concat(rows, ignore_index=True)
    return smiles.assign(type=np.where(smiles.strike < smiles.forward, "Put", "Call"), usable=True)


def surface(df: pd.DataFrame, field: str, fitted: pd.DataFrame | None = None) -> go.Figure:
    """field against strike and expiry; for essvi_vol the fitted surface with the market vols as dots"""
    is_vol = field in ("implied_vol", "essvi_vol")
    pivot = fitted_vols(df, fitted) if field == "essvi_vol" else by_strike(df, field)
    label = ESSVI if field == "essvi_vol" else to_title(field)
    value = "%{z:.2%}" if is_vol else f"%{{z:.{2 if field == 'price' else 4}f}}"
    fig = go.Figure(
        go.Surface(
            x=pd.to_datetime(pivot.columns),
            y=pivot.index,
            z=pivot.to_numpy(),
            colorscale=VOL_SCALE,
            showscale=False,
            hovertemplate=f"Strike %{{y:.2f}}<br>Expiry %{{x|%Y-%m-%d}}<br>{label} {value}<extra></extra>",
        )
    )
    if field == "essvi_vol":
        market = otm(df)
        fig.add_scatter3d(
            x=pd.to_datetime(market.maturity),
            y=market.strike,
            z=market.implied_vol,
            mode="markers",
            marker=dict(size=2.5, color=INK),
            name="Market implied vol",
            hovertemplate="Strike %{y:.2f}<br>Expiry %{x|%Y-%m-%d}<br>Market implied vol %{z:.2%}<extra></extra>",
        )
    fig.update_layout(
        title=dict(text=f"{label} surface", subtitle=dict(text="Click a point for its smile and term structure")),
        margin=dict(l=0, r=0, b=0, t=40),
        uirevision="keep",
        showlegend=False,
        scene=dict(
            camera=dict(
                eye=dict(x=1.7, y=1.7, z=1.0), center=dict(x=0, y=0, z=-0.2)
            ),  # keeps all labels in the narrow card
            aspectratio=dict(x=1, y=1, z=0.8),
            xaxis_title="Expiry",
            yaxis_title="Strike",
            zaxis=dict(title="Implied vol" if is_vol else label, tickformat=".0%" if is_vol else None),
        ),
    )
    return fig


def smile(df: pd.DataFrame, expiry: date, spot: float, fitted: pd.DataFrame | None = None) -> go.Figure:
    """Implied vol against strike for one expiry: puts below the forward, calls above, the out-of-the-money side, with
    bars from the bid vol to the ask vol, and the eSSVI curve (fitted, or fitted to df when None)"""
    chain = otm(df[df.maturity == expiry])
    fig = go.Figure(
        [
            go.Scatter(
                x=side.strike,
                y=side.implied_vol,
                mode="markers",
                name=f"OTM {kind.lower()}s",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(side.ask_vol - side.implied_vol).fillna(0),
                    arrayminus=(side.implied_vol - side.bid_vol).fillna(0),
                    thickness=1,
                    width=0,
                ),
                customdata=np.column_stack([side.bid_vol, side.ask_vol]),
                hovertemplate="Strike %{x:.2f}<br>Implied vol %{y:.2%}<br>bid %{customdata[0]:.2%}, "
                "ask %{customdata[1]:.2%}<extra></extra>",
            )
            for kind in ("Put", "Call")
            for side in [chain[chain.type == kind]]
        ]
    )
    excluded = otm(df[(df.maturity == expiry) & ~df.usable], usable_only=False)
    if not excluded.empty:
        fig.add_scatter(
            x=excluded.strike,
            y=excluded.implied_vol,
            mode="markers",
            name="Excluded quotes",
            marker=dict(symbol="circle-open", color=MUTED, size=7),
            customdata=np.column_stack([excluded.bid, excluded.ask]),
            hovertemplate="Excluded: one-sided or spread over half the mid<br>Strike %{x:.2f}<br>"
            "Implied vol %{y:.2%}<br>Bid %{customdata[0]:.2f}, ask %{customdata[1]:.2f}<extra></extra>",
        )
    found = arbitrage.scan(df[df.maturity == expiry])
    if not found.empty:
        strikes = found.assign(strike=found.strikes.str.split("/")).explode("strike").astype({"strike": float})
        marked = (
            chain.merge(strikes[["strike", "check"]], on="strike")
            .groupby("strike", as_index=False)
            .agg(implied_vol=("implied_vol", "first"), check=("check", lambda c: ", ".join(sorted(set(c)))))
        )
        fig.add_scatter(
            x=marked.strike,
            y=marked.implied_vol,
            mode="markers",
            name="In an arbitrage",
            marker=dict(symbol="x", size=10, color=ERROR),
            customdata=marked[["check"]],
            hovertemplate="Strike %{x:.2f} is in a %{customdata[0]} arbitrage<extra></extra>",
        )
    fig.add_vline(x=spot, line_dash="dot", line_color=MUTED, annotation_text=f"spot {spot:.2f}")
    atm = atm_vol(chain)
    if atm.notna().any():
        forward = chain.forward.iloc[0]
        fig.add_scatter(
            x=[forward],
            y=atm.to_numpy(),
            mode="markers+text",
            name=f"ATM (forward {forward:.2f})",
            text=[f"ATM {atm.iloc[0]:.2%}"],
            textposition="top center",
            marker=dict(size=11, symbol="diamond", color=INK),
            hovertemplate="ATM, strike = forward %{x:.2f}<br>Implied vol %{y:.2%}<extra></extra>",
        )
    fitted = ssvi_fit(df) if fitted is None else fitted
    if expiry in fitted.index and not chain.empty:
        theta, rho, psi = fitted.loc[expiry]
        strikes = np.linspace(chain.strike.min(), chain.strike.max(), 200)
        T, forward = chain["T"].iloc[0], chain.forward.iloc[0]
        curve = ssvi.vol(np.log(strikes / forward), T, theta, rho, psi)
        errors = ssvi.vol(np.log(chain.strike / forward), T, theta, rho, psi) - chain.implied_vol
        fig.add_scatter(
            x=strikes,
            y=curve,
            mode="lines",
            name=f"eSSVI fit (RMSE {np.sqrt(np.mean(errors**2)):.2%})",
            line=dict(color=MUTED, width=2),
            hovertemplate=f"eSSVI ρ {rho:.2f}, ψ {psi:.3f}"
            "<br>Strike %{x:.2f}<br>Implied vol %{y:.2%}<extra></extra>",
        )
    fig.update_layout(
        title=f"Smile, {expiry:%d %b %y} expiry",
        margin=MARGIN | dict(b=110),
        xaxis_title="Strike",
        yaxis=dict(title="Implied vol", tickformat=".0%"),
        legend=dict(orientation="h", x=0, y=-0.22, yanchor="top"),
    )
    return fig


def rr_bf(grid: pd.DataFrame) -> pd.DataFrame:
    """Risk reversals (call vol - put vol) and butterflies ((call vol + put vol) / 2 - ATM vol) at 25 and 10 delta,
    from a by_delta grid. ATM is the vol at strike = forward."""
    rows = {}
    for kind in ("RR", "BF"):
        for d in (25, 10):
            call, put = grid.loc[f"{d}Δ call"], grid.loc[f"{d}Δ put"]
            rows[f"{kind} {d}Δ"] = call - put if kind == "RR" else (call + put) / 2 - grid.loc[ATM]
    return pd.DataFrame(rows).T


def vol_cells(grid: pd.DataFrame, **colorbar) -> go.Heatmap:
    labelled = len(grid) <= 12  # print the vols in the cells when there are few enough rows to read them
    return go.Heatmap(
        x=[f"{e:%d %b %y}" for e in grid.columns],
        y=grid.index,
        z=grid.to_numpy(),
        colorscale=VOL_SCALE,
        texttemplate="%{z:.1%}" if labelled else None,
        textfont_size=10,
        colorbar=dict(tickformat=".0%", thickness=12, outlinewidth=0, **colorbar),
        hovertemplate="%{y}<br>Expiry %{x}<br>Implied vol %{z:.2%}<extra></extra>",
    )


def heatmap(grid: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(vol_cells(grid))
    fig.update_layout(title=title, margin=MARGIN, xaxis=dict(type="category", title="Expiry"))
    return fig


def delta_heatmap(df: pd.DataFrame, source: str = "market") -> go.Figure:
    """Vols by delta on top; risk reversals and butterflies below, on their own scale centred on 0"""
    grid = by_delta(df)
    spreads = rr_bf(grid)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(vol_cells(grid, y=0.65, len=0.7), row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=[f"{e:%d %b %y}" for e in spreads.columns],
            y=spreads.index,
            z=spreads.to_numpy(),
            colorscale="RdBu",
            zmid=0,
            texttemplate="%{z:+.1%}",
            textfont_size=10,
            colorbar=dict(tickformat="+.0%", y=0.14, len=0.3, thickness=12, outlinewidth=0),
            hovertemplate="%{y}<br>Expiry %{x}<br>%{z:+.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(type="category")
    fig.update_xaxes(title="Expiry", row=2, col=1)
    fig.update_layout(title=f"Implied vol by delta, risk reversals and butterflies, {source}", margin=MARGIN)
    return fig


def residuals(df: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """Usable out-of-the-money quotes and how far the eSSVI fit is from each: in vol and in half-spreads
    (vega x vol error / half-spread, the price error measured in what it costs to cross the quote), largest first"""
    smiles = otm(df).join(fitted.add_prefix("fit_"), on="maturity", how="inner")  # the chain has greeks theta, rho
    if smiles.empty:
        return smiles.assign(fit_vol=[], error=[], spreads=[])
    smiles["fit_vol"] = ssvi.vol(
        np.log(smiles.strike / smiles.forward), smiles["T"], smiles.fit_theta, smiles.fit_rho, smiles.fit_psi
    )
    smiles["error"] = smiles.fit_vol - smiles.implied_vol
    half_spread = ((smiles.ask - smiles.bid) / 2).where(lambda h: h > 0)  # Yahoo sometimes has bid = ask
    smiles["spreads"] = smiles.vega * smiles.error / half_spread
    return smiles.sort_values("spreads", key=abs, ascending=False, na_position="last")


def surface_pick(df: pd.DataFrame, strike_range: float, expiry, strike) -> tuple[int, float]:
    """A clicked surface point as (maturity slider index, strike picker value): the nearest expiry of df, which the
    slider indexes, and the nearest strike within the range listed on that expiry (any expiry if none is), so the
    smile has a quote there. The fitted surface's grid strikes aren't listed ones."""
    dates = expiries(df)
    picked = pd.Timestamp(expiry).date()
    index = min(range(len(dates)), key=lambda i: abs(dates[i] - picked))
    near = in_range(df, strike_range)
    strikes = near[near.maturity == dates[index]].strike.unique()
    strikes = strikes if len(strikes) else near.strike.unique()
    return index, float(min(strikes, key=lambda k: abs(k - strike)))


def default_strike(df: pd.DataFrame) -> float:
    """The strike nearest spot among those listed on the most expiries: odd strikes near spot (357.5) are often
    weeklies only"""
    listed = df.groupby("strike").maturity.nunique()
    widest = listed[listed == listed.max()].index
    return float(min(widest, key=lambda k: abs(k - df.spot.iloc[0])))


def term_structure(df: pd.DataFrame, strike: float, fitted: pd.DataFrame) -> go.Figure:
    """Implied vol against expiry at one strike: the out-of-the-money quote of each expiry listing the strike, with
    bars from the bid vol to the ask vol, and the eSSVI fit at every fitted expiry. Moneyness drifts with the forward
    from one expiry to the next."""
    quotes = otm(df[df.strike == strike])
    first = df.groupby("maturity")[["T", "forward"]].first()
    first = first[first.index.isin(fitted.index)]
    fit = fitted.loc[first.index]
    fig = go.Figure()
    for kind in ("Put", "Call"):
        side = quotes[quotes.type == kind]
        fig.add_scatter(
            x=pd.to_datetime(side.maturity),
            y=side.implied_vol,
            mode="markers",
            name=f"OTM {kind.lower()}",
            marker=dict(color=PUT if kind == "Put" else CALL),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(side.ask_vol - side.implied_vol).fillna(0),
                arrayminus=(side.implied_vol - side.bid_vol).fillna(0),
                thickness=1,
                width=0,
            ),
            customdata=np.column_stack([side.bid_vol, side.ask_vol]),
            hovertemplate="Expiry %{x|%d %b %y}<br>Implied vol %{y:.2%}<br>bid %{customdata[0]:.2%}, "
            "ask %{customdata[1]:.2%}<extra></extra>",
        )
    fig.add_scatter(
        x=pd.to_datetime(first.index),
        y=ssvi.vol(np.log(strike / first.forward), first["T"], fit.theta, fit.rho, fit.psi),
        mode="lines",
        name="eSSVI fit",
        line=dict(color=MUTED, width=2),
        hovertemplate="Expiry %{x|%d %b %y}<br>eSSVI vol %{y:.2%}<extra></extra>",
    )
    fig.update_layout(
        title=f"Term structure at strike {strike:g}",
        margin=MARGIN | dict(b=90),
        xaxis_title="Expiry",
        yaxis=dict(title="Implied vol", tickformat=".0%"),
        legend=dict(orientation="h", x=0, y=-0.22, yanchor="top"),
    )
    return fig


def total_variance(df: pd.DataFrame, source: str, fitted: pd.DataFrame | None = None) -> go.Figure:
    """Total variance vol^2 T against log-moneyness, one line per expiry, from market vols or the eSSVI fit. Without
    calendar arbitrage a later expiry's line never dips below an earlier one's; red crosses mark calendar arbitrage
    on executable prices (a later expiry's ask below an earlier one's bid, in total variance)."""
    smiles = otm(df)
    if smiles.empty:
        return empty("No data")
    smiles = smiles.assign(k=np.log(smiles.strike / smiles.forward), w=smiles.implied_vol**2 * smiles["T"])
    fitted = (ssvi_fit(df) if fitted is None else fitted) if source == "fit" else None
    dates = expiries(smiles)
    colors = sample_colorscale("Blues", list(np.linspace(0.3, 1, len(dates))) if len(dates) > 1 else [1.0])
    fig = go.Figure()
    for color, (maturity, s) in zip(colors, smiles.groupby("maturity"), strict=True):
        k, w = s.k, s.w
        if fitted is not None:
            if maturity not in fitted.index:
                continue
            k = np.linspace(s.k.min(), s.k.max(), 100)
            w = ssvi.total_variance(k, *fitted.loc[maturity])
        fig.add_scatter(
            x=k,
            y=w,
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
            hovertemplate=f"Expiry {maturity:%d %b %y}<br>k %{{x:.3f}}<br>σ²T %{{y:.4f}}<extra></extra>",
        )
    bad = arbitrage.calendars(df[(df.bid > 0) & (df.ask > 0) & df.usable])
    fig.add_scatter(
        x=bad.k,
        y=bad.w,
        mode="markers",
        marker=dict(symbol="x", size=9, color=ERROR),
        name=f"{len(bad)} calendar arbitrages at bid/ask",
        customdata=np.column_stack([bad.maturity.astype(str), bad.strikes, bad.profit * MULTIPLIER]),
        hovertemplate="Expiry %{customdata[0]}, strike %{customdata[1]}<br>σ²T at the ask %{y:.4f}, under an earlier "
        "expiry's bid<br>$%{customdata[2]:.0f} per contract<extra></extra>",
    )
    source_label = "eSSVI fit" if fitted is not None else "market"
    fig.update_layout(
        title=f"Total variance σ²T, {source_label} (darker: later expiry)",
        margin=MARGIN,
        xaxis_title="Log-moneyness ln(K/F)",
        yaxis=dict(title="σ²T (log scale)", type="log"),  # spreads the short expiries bunched near 0
        showlegend=True,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"),
    )
    return fig


def slider_marks(dates: list) -> dict:
    step = math.ceil(len(dates) / 8)  # about 8 labels whatever the number of expiries
    return {i: f"{d:%d %b %y}" if i % step == 0 else "" for i, d in enumerate(dates)}


def empty(message: str = "") -> go.Figure:
    """A blank chart with a message instead of empty axes"""
    fig = go.Figure()
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    if message:
        fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=14))
    return fig


def quote(**stats) -> list:
    """The quote figures in the masthead, label above value"""
    return [html.Span([html.Small(label.replace("_", " ")), value], className="stat") for label, value in stats.items()]


def field(label: str, control, control_id: str | None = None):
    """A toolbar control with its label above"""
    return html.Div([html.Label(label, htmlFor=control_id), control], className="field")


def segmented(component_id: str, options: list, value: str):
    """Radio items styled as a segmented control (assets/style.css)"""
    return dcc.RadioItems(options, value, id=component_id, inline=True)


def dropdown(component_id, options, value):
    return dcc.Dropdown(options, value, id=component_id, clearable=False)


def card(*children, class_name="card"):
    return html.Div(className=class_name, children=list(children))


def leg_columns(dates: list) -> list:
    """Columns of the legs table: what to trade is editable, what it costs is computed"""
    choice = {"editable": True, "cellClass": "editable", "cellEditor": "agSelectCellEditor"}
    number = {"editable": True, "cellClass": "editable", "cellEditor": "agNumberCellEditor"}
    return [
        {"field": "side", "headerName": "Side", "cellEditorParams": {"values": ["Buy", "Sell"]}} | choice,
        {"field": "quantity", "headerName": "Quantity"} | number,
        {"field": "instrument", "headerName": "Instrument", "cellEditorParams": {"values": ["Call", "Put", "Stock"]}}
        | choice,
        {"field": "expiry", "headerName": "Expiry", "cellEditorParams": {"values": [str(d) for d in dates]}} | choice,
        {"field": "strike", "headerName": "Strike"} | number,
        {"field": "fill", "headerName": "Fill (bid/ask)", "valueFormatter": DOLLARS},
        {"field": "mid", "headerName": "Mid", "valueFormatter": DOLLARS},
        {"field": "vol", "headerName": "Implied vol", "valueFormatter": PERCENT},
        {
            "field": "position_delta",
            "headerName": "Delta, shares",
            "valueFormatter": {"function": "params.value == null ? '' : d3.format(',.1f')(params.value)"},
        },
    ]


STRATEGY_TAB = html.Main(
    className="results",
    children=[
        html.Div(
            className="grid strategy-row",
            children=[
                card(
                    html.Div(
                        className="card-tools",
                        children=[
                            html.Label("Preset", htmlFor="preset"),
                            html.Div(dropdown("preset", list(strategy.PRESETS), "Covered call"), className="preset"),
                            html.Label("Expiry", htmlFor="preset-expiry"),
                            html.Div(dropdown("preset-expiry", [], None), className="expiry"),
                            html.Label("Apart", htmlFor="width"),
                            html.Div(
                                dropdown("width", [{"label": f"{w:.1%} of spot", "value": w} for w in WIDTHS], 0.05),
                                className="width",
                            ),
                            html.Button("Add leg", id="add-leg", className="secondary"),
                            html.Button("Remove ticked", id="remove-legs", className="secondary"),
                        ],
                    ),
                    html.P(
                        "Click a side, quantity, instrument, expiry or strike to change it; strikes snap to listed ones. "
                        "Quantity is contracts of 100 shares, or shares for the stock.",
                        className="hint",
                    ),
                    dag.AgGrid(
                        id="legs",
                        rowData=[],
                        columnDefs=leg_columns([]),
                        getRowId="params.data.id",
                        defaultColDef={"flex": 1, "sortable": False},
                        dashGridOptions={
                            "theme": GRID_THEME,
                            "domLayout": "autoHeight",
                            "rowSelection": {"mode": "multiRow"},
                            "singleClickEdit": True,
                            "stopEditingWhenCellsLoseFocus": True,
                        },
                    ),
                ),
                card(
                    html.Div(
                        className="card-tools",
                        children=[
                            html.H2("Position", className="card-title"),
                            html.Label("valued at", htmlFor="horizon"),
                            html.Div(dropdown("horizon", [], None), className="expiry"),
                            segmented("strategy-vols", VOL_MODES, "own"),
                        ],
                    ),
                    html.Div(id="position"),
                ),
            ],
        ),
        html.Div(
            className="grid",
            children=[
                card(dcc.Graph(id="payoff", className="chart")),
                card(dcc.Graph(id="scenarios", className="chart")),
            ],
        ),
    ],
)


PARITY_TAB = html.Main(
    className="results",
    children=[
        html.Div(
            className="grid",
            children=[
                card(dcc.Graph(id="parity-chart", className="chart")),
                card(
                    html.H2("Implied forward and yield by expiry", className="card-title"),
                    html.P(
                        "From put-call parity on the strikes nearest spot. The range is what bid/ask allows: outside "
                        "it a conversion or reversal would pay. Divs: the PV of dividends the forward implies. Borrow and "
                        "other: implied yield - trailing dividend yield.",
                        className="hint",
                    ),
                    dag.AgGrid(
                        id="parity-table",
                        rowData=[],
                        columnDefs=[
                            {"field": "maturity", "headerName": "Expiry", "minWidth": 110},
                            {"field": "forward", "headerName": "Forward", "valueFormatter": DOLLARS},
                            {"field": "implied_yield", "headerName": "Implied", "valueFormatter": PERCENT},
                            {"field": "low_yield", "headerName": "Low", "valueFormatter": PERCENT},
                            {"field": "high_yield", "headerName": "High", "valueFormatter": PERCENT},
                            {"field": "dividends", "headerName": "Divs, $", "valueFormatter": DOLLARS},
                            {"field": "borrow", "headerName": "Borrow", "valueFormatter": PERCENT},
                            {"field": "pairs", "headerName": "Pairs"},
                            {"field": "stale", "headerName": "Stale"},
                        ],
                        defaultColDef={"flex": 1, "sortable": True},
                        dashGridOptions={"theme": GRID_THEME, "domLayout": "autoHeight"},
                    ),
                ),
            ],
        ),
        card(
            html.H2(id="stale-title", className="card-title"),
            dag.AgGrid(
                id="stale-pairs",
                rowData=[],
                columnDefs=[
                    {"field": "maturity", "headerName": "Expiry"},
                    {"field": "strike", "headerName": "Strike"},
                    {"field": "bid", "headerName": "Call bid", "valueFormatter": DOLLARS},
                    {"field": "ask", "headerName": "Call ask", "valueFormatter": DOLLARS},
                    {"field": "bid_put", "headerName": "Put bid", "valueFormatter": DOLLARS},
                    {"field": "ask_put", "headerName": "Put ask", "valueFormatter": DOLLARS},
                    {"field": "low", "headerName": "Forward from, at bid/ask", "valueFormatter": DOLLARS},
                    {"field": "high", "headerName": "to", "valueFormatter": DOLLARS},
                    {"field": "forward", "headerName": "Expiry forward", "valueFormatter": DOLLARS},
                    {"field": "miss", "headerName": "Miss", "sort": "desc", "valueFormatter": PERCENT},
                ],
                defaultColDef={"flex": 1, "sortable": True, "filter": True},
                dashGridOptions={
                    "theme": GRID_THEME,
                    "domLayout": "autoHeight",
                    "pagination": True,
                    "paginationPageSize": 10,
                },
            ),
        ),
    ],
)


def parity_by_expiry(df: pd.DataFrame, dividend_yield: float, band=0.05) -> pd.DataFrame:
    """Per expiry: the implied forward and yield, the yield range the bid/ask of the live pairs within band of spot
    allow (the tightest of their forward ranges), the PV of dividends it implies, borrow and other (implied yield -
    trailing dividend yield), and how many pairs were checked and found stale"""
    pairs = parity_pairs(df)
    stale = pairs.miss > PARITY_TOL
    strikes = pairs.index.get_level_values("strike")
    live = pairs[~stale & ((strikes / pairs.spot - 1).abs() <= band)].groupby(level="maturity")
    first = df.groupby("maturity")[["spot", "r", "T", "forward", "div_yield"]].first()
    low, high = live.low.max().reindex(first.index), live.high.min().reindex(first.index)

    def to_yield(forward):
        return first.r - np.log(forward / first.spot) / first["T"]

    return pd.DataFrame(
        {
            "forward": first.forward,
            "implied_yield": first.div_yield,
            "low_yield": to_yield(high),
            "high_yield": to_yield(low),
            "dividends": first.spot - first.forward * np.exp(-first.r * first["T"]),
            "borrow": first.div_yield - dividend_yield,
            "pairs": pairs.groupby(level="maturity").size().reindex(first.index).fillna(0).astype(int),
            "stale": stale.groupby(level="maturity").sum().reindex(first.index).fillna(0).astype(int),
        }
    )


def parity_chart(table: pd.DataFrame, dividend_yield: float) -> go.Figure:
    """Implied yield per expiry with its bid/ask range, against the trailing dividend yield"""
    x = pd.to_datetime(table.index)
    fig = go.Figure()
    if pd.notna(dividend_yield):
        fig.add_hline(
            y=dividend_yield,
            line=dict(color=CALL, width=1.5, dash="dash"),
            annotation_text=f"trailing dividend yield {dividend_yield:.2%}",
            annotation_position="bottom right",
        )
    fig.add_scatter(
        x=x,
        y=table.implied_yield,
        mode="lines+markers",
        name="Implied yield",
        line=dict(color=PUT),
        error_y=dict(
            type="data",
            symmetric=False,
            array=(table.high_yield - table.implied_yield).clip(lower=0).fillna(0),
            arrayminus=(table.implied_yield - table.low_yield).clip(lower=0).fillna(0),
            thickness=1,
            width=0,
        ),
        hovertemplate="Expiry %{x|%d %b %y}<br>Implied yield %{y:.2%}<extra></extra>",
    )
    # front expiries, a session or two away, turn a cent of forward into tens of percent: keep the axis on the rest
    levels = pd.concat([table.implied_yield, pd.Series([dividend_yield])]).dropna()
    low, high = levels.quantile(0.1), levels.quantile(0.9)
    pad = max(high - low, 0.01)
    fig.update_layout(
        title=dict(
            text="Implied yield (dividends + borrow) by expiry",
            subtitle=dict(
                text="Bars: the range bid/ask allows near spot. Axis cut to the middle 80%: front expiries swing"
            ),
        ),
        yaxis_range=[low - pad, high + pad],
        margin=MARGIN | dict(t=70),
        xaxis_title="Expiry",
        yaxis=dict(title="Continuous yield", tickformat=".1%"),
        showlegend=False,
    )
    return fig


def strategy_spots(priced: pd.DataFrame, spot: float, points=401) -> np.ndarray:
    """Spots to chart: from 15% below the lowest of spot and the strikes to 15% above the highest"""
    levels = [spot, *priced.strike.dropna()]
    return np.linspace(min(levels) * 0.85, max(levels) * 1.15, points)


def payoff_chart(priced: pd.DataFrame, spot: float, horizon: date, elapsed: float, fitted=None) -> go.Figure:
    """P&L against spot, today and at the horizon, from the cost to open at bid/ask"""
    spots = strategy_spots(priced, spot)
    entry = strategy.cost(priced)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=RULE, width=1))
    fig.add_vline(x=spot, line=dict(color=MUTED, width=1, dash="dot"), annotation_text="spot")
    fig.add_scatter(
        x=spots, y=strategy.value(priced, spots, fitted=fitted) - entry, name="Today", line=dict(color=PUT, width=1.5)
    )
    fig.add_scatter(
        x=spots,
        y=strategy.value(priced, spots, elapsed, fitted=fitted) - entry,
        name=f"At {horizon:%d %b %y}",
        line=dict(color=INK, width=2.5),
    )
    fig.update_traces(hovertemplate="Spot %{x:,.2f}<br>P&L $%{y:,.0f}<extra></extra>")
    fig.update_layout(
        title=dict(
            text="P&L, $",
            subtitle=dict(
                text="Options repriced at their own vols (sticky strike)"
                if fitted is None
                else "Options repriced on the eSSVI fit at their moneyness; legs outside the fit at their own vol"
            ),
        ),
        margin=MARGIN | dict(b=90, t=70),
        xaxis_title="Spot",
        yaxis=dict(tickformat=",.0f"),
        legend=dict(orientation="h", x=0, y=-0.22, yanchor="top"),
    )
    return fig


def scenario_chart(priced: pd.DataFrame, spot: float, horizon: date, elapsed: float, fitted=None) -> go.Figure:
    """P&L at the horizon for spot moves against parallel vol shifts; one row, by spot alone, when the vol shift
    changes nothing (every option expired by then, or stock only)"""
    moves, shifts = np.arange(-20, 21, 5), np.arange(-10, 11, 5)
    entry = strategy.cost(priced)
    z = np.array(
        [strategy.value(priced, spot * (1 + moves / 100), elapsed, shift / 100, fitted) - entry for shift in shifts]
    )
    flat = np.allclose(z, z[shifts == 0], atol=0.5)  # within 50 cents
    rows = ["any vol"] if flat else [f"{v:+d} pts" for v in shifts]
    fig = go.Figure(
        go.Heatmap(
            x=[f"{m:+d}%" for m in moves],
            y=rows,
            z=z[shifts == 0] if flat else z,
            zmid=0,
            colorscale="RdBu",
            texttemplate="%{z:,.0f}",
            colorbar=dict(title="P&L, $"),
            hovertemplate="Spot %{x}, vol %{y}<br>P&L $%{z:,.0f}<extra></extra>",
        )
    )
    title = f"P&L at {horizon:%d %b %y}: " + (
        "spot move (no vol exposure left by then)" if flat else "spot move against vol shift"
    )
    fig.update_layout(title=title, margin=MARGIN, xaxis_title="Spot move")
    return fig


def money(value: float | None, decimals=0) -> str:
    """-$1,234 rather than $-1,234"""
    return "unlimited" if value is None else f"{'-' if value < 0 else ''}${abs(value):,.{decimals}f}"


def position_summary(priced: pd.DataFrame, chain: pd.DataFrame, horizon: date, elapsed: float, fitted=None) -> html.Dl:
    """Cost to open, greeks and, at the horizon, max profit and loss, breakevens and a rough chance of profit"""
    spot = chain.spot.iloc[0]
    entry, mid = strategy.cost(priced), strategy.cost(priced, "mid")
    spots = np.linspace(spot * 1e-3, spot * 4, 4001)
    pnl = strategy.value(priced, spots, elapsed, fitted=fitted) - entry
    stats = strategy.payoff_stats(spots, pnl)
    at = chain[chain.maturity == horizon]
    vol = atm_vol(chain).get(horizon, np.nan)
    vol = vol if pd.notna(vol) else priced.vol.median()
    drift = float((at.r - at.div_yield).iloc[0]) if len(at) else 0.0
    chance = strategy.chance_of_profit(spots, pnl, spot, elapsed, vol, drift)
    greeks = strategy.greeks(priced)
    rows = {
        "Cost to open, at bid/ask": f"{'pay' if entry >= 0 else 'receive'} ${abs(entry):,.2f}",
        "At mid": f"{'pay' if mid >= 0 else 'receive'} ${abs(mid):,.2f}; the spread costs {money(entry - mid, 2)}",
        **(
            {}
            if fitted is None
            else {"eSSVI value - mid": money(strategy.value(priced, [spot], fitted=fitted)[0] - mid, 2)}
        ),
        "Delta": f"{greeks['delta']:,.1f} shares",
        "Gamma": f"{greeks['gamma']:,.2f} shares per $1",
        "Vega": f"{money(greeks['vega'], 2)} per vol point",
        "Theta": f"{money(greeks['theta'], 2)} per trading day",
        f"Max profit at {horizon:%d %b}": money(stats["max_profit"]),
        f"Max loss at {horizon:%d %b}": money(stats["max_loss"]),
        "Breakevens": ", ".join(f"{b:,.2f}" for b in stats["breakevens"]) or "none",
        "Chance of profit": f"{chance:.0%} (rough: lognormal, ATM vol {vol:.0%})",
    }
    return html.Dl([part for k, v in rows.items() for part in (html.Dt(k), html.Dd(v))], className="stats")


app = Dash(__name__, title="Listed options dashboard")
app.layout = html.Div(
    className="page",
    children=[
        dcc.Store(id="loaded"),
        html.Header(
            className="masthead",
            children=[
                html.Div(
                    className="masthead-top",
                    children=[
                        html.H1("Listed options"),
                        html.Div(
                            dropdown(
                                "ticker",
                                [{"label": f"{t} · {name}", "value": t} for t, name in TICKERS.items()],
                                "TSLA",
                            ),
                            className="ticker-picker",
                        ),
                        html.Div(id="status", className="quote", role="status"),
                        html.Button("Reload", id="load", title="Fetch fresh quotes instead of the 15-minute cache"),
                    ],
                ),
                html.Div(
                    className="toolbar",
                    children=[
                        field("Calls or puts (surface and chain)", segmented("type", ["Call", "Put"], "Call")),
                        field(
                            "Surface",
                            dropdown(
                                "field",
                                [{"label": to_title(f), "value": f} for f in FIELDS]
                                + [{"label": ESSVI, "value": "essvi_vol"}],
                                "implied_vol",
                            ),
                            "field",
                        ),
                        field(
                            "Strike range around spot",
                            dropdown("range", [{"label": f"± {p}%", "value": p} for p in (5, 10, 15, 25, 50)], 15),
                            "range",
                        ),
                        field(
                            "Max maturity",
                            dropdown(
                                "months", [{"label": f"{m}M", "value": m} for m in (1, 2, 3, 6, 9, 12, 24, 36)], 6
                            ),
                            "months",
                        ),
                    ],
                ),
            ],
        ),
        dcc.Tabs(
            className="tabs",
            parent_className="tabs-bar",
            children=[
                dcc.Tab(
                    label="Vol surface",
                    className="tab",
                    selected_className="tab--selected",
                    children=html.Main(
                        className="results",
                        children=[
                            html.Div(
                                className="grid surface-row",
                                children=[
                                    card(
                                        dcc.Loading(
                                            dcc.Graph(id="surface", className="chart", config={"scrollZoom": True})
                                        )
                                    ),
                                    card(
                                        html.Div(
                                            className="card-tools",
                                            children=[
                                                segmented(
                                                    "heatmap-axis",
                                                    [
                                                        {"label": "By delta", "value": "delta"},
                                                        {"label": "By strike", "value": "strike"},
                                                    ],
                                                    "delta",
                                                ),
                                                segmented("heatmap-source", SOURCES, "market"),
                                            ],
                                        ),
                                        dcc.Graph(id="heatmap", className="chart"),
                                    ),
                                ],
                            ),
                            html.Div(
                                className="grid",
                                children=[
                                    card(
                                        dcc.Graph(id="smile", className="chart"),
                                        html.Label("Maturity", htmlFor="maturity"),
                                        dcc.Slider(
                                            0,
                                            0,
                                            1,
                                            value=0,
                                            id="maturity",
                                            marks={},
                                            included=False,
                                            allow_direct_input=False,
                                        ),
                                    ),
                                    card(
                                        html.Div(
                                            className="card-tools",
                                            children=[
                                                html.Label("Strike", htmlFor="strike"),
                                                html.Div(
                                                    dcc.Dropdown([], None, id="strike", clearable=False),
                                                    className="strike",
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(id="term-structure", className="chart"),
                                    ),
                                    card(
                                        html.Div(
                                            segmented("variance-source", SOURCES, "market"), className="card-tools"
                                        ),
                                        dcc.Graph(id="total-variance", className="chart"),
                                    ),
                                ],
                            ),
                            html.Div(
                                className="grid tables-row",
                                children=[
                                    card(
                                        html.H2("Quotes furthest from the eSSVI fit", className="card-title"),
                                        dag.AgGrid(
                                            id="residuals",
                                            rowData=[],
                                            columnDefs=[
                                                {"field": "maturity", "headerName": "Expiry"},
                                                {"field": "type", "headerName": "Type"},
                                                {"field": "strike", "headerName": "Strike"},
                                                {
                                                    "field": "implied_vol",
                                                    "headerName": "Market vol",
                                                    "valueFormatter": PERCENT,
                                                },
                                                {
                                                    "field": "fit_vol",
                                                    "headerName": "Fit vol",
                                                    "valueFormatter": PERCENT,
                                                },
                                                {
                                                    "field": "error",
                                                    "headerName": "Fit - market",
                                                    "valueFormatter": PERCENT,
                                                },
                                                {
                                                    "field": "spreads",
                                                    "headerName": "In half-spreads",
                                                    "valueFormatter": {
                                                        "function": "params.value == null ? '' : d3.format('+.1f')(params.value)"
                                                    },
                                                },
                                            ],
                                            defaultColDef={"sortable": True, "filter": True, "flex": 1},
                                            dashGridOptions={
                                                "theme": GRID_THEME,
                                                "pagination": True,
                                                "paginationPageSize": 10,
                                                "domLayout": "autoHeight",
                                            },
                                        ),
                                    ),
                                    card(
                                        html.H2(id="arbitrage-title", className="card-title"),
                                        dag.AgGrid(
                                            id="arbitrage",
                                            rowData=[],
                                            columnDefs=[
                                                {"field": "check", "headerName": "Check"},
                                                {"field": "maturity", "headerName": "Expiry"},
                                                {"field": "strikes", "headerName": "Strikes"},
                                                {
                                                    "field": "dollars",
                                                    "headerName": "Profit, $ per contract",
                                                    "filter": "agNumberColumnFilter",
                                                    "sort": "desc",
                                                    "valueFormatter": {"function": "d3.format(',.2f')(params.value)"},
                                                },
                                                {"field": "trade", "headerName": "Trade", "flex": 3},
                                            ],
                                            defaultColDef={"sortable": True, "filter": True, "flex": 1},
                                            dashGridOptions={
                                                "theme": GRID_THEME,
                                                "pagination": True,
                                                "paginationPageSize": 10,
                                                "domLayout": "autoHeight",
                                            },
                                        ),
                                    ),
                                ],
                            ),
                            card(
                                dag.AgGrid(
                                    id="table",
                                    rowData=[],
                                    columnDefs=[
                                        {"field": c, "headerName": to_title(c)}
                                        | ({} if c in ("name", "maturity") else {"filter": "agNumberColumnFilter"})
                                        | (
                                            {
                                                "valueFormatter": {
                                                    "function": f"params.value == null ? '' : d3.format('{TABLE_FORMATS[c]}')(params.value)"
                                                }
                                            }
                                            if c in TABLE_FORMATS
                                            else {}
                                        )
                                        for c in COLUMNS
                                    ],
                                    defaultColDef={"sortable": True, "filter": True, "floatingFilter": True, "flex": 1},
                                    dashGridOptions={
                                        "theme": GRID_THEME,
                                        "pagination": True,
                                        "paginationPageSize": 20,
                                        "domLayout": "autoHeight",
                                    },
                                )
                            ),
                        ],
                    ),
                ),
                dcc.Tab(label="Strategy", className="tab", selected_className="tab--selected", children=STRATEGY_TAB),
                dcc.Tab(label="Parity", className="tab", selected_className="tab--selected", children=PARITY_TAB),
            ],
        ),
    ],
)


@app.callback(
    Output("surface", "figure"),
    Output("table", "rowData"),
    Output("status", "children"),
    Output("loaded", "data"),
    Output("maturity", "max"),
    Output("maturity", "marks"),
    Output("maturity", "value"),
    Input("load", "n_clicks"),
    Input("ticker", "value"),
    Input("type", "value"),
    Input("field", "value"),
    Input("range", "value"),
    Input("months", "value"),
    State("maturity", "value"),
)
def update(_clicks, ticker, option_type, field, strike_range, months, maturity_index):
    """Load the chain, then draw the surface and table; the smile and heatmap follow from the `loaded` store"""
    try:  # show data-source errors (Yahoo down, no options listed) in the page instead of failing the callback
        df = load_options(ticker, months, refresh=ctx.triggered_id == "load")
    except Exception as error:  # noqa: BLE001  (any failure is shown in the status line)
        message = html.Span(f"Could not load {ticker}: {error}. Press Reload to try again.", className="error")
        return empty("No data"), [], message, None, 0, {}, 0
    loaded = dict(ticker=ticker, months=months, range=strike_range, type=option_type)
    dates = expiries(df)
    slider = (len(dates) - 1, slider_marks(dates), min(maturity_index or 0, len(dates) - 1))
    view = select(df, option_type, strike_range)
    atm = atm_vol(df).dropna()
    status = quote(
        spot=f"{df.spot.iloc[0]:.2f}",
        rate=f"{df.r.iloc[0]:.2%}",
        front_ATM_vol=f"{atm.iloc[0]:.1%}" if len(atm) else "n/a",
        options=f"{len(view)} {option_type.lower()}s",
        usable_quotes=f"{df.usable.mean():.0%}" if (df.bid > 0).any() else "none: market closed, last trades",
        expiries=f"{view.maturity.nunique()} to {view.maturity.max():%d %b %y}" if len(view) else "none",
    )
    if view.empty:
        widen = f"No {option_type.lower()}s within ± {strike_range}% of spot. Widen the strike range."
        return empty(widen), [], status, loaded, *slider
    fitted = fit_for(ticker, months, strike_range) if field == "essvi_vol" else None
    return (
        surface(vol_view(df, strike_range) if field in ("implied_vol", "essvi_vol") else view, field, fitted),
        view[COLUMNS].round(5).to_dict("records"),
        status,
        loaded,
        *slider,
    )


@app.callback(Output("smile", "figure"), Input("maturity", "value"), Input("loaded", "data"))
def update_smile(maturity_index, loaded):
    if not loaded:
        return empty("No data")
    df = load_options(loaded["ticker"], loaded["months"])
    dates = expiries(df)  # the slider indexes the same list
    fitted = fit_for(loaded["ticker"], loaded["months"], loaded["range"])
    expiry = dates[min(maturity_index, len(dates) - 1)]
    return smile(in_range(df, loaded["range"]), expiry, df.spot.iloc[0], fitted)


@app.callback(
    Output("heatmap", "figure"),
    Input("heatmap-axis", "value"),
    Input("heatmap-source", "value"),
    Input("loaded", "data"),
)
def update_heatmap(axis, source, loaded):
    """By delta reads the whole chain (the 5-delta wings are often beyond the strike range); the fit, within the
    strike range, is extended over the whole chain's strikes, arbitrage-free"""
    if not loaded:
        return empty("No data")
    df = load_options(loaded["ticker"], loaded["months"])
    if source == "fit":
        fitted = fit_for(loaded["ticker"], loaded["months"], loaded["range"])
        if axis == "delta":
            return delta_heatmap(fitted_smiles(df, fitted), "eSSVI fit")
        grid = fitted_vols(in_range(df, loaded["range"]), fitted)
        return heatmap(grid, "Implied vol by strike, eSSVI fit")
    if axis == "delta":
        return delta_heatmap(df)
    return heatmap(by_strike(vol_view(df, loaded["range"]), "implied_vol"), "Implied vol by strike (OTM)")


@app.callback(Output("total-variance", "figure"), Input("variance-source", "value"), Input("loaded", "data"))
def update_total_variance(source, loaded):
    if not loaded:
        return empty("No data")
    fitted = fit_for(loaded["ticker"], loaded["months"], loaded["range"])
    return total_variance(in_range(load_options(loaded["ticker"], loaded["months"]), loaded["range"]), source, fitted)


@app.callback(Output("arbitrage", "rowData"), Output("arbitrage-title", "children"), Input("loaded", "data"))
def update_arbitrage(loaded):
    if not loaded:
        return [], "Static arbitrage"
    found = arbitrage.scan(in_range(load_options(loaded["ticker"], loaded["months"]), loaded["range"]))
    title = (
        f"Static arbitrage at bid/ask: {len(found)} to check (free quotes can be stale; out-of-the-money and "
        f"near-the-money strikes only)"
        if len(found)
        else "Static arbitrage at bid/ask: none (out-of-the-money and near-the-money strikes)"
    )
    rows = found.assign(dollars=found.profit * MULTIPLIER, maturity=found.maturity.astype(str))
    return rows.drop(columns=["k", "w", "profit"], errors="ignore").to_dict("records"), title


@app.callback(Output("residuals", "rowData"), Input("loaded", "data"))
def update_residuals(loaded):
    if not loaded:
        return []
    df = in_range(load_options(loaded["ticker"], loaded["months"]), loaded["range"])
    found = residuals(df, fit_for(loaded["ticker"], loaded["months"], loaded["range"]))
    columns = ["maturity", "type", "strike", "implied_vol", "fit_vol", "error", "spreads"]
    return found[columns].head(200).astype({"maturity": str}).to_dict("records")


@app.callback(
    Output("strike", "options"),
    Output("strike", "value"),
    Input("loaded", "data"),
    State("strike", "value"),
)
def update_strikes(loaded, strike):
    """Strikes within the strike range with how many expiries list each; keeps the chosen one if still listed"""
    if not loaded:
        return [], None
    df = in_range(load_options(loaded["ticker"], loaded["months"]), loaded["range"])
    listed = df.groupby("strike").maturity.nunique()
    if strike not in listed.index:
        strike = default_strike(df)
    return [{"label": f"{k:g}  ({n} expiries)", "value": k} for k, n in listed.items()], strike


@app.callback(Output("term-structure", "figure"), Input("strike", "value"), Input("loaded", "data"))
def update_term_structure(strike, loaded):
    if not loaded or strike is None:
        return empty("No data")
    df = in_range(load_options(loaded["ticker"], loaded["months"]), loaded["range"])
    return term_structure(df, strike, fit_for(loaded["ticker"], loaded["months"], loaded["range"]))


@app.callback(
    Output("maturity", "value", allow_duplicate=True),
    Output("strike", "value", allow_duplicate=True),
    Input("surface", "clickData"),
    State("loaded", "data"),
    prevent_initial_call=True,
)
def pick_from_surface(click, loaded):
    if not click or not loaded:
        return no_update, no_update
    point = click["points"][0]
    df = load_options(loaded["ticker"], loaded["months"])
    return surface_pick(df, loaded["range"], point["x"], point["y"])


@app.callback(
    Output("preset-expiry", "options"),
    Output("preset-expiry", "value"),
    Output("legs", "columnDefs"),
    Input("loaded", "data"),
    State("preset-expiry", "value"),
)
def update_strategy_expiries(loaded, expiry):
    """Expiries of the loaded chain; keeps the chosen one if still listed, else the one nearest a month out"""
    if not loaded:
        return [], None, leg_columns([])
    dates = expiries(load_options(loaded["ticker"], loaded["months"]))
    if expiry not in [str(d) for d in dates]:
        expiry = str(min(dates, key=lambda d: abs((d - date.today()).days - 30)))
    return [{"label": f"{d:%d %b %y}", "value": str(d)} for d in dates], expiry, leg_columns(dates)


@app.callback(
    Output("legs", "rowData"),
    Input("preset", "value"),
    Input("preset-expiry", "value"),
    Input("width", "value"),
    Input("add-leg", "n_clicks"),
    Input("remove-legs", "n_clicks"),
    Input("legs", "cellValueChanged"),
    Input("loaded", "data"),
    State("legs", "rowData"),
    State("legs", "selectedRows"),
)
def update_legs(preset, expiry, width, _add, _remove, changed, loaded, rows, selected):
    """A preset, its expiry or width (re)builds the legs; adding, removing or editing a leg reprices them as they are"""
    if not loaded or expiry is None:
        return []
    chain = load_options(loaded["ticker"], loaded["months"])
    if ctx.triggered_id in (None, "loaded", "preset", "preset-expiry", "width"):
        return strategy.build(preset, chain, expiry, width)
    legs = {row["id"]: row for row in rows or []}
    if ctx.triggered_id == "add-leg":
        legs[max(legs, default=-1) + 1] = dict(side="Buy", quantity=1, instrument="Call", expiry=expiry, strike=None)
    elif ctx.triggered_id == "remove-legs":
        for row in selected or []:
            legs.pop(row["id"], None)
    else:  # an edited cell
        legs |= {change["data"]["id"]: change["data"] for change in changed or []}
    return strategy.as_rows(strategy.price(list(legs.values()), chain)) if legs else []


@app.callback(
    Output("horizon", "options"),
    Output("horizon", "value"),
    Input("legs", "rowData"),
    State("loaded", "data"),
    State("horizon", "value"),
)
def update_horizons(rows, loaded, horizon):
    """Expiries up to the first option leg's, the default: until then every leg is still alive"""
    first = min((row["expiry"] for row in rows or [] if row.get("expiry")), default=None)
    if not loaded or first is None:
        return [], None
    dates = [str(d) for d in expiries(load_options(loaded["ticker"], loaded["months"])) if str(d) <= first]
    options = [{"label": f"{date.fromisoformat(d):%d %b %y}", "value": d} for d in dates]
    return options, horizon if horizon in dates else first


@app.callback(
    Output("payoff", "figure"),
    Output("position", "children"),
    Output("scenarios", "figure"),
    Input("legs", "rowData"),
    Input("horizon", "value"),
    Input("strategy-vols", "value"),
    State("loaded", "data"),
)
def update_strategy_view(rows, horizon, vols, loaded):
    if not loaded or not rows or horizon is None:
        return empty("Pick a preset or add a leg"), "", empty()
    chain = load_options(loaded["ticker"], loaded["months"])
    priced = strategy.price(rows, chain)
    horizon = date.fromisoformat(horizon)
    elapsed = float(chain[chain.maturity == horizon]["T"].max())  # trading years until the horizon expiry
    spot = chain.spot.iloc[0]
    fitted = fit_for(loaded["ticker"], loaded["months"], loaded["range"]) if vols == "fit" else None
    return (
        payoff_chart(priced, spot, horizon, elapsed, fitted),
        position_summary(priced, chain, horizon, elapsed, fitted),
        scenario_chart(priced, spot, horizon, elapsed, fitted),
    )


@app.callback(
    Output("parity-chart", "figure"),
    Output("parity-table", "rowData"),
    Output("stale-pairs", "rowData"),
    Output("stale-title", "children"),
    Input("loaded", "data"),
)
def update_parity(loaded):
    if not loaded:
        return empty("No data"), [], [], "Pairs dropped by the parity check"
    ticker = loaded["ticker"]
    df = in_range(load_options(ticker, loaded["months"]), loaded["range"])
    try:
        dividend_yield = Environment().get_div_yield(ticker)
    except Exception:  # noqa: BLE001  (a missing dividend history only blanks one column)
        dividend_yield = np.nan
    table = parity_by_expiry(df, dividend_yield)
    pairs = parity_pairs(df)
    stale = pairs[pairs.miss > PARITY_TOL].reset_index()
    title = (
        f"Pairs dropped by the parity check: {len(stale)} of {len(pairs)}. Their bid/ask can't reach the expiry's "
        f"forward within {PARITY_TOL:.2%}: one quote is stale, so neither is used by the fit or the arbitrage scan."
    )
    columns = ["maturity", "strike", "bid", "ask", "bid_put", "ask_put", "low", "high", "forward", "miss"]
    return (
        parity_chart(table, dividend_yield),
        table.reset_index(names="maturity").astype({"maturity": str}).round(6).to_dict("records"),
        stale[columns].astype({"maturity": str}).round(6).to_dict("records"),
        title,
    )


if __name__ == "__main__":
    app.run()
