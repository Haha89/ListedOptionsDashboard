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
from dash import Dash, Input, Output, State, ctx, dcc, html
from dateutil.relativedelta import relativedelta
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from src.option_vol import arbitrage, ssvi
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
MAX_SPREAD = 0.5  # widest usable quote, (ask - bid) / mid
ATM = "ATM (fwd)"
ESSVI = "Implied vol, eSSVI fit"
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
        df["div_yield"] = implied_yield(df)
        df["forward"] = df.spot * np.exp((df.r - df.div_yield) * df["T"])
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
    put-call parity F = K + (C - P) e^(rT) at the 3 strikes nearest spot (median). Without it calls and puts of one
    strike give different vols; it absorbs dividends, borrow costs and a spot taken at another time than the quotes.
    Usable quotes only; 0 for expiries with no usable call/put pair at a strike."""
    quotes = df[df.usable]
    calls = quotes[quotes.type == "Call"].set_index(["maturity", "strike"])
    puts = quotes[quotes.type == "Put"].set_index(["maturity", "strike"]).price
    pairs = calls.assign(put=puts).dropna(subset=["put"])
    strikes = pairs.index.get_level_values("strike")
    pairs["forward"] = strikes + (pairs.price - pairs.put) * np.exp(pairs.r * pairs["T"])
    pairs["distance"] = abs(strikes / pairs.spot - 1)
    nearest = pairs.sort_values("distance").groupby(level="maturity").head(3).groupby(level="maturity")
    first = nearest[["spot", "r", "T"]].first()
    q = first.r - np.log(nearest.forward.median() / first.spot) / first["T"]
    return df.maturity.map(q).fillna(0.0)


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
        title=f"{label} surface",
        margin=dict(l=0, r=0, b=0, t=40),
        uirevision="keep",
        showlegend=False,
        scene=dict(
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
    bad = arbitrage.calendars(df[(df.bid > 0) & (df.ask > 0)])
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
        title=f"Total variance σ²T by log-moneyness, {source_label} (darker = later expiry)",
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


def quote(ticker: str, **stats) -> list:
    """The quote strip above the charts: ticker, then label/value pairs"""
    return [html.Strong(ticker, className="ticker")] + [
        html.Span([html.Small(label.replace("_", " ")), value], className="stat") for label, value in stats.items()
    ]


def dropdown(component_id, options, value):
    return dcc.Dropdown(options, value, id=component_id, clearable=False)


def card(*children, class_name="card"):
    return html.Div(className=class_name, children=list(children))


app = Dash(
    __name__,
    title="Listed options dashboard",
    external_stylesheets=["https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap"],
)
app.layout = html.Div(
    className="page",
    children=[
        dcc.Store(id="loaded"),
        html.Div(
            className="controls",
            children=[
                html.H1("Listed options"),
                html.Label("Underlying", htmlFor="ticker"),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            dropdown(
                                "ticker",
                                [{"label": f"{t} · {name}", "value": t} for t, name in TICKERS.items()],
                                "TSLA",
                            ),
                            className="grow",
                        ),
                        html.Button("Reload", id="load", title="Fetch fresh quotes instead of the 15-minute cache"),
                    ],
                ),
                dcc.RadioItems(["Call", "Put"], "Call", id="type", inline=True),
                html.Label("Surface field", htmlFor="field"),
                dropdown(
                    "field",
                    [{"label": to_title(f), "value": f} for f in FIELDS] + [{"label": ESSVI, "value": "essvi_vol"}],
                    "implied_vol",
                ),
                html.Label("Strike range around spot", htmlFor="range"),
                dropdown("range", [{"label": f"± {p}%", "value": p} for p in (5, 10, 15, 25, 50)], 15),
                html.Label("Max maturity", htmlFor="months"),
                dropdown("months", [{"label": f"{m}M", "value": m} for m in (1, 2, 3, 6, 9, 12, 24, 36)], 6),
            ],
        ),
        html.Div(
            className="results",
            children=[
                html.Div(id="status", className="quote", role="status"),
                html.Div(
                    className="grid",
                    children=[
                        card(dcc.Loading(dcc.Graph(id="surface", className="chart", config={"scrollZoom": True}))),
                        card(
                            dcc.RadioItems(
                                [{"label": "By delta", "value": "delta"}, {"label": "By strike", "value": "strike"}],
                                "delta",
                                id="heatmap-axis",
                                inline=True,
                            ),
                            dcc.RadioItems(
                                [{"label": "Market", "value": "market"}, {"label": "eSSVI fit", "value": "fit"}],
                                "market",
                                id="heatmap-source",
                                inline=True,
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
                                0, 0, 1, value=0, id="maturity", marks={}, included=False, allow_direct_input=False
                            ),
                        ),
                        card(
                            dcc.RadioItems(
                                [{"label": "Market", "value": "market"}, {"label": "eSSVI fit", "value": "fit"}],
                                "market",
                                id="variance-source",
                                inline=True,
                            ),
                            dcc.Graph(id="total-variance", className="chart"),
                        ),
                    ],
                ),
                card(
                    html.H2("Quotes furthest from the eSSVI fit", className="card-title"),
                    dag.AgGrid(
                        id="residuals",
                        rowData=[],
                        columnDefs=[
                            {"field": "maturity", "headerName": "Expiry"},
                            {"field": "type", "headerName": "Type"},
                            {"field": "strike", "headerName": "Strike"},
                            {"field": "implied_vol", "headerName": "Market vol", "valueFormatter": PERCENT},
                            {"field": "fit_vol", "headerName": "Fit vol", "valueFormatter": PERCENT},
                            {"field": "error", "headerName": "Fit - market", "valueFormatter": PERCENT},
                            {
                                "field": "spreads",
                                "headerName": "In half-spreads",
                                "valueFormatter": {
                                    "function": "params.value == null ? '' : d3.format('+.1f')(params.value)"
                                },
                            },
                        ],
                        defaultColDef={"sortable": True, "filter": True, "flex": 1},
                        dashGridOptions={"pagination": True, "paginationPageSize": 10, "domLayout": "autoHeight"},
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
                        dashGridOptions={"pagination": True, "paginationPageSize": 10, "domLayout": "autoHeight"},
                    ),
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
                        dashGridOptions={"pagination": True, "paginationPageSize": 20, "domLayout": "autoHeight"},
                    )
                ),
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
        ticker,
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


if __name__ == "__main__":
    app.run()
