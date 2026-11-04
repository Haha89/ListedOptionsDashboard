"""Dash dashboard of listed options: greek surface, smile per maturity, implied vol heatmap and a table.
Run with `uv run python app.py` and open http://127.0.0.1:8050 (HOST, PORT and DASH_DEBUG env vars apply)."""

import math
import time
from datetime import date

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots

from src.option_vol import black_scholes as bs
from src.option_vol.models import Environment
from src.option_vol.scrapping import Scrapping
from src.option_vol.utils import to_title

FIELDS = ["implied_vol", "price", "delta", "gamma", "vega", "theta", "rho"]
COLUMNS = ["name", "strike", "maturity", *FIELDS]
DELTAS = (5, 10, 25, 40)
ATM = "ATM (fwd)"
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
MARGIN = dict(l=50, r=10, b=40, t=40)
# shortcut: per-process, never evicted; fine for one user, use flask-caching to share between gunicorn workers
_cache: dict = {}


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
        df["div_yield"] = implied_yield(df)
        df["forward"] = df.spot * np.exp((df.r - df.div_yield) * df["T"])
        market = dict(S=df.spot, K=df.strike, T=df["T"], r=df.r, is_call=df.type == "Call", q=df.div_yield)
        df["implied_vol"] = bs.implied_vol(df.price, **market)
        df = df.assign(**bs.greeks(vol=df.implied_vol, **market))
        hit = _cache[ticker] = (time.monotonic(), max_expiry, df)
    df = hit[2]
    return df[df.maturity <= max_expiry]


def implied_yield(df: pd.DataFrame) -> pd.Series:
    """Per expiry, the continuous yield q making the model forward S e^((r-q)T) equal the market's, read from
    put-call parity F = K + (C - P) e^(rT) at the 3 strikes nearest spot (median). Without it calls and puts of one
    strike give different vols; it absorbs dividends, borrow costs and a spot taken at another time than the quotes.
    0 for expiries with no call/put pair at a strike."""
    # shortcut: prices are bid/ask mids or, without a two-sided quote, last trades that can be stale; restrict to
    # two-sided quotes (keep bid/ask in Scrapping) if the forward looks off on illiquid names
    calls = df[df.type == "Call"].set_index(["maturity", "strike"])
    puts = df[df.type == "Put"].set_index(["maturity", "strike"]).price
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


def otm(df: pd.DataFrame) -> pd.DataFrame:
    """One implied vol per strike from the liquid side: puts below the forward, calls at and above it"""
    side = ((df.type == "Put") & (df.strike < df.forward)) | ((df.type == "Call") & (df.strike >= df.forward))
    return df[side & df.implied_vol.notna()].sort_values(["maturity", "strike"])


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


def surface(df: pd.DataFrame, field: str) -> go.Figure:
    pivot = by_strike(df, field)
    label = to_title(field)
    fig = go.Figure(
        go.Surface(
            x=pd.to_datetime(pivot.columns),
            y=pivot.index,
            z=pivot.to_numpy(),
            colorscale="Portland",
            showscale=False,
            hovertemplate=f"Strike %{{y:.2f}}<br>Expiry %{{x|%Y-%m-%d}}<br>{label} %{{z:.{2 if field == 'price' else 4}f}}"
            "<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="keep",
        scene=dict(xaxis_title="Expiry", yaxis_title="Strike", zaxis_title=label),
    )
    return fig


def smile(df: pd.DataFrame, expiry: date, spot: float) -> go.Figure:
    """Implied vol against strike for one expiry: puts below the forward, calls above, the out-of-the-money side"""
    chain = otm(df[df.maturity == expiry])
    fig = go.Figure(
        [
            go.Scatter(
                x=chain[chain.type == kind].strike,
                y=chain[chain.type == kind].implied_vol,
                mode="lines+markers",
                name=f"OTM {kind.lower()}s",
                hovertemplate="Strike %{x:.2f}<br>Implied vol %{y:.2%}<extra></extra>",
            )
            for kind in ("Put", "Call")
        ]
    )
    fig.add_vline(x=spot, line_dash="dot", annotation_text=f"spot {spot:.2f}")
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
            marker=dict(size=12, symbol="diamond", color="black"),
            hovertemplate="ATM, strike = forward %{x:.2f}<br>Implied vol %{y:.2%}<extra></extra>",
        )
    fig.update_layout(
        title=f"Smile {expiry:%Y-%m-%d}",
        margin=MARGIN,
        xaxis_title="Strike",
        yaxis=dict(title="Implied vol", tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
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
        x=[f"{e:%Y-%m-%d}" for e in grid.columns],
        y=grid.index,
        z=grid.to_numpy(),
        colorscale="Portland",
        texttemplate="%{z:.1%}" if labelled else None,
        colorbar=dict(tickformat=".0%", **colorbar),
        hovertemplate="%{y}<br>Expiry %{x}<br>Implied vol %{z:.2%}<extra></extra>",
    )


def heatmap(grid: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(vol_cells(grid))
    fig.update_layout(title=title, margin=MARGIN, xaxis=dict(type="category", title="Expiry"))
    return fig


def delta_heatmap(df: pd.DataFrame) -> go.Figure:
    """Vols by delta on top; risk reversals and butterflies below, on their own scale centred on 0"""
    grid = by_delta(df)
    spreads = rr_bf(grid)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(vol_cells(grid, y=0.65, len=0.7), row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=[f"{e:%Y-%m-%d}" for e in spreads.columns],
            y=spreads.index,
            z=spreads.to_numpy(),
            colorscale="RdBu",
            zmid=0,
            texttemplate="%{z:+.1%}",
            colorbar=dict(tickformat="+.0%", y=0.14, len=0.3),
            hovertemplate="%{y}<br>Expiry %{x}<br>%{z:+.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(type="category")
    fig.update_xaxes(title="Expiry", row=2, col=1)
    fig.update_layout(title="Implied vol by delta, risk reversals and butterflies", margin=MARGIN)
    return fig


def slider_marks(dates: list) -> dict:
    step = math.ceil(len(dates) / 8)  # about 8 labels whatever the number of expiries
    return {i: f"{d:%d %b %y}" if i % step == 0 else "" for i, d in enumerate(dates)}


def dropdown(component_id, options, value):
    return dcc.Dropdown(options, value, id=component_id, clearable=False)


def card(*children, class_name="card"):
    return html.Div(className=class_name, children=list(children))


app = Dash(__name__, title="Listed options dashboard")
app.layout = html.Div(
    className="page",
    children=[
        dcc.Store(id="loaded"),
        html.Div(
            className="controls",
            children=[
                html.H2("Listed options"),
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
                dropdown("field", [{"label": to_title(f), "value": f} for f in FIELDS], "implied_vol"),
                html.Label("Strike range around spot", htmlFor="range"),
                dropdown("range", [{"label": f"± {p}%", "value": p} for p in (5, 10, 15, 25, 50)], 15),
                html.Label("Max maturity", htmlFor="months"),
                dropdown("months", [{"label": f"{m}M", "value": m} for m in (1, 2, 3, 6, 9, 12, 24, 36)], 6),
                html.P(id="status", role="status"),
            ],
        ),
        html.Div(
            className="results",
            children=[
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
                            dcc.Graph(id="heatmap", className="chart"),
                        ),
                    ],
                ),
                card(
                    dcc.Graph(id="smile", className="chart"),
                    html.Label("Maturity", htmlFor="maturity"),
                    dcc.Slider(0, 0, 1, value=0, id="maturity", marks={}, included=False),
                ),
                card(
                    dag.AgGrid(
                        id="table",
                        rowData=[],
                        columnDefs=[
                            {"field": c, "headerName": to_title(c)}
                            | ({} if c in ("name", "maturity") else {"filter": "agNumberColumnFilter"})
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
        return go.Figure(), [], f"Could not load {ticker}: {error}", None, 0, {}, 0
    loaded = dict(ticker=ticker, months=months, range=strike_range, type=option_type)
    dates = expiries(df)
    slider = (len(dates) - 1, slider_marks(dates), min(maturity_index or 0, len(dates) - 1))
    view = select(df, option_type, strike_range)
    spot = df.spot.iloc[0]
    if view.empty:
        return (
            go.Figure(),
            [],
            f"{ticker} spot {spot:.2f}: no {option_type.lower()}s within ± {strike_range}%",
            loaded,
            *slider,
        )
    status = (
        f"{ticker} spot {spot:.2f} · rate {df.r.iloc[0]:.2%} · {len(view)} {option_type.lower()}s · "
        f"{view.maturity.nunique()} expiries up to {view.maturity.max():%Y-%m-%d}"
    )
    return (
        surface(vol_view(df, strike_range) if field == "implied_vol" else view, field),
        view[COLUMNS].round(5).to_dict("records"),
        status,
        loaded,
        *slider,
    )


@app.callback(Output("smile", "figure"), Input("maturity", "value"), Input("loaded", "data"))
def update_smile(maturity_index, loaded):
    if not loaded:
        return go.Figure()
    df = load_options(loaded["ticker"], loaded["months"])
    dates = expiries(df)  # the slider indexes the same list
    return smile(in_range(df, loaded["range"]), dates[min(maturity_index, len(dates) - 1)], df.spot.iloc[0])


@app.callback(Output("heatmap", "figure"), Input("heatmap-axis", "value"), Input("loaded", "data"))
def update_heatmap(axis, loaded):
    if not loaded:
        return go.Figure()
    df = load_options(loaded["ticker"], loaded["months"])
    if axis == "delta":
        return delta_heatmap(df)
    return heatmap(by_strike(vol_view(df, loaded["range"]), "implied_vol"), "Implied vol by strike (OTM)")


if __name__ == "__main__":
    app.run()
