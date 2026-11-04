# Roadmap

In order: each step relies on the ones before.

## 1. Quote quality (prerequisite)
Today an option's price is the bid/ask mid, or the last trade when there is no two-sided quote. Stale quotes go
through: deep in-the-money SPX calls quoted around an index level of ~7700 when the index is at 7569 gave vols of
280%. Without this step, an arbitrage scan would mostly find bad data.
- Keep `bid`, `ask`, `lastTradeDate`, `volume` and `openInterest` from Yahoo (`Scrapping`).
- Flag quotes that are one-sided, too wide (spread / mid above a threshold) or not traded for N days, and leave them
  out of fits and scans.
- Implied forward (`implied_yield`) from two-sided quotes only.
- Intraday time to expiry: options expire at 16:00 New York time (AM-settled SPX monthlies at the open), not at
  midnight. With T counted in whole days, 1-day options get wrong vols and wrong implied yields.
- SPX: tell apart the SPX root (AM settlement) from SPXW (PM settlement) on the same expiry.

## 2. Static arbitrage checks on market quotes (model-free)
Check on executable prices (buy at the ask, sell at the bid), so a violation is a real opportunity before costs, not
an effect of the mid.
- **Bounds**: D·max(F − K, 0) ≤ C ≤ D·F, and the same for puts (D the discount factor, F the forward).
- **Vertical spreads**: call prices fall with strike, with slope between −D and 0 (rising for puts).
- **Butterflies**: call prices are convex in strike, which means a non-negative risk-neutral density
  (Breeden-Litzenberger). Use the formula for unequal strike gaps.
- **Calendar spreads**: at fixed moneyness k = ln(K/F), total variance w = σ²T does not fall with maturity.
- **Put-call parity** against the implied forward, an exact identity only for European options.

Dashboard: a table of violations (type, strikes, expiries, size in $ after the spread) and markers on the smile.
Caveats: listed US equity options are American (early exercise makes parity and calendar spreads on puts an
inequality, not an equality); Yahoo quotes are delayed about 15 minutes.

## 3. Arbitrage-free smoothed surface
Fit the out-of-the-money vols, weighted by vega or by the inverse of the spread:
- **SVI** per expiry (Gatheral): 5 parameters, constrained to have no butterfly arbitrage (Durrleman condition g(k) ≥ 0).
- **SSVI** over the whole surface (Gatheral-Jacquier 2014): free of calendar and butterfly arbitrage by construction
  under simple constraints on its parameters. Interpolates between expiries in total variance.

Then surface, heatmaps, ATM, RR and BF read from the fit rather than raw quotes, and the smile shows market points,
bid/ask bars and the fitted curve. A fit residuals table points to quotes that are off.

## 4. History
- Store each snapshot (DuckDB or SQLite file): spot, forward, rate, fitted parameters, ATM / RR / BF per expiry.
- Time series of ATM vol, 25Δ RR and 25Δ BF; term structure today against N days ago.
- Realized vs implied vol (close-to-close or Parkinson from yfinance history).

## 5. Models and products
- American options: implied vol by inverting a binomial tree with early exercise (`binomial/pricer.py`, to extend to
  puts and early exercise), at least for single stocks.
- Options on futures (e.g. heating oil HO): Black-76, and a data source other than Yahoo, which doesn't list them.
  `snippets/cme_settlements.py` fetches CME futures settlements.
- Dividends: discrete dividends instead of a continuous yield, for names that pay quarterly.

## 6. Engineering
- CI: run `uv run pytest` and a linter (ruff) instead of pylint over every file.
- Rate curve (Treasury) instead of a single 13-week T-bill rate.
- Shared cache (flask-caching / Redis) if the dashboard is served to several users.
