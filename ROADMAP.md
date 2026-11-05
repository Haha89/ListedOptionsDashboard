# Roadmap

In order: each step relies on the ones before.

## 1. Quote quality: done
- `Scrapping` keeps bid, ask, volume and open interest. A quote is usable when two-sided with a spread of at most
  half the mid; the forward, the fit and the arbitrage checks use usable quotes only (last trades when the market is
  closed). Excluded quotes show as hollow markers on the smile.
- T runs to 16:00 New York, 09:30 for third-Friday SPX / NDX / RUT monthlies. Where one date lists an AM root and a
  PM one (SPX / SPXW), the PM contracts are kept.
- Not done: excluding quotes by trade age (`lastTradeDate`). Long-dated options with tight live quotes often go days
  without a trade, and the spread rule already drops minimum-tick quotes.

## 2. Static arbitrage checks at bid/ask: done
`src/option_vol/arbitrage.py`: bounds, verticals, butterflies (unequal gaps), boxes (put-call parity across
strikes) and calendars in total variance, profit after crossing the spread; a table in $ per contract, crosses on
the smile and on the σ²T chart.
- Yahoo quotes are snapshots of each contract taken at different times, and deep in-the-money ones are often hours
  stale. They made most violations (SPX: 418 flags, down to 33 without them), so only out-of-the-money strikes and
  strikes within 2% of the forward are scanned. A violation is a quote to check, not a guaranteed trade.
- Next: a data source with quote timestamps, or keep a violation only when it persists over several snapshots.
- Butterflies and boxes assume European exercise; listed equity options are American.

## 3. Arbitrage-free smoothed surface: done (eSSVI)
`src/option_vol/ssvi.py`: eSSVI (Hendriks-Martini 2019), SSVI with its own rho per expiry. Expiries are fitted from
the shortest, each free of butterfly arbitrage (Gatheral-Jacquier theorem 4.2) and of calendar arbitrage against
the one before. The calendar condition is derived in the module docstring and checked against brute force in
tests. theta is the market ATM total variance; errors are weighted by (vega / half-spread)^2.
- Dashboard: fitted curve and bid/ask bars on the smile, fitted surface with market dots, Market / fit toggles on
  the heatmaps and the σ²T chart, residuals table. One fit per snapshot, shared by every chart.
- Fit error: 0.1-0.4% of vol from about two weeks out; 1-3% on 1-to-5-day expiries, where 2 free parameters per
  expiry are not enough to bend through the smile.
- Next: let theta float per expiry, or SVI (5 parameters) for the front expiries; interpolate between expiries in
  total variance for a continuous surface.

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
- Rate curve (Treasury) instead of a single 13-week T-bill rate.
- Shared cache (flask-caching / Redis) if the dashboard is served to several users.
