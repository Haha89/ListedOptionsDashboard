# ListedOptionDashboard
A [Dash](https://dash.plotly.com/) dashboard to visualize the implied volatility and Greek surfaces of listed options.

This repo contains a library to price European calls and puts with the Black-Scholes model. It retrieves listed options of any underlying from Yahoo Finance using [yfinance](https://github.com/ranaroussi/yfinance).
Implied volatility is computed from market prices (bid/ask mid, or last trade when there is no two-sided quote) and the greeks are computed at that vol.

# Getting Started

## Prerequisites
- [uv](https://docs.astral.sh/uv/), or Docker

## Running the app locally
```
cd backend
uv sync
uv run python app.py
```
The dashboard runs on http://127.0.0.1:8050. Set `DASH_DEBUG=1` for hot reload and Dash dev tools (local use only).

With Docker:
```
docker build -t options-dashboard backend
docker run -p 8050:8050 options-dashboard
```

## Tests
```
cd backend
uv run pytest
```

## Behind an HTTPS-scanning antivirus or proxy
If downloads fail with a certificate error, point curl (used by yfinance) at a CA bundle that includes your system's
root certificates: `CURL_CA_BUNDLE=<path to bundle.pem>`. For uv, set `UV_SYSTEM_CERTS=1`.

# Conventions
Time to maturity in calendar years (days / 365), vega per 1.00 of vol, theta per calendar day, rho per 1% move in rate.

# Built With
- Dash and Plotly: dashboard and surface plots
- NumPy and SciPy: vectorised Black-Scholes pricing and implied vol
- yfinance: listed option chains and spot prices

# License
This project is licensed under the MIT License - see the LICENSE.md file for details

# Next steps
See [ROADMAP.md](ROADMAP.md).
