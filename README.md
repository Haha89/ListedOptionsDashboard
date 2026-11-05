# ListedOptionDashboard
A [Dash](https://dash.plotly.com/) dashboard to visualize the implied volatility and Greek surfaces of listed options.

This repo contains a library to price European calls and puts with the Black-Scholes model. It retrieves listed options of any underlying from Yahoo Finance using [yfinance](https://github.com/ranaroussi/yfinance).
Implied volatility is computed from market prices (bid/ask mid, or last trade when there is no two-sided quote) and the greeks are computed at that vol.
An arbitrage-free eSSVI surface is fitted to the out-of-the-money vols, and the chain is scanned for static arbitrage at bid/ask.

# Getting Started

## Prerequisites
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.15, pinned in `backend/.python-version`), or Docker

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

On Windows, this writes such a bundle (certifi's roots plus the Windows store, which holds the antivirus root):
```
cd backend
uv run python -c "import ssl, certifi; c = ssl.create_default_context(); c.load_default_certs(); open('ca.pem', 'w').write(open(certifi.where()).read() + ''.join(ssl.DER_cert_to_PEM_cert(d) for d in c.get_ca_certs(binary_form=True)))"
```
then run the app with `CURL_CA_BUNDLE=ca.pem` (bash) or `$env:CURL_CA_BUNDLE = "ca.pem"` (PowerShell).

# Conventions
Time to maturity in calendar years (seconds to expiry at 16:00 New York / 365 days), vega per 1.00 of vol, theta per calendar day, rho per 1% move in rate.

# Built With
- Dash and Plotly: dashboard and surface plots
- NumPy and SciPy: vectorised Black-Scholes pricing and implied vol
- yfinance: listed option chains and spot prices

# License
This project is licensed under the MIT License - see the LICENSE.md file for details

# Next steps
See [ROADMAP.md](ROADMAP.md).
