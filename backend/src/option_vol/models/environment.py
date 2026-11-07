import math

import pandas as pd
from yfinance import Ticker, download

from src.option_vol.utils import YAHOO_MAPPING


class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
        return cls.__instance


class Environment(Singleton):
    spots, div_yield, vol = {}, {}, {}
    risk_free_rate = None
    listed_options = {}  # noqa: RUF012  (class-level caches, shared by the singleton like spots above)

    def get_risk_free_rate(self) -> float:
        """13-week US T-bill yield (^IRX, quoted in percent) as a continuous rate"""
        # shortcut: fetched once per process and one rate for all maturities; the implied dividend yield per expiry
        # (see app.implied_yield) absorbs the gap on the forward. Use a Treasury curve if discounting matters.
        if Environment.risk_free_rate is None:
            Environment.risk_free_rate = float(Ticker("^IRX").fast_info["lastPrice"]) / 100
        return Environment.risk_free_rate

    def get_spot(self, underlying):
        return self._query(underlying, "spots")

    def get_div_yield(self, underlying):
        return self._query(underlying, "div_yield")

    def get_listed_options(self, underlying):
        return self._query(underlying, "listed_options")

    def _query(self, underlying, shelf):
        """Retrieves market data of 3 types: spots, div_yields and listed options. Results are cached for later use"""
        content = getattr(self, shelf)
        if underlying not in content:
            match shelf:
                case "spots":
                    val = self._get_spot(underlying)
                case "div_yield":
                    val = self.get_dividend_yield(underlying)
                case "listed_options":
                    from src.option_vol.scrapping import Scrapping

                    val = Scrapping().parse_option(underlying=underlying)
                case _:
                    raise ValueError(f"Store {shelf} not found")
            content[underlying] = val
        return content[underlying]

    @staticmethod
    def get_dividend_yield(underlying) -> float:
        """Dividends paid over the last 365 days over spot; nan for indices, whose dividends Yahoo doesn't list"""
        if underlying in YAHOO_MAPPING:
            return math.nan
        divs = Ticker(underlying).dividends
        if divs.empty:
            return 0.0
        last_year = divs[divs.index >= divs.index[-1].now(divs.index.tz) - pd.Timedelta(days=365)]
        return float(last_year.sum()) / Environment().get_spot(underlying)

    @staticmethod
    def _get_spot(underlying):
        return float(Ticker(YAHOO_MAPPING.get(underlying, underlying)).fast_info["lastPrice"])

    def get_historical_spots(self, underlyings):
        return download(tickers=[YAHOO_MAPPING.get(u, u) for u in underlyings], period="1y", interval="1d")

    def get_historical_volatility(self, underlying):
        prices = self.get_historical_spots([underlying])
        print(prices)
