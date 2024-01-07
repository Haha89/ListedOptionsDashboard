from datetime import date

from yfinance import download, Ticker

from src.option_vol.utils import YAHOO_MAPPING


class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance


class Environment(Singleton):
    spots, div_yield, vol = {}, {}, {}
    risk_free_rate = 0
    listed_options = {}

    def get_spot(self, underlying):
        return self._query(underlying, "spots")

    def get_div_yield(self, underlying):
        return self._query(underlying, "div_yield")

    def get_listed_options(self, underlying):
        return self._query(underlying, "listed_options")

    def _query(self, underlying, shelf):
        """ Retrieves market data of 3 types: spots, div_yields and listed options. Results are cached for later use """
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
    def get_dividend_yield(underlying):
        divs = Ticker(underlying).dividends
        if divs.empty:
            return 0

        last10y = divs[divs.index.year > (date.today().year - 10)]
        last10y = last10y.groupby(last10y.index.year).sum()

        tot_div = divs[[x >= date(2022, 1, 1) for x in divs.index]].sum()
        last_spot_year = \
            download(tickers=underlying, start=f"{last10y.index[0]}-12-31", end=f"{last10y.index[-1]}-12-31",
                     interval="1m")["Adj Close"][0]
        return tot_div / last_spot_year

    @staticmethod
    def _get_spot(underlying):
        return download(tickers=YAHOO_MAPPING.get(underlying, underlying), period='1d', interval='1d')["Adj Close"].iloc[0]

    def get_historical_spots(self, underlyings):
        return download(tickers=[YAHOO_MAPPING.get(u, u) for u in underlyings], period='1y', interval='1d')

    def get_historical_volatility(self, underlying):
        prices = self.get_historical_spots([underlying])
        print(prices)
