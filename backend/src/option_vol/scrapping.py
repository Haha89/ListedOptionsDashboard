from concurrent.futures import ThreadPoolExecutor
from datetime import date

from yfinance import Ticker

from src.option_vol.models import BaseOption, Call, Put
from src.option_vol.utils import YAHOO_MAPPING


class Scrapping:
    """Listed options from Yahoo Finance"""

    def parse_option(self, underlying) -> list[BaseOption]:
        """Listed options of the nearest expiry"""
        return self.parse_underlying_until(underlying, date.today())

    @staticmethod
    def parse_underlying_until(underlying, expiry_date: date) -> list[BaseOption]:
        """Listed calls and puts expiring after today and up to expiry_date, with at least the nearest expiry.
        Price is the bid/ask mid, or the last trade when there is no two-sided quote (e.g. market closed)."""
        symbol = YAHOO_MAPPING.get(underlying, underlying)
        expiries = [e for e in map(date.fromisoformat, Ticker(symbol).options) if e > date.today()]
        expiries = [e for e in expiries if e <= expiry_date] or expiries[:1]
        with ThreadPoolExecutor(8) as pool:
            chains = list(pool.map(lambda e: Ticker(symbol).option_chain(e.isoformat()), expiries))

        options = []
        for expiry, chain in zip(expiries, chains):
            for cls, quotes in ((Call, chain.calls), (Put, chain.puts)):
                for row in quotes.itertuples():
                    price = (row.bid + row.ask) / 2 if row.bid > 0 and row.ask > 0 else row.lastPrice
                    if price > 0:  # also drops nan
                        options.append(cls(float(row.strike), expiry, underlying, float(price)))
        return options


if __name__ == "__main__":
    from datetime import timedelta

    print(Scrapping.parse_underlying_until("TSLA", date.today() + timedelta(days=30))[:5])
