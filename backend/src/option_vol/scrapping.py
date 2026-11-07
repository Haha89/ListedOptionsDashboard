from concurrent.futures import ThreadPoolExecutor
from datetime import date

from yfinance import Ticker

from src.option_vol.models import BaseOption, Call, Put
from src.option_vol.models.base_option import CLOSE, OPEN
from src.option_vol.utils import YAHOO_MAPPING

AM_SETTLED = {"SPX", "NDX", "RUT"}  # roots of index monthlies settled at the open; SPXW, NDXP, RUTW settle at the close


def settles_at_open(root: str, expiry: date) -> bool:
    """Only the third-Friday monthlies of AM_SETTLED roots; Yahoo also lists e.g. SPX quarterlies (PM) under SPX"""
    return root in AM_SETTLED and expiry.weekday() == 4 and 15 <= expiry.day <= 21


class Scrapping:
    """Listed options from Yahoo Finance"""

    def parse_option(self, underlying) -> list[BaseOption]:
        """Listed options of the nearest expiry"""
        return self.parse_underlying_until(underlying, date.today())

    @staticmethod
    def parse_underlying_until(underlying, expiry_date: date) -> list[BaseOption]:
        """Listed calls and puts expiring after today and up to expiry_date, with at least the nearest expiry.
        Price is the bid/ask mid, or the last trade when there is no two-sided quote (e.g. market closed); bid and ask
        stay nan then. Where one date lists both an AM-settled root (SPX) and a PM-settled one (SPXW), only the PM
        contracts are kept, so each expiry has one expiry time."""
        symbol = YAHOO_MAPPING.get(underlying, underlying)
        expiries = [e for e in map(date.fromisoformat, Ticker(symbol).options) if e > date.today()]
        expiries = [e for e in expiries if e <= expiry_date] or expiries[:1]
        with ThreadPoolExecutor(8) as pool:
            chains = list(pool.map(lambda e: Ticker(symbol).option_chain(e.isoformat()), expiries))

        options = []
        for expiry, chain in zip(expiries, chains):
            for cls, quotes in ((Call, chain.calls), (Put, chain.puts)):
                if quotes.empty:  # e.g. NDX 2026-12-03 puts; Yahoo's empty table has float columns, no .str
                    continue
                roots = quotes.contractSymbol.str.extract(r"^([A-Z]+)\d", expand=False)
                if roots.nunique() > 1:
                    quotes, roots = quotes[~roots.isin(AM_SETTLED)], roots[~roots.isin(AM_SETTLED)]
                for row, root in zip(quotes.itertuples(), roots, strict=True):
                    two_sided = row.bid > 0 and row.ask > 0
                    price = (row.bid + row.ask) / 2 if two_sided else row.lastPrice
                    if not price > 0:  # also drops nan
                        continue
                    expires_at = OPEN if settles_at_open(root, expiry) else CLOSE
                    option = cls(float(row.strike), expiry, underlying, float(price), expires_at)
                    if two_sided:
                        option.bid, option.ask = float(row.bid), float(row.ask)
                    option.volume = int(row.volume) if row.volume > 0 else 0  # Yahoo leaves nan for no trade today
                    option.open_interest = int(row.openInterest) if row.openInterest > 0 else 0
                    options.append(option)
        return options


if __name__ == "__main__":
    from datetime import timedelta

    print(Scrapping.parse_underlying_until("TSLA", date.today() + timedelta(days=30))[:5])
