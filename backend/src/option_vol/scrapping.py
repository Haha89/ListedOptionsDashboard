import calendar
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, datetime
from typing import List

import pandas as pd
from dateutil.utils import today

from src.option_vol.models import BaseOption, Call, Put
from src.option_vol.utils import is_float

URL_OPTION_CHAIN = "https://bigcharts.marketwatch.com/Options/OptionsChainMonthPartial"
INDEX_TABLE = 2


class Scrapping:

    def __init__(self):
        self.mat = datetime.today().date()

    def parse_option(self, underlying) -> List[BaseOption]:
        """
        Retrieves listed options for an underlying with their Strike and ask price
        :param underlying: string, name of the underlying
        :return: a list of BaseOption
        """
        return self.parse_options_per_month(underlying, today().month, today().year)

    def parse_underlying_until(self, underlying, expiry_date):
        from dateutil.rrule import rrule, MONTHLY
        dates = [dt for dt in rrule(MONTHLY, dtstart=today(), until=expiry_date)]
        pool = ThreadPoolExecutor(max(len(dates), 1))
        reqs = [pool.submit(self.parse_options_per_month, underlying, d.month, d.year) for d in dates]
        wait(reqs)
        found_options = []
        for o in reqs:
            found_options.extend(o.result())
        return found_options

    def parse_options_per_month(self, underlying, month, year):
        import requests

        found_options = []
        x = requests.post(f"{URL_OPTION_CHAIN}?symb={underlying}&month={calendar.month_name[month]}&year={year}")
        options = pd.read_html(f"<table>{x.text}</table>")[0]
        options = options[[c for c in options.columns if options[c].isin(["Ask", "StrikePrice"]).any()]]
        options.columns = ["call_ask", "strike", "put_ask"]

        def read_row(row):
            if "Expires " in str(row.call_ask):
                self.mat = datetime.strptime(row.call_ask.replace("Expires ", ""), "%B %d, %Y").date()

            elif is_float(row.call_ask) and (self.mat > date.today()):
                found_options.append(Call(float(row.strike), self.mat, underlying, float(row.call_ask)))
                found_options.append(Put(float(row.strike), self.mat, underlying, float(row.put_ask)))

        options.apply(read_row, axis=1)
        return found_options


if __name__ == "__main__":
    print(Scrapping().parse_underlying_until("TSLA", date(2024, 3, 19)))
