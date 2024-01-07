from typing import List

import pandas as pd
from sqlalchemy.orm import Session

from .models.option import OptionORM
from .models.surface import Surface


class ExplorerService:

    def get_options(self, session: Session, ticker: str, refresh=False) -> List[OptionORM]:
        options = OptionORM.get_by_underlying(session, ticker)
        if options.first() and not refresh:
            return options.all()

        print("No option found; will fetch them and price them")
        from src.option_vol.scrapping import Scrapping
        listed_options = Scrapping().parse_option(ticker)
        option_db = []
        for o in listed_options:
            o.set_implied_volatility()
            o.set_greeks()
            option_db.append(OptionORM.from_option(o))

        OptionORM.remove_by_underlying(session, ticker)
        session.add_all(option_db)
        session.commit()
        # store those listed options in OptionORM
        return option_db

    def get_surface(self, session: Session, ticker: str, option_type: str, field: str, depth: int) -> Surface:
        options = OptionORM.get_options_near_spot(session, ticker, option_type, depth).all()
        df = pd.DataFrame([vars(o) for o in options], columns=["strike", "maturity", field])
        pivot = df.pivot_table(index="strike", values=field, columns="maturity", aggfunc='first')
        pivot = pivot.interpolate(method='polynomial', order=2).ffill().bfill()
        s = dict(x=pivot.columns.to_list(), y=pivot.index.to_list(), z=pivot.values.tolist())
        return Surface(**s)
