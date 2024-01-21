from datetime import date
from typing import List

import pandas as pd
from sqlalchemy.orm import Session

from .models.option import OptionORM
from .models.surface import Surface, SurfaceRequest


class ExplorerService:

    @staticmethod
    def get_options( session: Session, ticker: str, refresh=False, maturity: date = None) -> List[OptionORM]:
        options = OptionORM.get_by_underlying(session, ticker, maturity)
        if options.first() and not refresh:
            return options.all()

        print("No option found; will fetch them and price them")
        from src.option_vol.scrapping import Scrapping
        listed_options = Scrapping().parse_underlying_until(ticker, maturity or date.today())
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

    @staticmethod
    def get_surface(session: Session, request: SurfaceRequest) -> Surface:
        options = OptionORM.get_options_near_spot(session, request.ticker, request.option_type, request.range,
                                                  request.maturity).all()
        df = pd.DataFrame([vars(o) for o in options], columns=["strike", "maturity", request.field])
        pivot = df.pivot_table(index="strike", values=request.field, columns="maturity", aggfunc='first')
        pivot = pivot.interpolate(method='polynomial', order=2).ffill().bfill()
        s = dict(x=pivot.columns.to_list(), y=pivot.index.to_list(), z=pivot.values.tolist())
        return Surface(**s)
