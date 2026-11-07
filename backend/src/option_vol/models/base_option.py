import math
from datetime import date, datetime, time, timedelta
from functools import cache
from zoneinfo import ZoneInfo

import numpy as np
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

from src.option_vol import black_scholes as bs
from src.option_vol.models import Environment
from src.option_vol.utils import display_options

NEW_YORK = ZoneInfo("America/New_York")
CLOSE, OPEN = time(16), time(9, 30)  # US listed options expire at the close; AM-settled index monthlies at the open
TRADING_DAYS = 252  # sessions per year
SESSION = 6.5 * 3600  # seconds, 09:30 to 16:00


class NYSEHolidays(AbstractHolidayCalendar):
    """Days the NYSE is closed. shortcut: early closes (13:00 after Thanksgiving, on Christmas Eve) count as full
    sessions, and one-off closures (national days of mourning) are missing; add them if a front expiry falls there"""

    rules = [  # noqa: RUF012  (pandas reads rules as a class attribute)
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),  # Saturday: no Friday holiday
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@cache
def _holidays() -> np.ndarray:
    return NYSEHolidays().holidays("2000-01-01", "2060-12-31").to_numpy().astype("datetime64[D]")


def _session_part(day: date, start: datetime, end: datetime) -> float:
    """Fraction of day's NYSE session between start and end"""
    if not np.is_busday(day, holidays=_holidays()):
        return 0.0
    opened = max(datetime.combine(day, OPEN, NEW_YORK), start)
    closed = min(datetime.combine(day, CLOSE, NEW_YORK), end)
    return max((closed - opened).total_seconds(), 0.0) / SESSION


def year_fraction(maturity: date, expires_at: time = CLOSE, now: datetime | None = None) -> float:
    """Trading years from now to maturity at expires_at: the NYSE session time in between, over 252 sessions a year.
    Variance accrues while the market trades, so a Monday expiry seen on Friday evening holds one session, not three
    calendar days: in calendar time its vol would read about 30% low."""
    now = now or datetime.now(NEW_YORK)
    expiry = datetime.combine(maturity, expires_at, NEW_YORK)
    sessions = _session_part(now.date(), now, expiry)
    if maturity > now.date():
        between = np.busday_count(now.date() + timedelta(days=1), maturity, holidays=_holidays())
        sessions += between + _session_part(maturity, now, expiry)
    # shortcut: AM-settled options seen after the last close before settlement have no session left; count one hour so
    # vols stay finite. Weight overnight variance properly if those options matter.
    return max(sessions, 1 / 6.5) / TRADING_DAYS


class BaseOption:
    """A listed European option; Call and Put only set is_call. The math lives in black_scholes."""

    is_call: bool

    def __init__(self, strike: float, maturity: date, underlying: str, price=None, expires_at: time = CLOSE):
        self.type = self.__class__.__name__
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity
        self.price = price

        self.name = self.get_name()
        self.implied_vol = None
        self.expires_at = expires_at
        # shortcut: one clock, trading time, also discounts at r; vs calendar time the gap is a few days of interest
        # over the option's life, which the implied dividend yield (app.implied_yield) absorbs into the forward
        self.T = year_fraction(maturity, expires_at)
        self.bid = self.ask = math.nan  # nan when there is no two-sided quote
        self.volume = self.open_interest = 0
        self.r = Environment().get_risk_free_rate()
        self.spot = Environment().get_spot(self.underlying)
        self.div_yield = 0  # Environment().get_div_yield(self.underlying)

        for g in ["delta", "gamma", "theta", "vega", "rho"]:  # Greeks
            self.__setattr__(g, 0)

    def __eq__(self, other):
        # Two options are equal if same type, strike, maturity and underlying
        return (
            self.type == other.type
            and self.strike == other.strike
            and self.maturity == other.maturity
            and self.underlying == other.underlying
        )

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return str(display_options([self]))

    def get_name(self):
        und, mat, typ = self.underlying, self.maturity.strftime("%y%m%d"), self.type[0]
        return f"{und}{mat}{typ}{self.strike:09.3f}".replace(".", "")

    def _market(self):
        return dict(S=self.spot, K=self.strike, T=self.T, r=self.r, is_call=self.is_call, q=self.div_yield)

    def price_with_black_scholes(self, vol: float) -> float:
        return float(bs.price(vol=vol, **self._market()))

    def get_vega(self, vol=None) -> float:
        return float(bs.greeks(vol=vol or self.implied_vol, **self._market())["vega"])

    def set_implied_volatility(self) -> float:
        if not self.implied_vol:
            self.implied_vol = self.solve_implied_vol(self.price)
        return self.implied_vol

    def solve_implied_vol(self, target_price) -> float:
        """Vol whose Black-Scholes price equals target_price, nan if none does"""
        return float(bs.implied_vol(target_price, **self._market()))

    def set_greeks(self):
        """Store delta, gamma, vega, theta and rho at the implied vol as attributes"""
        for greek, value in bs.greeks(vol=self.implied_vol, **self._market()).items():
            setattr(self, greek, float(value))

    def find_price(self):
        listed_options = Environment().get_listed_options(self.underlying)
        if found_options := list(filter(lambda x: x == self, listed_options)):  # Could use the name also
            self.price = found_options[0].price
            return self.price
        return 0
