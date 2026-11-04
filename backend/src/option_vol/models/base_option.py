from datetime import date, datetime

import pytz

from src.option_vol import black_scholes as bs
from src.option_vol.models import Environment
from src.option_vol.utils import display_options

newYorkTz = pytz.timezone("America/New_York")


class BaseOption:
    """ A listed European option; Call and Put only set is_call. The math lives in black_scholes. """
    is_call: bool

    def __init__(self, strike: float, maturity: date, underlying: str, price=None):
        self.type = self.__class__.__name__
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity
        self.price = price

        self.name = self.get_name()
        self.implied_vol = None
        self.T = (self.maturity - datetime.now(newYorkTz).date()).days / 365  # calendar days -> years
        self.r = Environment().get_risk_free_rate()
        self.spot = Environment().get_spot(self.underlying)
        self.div_yield = 0  # Environment().get_div_yield(self.underlying)

        for g in ["delta", "gamma", "theta", "vega", "rho"]:  # Greeks
            self.__setattr__(g, 0)

    def __eq__(self, other):
        # Two options are equal if same type, strike, maturity and underlying
        return self.type == other.type and self.strike == other.strike and \
            self.maturity == other.maturity and self.underlying == other.underlying

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return str(display_options([self]))

    def get_name(self):
        und, mat, typ = self.underlying, self.maturity.strftime('%y%m%d'), self.type[0]
        return f"{und}{mat}{typ}{self.strike:09.3f}".replace('.', '')

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
        """ Vol whose Black-Scholes price equals target_price, nan if none does """
        return float(bs.implied_vol(target_price, **self._market()))

    def set_greeks(self):
        """ Store delta, gamma, vega, theta and rho at the implied vol as attributes """
        for greek, value in bs.greeks(vol=self.implied_vol, **self._market()).items():
            setattr(self, greek, float(value))

    def find_price(self):
        listed_options = Environment().get_listed_options(self.underlying)
        if found_options := list(filter(lambda x: x == self, listed_options)):  # Could use the name also
            self.price = found_options[0].price
            return self.price
        return 0
