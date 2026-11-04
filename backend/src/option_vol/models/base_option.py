from datetime import date, datetime
from math import log, sqrt, exp

import pytz
from scipy.stats import norm

from src.option_vol.models import Environment
from src.option_vol.utils import display_options

newYorkTz = pytz.timezone("America/New_York")


class BaseOption:

    def __init__(self, strike: float, maturity: date, underlying: str, price=None):
        self.type = self.__class__.__name__
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity
        self.price = price

        self.name = self.get_name()
        self.implied_vol = None
        self.T = (self.maturity - datetime.now(newYorkTz).date()).days / 365  # calendar days -> years
        self.r = Environment().risk_free_rate
        self.spot = Environment().get_spot(self.underlying)
        self.div_yield = 0 #Environment().get_div_yield(self.underlying)
        self.sqrt_t = sqrt(self.T)
        self.exp_rt = exp(- self.r * self.T)
        self.exp_div = exp(- self.div_yield * self.T)

        self.cache_d1 = {}

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

    def price_with_black_scholes(self, vol: float) -> float:
        """ Placeholder, """
        print("This function should not be called directly")
        raise NotImplementedError("BaseOption doesnt have a price")

    def get_d1(self, vol) -> float:
        if vol not in self.cache_d1:
            self.cache_d1[vol] = (log(self.spot / self.strike) + self.T * (self.r - self.div_yield + .5 * vol ** 2)) \
                                 / (vol * self.sqrt_t)
        return self.cache_d1[vol]

    def get_d2(self, vol) -> float:
        return self.get_d1(vol) - vol * self.sqrt_t

    def get_d1_and_d2(self, vol) -> (float, float):
        d1 = self.get_d1(vol=vol)
        d2 = d1 - vol * self.sqrt_t
        return d1, d2

    def get_gamma(self) -> float:
        return self.exp_div * norm.pdf(self.get_d1(vol=self.implied_vol)) / (
                self.spot * self.implied_vol * self.sqrt_t)

    def get_vega(self, vol=None) -> float:
        """ calculate the vega of an option using the Black-Scholes model, given the volatility.
        It corresponds to the rate of change of the price of an option with respect to the volatility """
        v = vol or self.implied_vol
        return self.spot * self.sqrt_t * self.exp_div * norm.pdf(self.get_d1(vol=v))

    def set_implied_volatility(self) -> float:
        if not self.implied_vol:
            self.implied_vol = self.solve_implied_vol(self.price)
        return self.implied_vol

    def solve_implied_vol(self, target_price, max_iter=100, tol=1e-8, low=1e-4, high=5.0) -> float:
        """ Find the volatility whose Black-Scholes price equals target_price, or nan if no vol in [low, high] does.
        Newton-Raphson steps, falling back to bisection when a step leaves the bracket: the price is increasing
        in vol, so the bracket always holds the root. """
        if not self.price_with_black_scholes(low) <= target_price <= self.price_with_black_scholes(high):
            return float("nan")
        vol = 0.2
        for _ in range(max_iter):
            diff = self.price_with_black_scholes(vol) - target_price
            if abs(diff) < tol:
                break
            if diff > 0:
                high = vol
            else:
                low = vol
            vega = self.get_vega(vol=vol)
            vol = vol - diff / vega if vega > 0 else low
            if not low < vol < high:
                vol = (low + high) / 2
        return vol

    def set_greeks(self):
        """ calculate greeks values (sensitivity of an option value to underlying price,
         volatility, time and interest rate) of the option only once and store it as a object attribute for future use """
        for g in ["delta", "gamma", "theta", "rho", "vega"]:  # Greeks
            if not getattr(self, f"{g}"):
                greek = getattr(self, f"get_{g}")()
                setattr(self, g, greek)

    def find_price(self):
        listed_options = Environment().get_listed_options(self.underlying)
        if found_options := list(filter(lambda x: x == self, listed_options)):  # Could use the name also
            self.price = found_options[0].price
            return self.price
        return 0