import math
from datetime import date

import numpy as np

from src.option_vol.models import Call


def price_call(option: Call, vol: float, periods=2, div_yield=0.0) -> float:
    """Price a European call on a Cox-Ross-Rubinstein tree, using the option's spot, rate and maturity"""
    r = option.r
    dt = option.T / periods
    u = math.exp(vol * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - div_yield) * dt) - d) / (u - d)  # risk-neutral up probability
    discount_factor = math.exp(-r * dt)

    # terminal spots from the top node (all ups) to the bottom one (all downs), then roll back one step at a time
    spots = option.spot * u ** np.arange(periods, -periods - 1, -2)
    values = np.maximum(spots - option.strike, 0)
    for _ in range(periods):
        values = discount_factor * (p * values[:-1] + (1 - p) * values[1:])
    return float(values[0])


if __name__ == "__main__":
    call = Call(3800, date(2027, 3, 19), "SPX")
    print(price_call(call, vol=0.1961, periods=500, div_yield=0.0163))
