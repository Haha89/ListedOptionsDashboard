import math
from datetime import date

from src.option_vol.models import Call, Environment

env = Environment()
env.risk_free_rate = 0.0353
env.div_yield = {"SPX": 0.0163}
env.vol = {"SPX": 0.1961}


def price_call(option: Call, periods=2):
    r = env.risk_free_rate
    vol = env.vol[option.underlying]
    div_yield = env.div_yield[option.underlying]

    dt = option.T / periods
    u = math.exp(vol * math.sqrt(dt))
    d = 1 / u

    a = math.exp((r - div_yield) * dt)
    p = (a - d) / (u - d)
    discount_factor = math.exp(-(r - div_yield) * math.sqrt(dt))

    def price_step(step, spot):
        if step == periods:
            return max(0, spot - option.strike)
        else:
            return discount_factor * (p * price_step(step + 1, spot * u) + (1 - p) * price_step(step + 1, spot * d))

    return price_step(0, env.spots[option.underlying])


if __name__ == "__main__":
    print(env.get_spot("SPX"))
    call = Call(3800, date(2024, 3, 15), "SPX")
    print(price_call(call, periods=3))
