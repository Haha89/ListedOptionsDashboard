""" Black-Scholes prices, greeks and implied vol. Every argument can be a scalar or a numpy array (they broadcast).
Conventions: T in years, r and q continuous rates, vega per 1.00 of vol, theta per calendar day, rho per 1% of rate. """
import numpy as np
from scipy.special import ndtr  # standard normal cdf


def _d1_d2(S, K, T, r, vol, q):
    vol_t = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + vol ** 2 / 2) * T) / vol_t
    return d1, d1 - vol_t


def _pdf(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def price(S, K, T, r, vol, is_call, q=0.0):
    d1, d2 = _d1_d2(S, K, T, r, vol, q)
    w = np.where(is_call, 1.0, -1.0)
    return w * (S * np.exp(-q * T) * ndtr(w * d1) - K * np.exp(-r * T) * ndtr(w * d2))


def greeks(S, K, T, r, vol, is_call, q=0.0) -> dict:
    d1, d2 = _d1_d2(S, K, T, r, vol, q)
    w = np.where(is_call, 1.0, -1.0)
    disc_q, disc_r, pdf, sqrt_t = np.exp(-q * T), np.exp(-r * T), _pdf(d1), np.sqrt(T)
    return {
        "delta": w * disc_q * ndtr(w * d1),
        "gamma": disc_q * pdf / (S * vol * sqrt_t),
        "vega": S * disc_q * pdf * sqrt_t,
        "theta": (-S * disc_q * pdf * vol / (2 * sqrt_t) - w * r * K * disc_r * ndtr(w * d2)
                  + w * q * S * disc_q * ndtr(w * d1)) / 365,
        "rho": w * K * T * disc_r * ndtr(w * d2) / 100,
    }


def implied_vol(target, S, K, T, r, is_call, q=0.0, low=1e-4, high=5.0, iterations=60):
    """ Vol whose price equals target, by bisection (price increases with vol): 60 halvings of [low, high] reach
    machine precision. nan where no vol in [low, high] gives the target, e.g. a price below intrinsic value. """
    target, S, K, T, r, is_call, q = np.broadcast_arrays(target, S, K, T, r, is_call, q)
    lo, hi = np.full(target.shape, low), np.full(target.shape, high)
    attainable = (price(S, K, T, r, lo, is_call, q) <= target) & (target <= price(S, K, T, r, hi, is_call, q))
    for _ in range(iterations):
        mid = (lo + hi) / 2
        above = price(S, K, T, r, mid, is_call, q) > target
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return np.where(attainable, (lo + hi) / 2, np.nan)
