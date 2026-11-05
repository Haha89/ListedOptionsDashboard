"""eSSVI implied vol surface: SSVI (Gatheral & Jacquier 2014, "Arbitrage-free SVI volatility surfaces") with its own
correlation per expiry (Hendriks & Martini 2019, "The extended SSVI volatility surface").

Total implied variance w = vol^2 T of one expiry at log-moneyness k = ln(K/F):
    w(k) = theta / 2 * (1 + rho psi k / theta + sqrt((psi k / theta + rho)^2 + 1 - rho^2))
theta the ATM total variance, rho the skew, psi = theta phi the ATM slope scale.

Butterfly arbitrage: none in a slice when psi (1 + |rho|) <= 4 and psi^2 (1 + |rho|) <= 4 theta (Gatheral-Jacquier
theorem 4.2).

Calendar arbitrage: writing sqrt(x^2 + y^2) = max over unit (c, s) of (c x + s y),
    2 w(k) = max over c in [-1, 1] of a(c) + b(c) k,  a(c) = theta (1 + rho c + sqrt(1 - rho^2) sqrt(1 - c^2)),
                                                       b(c) = psi (rho + c)
so 2w is the support function of an ellipse in the (a, b) plane, and w2 >= w1 for every k exactly when ellipse 1 lies
inside ellipse 2 extended to a = -infinity: the b-range of slice 2 covers that of slice 1 (psi2 (1 +/- rho2) >=
psi1 (1 +/- rho1), the wing slopes), and over that b-range the right edge a(b) of slice 2 is at or right of slice 1's."""

import numpy as np
from scipy.optimize import minimize

C = np.cos(np.linspace(0, np.pi, 101))  # where the calendar condition is checked, denser near the wings


def total_variance(k, theta, rho, psi):
    x = psi * np.asarray(k) / theta
    return theta / 2 * (1 + rho * x + np.sqrt((x + rho) ** 2 + 1 - rho**2))


def vol(k, T, theta, rho, psi):
    return np.sqrt(total_variance(k, theta, rho, psi) / T)


def right_edge(c, theta, rho):
    c = np.clip(c, -1, 1)
    return theta * (1 + rho * c + np.sqrt(1 - rho**2) * np.sqrt(1 - c**2))


def calendar_margin(earlier, later) -> np.ndarray:
    """Non-negative everywhere when slice later = (theta, rho, psi) has no calendar arbitrage against slice earlier.
    shortcut: the right-edge condition is checked at the 101 points C, so a violation between them can slip by;
    solve for its minimum in closed form if fits ever show a tiny dip."""
    theta1, rho1, psi1 = earlier
    theta2, rho2, psi2 = later
    wings = [psi2 * (1 - rho2) - psi1 * (1 - rho1), psi2 * (1 + rho2) - psi1 * (1 + rho1)]
    b = psi1 * (rho1 + C)  # slice 1's b-range, mapped to slice 2's c below
    c2 = b / psi2 - rho2 if psi2 > 0 else np.zeros_like(C)
    return np.concatenate([wings, right_edge(c2, theta2, rho2) - right_edge(C, theta1, rho1)])


def fit_slice(k, T, theta, market_vol, weight, earlier=None) -> tuple[float, float]:
    """(rho, psi) of one expiry minimising weighted squared vol errors, theta fixed (the market ATM total variance),
    free of butterfly arbitrage and of calendar arbitrage against slice earlier = (theta, rho, psi), whose theta must
    be no higher; None for the first expiry. Vega weights make vol errors count like price errors, so stale far wings
    pull little."""
    k, market_vol = np.asarray(k, dtype=float), np.asarray(market_vol, dtype=float)
    weight = np.asarray(weight, dtype=float) / np.mean(weight)

    def error(x):
        return np.sum(weight * (vol(k, T, theta, *x) - market_vol) ** 2)

    constraints = [  # butterfly, both signs of rho so the constraints stay smooth
        {"type": "ineq", "fun": lambda x: [4 - x[1] * (1 + x[0]), 4 - x[1] * (1 - x[0])]},
        {"type": "ineq", "fun": lambda x: [4 * theta - x[1] ** 2 * (1 + x[0]), 4 * theta - x[1] ** 2 * (1 - x[0])]},
    ]
    if earlier is None:
        start = (0.0, min(1.0, np.sqrt(2 * theta)) / 2)
    else:
        constraints.append({"type": "ineq", "fun": lambda x: calendar_margin(earlier, (theta, *x))})
        start = earlier[1:]  # feasible: theta is no lower, so same rho and psi keep both conditions
    result = minimize(
        error,
        start,
        method="SLSQP",
        bounds=[(-0.999, 0.999), (0.0, None)],
        constraints=constraints,
        options=dict(ftol=1e-12, maxiter=200),
    )
    return float(result.x[0]), float(result.x[1])
