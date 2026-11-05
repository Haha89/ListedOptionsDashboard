import numpy as np
import pytest

from src.option_vol import ssvi

K = np.linspace(-1.5, 1.5, 3001)
WIDE = np.concatenate([-np.geomspace(1e4, 1e-4, 4000), np.geomspace(1e-4, 1e4, 4000)])  # out to where only slopes count


def durrleman(k, w):
    """Butterfly arbitrage density condition g(k) >= 0 (Gatheral & Jacquier eq. 2.1), derivatives by finite differences"""
    dw, d2w = np.gradient(w, k), np.gradient(np.gradient(w, k), k)
    return (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 1 / 4) + d2w / 2


def max_psi(theta, rho):
    """Largest psi without butterfly arbitrage"""
    return min(4 / (1 + abs(rho)), 2 * np.sqrt(theta / (1 + abs(rho))))


@pytest.mark.parametrize("theta", [1e-4, 0.01, 0.1, 1.0, 4.0])
@pytest.mark.parametrize("rho", [-0.999, -0.7, 0.0, 0.5, 0.999])
def test_butterfly_bound_is_arbitrage_free(theta, rho):
    w = ssvi.total_variance(K, theta, rho, max_psi(theta, rho))
    assert durrleman(K, w)[1:-1].min() > -1e-6  # np.gradient is one-sided at the ends


def test_butterfly_bound_is_needed():
    w = ssvi.total_variance(K, 0.01, -0.7, 2 * max_psi(0.01, -0.7))
    assert durrleman(K, w)[1:-1].min() < 0


def test_calendar_margin_matches_a_brute_force_check():
    """calendar_margin >= 0 exactly when the later slice's w stays above the earlier one's, both ways"""
    rng = np.random.default_rng(0)
    seen = {True: 0, False: 0}
    for _ in range(3000):
        theta1 = rng.uniform(1e-3, 1)
        theta2 = theta1 * rng.uniform(1, 2)
        rho1, rho2 = rng.uniform(-0.99, 0.99, 2)
        psi1, psi2 = rng.uniform(0, max_psi(theta1, rho1)), rng.uniform(0, max_psi(theta2, rho2))
        margin = ssvi.calendar_margin((theta1, rho1, psi1), (theta2, rho2, psi2)).min()
        if abs(margin) < 1e-3:  # too close to the boundary for a grid in k to tell
            continue
        dip = (ssvi.total_variance(WIDE, theta2, rho2, psi2) - ssvi.total_variance(WIDE, theta1, rho1, psi1)).min()
        assert (margin > 0) == (dip >= -1e-12), (theta1, rho1, psi1, theta2, rho2, psi2)
        seen[margin > 0] += 1
    assert min(seen.values()) > 100


def test_wing_slopes_alone_allow_a_calendar_dip():
    """Wing slopes ordered, but the later slice's left intercept theta (1 - rho) is lower: it dips in between"""
    earlier, later = (0.267, -0.11, 0.040), (0.555, 0.85, 0.359)
    assert ssvi.calendar_margin(earlier, later)[:2].min() >= 0 > ssvi.calendar_margin(earlier, later).min()
    assert (ssvi.total_variance(K, *later) - ssvi.total_variance(K, *earlier)).min() < 0


def test_fit_slice_recovers_the_parameters():
    k, T, theta = np.linspace(-0.4, 0.3, 30), 0.25, 0.3**2 * 0.25
    market = ssvi.vol(k, T, theta, -0.6, 0.2)
    assert ssvi.fit_slice(k, T, theta, market, np.ones_like(k)) == pytest.approx((-0.6, 0.2), abs=1e-4)


def test_fit_slice_respects_the_previous_slice():
    k, T, theta = np.linspace(-0.4, 0.3, 30), 0.25, 0.3**2 * 0.25
    flat = np.full_like(k, 0.3)  # wants psi = 0, but the slice before had a skew
    earlier = (0.3**2 * 0.2, -0.6, 0.2)
    rho, psi = ssvi.fit_slice(k, T, theta, flat, np.ones_like(k), earlier)
    assert psi <= max_psi(theta, rho) + 1e-9 and ssvi.calendar_margin(earlier, (theta, rho, psi)).min() >= -1e-9
    assert (ssvi.total_variance(WIDE, theta, rho, psi) - ssvi.total_variance(WIDE, *earlier)).min() >= -1e-9


def test_atm_vol_is_theta():
    assert ssvi.vol(0.0, 0.5, 0.3**2 * 0.5, rho=-0.5, psi=0.3) == pytest.approx(0.3)
