"""Tests for the Erlang-A model (spec section 9, "Erlang-A")."""

import numpy as np
import pytest
from scipy import stats

from aicap.capacity_rules import min_second_difference
from aicap.erlang_a import (SECTION4, CostFunction, GammaRate, LognormalRate, PointRate,
                            UniformRate, correction_D, expected_cost, expected_queue,
                            gauss_gamma, newsvendor_cost, queue_moments,
                            stationary_distribution)

LAMS = [0.5, 3.0, 37.5, 75.0, 300.0]


def poisson_excess(mean, b):
    """E[(N - b)^+] for N ~ Poisson(mean), by direct summation."""
    n = np.arange(0, int(mean + 40 * np.sqrt(mean + 1) + 50))
    return float(np.sum(np.maximum(n - b, 0) * stats.poisson.pmf(n, mean)))


@pytest.mark.parametrize("lam", LAMS)
@pytest.mark.parametrize("b", [0, 1, 5, 40, 80, 310])
def test_mm_infinity_when_gamma_equals_mu(lam, b):
    """gamma = theta: every customer leaves at rate theta, so N ~ Poisson(lam/theta)."""
    theta = 1.7
    qm = queue_moments(lam, b, theta=theta, gamma=theta)
    assert qm.EQ[0] == pytest.approx(poisson_excess(lam / theta, b), rel=1e-10, abs=1e-13)
    assert qm.EN[0] == pytest.approx(lam / theta, rel=1e-10)


@pytest.mark.parametrize("lam", LAMS)
def test_no_servers_gives_poisson_lam_over_gamma(lam):
    """b = 0: N ~ Poisson(lam/gamma) and E[Q] = E[N] = lam/gamma."""
    pi = stationary_distribution(lam, 0, theta=1.0, gamma=3.0)
    n = np.arange(pi.size)
    assert np.allclose(pi, stats.poisson.pmf(n, lam / 3.0), atol=1e-13)
    assert expected_queue(lam, 0) == pytest.approx(lam / 3.0, rel=1e-12)


@pytest.mark.parametrize("lam", LAMS + [1500.0])
@pytest.mark.parametrize("b", [0, 1, 30, 75, 160, 1600])
@pytest.mark.parametrize("gamma", [1 / 3, 3.0])
def test_flow_balance(lam, b, gamma):
    """lam = theta E[min(N, b)] + gamma E[Q] (arrivals = services + abandonments)."""
    qm = queue_moments(lam, b, theta=1.0, gamma=gamma)
    rhs = 1.0 * qm.E_busy[0] + gamma * qm.EQ[0]
    assert abs(rhs - lam) / lam < 1e-9


@pytest.mark.parametrize("lam", [37.5, 300.0])
def test_queue_decreasing_in_b(lam):
    eq = np.array([expected_queue(lam, b) for b in range(0, int(1.5 * lam))])
    assert np.all(np.diff(eq) < 0)


@pytest.mark.parametrize("lam,b", [(37.5, 37), (300.0, 315), (5000.0, 4000), (0.5, 3)])
def test_truncation_error_below_1e12(lam, b):
    """The geometric tail bound is < 1e-12, and a 4x longer chain changes E[Q] by < 1e-12
    relative (the comparison itself carries ~1e-13 floating-point summation noise)."""
    qm = queue_moments(lam, b)
    assert qm.tail_mass[0] < 1e-12 and qm.tail_EQ[0] < 1e-12
    # brute-force reference with a much longer chain
    n = np.arange(4 * qm.n_max + 1)
    d = np.minimum(n[1:], b) * 1.0 + np.maximum(n[1:] - b, 0) * 3.0
    logw = n * np.log(lam) - np.concatenate(([0.0], np.cumsum(np.log(d))))
    pi = np.exp(logw - logw.max())
    pi /= pi.sum()
    assert abs(np.sum(np.maximum(n - b, 0) * pi) - qm.EQ[0]) < 1e-12 * max(1.0, qm.EQ[0])


SECTION4_CASES = [UniformRate(25, 50), UniformRate(50, 100), UniformRate(200, 400),
                  PointRate(37.5), PointRate(75), PointRate(300)]


@pytest.mark.parametrize("rate", SECTION4_CASES, ids=lambda r: r.label)
def test_cost_unimodal_in_b(rate):
    """Pi(b) is discretely convex (hence unimodal) on a grid covering the optimum."""
    cost = CostFunction(rate, SECTION4)
    top = int(1.5 * (rate.hi if isinstance(rate, UniformRate) else rate.mean)) + 10
    vals = np.array([cost(b) for b in range(0, top)])
    assert min_second_difference(vals) > -1e-10
    d = np.diff(vals)
    assert np.all(np.diff(np.sign(d)) >= 0)  # one sign change, from - to +


def test_gauss_gamma_moments():
    """Generalized Gauss-Laguerre integrates the gamma moments exactly, even for huge shapes."""
    for shape, scale in [(2.5, 3.0), (400.0, 0.1), (1e4, 0.3)]:
        x, w = gauss_gamma(shape, scale, 20)
        assert np.sum(w) == pytest.approx(1.0)
        assert np.sum(w * x) == pytest.approx(shape * scale, rel=1e-12)
        assert np.sum(w * x**2) == pytest.approx(shape * scale**2 * (shape + 1), rel=1e-11)


@pytest.mark.parametrize("rate", [UniformRate(10, 30), GammaRate(50, 0.4), LognormalRate(50, 0.4),
                                  PointRate(12.3)], ids=lambda r: r.label)
def test_partial_expectation_closed_forms(rate):
    """E[(Lambda - x)^+] closed forms vs numerical integration over the quantile function."""
    u = (np.arange(400_000) + 0.5) / 400_000
    lam = rate.ppf(u)
    for x in [0.0, 20.0, 50.0, 80.0]:
        mc = float(np.mean(np.maximum(lam - x, 0)))
        assert rate.partial_expectation(x) == pytest.approx(mc, rel=2e-4, abs=1e-4)


def test_split_identity_and_correction_nonnegative():
    """E[Q] = (lam - b theta)^+/gamma + D with D >= 0, and Pi >= Pibar (paper eq. 6)."""
    lam = np.linspace(0.1, 120, 300)
    b = 60
    D = correction_D(lam, b)
    eq = queue_moments(lam, b).EQ
    assert np.all(D >= -1e-14)
    assert np.allclose(eq, np.maximum(lam - b, 0) / 3.0 + D, atol=1e-11)
    rate = UniformRate(25, 50)
    for bb in range(30, 55):
        assert expected_cost(bb, rate) >= newsvendor_cost(bb, rate) - 1e-12


@pytest.mark.parametrize("rate", [UniformRate(25, 50), UniformRate(0, 300), UniformRate(200, 400)],
                         ids=lambda r: r.label)
def test_split_matches_gauss_legendre(rate):
    for b in [int(rate.mean), int(rate.ppf(0.75)), int(rate.ppf(0.9))]:
        g = expected_cost(b, rate, method="gauss", n_nodes=128)
        s = expected_cost(b, rate, method="split")
        assert s == pytest.approx(g, rel=1e-10)


@pytest.mark.parametrize("rate", [GammaRate(1000, 0.3), GammaRate(30, 0.01), LognormalRate(3000, 0.6)],
                         ids=lambda r: r.label)
def test_split_method_converged_for_gamma_and_lognormal(rate):
    for b in [int(rate.mean), int(rate.ppf(0.75))]:
        s1 = expected_cost(b, rate, method="split")
        s2 = expected_cost(b, rate, method="split", nodes_per_panel=16, width_factor=0.5)
        assert s1 == pytest.approx(s2, rel=1e-10)
