"""Tests for the capacity rules and the reproduction of the paper (spec section 9, "Rules")."""

import numpy as np
import pytest

from aicap.capacity_rules import (brute_force_optimal, evaluate_rules, known_rate_capacity,
                                  newsvendor_capacity, optimal_capacity)
from aicap.erlang_a import SECTION4, CostFunction, GammaRate, PointRate, UniformRate, expected_cost


def test_section4_fractile_is_one_quarter():
    assert SECTION4.fractile == pytest.approx(0.25)
    assert SECTION4.queue_weight == pytest.approx(4.0)
    assert SECTION4.underage == pytest.approx(4.0 / 3.0)


def test_newsvendor_uniform_cases_match_paper():
    """Paper Table 2: b-bar = 43, 87, 350 (75th percentile rounded down)."""
    got = [newsvendor_capacity(UniformRate(a, 2 * a)) for a in (25, 50, 200)]
    assert got == [43, 87, 350]


def test_newsvendor_fixed_rate_is_lambda_over_mu():
    """With a known rate the newsvendor rule is b = lambda/theta (rounded down, as in Table 3)."""
    assert [newsvendor_capacity(PointRate(lam)) for lam in (37.5, 75, 300)] == [37, 75, 300]


def test_newsvendor_all_paper_table_entries():
    """All 20 b-bar values in Tables 2, 3, 4 and 7 equal floor(F^{-1}(3/4))."""
    cases = {(1, 2): 1, (2, 4): 3, (5, 10): 8, (10, 20): 17, (15, 30): 26, (20, 40): 35, (25, 50): 43,
             (50, 100): 87, (200, 400): 350, (0, 300): 225, (125, 175): 162, (135, 165): 157,
             (140, 160): 155, (145, 155): 152}
    for (lo, hi), bbar in cases.items():
        assert newsvendor_capacity(UniformRate(lo, hi)) == bbar
    for lam, bbar in {37.5: 37, 75: 75, 150: 150, 300: 300}.items():
        assert newsvendor_capacity(PointRate(lam)) == bbar


@pytest.mark.parametrize("a", [25, 50, 200])
def test_gauss_legendre_converges(a):
    """Doubling the Gauss-Legendre nodes changes Pi(b) by < 1e-6 (relative)."""
    rate = UniformRate(a, 2 * a)
    for b in [a, int(1.5 * a), int(1.75 * a), 2 * a]:
        p64 = expected_cost(b, rate, method="gauss", n_nodes=64)
        p128 = expected_cost(b, rate, method="gauss", n_nodes=128)
        assert abs(p64 - p128) / p128 < 1e-6


ALL_CASES = [UniformRate(25, 50), UniformRate(50, 100), UniformRate(200, 400), PointRate(37.5),
             PointRate(75), PointRate(300), UniformRate(0, 300), UniformRate(125, 175),
             UniformRate(135, 165), UniformRate(140, 160), UniformRate(145, 155), PointRate(150)]


@pytest.mark.parametrize("rate", ALL_CASES, ids=lambda r: r.label)
def test_bisection_equals_brute_force(rate):
    cost = CostFunction(rate, SECTION4)
    top = int(1.6 * (rate.hi if isinstance(rate, UniformRate) else rate.mean)) + 10
    b_bf, _, _ = brute_force_optimal(cost, np.arange(0, top))
    assert optimal_capacity(cost, 0, 5) == b_bf
    assert evaluate_rules(rate).b_star == b_bf


# Paper Table 4 (p.13): (b*, Pi*, b-bar, Pi(b-bar)). Our exact computation matches it.
TABLE4 = {(0, 300): (224, 88.34, 225, 88.34), (125, 175): (165, 59.06, 162, 59.16),
          (135, 165): (162, 57.40, 157, 57.78), (140, 160): (162, 56.78, 155, 57.42),
          (145, 155): (161, 56.40, 152, 57.73), 150: (161, 56.26, 150, 58.25)}


@pytest.mark.parametrize("key", list(TABLE4), ids=str)
def test_reproduces_paper_table4(key):
    rate = PointRate(key) if key == 150 else UniformRate(*key)
    res = evaluate_rules(rate)
    b_star, pi_star, b_bar, pi_bar = TABLE4[key]
    assert (res.b_star, res.b_bar) == (b_star, b_bar)
    assert res.Pi_star == pytest.approx(pi_star, abs=0.025)
    assert res.Pi_bar == pytest.approx(pi_bar, abs=0.025)


# The spec's prototype numbers (same convention); a regression check on our implementation.
PROTOTYPE = {(25, 50): (45, 17.10, 43, 0.0041), (50, 100): (88, 32.92, 87, 0.0005),
             (200, 400): (350, 126.90, 350, 0.0000), 37.5: (43, 15.66, 37, 0.073),
             75: (83, 29.45, 75, 0.047), 300: (315, 108.83, 300, 0.026)}


@pytest.mark.parametrize("key", list(PROTOTYPE), ids=str)
def test_matches_prototype(key):
    rate = PointRate(key) if not isinstance(key, tuple) else UniformRate(*key)
    res = evaluate_rules(rate)
    b_star, pi_star, b_bar, gap = PROTOTYPE[key]
    assert (res.b_star, res.b_bar) == (b_star, b_bar)
    assert res.Pi_star == pytest.approx(pi_star, abs=0.006)
    assert res.gap_bar == pytest.approx(gap, abs=0.0006)


def test_known_rate_rule_is_optimal_for_fixed_rates():
    for lam in (37.5, 75, 300):
        res = evaluate_rules(PointRate(lam))
        assert res.b_det == res.b_star and res.gap_det == 0.0


def test_qualitative_result():
    """Newsvendor within 1% when the rate is uncertain; several % off when it is known."""
    unc = [evaluate_rules(UniformRate(a, 2 * a)).gap_bar for a in (25, 50, 200)]
    det = [evaluate_rules(PointRate(lam)).gap_bar for lam in (37.5, 75, 300)]
    assert max(unc) < 0.01
    assert min(det) > 0.02
    assert det[0] > det[1] > det[2]  # gap shrinks like 1/sqrt(E)


def test_known_rate_capacity_gamma_mean():
    rate = GammaRate(300, 0.2)
    assert known_rate_capacity(rate) == evaluate_rules(PointRate(300)).b_star
