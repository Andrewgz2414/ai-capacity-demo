"""Tests for the Cox-process model (spec section 9, "Cox process")."""

import numpy as np
import pytest
from scipy import integrate, stats

from aicap import cox

Z_CRIT = stats.norm.ppf(1 - 0.001 / 2)  # two-sided z-test at the 0.1% level (3.29)


@pytest.mark.parametrize("median,sigma", [(0.3, 0.5), (0.34, 1.0), (1.0, 2.0)])
def test_lognormal_moment_formulas(median, sigma):
    m1, m2 = cox.lognormal_moments(median, sigma)
    dist = stats.lognorm(sigma, scale=median)
    assert m1 == pytest.approx(dist.mean(), rel=1e-12)
    assert m2 == pytest.approx(dist.moment(2), rel=1e-12)
    assert dist.median() == pytest.approx(median)
    assert cox.mean_over_median(sigma) == pytest.approx(np.exp(sigma**2 / 2))


@pytest.mark.parametrize("sigma,q", [(1.0, 0.9), (2.0, 0.99), (3.0, 0.5)])
def test_energy_share_above_quantile(sigma, q):
    dist = stats.lognorm(sigma, scale=1.0)
    t = dist.ppf(q)
    share, _ = integrate.quad(lambda x: x * dist.pdf(x), t, np.inf, limit=200)
    assert cox.energy_share_above_quantile(sigma, q) == pytest.approx(share / dist.mean(), rel=1e-6)


CASES = [(5, 0.5, 0.0), (5, 1.0, 0.3), (100, 0.5, 0.3), (100, 1.0, 0.0), (40, 1.0, 0.5)]


@pytest.mark.parametrize("nbar,sigma,cv_rate", CASES)
def test_simulated_mean_and_variance_match_theory(nbar, sigma, cv_rate):
    """E[S] = nbar m1 and Var S = nbar m2 + nbar^2 m1^2 CV_rate^2, z-tests at the 0.1% level."""
    rng = np.random.default_rng(1234 + int(nbar) + int(100 * sigma) + int(100 * cv_rate))
    R = 200_000
    S = cox.simulate_daily_energy(nbar, sigma, cv_rate, R, rng, median=0.34)["S"]
    th = cox.daily_energy_moments(nbar, sigma, cv_rate, median=0.34)
    z_mean = (S.mean() - th["mean"]) / (S.std(ddof=1) / np.sqrt(R))
    c = S - S.mean()
    se_var = np.sqrt((np.mean(c**4) - np.var(c) ** 2) / R)  # SE of the sample variance
    z_var = (S.var(ddof=1) - th["var"]) / se_var
    assert abs(z_mean) < Z_CRIT, z_mean
    assert abs(z_var) < Z_CRIT, z_var


def test_cv_formula_and_floor():
    nbar = np.array([10.0, 1e3, 1e6, 1e10])
    cv = cox.cv_daily_energy(nbar, 2.0, 0.5)
    m = cox.daily_energy_moments(1e3, 2.0, 0.5)
    assert cv[1] == pytest.approx(m["cv"])
    assert np.all(np.diff(cv) < 0) and cv[-1] == pytest.approx(0.5, rel=1e-6)


def test_crossover_balances_the_two_variance_parts():
    for sigma in (1.0, 2.0, 3.0):
        for cvr in (0.5, 1.0):
            n_star = cox.crossover_n(sigma, cvr)
            m = cox.daily_energy_moments(n_star, sigma, cvr)
            assert m["var_aleatory"] == pytest.approx(m["var_epistemic"], rel=1e-12)
    # spec section 11 with CV_rate = 1: about 3, 55 and 8,103 queries
    assert [round(cox.crossover_n(s, 1.0)) for s in (1, 2, 3)] == [3, 55, 8103]


def test_effective_load_maps_noise_to_paper_axis():
    """Per-query CV of S with a known rate is 1/sqrt(E_eff)."""
    for nbar, sigma in [(1e4, 1.0), (1e10, 3.0)]:
        assert cox.cv_daily_energy(nbar, sigma, 0.0) == pytest.approx(1 / np.sqrt(cox.effective_load(nbar, sigma)))


def test_rao_blackwell_keeps_mean_and_lowers_variance():
    rng = np.random.default_rng(7)
    rb = cox.rao_blackwell_demo(50, 1.0, 0.4, 100_000, rng)
    assert rb["mean_RB"] == pytest.approx(rb["mean_S"], rel=0.01)
    assert rb["var_RB"] < rb["var_S"]
    # Var(E[S|xi]) is the epistemic part; the dropped part is the aleatory one
    assert rb["var_RB"] == pytest.approx(rb["theory"]["var_epistemic"], rel=0.03)


def test_concurrency_units():
    """2.5e9 prompts/day = 28,935/s; with a 10 s mean service time, about 2.9e5 in service."""
    assert cox.concurrency(2.5e9, 1.0) == pytest.approx(28_935.2, rel=1e-5)
    assert cox.concurrency(2.5e9, 10.0) == pytest.approx(2.894e5, rel=1e-3)
