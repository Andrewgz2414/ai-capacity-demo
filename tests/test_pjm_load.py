"""Tests for the PJM load model (spec section 9, "PJM model")."""

import numpy as np
import pytest

from aicap import pjm_load as pl
from aicap.params import DELIVERY_YEARS, PRIORS, Prior, scaled_priors


def test_units_hand_calculation():
    """2.5e9 prompts/day x 0.34 Wh / 24 h = 35.4 MW."""
    assert pl.to_gw(2.5e9 * 0.34) * 1e3 == pytest.approx(35.4167, rel=1e-4)
    ladder = pl.fermi_ladder()
    assert ladder[0]["MW"] == pytest.approx(35.4167, rel=1e-4)


def test_fermi_ladder_matches_prototype():
    """Every parameter at its median: the ladder ends within 2% of the prototype's 4,763 MW."""
    ladder = pl.fermi_ladder(T=5, structure="matures")
    assert ladder[-1]["MW"] == pytest.approx(4763, rel=0.02)
    proto = [35, 56, 449, 1187, 10523, 32283, 12854, 20718, 4140, 4763]
    for row, ref in zip(ladder, proto):
        assert row["MW"] == pytest.approx(ref, rel=0.02)


def test_ladder_ends_at_model_value():
    th = pl.median_params()
    assert pl.fermi_ladder()[-1]["MW"] == pytest.approx(float(pl.pT_gw(th, 5)[0]) * 1e3, rel=1e-12)


def test_lognormal_prior_percentiles():
    p = PRIORS["g1"]
    assert p.ppf(0.05) == pytest.approx(1.8, rel=1e-4) and p.ppf(0.95) == pytest.approx(5.0, rel=1e-4)
    assert p.median == pytest.approx(3.0, rel=1e-12)


def test_scaled_prior_halves_spread_keeps_centre():
    for name in ("g1", "lam", "w_AT"):
        p, q = PRIORS[name], PRIORS[name].scaled(0.5)
        assert q.median == pytest.approx(p.median)
        if p.dist == "lognormal":
            assert q.s == pytest.approx(0.5 * p.s)
        else:
            assert (q.hi - q.lo) == pytest.approx(0.5 * (p.hi - p.lo))
    sp = scaled_priors(["growth"], 0.5)
    assert sp["g1"].s == pytest.approx(0.5 * PRIORS["g1"].s) and sp["N0"] is PRIORS["N0"]


def test_quantiles_stable_across_seeds():
    """Changing the seed moves each quantile by less than 3 combined bootstrap SEs."""
    a = pl.simulate(seed=1)
    b = pl.simulate(seed=2)
    for key in [(2030, "matures"), (2030, "compounding"), (2028, "matures")]:
        qa, qb = pl.quantile_table(a[key]["dQ"], n_boot=100), pl.quantile_table(b[key]["dQ"], n_boot=100)
        for k in ("P10", "P50", "P90"):
            se = np.hypot(qa[f"{k}_se"], qb[f"{k}_se"])
            assert abs(qa[k] - qb[k]) < 3 * se, (key, k, qa[k], qb[k], se)
        se_p = np.hypot(qa["p_exceed_se"], qb["p_exceed_se"])
        assert abs(qa["p_exceed"] - qb["p_exceed"]) < 3 * se_p


def test_prototype_reference_values():
    """Spec section 11 (400k draws, seed 1): our 200k-draw quantiles are within 10%."""
    sim = pl.simulate()
    ref = {(2028, "matures"): (0.82, 4.17, 22.2, 10.5), (2028, "compounding"): (0.18, 1.99, 13.7, 6.39),
           (2030, "matures"): (0.94, 5.42, 34.2, 16.95), (2030, "compounding"): (-0.19, 0.97, 13.07, 6.89)}
    for key, vals in ref.items():
        q = pl.quantile_table(sim[key]["dQ"], n_boot=20)
        for got, want in zip((q["P10"], q["P50"], q["P90"], q["mean"]), vals):
            assert got == pytest.approx(want, rel=0.10)
    assert pl.quantile_table(sim[(2030, "matures")]["dQ"], n_boot=10)["p_exceed"] == pytest.approx(0.116, abs=0.005)


def test_common_random_numbers_and_sign():
    sim = pl.simulate(n=20_000)
    m, c = sim[(2030, "matures")], sim[(2030, "compounding")]
    assert np.array_equal(m["P0"], c["P0"])  # same draws, same baseline
    # (1 - e^{-lam T})/lam < T, so for h >= 1 "matures" removes less load than "compounding"
    h = sim["theta"]["h"]
    assert np.all(m["PT"][h >= 1] >= c["PT"][h >= 1])
    assert np.any(c["dQ"] < 0)  # Delta Q can be negative; the sign is kept


def test_quantile_capacity_is_the_fractile():
    x = np.random.default_rng(0).lognormal(1.0, 1.0, 100_000)
    qc = pl.quantile_capacity(x, (0.10, 0.50))
    assert np.mean(x > qc["0.10"]) == pytest.approx(0.10, abs=1e-3)
    assert qc["0.50"] == pytest.approx(np.median(x))
    pos = pl.position_of(x, qc["0.10"])
    assert pos["implied_cost_ratio"] == pytest.approx(0.10, abs=1e-3)


def test_lognormal_approximation_with_exact_mix_moments():
    """With exact log-moments for the mix, the lognormal approximation of P_T is within 10%."""
    sim = pl.simulate()
    for year, dy in DELIVERY_YEARS.items():
        for s in ("matures", "compounding"):
            chk = pl.lognormal_check(sim[(year, s)]["PT"], dy["T"], s)
            assert max(abs(r) for r in chk["exact"]["rel_diff"]) < 0.10
            assert chk["exact"]["var_log"] == pytest.approx(chk["mc_var_log"], rel=0.02)


def test_prior_validation_shapes():
    assert all(isinstance(p, Prior) for p in PRIORS.values())
    for p in PRIORS.values():
        assert p.lo < p.hi and p.lo > 0
