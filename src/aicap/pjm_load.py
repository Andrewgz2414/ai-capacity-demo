"""New PJM summer-peak load from AI, Delta Q (GW), and the quantile (newsvendor) reading.

Model (spec section 6), with independent epistemic parameters theta drawn from `params.PRIORS`
and horizon T years from 2025:

    m_c   = M_c e^{sigma_c^2/2} beta                         mean Wh per chat query incl. overhead
    mix_t = (1 - w_R,t - w_A,t) + w_R,t r + w_A,t a          energy per request relative to chat
    G_T   = exp( ln(g1)/lam (1 - e^{-lam T}) )               volume growth (Gompertz)
    H_T   = exp( ln(h)/lam (1 - e^{-lam T}) )  ["matures"]   efficiency gain (default)
          = h^T                                ["compounding"]
    scope = (1 + tau) s_US s_PJM kappa
    P_0   = N0 m_c mix_0 scope / 24 / 1e9                    GW (Wh/day / 24 h = W)
    P_T   = N0 G_T m_c mix_T / H_T scope / 24 / 1e9          GW
    Delta Q = P_T - P_0                                       (can be negative; sign kept)

The grid model uses E[S | theta] (daily energy given the parameters), not a simulated sum
over ~1e10 queries: by Rao-Blackwellization the mean is unchanged and the variance can only
fall, and part C shows the per-query noise is ~1e-5 of the epistemic spread.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from .params import (ALTMAN_WH, CHATGPT_PROMPTS_PER_DAY, COST_RATIOS, DELIVERY_YEARS,
                     EFFICIENCY_STRUCTURES, N_DRAWS, PJM_DC_GROWTH_GW, PJM_SUMMER_PEAK_2026_GW,
                     PRESENTER1_STATUS, PRIORS, PRIORS_VERSION, SEED, priors_table, scaled_priors)

HOURS_PER_DAY = 24.0
W_PER_GW = 1e9
DAYS_PER_YEAR = 365.0
TWH_TO_GWH_PER_DAY = 1e3 / DAYS_PER_YEAR  # TWh/yr -> average GWh/day
REGIONS = ("global", "US", "PJM")

# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------


def sample_params(n: int, rng: np.random.Generator, priors: dict | None = None) -> dict:
    """One array of n independent draws per parameter (fixed order -> reproducible)."""
    priors = priors or PRIORS
    return {name: p.sample(n, rng) for name, p in priors.items()}


def median_params(priors: dict | None = None) -> dict:
    priors = priors or PRIORS
    return {name: np.atleast_1d(p.median) for name, p in priors.items()}


def mean_wh_chat(th: dict):
    """m_c = M_c e^{sigma_c^2/2} beta (Wh per chat query, fleet average)."""
    return th["M_c"] * np.exp(th["sigma_c"] ** 2 / 2) * th["beta"]


def mix_factor(th: dict, when: str):
    """Energy per request relative to a chat query, in 2025 ("0") or the delivery year ("T").

    Shares must satisfy w_R + w_A <= 1; the lognormal prior on w_A,T is unbounded, so w_A is
    capped at 1 - w_R (under the priors the cap binds with probability 2.1e-5 in the delivery
    year, about 4 of 200,000 draws; never in 2025)."""
    wR = th[f"w_R{when}"]
    wA = np.minimum(th[f"w_A{when}"], 1 - wR)
    return (1 - wR - wA) + wR * th["r"] + wA * th["a"]


def gompertz(first_year_factor, lam, T: float):
    """exp( ln(g1)/lam (1 - e^{-lam T}) ): log growth rate ln(g1) decaying at rate lam."""
    return np.exp(np.log(first_year_factor) / lam * (1 - np.exp(-lam * T)))


def growth_factor(th: dict, T: float):
    return gompertz(th["g1"], th["lam"], T)


def efficiency_factor(th: dict, T: float, structure: str = "matures"):
    if structure == "matures":
        return gompertz(th["h"], th["lam"], T)
    if structure == "compounding":
        return th["h"] ** T
    raise ValueError(structure)


def scope_factor(th: dict):
    """(1 + tau) s_US s_PJM kappa: training uplift, siting in the US and PJM, summer-peak coincidence."""
    return (1 + th["tau"]) * th["s_US"] * th["s_PJM"] * th["kappa"]


def to_gw(wh_per_day):
    """Wh/day -> average GW."""
    return wh_per_day / HOURS_PER_DAY / W_PER_GW


def p0_gw(th: dict):
    return to_gw(th["N0"] * mean_wh_chat(th) * mix_factor(th, "0") * scope_factor(th))


def pT_gw(th: dict, T: float, structure: str = "matures"):
    return to_gw(th["N0"] * growth_factor(th, T) * mean_wh_chat(th) * mix_factor(th, "T")
                 / efficiency_factor(th, T, structure) * scope_factor(th))


def delta_q(th: dict, T: float, structure: str = "matures") -> dict:
    P0, PT = p0_gw(th), pT_gw(th, T, structure)
    return {"dQ": PT - P0, "P0": P0, "PT": PT}


def region_share(th: dict, region: str):
    """Fraction of global AI load in the region (fixed shares: siting is not a decision here)."""
    if region == "global":
        return 1.0
    if region == "US":
        return th["s_US"]
    if region == "PJM":
        return th["s_US"] * th["s_PJM"]
    raise ValueError(region)


def energy_twh(th: dict, T: float | None = None, structure: str = "matures", region: str = "PJM"):
    """Annualized AI electricity use, TWh/yr: daily energy x 365 (= average power x 8,760 h).

    Inference for all providers plus training as (1 + tau) x inference, times the regional share.
    No summer-peak factor kappa (energy, not peak). T=None gives 2025; otherwise the annualized
    rate at year 2025 + T. Boundary: the per-query energies are as measured by the cited studies
    (Google's 0.24 Wh already includes data-centre overhead), so this is close to facility energy,
    not server (IT) energy."""
    wh_day = th["N0"] * mean_wh_chat(th) * (1 + th["tau"]) * region_share(th, region)
    if T is None:
        wh_day = wh_day * mix_factor(th, "0")
    else:
        wh_day = wh_day * growth_factor(th, T) * mix_factor(th, "T") / efficiency_factor(th, T, structure)
    return wh_day * DAYS_PER_YEAR / 1e12


def simulate(n: int = N_DRAWS, seed: int = SEED, priors: dict | None = None) -> dict:
    """Common random numbers: one parameter draw serves both delivery years and both
    efficiency structures. Returns {"theta": draws, "E0": {region: TWh/yr in 2025},
    (year, structure): {"dQ", "P0", "PT", "ET": {region: TWh/yr in that year}}}."""
    rng = np.random.default_rng(seed)
    th = sample_params(n, rng, priors)
    out = {"theta": th, "n": n, "seed": seed, "E0": {r: energy_twh(th, None, region=r) for r in REGIONS}}
    for year, dy in DELIVERY_YEARS.items():
        for s in EFFICIENCY_STRUCTURES:
            out[(year, s)] = delta_q(th, dy["T"], s)
            out[(year, s)]["ET"] = {r: energy_twh(th, dy["T"], s, r) for r in REGIONS}
    return out


# ---------------------------------------------------------------------------
# Monte Carlo summaries
# ---------------------------------------------------------------------------


def quantile_table(x, probs=(0.10, 0.50, 0.90), n_boot: int = 200, seed: int = 0,
                   threshold: float | None = PJM_DC_GROWTH_GW) -> dict:
    """Quantiles with bootstrap SEs, mean with its SE, and P(x > threshold) with its SE.
    n_boot=0 skips the bootstrap (SEs None); threshold=None skips the exceedance probability."""
    x = np.asarray(x, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    q = np.quantile(x, probs)
    out = {f"P{round(100 * p)}": float(v) for p, v in zip(probs, q)}
    if n_boot > 0:
        boot = np.empty((n_boot, len(probs)))
        for b in range(n_boot):
            boot[b] = np.quantile(x[rng.integers(0, n, n)], probs)
        out.update({f"P{round(100 * p)}_se": float(s) for p, s in zip(probs, boot.std(axis=0, ddof=1))})
    else:
        out.update({f"P{round(100 * p)}_se": None for p in probs})
    out.update(mean=float(x.mean()), mean_se=float(x.std(ddof=1) / np.sqrt(n)),
               p_negative=float(np.mean(x < 0)), n=n, n_boot=n_boot, threshold=threshold)
    if threshold is not None:
        p_exc = float(np.mean(x > threshold))
        out.update(p_exceed=p_exc, p_exceed_se=float(np.sqrt(p_exc * (1 - p_exc) / n)))
    return out


def energy_tables(sim: dict, n_boot: int = 50) -> dict:
    """Quantiles of annualized energy (TWh/yr) for 2025 and each delivery year x structure x region,
    and of the new energy dE = E_T - E_2025."""
    probs = (0.10, 0.50, 0.90)
    out = {"units": "TWh/yr (annualized: daily energy x 365); GWh/day = TWh/yr x 1000/365",
           "twh_to_gwh_per_day": TWH_TO_GWH_PER_DAY, "regions": list(REGIONS),
           "boundary": "approximately facility energy (per-query energies as measured by the cited studies)",
           "baseline_2025": {r: quantile_table(sim["E0"][r], probs, n_boot, threshold=None) for r in REGIONS},
           "cases": {}}
    for year in DELIVERY_YEARS:
        for s in EFFICIENCY_STRUCTURES:
            ET = sim[(year, s)]["ET"]
            out["cases"][f"{year}_{s}"] = {
                r: {"E": quantile_table(ET[r], probs, n_boot, seed=year, threshold=None),
                    "dE": quantile_table(ET[r] - sim["E0"][r], probs, n_boot, seed=year, threshold=None)}
                for r in REGIONS}
    return out


def quantile_capacity(dQ, ratios=COST_RATIOS) -> dict:
    """Newsvendor reading: buy Q with P(Delta Q > Q) = c_cap/c_short, i.e. the (1 - ratio) quantile."""
    return {f"{r:.2f}": float(np.quantile(dQ, 1 - r)) for r in ratios}


def position_of(dQ, level: float = PJM_DC_GROWTH_GW) -> dict:
    """Quantile at which `level` sits and the implied cost ratio c_cap/c_short = P(Delta Q > level)."""
    F = float(np.mean(np.asarray(dQ) <= level))
    return {"level_GW": level, "quantile": F, "implied_cost_ratio": 1 - F}


def cost_of_uncertainty(dQ, q: float = 0.90) -> dict:
    """Q_q - median: the capacity bought only because the future is uncertain."""
    Qq, med = np.quantile(dQ, [q, 0.5])
    return {"Q": float(Qq), "median": float(med), "gap": float(Qq - med), "q": q}


# ---------------------------------------------------------------------------
# Fermi ladder: median path from ChatGPT today to the PJM number
# ---------------------------------------------------------------------------


def fermi_ladder(T: float = 5, structure: str = "matures", priors: dict | None = None) -> list[dict]:
    """One multiplicative factor per step, every parameter at its prior median.

    Starts at 2.5e9 prompts/day x 0.34 Wh / 24 h = 35.4 MW and ends at P_T(median parameters)."""
    th = {k: float(v[0]) for k, v in median_params(priors).items()}
    th_arr = {k: np.array([v]) for k, v in th.items()}
    start = to_gw(CHATGPT_PROMPTS_PER_DAY * ALTMAN_WH) * 1e3  # MW
    steps = [
        ("ChatGPT today: 2.5e9 prompts/day x 0.34 Wh / 24 h", None, start),
        ("mean Wh incl. overhead: M_c e^(σ²/2) β / 0.34 Wh", float(mean_wh_chat(th_arr)[0]) / ALTMAN_WH, None),
        ("all providers: N₀ / 2.5e9", th["N0"] / CHATGPT_PROMPTS_PER_DAY, None),
        ("2025 query mix: mix₀", float(mix_factor(th_arr, "0")[0]), None),
        ("volume growth to the delivery year: G_T", float(growth_factor(th_arr, T)[0]), None),
        ("mix shift: mix_T / mix₀", float(mix_factor(th_arr, "T")[0] / mix_factor(th_arr, "0")[0]), None),
        ("efficiency gains: 1 / H_T", float(1 / efficiency_factor(th_arr, T, structure)[0]), None),
        ("training: 1 + τ", 1 + th["tau"], None),
        ("US and PJM share: s_US s_PJM", th["s_US"] * th["s_PJM"], None),
        ("summer-peak coincidence: κ", th["kappa"], None),
    ]
    rows, value = [], start
    for label, factor, v in steps:
        if factor is not None:
            value *= factor
        rows.append({"step": label, "factor": factor, "MW": value})
    check = float(pT_gw(th_arr, T, structure)[0]) * 1e3
    assert abs(rows[-1]["MW"] / check - 1) < 1e-12, "ladder must end at P_T(median parameters)"
    return rows


# ---------------------------------------------------------------------------
# Lognormal (Fenton-Wilkinson) cross-check of the Monte Carlo
# ---------------------------------------------------------------------------


def _uniform_log_moments(lo: float, hi: float) -> tuple[float, float]:
    """E[ln U], Var[ln U] for U ~ Uniform(lo, hi), lo > 0 (closed form)."""
    F1 = lambda x: x * np.log(x) - x  # noqa: E731  antiderivative of ln x
    F2 = lambda x: x * (np.log(x) ** 2 - 2 * np.log(x) + 2)  # noqa: E731  of ln^2 x
    m1 = (F1(hi) - F1(lo)) / (hi - lo)
    m2 = (F2(hi) - F2(lo)) / (hi - lo)
    return m1, m2 - m1**2


def _moments_of(prior, power: int = 1) -> tuple[float, float]:
    """E[X^power], E[X^{2 power}] for one prior."""
    if prior.dist == "lognormal":
        mu, s = prior.mu, prior.s
        return np.exp(power * mu + (power * s) ** 2 / 2), np.exp(2 * power * mu + 2 * (power * s) ** 2)
    a, b = prior.lo, prior.hi
    m = lambda k: (b ** (k + 1) - a ** (k + 1)) / ((k + 1) * (b - a))  # noqa: E731
    return m(power), m(2 * power)


def _nodes(prior, n: int):
    """Gauss nodes/weights (weights sum to 1) for one prior: Legendre for uniform, Hermite for lognormal."""
    if prior.dist == "lognormal":
        x, w = np.polynomial.hermite.hermgauss(n)
        return np.exp(prior.mu + np.sqrt(2) * prior.s * x), w / np.sqrt(np.pi)
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (prior.hi - prior.lo) * x + 0.5 * (prior.hi + prior.lo), 0.5 * w


def mix_log_moments_exact(when: str = "T", priors: dict | None = None, n: int = 32) -> tuple[float, float]:
    """E[ln mix] and Var[ln mix] by 4-D tensor Gauss quadrature over (w_R, w_A, r, a)."""
    pr = priors or PRIORS
    grids = [_nodes(pr[k], n) for k in (f"w_R{when}", f"w_A{when}", "r", "a")]
    X = np.meshgrid(*[g[0] for g in grids], indexing="ij")
    W = np.einsum("i,j,k,l->ijkl", *[g[1] for g in grids])
    wA = np.minimum(X[1], 1 - X[0])  # same share cap as mix_factor
    lm = np.log((1 - X[0] - wA) + X[0] * X[2] + wA * X[3])
    m1 = float(np.sum(W * lm))
    return m1, float(np.sum(W * lm**2) - m1**2)


def log_moments(T: float, structure: str = "matures", priors: dict | None = None,
                when: str = "T", mix_method: str = "fenton-wilkinson") -> dict:
    """Mean and variance of ln P_t factor by factor (factors are independent).

    mix_method="fenton-wilkinson": the mix sum is replaced by the lognormal with the same mean
    and variance (Fenton 1960); "exact": E[ln mix], Var[ln mix] by quadrature."""
    pr = priors or PRIORS
    parts = {}
    # base volume
    parts["base volume"] = (pr["N0"].mu, pr["N0"].s ** 2)
    # Wh per query: ln M_c + sigma_c^2/2 + ln beta
    e_s2, e_s4 = _moments_of(pr["sigma_c"], 2)
    mb, vb = _uniform_log_moments(pr["beta"].lo, pr["beta"].hi)
    parts["Wh per query"] = (pr["M_c"].mu + e_s2 / 2 + mb, pr["M_c"].s ** 2 + (e_s4 - e_s2**2) / 4 + vb)
    # mix: 1 + w_R (r - 1) + w_A (a - 1), two independent products; Fenton-Wilkinson on the sum
    wR, wA = pr[f"w_R{when}"], pr[f"w_A{when}"]
    Er, Er2 = _moments_of(pr["r"])
    Ea, Ea2 = _moments_of(pr["a"])
    EwR, EwR2 = _moments_of(wR)
    EwA, EwA2 = _moments_of(wA)
    EX, EY = EwR * (Er - 1), EwA * (Ea - 1)
    VX = EwR2 * (Er2 - 2 * Er + 1) - EX**2
    VY = EwA2 * (Ea2 - 2 * Ea + 1) - EY**2
    m, v = 1 + EX + EY, VX + VY
    s2 = np.log1p(v / m**2)
    parts["mix"] = (np.log(m) - s2 / 2, s2) if mix_method == "fenton-wilkinson" else mix_log_moments_exact(when, pr)
    if when == "T":
        # growth and efficiency share lam: ln G_T - ln H_T = (ln g1 - ln h) phi(lam) [matures]
        lam_lo, lam_hi = pr["lam"].lo, pr["lam"].hi
        phi = lambda l: (1 - np.exp(-l * T)) / l  # noqa: E731
        Ephi = integrate.quad(phi, lam_lo, lam_hi)[0] / (lam_hi - lam_lo)
        Ephi2 = integrate.quad(lambda l: phi(l) ** 2, lam_lo, lam_hi)[0] / (lam_hi - lam_lo)
        mg, vg, mh, vh = pr["g1"].mu, pr["g1"].s ** 2, pr["h"].mu, pr["h"].s ** 2
        if structure == "matures":
            md, vd = mg - mh, vg + vh
            parts["growth / efficiency"] = (md * Ephi, (vd + md**2) * Ephi2 - (md * Ephi) ** 2)
        else:
            parts["growth / efficiency"] = (mg * Ephi - T * mh, (vg + mg**2) * Ephi2 - (mg * Ephi) ** 2 + T**2 * vh)
    # training: ln(1 + tau), tau lognormal -> Gauss-Hermite
    x, w = np.polynomial.hermite.hermgauss(80)
    v_ = np.log1p(np.exp(pr["tau"].mu + np.sqrt(2) * pr["tau"].s * x))
    e1, e2 = np.sum(w * v_) / np.sqrt(np.pi), np.sum(w * v_**2) / np.sqrt(np.pi)
    parts["training"] = (e1, e2 - e1**2)
    # siting and peak
    ms = vs = 0.0
    for k in ("s_US", "s_PJM", "kappa"):
        a_, b_ = _uniform_log_moments(pr[k].lo, pr[k].hi)
        ms, vs = ms + a_, vs + b_
    parts["siting and peak"] = (ms, vs)
    mean = sum(p[0] for p in parts.values()) - np.log(HOURS_PER_DAY * W_PER_GW)
    var = sum(p[1] for p in parts.values())
    return {"mean_log": float(mean), "var_log": float(var), "parts": {k: (float(a), float(b)) for k, (a, b) in parts.items()}}


def lognormal_check(PT_samples, T: float, structure: str = "matures", probs=(0.10, 0.50, 0.90),
                    when: str = "T") -> dict:
    """Compare Monte Carlo quantiles of P_t with two lognormal approximations:
    Fenton-Wilkinson for the mix sum (spec) and exact log-moments for the mix (quadrature)."""
    z = special.ndtri(np.asarray(probs))
    mc = np.quantile(PT_samples, probs)
    lnx = np.log(PT_samples)
    out = {"probs": list(probs), "mc": mc.tolist(), "mc_mean_log": float(lnx.mean()),
           "mc_var_log": float(lnx.var())}
    for key, method in (("fw", "fenton-wilkinson"), ("exact", "exact")):
        lm = log_moments(T, structure, when=when, mix_method=method)
        approx = np.exp(lm["mean_log"] + np.sqrt(lm["var_log"]) * z)
        out[key] = {"quantiles": approx.tolist(), "rel_diff": (approx / mc - 1).tolist(),
                    "mean_log": lm["mean_log"], "var_log": lm["var_log"], "parts": lm["parts"]}
    return out


def mc_cv(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std() / x.mean())


def lognormal_from_samples_quantile(x, q: float) -> float:
    """Quantile of the lognormal with the same log-mean and log-variance as the samples."""
    lnx = np.log(x)
    return float(np.exp(lnx.mean() + lnx.std() * stats.norm.ppf(q)))


# ---------------------------------------------------------------------------
# Summaries and handoff (used by scripts/run_all.py and the notebook)
# ---------------------------------------------------------------------------


def default_handoff_dir() -> Path:
    from . import OUTPUTS
    return OUTPUTS / "handoff"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def summarize(sim: dict, n_boot: int = 200, verbose: bool = True, energy_n_boot: int = 50) -> dict:
    """All Part D numbers for summary.json: quantiles with SEs, capacity, 30 GW position, cost of uncertainty,
    tail, lognormal checks, Fermi ladders, halved-uncertainty rerun, flexible load, per-type table."""
    t = time.time()
    out = {"n_draws": sim["n"], "seed": sim["seed"], "priors_version": PRIORS_VERSION,
           "presenter1_status": PRESENTER1_STATUS, "priors": priors_table(),
           "delivery_years": DELIVERY_YEARS, "cost_ratios": list(COST_RATIOS), "pjm_dc_growth_GW": PJM_DC_GROWTH_GW,
           "pjm_summer_peak_2026_GW": PJM_SUMMER_PEAK_2026_GW}
    q, cap, pos, cou, tail, lnchk, cvs = {}, {}, {}, {}, {}, {}, {}
    for year, dy in DELIVERY_YEARS.items():
        for s in EFFICIENCY_STRUCTURES:
            d = sim[(year, s)]
            key = f"{year}_{s}"
            q[key] = quantile_table(d["dQ"], probs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95), n_boot=n_boot,
                                    seed=year)
            cap[key] = quantile_capacity(d["dQ"])
            pos[key] = position_of(d["dQ"])
            cou[key] = cost_of_uncertainty(d["dQ"])
            tail[key] = {"P99": float(np.quantile(d["dQ"], 0.99)), "P999": float(np.quantile(d["dQ"], 0.999)),
                         "max": float(d["dQ"].max()),
                         "p_exceed_pjm_peak": float(np.mean(d["dQ"] > PJM_SUMMER_PEAK_2026_GW))}
            lnchk[key] = lognormal_check(d["PT"], dy["T"], s)
            lv = lnchk[key]["mc_var_log"]
            cvs[key] = {"cv_lognormal_equivalent": float(np.sqrt(np.expm1(lv))), "cv_sample": mc_cv(d["PT"]),
                        "log_sd": float(np.sqrt(lv))}
            q[key]["PT"] = quantile_table(d["PT"], probs=(0.10, 0.50, 0.90), n_boot=50, seed=year)
    p0 = sim[(2030, "matures")]["P0"]
    out["P0"] = quantile_table(p0, probs=(0.10, 0.50, 0.90), n_boot=50)
    out["P0_lognormal_check"] = lognormal_check(p0, 0, when="0")
    out.update(quantiles=q, quantile_capacity=cap, pjm_30GW=pos, cost_of_uncertainty=cou, tail=tail,
               lognormal_check=lnchk, cv_PT=cvs)
    out["fermi_ladder"] = {f"{y}_{s}": fermi_ladder(DELIVERY_YEARS[y]["T"], s)
                           for y in DELIVERY_YEARS for s in EFFICIENCY_STRUCTURES}
    th_med = median_params()
    out["at_prior_medians"] = {f"{y}": {"P0_GW": float(p0_gw(th_med)[0]),
                                        "PT_GW": float(pT_gw(th_med, DELIVERY_YEARS[y]["T"])[0])}
                               for y in DELIVERY_YEARS}
    # better forecasting: halve the log-sd (uniform: half-width) of the growth and mix priors
    half = simulate(n=sim["n"], seed=sim["seed"], priors=scaled_priors(["growth", "mix"], 0.5))
    hcou = {}
    for year in DELIVERY_YEARS:
        for s in EFFICIENCY_STRUCTURES:
            key = f"{year}_{s}"
            c = cost_of_uncertainty(half[(year, s)]["dQ"])
            hcou[key] = c | {"base_gap": cou[key]["gap"], "shrink_fraction": 1 - c["gap"] / cou[key]["gap"],
                             "p_exceed_30": float(np.mean(half[(year, s)]["dQ"] > PJM_DC_GROWTH_GW))}
    out["halved_growth_mix_uncertainty"] = hcou
    # flexible load: curtailment halves the shortage cost -> ratio 0.10 -> 0.20 -> P90 -> P80
    d30 = sim[(2030, "matures")]["dQ"]
    q90, q80 = float(np.quantile(d30, 0.90)), float(np.quantile(d30, 0.80))
    out["flexible_load"] = {"ratio_before": 0.10, "ratio_after": 0.20, "Q_before": q90, "Q_after": q80,
                            "reduction_GW": q90 - q80, "case": "2030_matures"}
    # presenter 1's per-type table at prior medians (placeholders)
    Mc, sc = PRIORS["M_c"].median, PRIORS["sigma_c"].median
    types = {"chat": 1.0, "reasoning": PRIORS["r"].median, "agentic": PRIORS["a"].median}
    out["per_type_at_medians"] = {k: {"median_Wh": Mc * m, "sigma": sc, "mean_Wh": Mc * m * np.exp(sc**2 / 2)}
                                  for k, m in types.items()}
    out["share_cap_binding_draws"] = int(np.sum(sim["theta"]["w_AT"] > 1 - sim["theta"]["w_RT"]))
    out["energy"] = energy_tables(sim, energy_n_boot)
    if verbose:
        print(f"[D] PJM model summaries in {time.time() - t:.1f} s")
    return out


def _handoff_dir(outdir=None) -> Path:
    path = Path(outdir) if outdir else default_handoff_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _provenance(sim: dict, info: dict) -> str:
    return (f"# seed={sim['seed']} n_draws={sim['n']} per efficiency structure; priors_version={PRIORS_VERSION}; "
            f"date={info['timestamp']}; git={info['git_commit']}\n")


def write_peak_csvs(sim: dict, info: dict, outdir=None) -> list[Path]:
    """dQ_samples_DY{year}.csv: new PJM summer-peak load (GW), both efficiency structures."""
    out, paths = _handoff_dir(outdir), []
    for year, dy in DELIVERY_YEARS.items():
        frames = []
        for s in EFFICIENCY_STRUCTURES:
            r = sim[(year, s)]
            frames.append(pd.DataFrame({"draw_id": np.arange(sim["n"]), "dQ_GW": r["dQ"], "P0_GW": r["P0"],
                                        "PT_GW": r["PT"], "efficiency_structure": s}))
        path = out / f"dQ_samples_DY{year}.csv"
        header = (f"# AI new PJM summer-peak load samples, delivery year {dy['label']} (T = {dy['T']} years from 2025)\n"
                  + _provenance(sim, info) +
                  "# units: GW of summer-peak load in PJM. dQ_GW = PT_GW - P0_GW (can be negative). Same draw_id = "
                  "same parameter draw in every handoff file.\n"
                  "# efficiency_structure: matures (default) or compounding. See outputs/handoff/README.md.\n")
        with open(path, "w") as f:
            f.write(header)
            pd.concat(frames, ignore_index=True).to_csv(f, index=False, float_format="%.6g")
        paths.append(path)
    return paths


def write_energy_csvs(sim: dict, info: dict, outdir=None) -> list[Path]:
    """energy_samples_{year}.csv: annualized AI electricity use (TWh/yr) in 2025 and in the delivery
    year, globally, in the US and in PJM (for the water footprint)."""
    out, paths = _handoff_dir(outdir), []
    for year, dy in DELIVERY_YEARS.items():
        frames = []
        for s in EFFICIENCY_STRUCTURES:
            cols = {"draw_id": np.arange(sim["n"]), "efficiency_structure": s}
            for r in REGIONS:
                cols[f"E2025_TWh_{r}"] = sim["E0"][r]
                cols[f"E{year}_TWh_{r}"] = sim[(year, s)]["ET"][r]
            frames.append(pd.DataFrame(cols))
        path = out / f"energy_samples_{year}.csv"
        header = (f"# Annualized AI electricity use, 2025 and {year} (T = {dy['T']} years from 2025)\n"
                  + _provenance(sim, info) +
                  "# units: TWh per year (daily energy x 365); GWh/day = TWh/yr x 1000/365. Inference for all "
                  "providers + training (1 + tau); US = x s_US, PJM = x s_US s_PJM; no summer-peak factor.\n"
                  "# boundary: approximately FACILITY energy (Google's 0.24 Wh/query includes data-centre overhead); "
                  "server (IT) energy = facility / PUE. Same draw_id = same parameter draw in every handoff file.\n")
        with open(path, "w") as f:
            f.write(header)
            pd.concat(frames, ignore_index=True).to_csv(f, index=False, float_format="%.6g")
        paths.append(path)
    return paths


def handoff_readme(sim: dict, d: dict, info: dict) -> str:
    """README.md for the handoff folder, with the key numbers filled in."""
    q = d["quantiles"]["2030_matures"]
    t = d["tail"]["2030_matures"]
    E = d["energy"]
    g = TWH_TO_GWH_PER_DAY
    erows = []
    for r in REGIONS:
        b, e = E["baseline_2025"][r], E["cases"]["2030_matures"][r]["E"]
        erows.append(f"| {r} | {b['P50']:.1f} ({b['P50'] * g:.0f}) | {e['P10']:.0f} | {e['P50']:.0f} ({e['P50'] * g:.0f}) "
                     f"| {e['P90']:.0f} ({e['P90'] * g:,.0f}) | {e['mean']:.0f} |")
    erows = "\n".join(erows)
    pjm_c = E["cases"]["2030_compounding"]["PJM"]["E"]
    return f"""# Handoff: AI electricity use and new PJM peak load (samples)

Generated by `scripts/run_all.py` on {info['timestamp']} (git {info['git_commit']}).
Seed {sim['seed']}, {sim['n']:,} parameter draws per efficiency structure, priors version {PRIORS_VERSION}.

## Files

| File | For | Content |
|---|---|---|
| `energy_samples_2028.csv`, `energy_samples_2030.csv` | **Presenter 3 (water footprint)** | Annualized AI electricity use (TWh/yr) in 2025 and in 2028 or 2030: global, US, PJM |
| `dQ_samples_DY2028.csv`, `dQ_samples_DY2030.csv` | Anyone working with peak load or prices (capacity prices, bills) | New PJM summer-peak load ΔQ (GW) for delivery years 2028/29 and 2030/31 |
| `handoff_summary.json` | Everyone | Quantile tables with Monte Carlo standard errors |

Lines starting with `#` are comments (`pandas.read_csv(path, comment="#")`). Every file holds the same
{sim['n']:,} parameter draws under both efficiency structures (`matures` = default, `compounding`):
rows with the same `draw_id` share every parameter **across all four CSVs**, so energy and peak load can be paired.

## Energy files (for the water footprint)

Columns: `draw_id, efficiency_structure, E2025_TWh_global, E{{year}}_TWh_global, E2025_TWh_US, E{{year}}_TWh_US,
E2025_TWh_PJM, E{{year}}_TWh_PJM`.

- **Units:** TWh per year, annualized (daily energy × 365, i.e. average power × 8,760 h).
  GWh/day = TWh/yr × 1000/365.
- **Included:** inference for all providers (requests/day × mean Wh per request, with the reasoning/agentic
  mix and fleet overhead β) plus training as (1 + τ) × inference. US = global × s_US; PJM = global × s_US × s_PJM.
  No summer-peak factor (that is only in the peak-load files).
- **Energy boundary: approximately facility energy.** The per-query energies follow the cited
  measurements, and Google's 0.24 Wh already includes data-centre overhead and idle machines.
  Li et al.'s operational water footprint (arXiv:2304.03271 v5, §3.1) is
  Water = Σₜ eₜ·[ρ_s1,t + θₜ·ρ_s2,t], with eₜ the **server** energy, ρ_s1 the on-site WUE, θ the PUE and
  ρ_s2 the off-site EWIF. With facility energy F = θ·e this becomes **Water = F·(ρ_s1/θ + ρ_s2)**:
  divide by PUE in the on-site term only, or PUE is counted twice. Li et al.'s Table 1 lists PUEs of
  1.11–1.43 by location (U.S. average 1.17; Virginia, inside PJM, 1.14).

2030 (default efficiency), TWh/yr (GWh/day in parentheses):

| Region | 2025 median | 2030 P10 | 2030 median | 2030 P90 | 2030 mean |
|---|---|---|---|---|---|
{erows}

Under compounding efficiency the 2030 PJM median is {pjm_c['P50']:.0f} TWh/yr.

### Why samples and not one number

Water is linear in energy for fixed intensities, so **expected** water needs only expected energy. A
**water budget** is a threshold, though: the probability of exceeding it (and any nonlinear cost of
scarcity) depends on the whole distribution. This is the same quantile logic as capacity: if a budget
must hold with probability 1 − α, plan for the (1 − α) quantile of water use, not its median.

## Peak-load files

Columns: `draw_id, dQ_GW, P0_GW, PT_GW, efficiency_structure`. All values are **GW of summer-peak load inside
PJM** (average power × a summer-peak coincidence factor κ). `P0_GW`: 2025; `PT_GW`: delivery year;
`dQ_GW = PT_GW − P0_GW` (can be negative; the sign is kept). AI only: PJM's "up to ~30 GW" covers all
data-centre growth 2025–2030.

DY 2030/31, default efficiency (GW):

| P10 | P50 | P90 | P95 | mean | P(ΔQ > 30 GW) |
|---|---|---|---|---|---|
| {q['P10']:.2f} | {q['P50']:.2f} | {q['P90']:.1f} | {q['P95']:.1f} | {q['mean']:.1f} | {q['p_exceed']:.3f} |

For prices or bills: push **every sample** through the price function and average. For a convex price
curve p, E[p(ΔQ)] ≥ p(E[ΔQ]) (Jensen), so plugging in the mean or median understates the expected impact.

## Caveats (both kinds of files)

- **The upper tail is not physical.** It comes from multiplying independent, wide lognormal priors. For
  ΔQ in 2030/31: P99 = {t['P99']:.0f} GW, P99.9 = {t['P999']:.0f} GW, largest draw {t['max']:,.0f} GW, while PJM's whole
  forecast summer peak for 2026 is about {PJM_SUMMER_PEAK_2026_GW:.0f} GW ({100 * t['p_exceed_pjm_peak']:.1f}% of draws exceed it).
  The energy files share these draws. **Means are pulled up by this tail; quantiles up to P95 are robust.**
  Cap or winsorize at a level you can defend before averaging a nonlinear function.
- Energy is annualized from the level reached in the given year; seasonal variation (which raises
  on-site WUE in summer) is not modelled.
- Regional totals use fixed shares (s_US, s_PJM); where computing runs is not a decision in this model.
- Priors are judgment-based and independent; presenter 1's per-query numbers are still placeholders.
"""


def write_handoff(sim: dict, d: dict, info: dict, outdir=None, csv: bool = True) -> Path:
    """Write the sample CSVs (energy and peak load), handoff_summary.json and README.md."""
    out = _handoff_dir(outdir)
    if csv:
        write_energy_csvs(sim, info, out)
        write_peak_csvs(sim, info, out)
    hs = {"run": info, "n_draws": sim["n"], "seed": sim["seed"], "priors_version": PRIORS_VERSION,
          "delivery_years": DELIVERY_YEARS,
          "energy": d["energy"],
          "peak_load_GW": {"quantiles": d["quantiles"], "quantile_capacity": d["quantile_capacity"],
                           "pjm_30GW": d["pjm_30GW"], "cost_of_uncertainty": d["cost_of_uncertainty"],
                           "tail": d["tail"], "P0_2025": d["P0"]}}
    (out / "handoff_summary.json").write_text(json.dumps(_jsonable(hs), indent=2))
    (out / "README.md").write_text(handoff_readme(sim, d, info))
    return out




def cv_lognormal_equivalent(x) -> float:
    """CV of the lognormal with the same log-variance as the samples: sqrt(e^{Var ln x} - 1)."""
    return float(np.sqrt(np.expm1(np.log(np.asarray(x, dtype=float)).var())))


def cv_2030_range(sim: dict) -> tuple[float, float]:
    """Delivery-year (2030/31) rate uncertainty used to place AI on the regime map: the
    lognormal-equivalent CV of P_T under the two efficiency structures."""
    v = sorted(cv_lognormal_equivalent(sim[(2030, s)]["PT"]) for s in EFFICIENCY_STRUCTURES)
    return v[0], v[1]
