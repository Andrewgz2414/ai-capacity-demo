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

import numpy as np
from scipy import integrate, special, stats

from .params import (ALTMAN_WH, CHATGPT_PROMPTS_PER_DAY, COST_RATIOS, DELIVERY_YEARS,
                     EFFICIENCY_STRUCTURES, N_DRAWS, PJM_DC_GROWTH_GW, PRIORS, SEED)

HOURS_PER_DAY = 24.0
W_PER_GW = 1e9

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


def simulate(n: int = N_DRAWS, seed: int = SEED, priors: dict | None = None) -> dict:
    """Common random numbers: one parameter draw serves both delivery years and both
    efficiency structures. Returns {"theta": draws, (year, structure): {"dQ","P0","PT"}}."""
    rng = np.random.default_rng(seed)
    th = sample_params(n, rng, priors)
    out = {"theta": th, "n": n, "seed": seed}
    for year, dy in DELIVERY_YEARS.items():
        for s in EFFICIENCY_STRUCTURES:
            out[(year, s)] = delta_q(th, dy["T"], s)
    return out


# ---------------------------------------------------------------------------
# Monte Carlo summaries
# ---------------------------------------------------------------------------


def quantile_table(x, probs=(0.10, 0.50, 0.90), n_boot: int = 200, seed: int = 0,
                   threshold: float = PJM_DC_GROWTH_GW) -> dict:
    """Quantiles with bootstrap SEs, mean with its SE, and P(x > threshold) with its SE."""
    x = np.asarray(x, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    q = np.quantile(x, probs)
    boot = np.empty((n_boot, len(probs)))
    for b in range(n_boot):
        boot[b] = np.quantile(x[rng.integers(0, n, n)], probs)
    p_exc = float(np.mean(x > threshold))
    out = {f"P{round(100 * p)}": float(v) for p, v in zip(probs, q)}
    out.update({f"P{round(100 * p)}_se": float(s) for p, s in zip(probs, boot.std(axis=0, ddof=1))})
    out.update(mean=float(x.mean()), mean_se=float(x.std(ddof=1) / np.sqrt(n)),
               p_exceed=p_exc, p_exceed_se=float(np.sqrt(p_exc * (1 - p_exc) / n)),
               threshold=threshold, p_negative=float(np.mean(x < 0)), n=n, n_boot=n_boot)
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
