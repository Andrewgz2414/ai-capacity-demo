"""AI queries as a Cox (doubly stochastic Poisson) process with lognormal per-query energy.

Model (spec section 5): daily count N | xi ~ Poisson(nbar * xi), E[xi] = 1, CV(xi) = CV_rate
(xi gamma-distributed here, so N is negative binomial); energies X_i i.i.d. lognormal with
median M and log-sd sigma; daily energy S = sum_{i <= N} X_i.

    m1 = E[X] = M e^{sigma^2/2},   m2 = E[X^2] = M^2 e^{2 sigma^2}
    E[S | xi] = nbar xi m1,   Var(S | xi) = nbar xi m2          (compound Poisson)
    Var S = E[Var(S | xi)] + Var(E[S | xi]) = nbar m2 + nbar^2 m1^2 CV_rate^2
          = aleatory (per-query noise)      + epistemic (rate uncertainty)
    CV^2(S) = e^{sigma^2}/nbar + CV_rate^2     -> floor CV_rate as nbar -> infinity
    crossover nbar* = e^{sigma^2}/CV_rate^2;   effective load E_eff = nbar/e^{sigma^2}
"""

from __future__ import annotations

import numpy as np
from scipy import special

SECONDS_PER_DAY = 86_400.0

# ---------------------------------------------------------------------------
# Lognormal per-query energy
# ---------------------------------------------------------------------------


def lognormal_moments(median: float, sigma: float) -> tuple[float, float]:
    """(m1, m2) = (E[X], E[X^2]) for X lognormal with the given median and log-sd."""
    return median * np.exp(sigma**2 / 2), median**2 * np.exp(2 * sigma**2)


def mean_over_median(sigma: float) -> float:
    """E[X]/median = e^{sigma^2/2}: the only per-query statistic that survives aggregation."""
    return float(np.exp(sigma**2 / 2))


def energy_share_above_quantile(sigma: float, q: float) -> float:
    """Share of total energy from queries above the q-quantile: 1 - Phi(z_q - sigma).

    E[X 1{X > M e^{sigma z}}]/E[X] = 1 - Phi(z - sigma) (complete the square)."""
    z = special.ndtri(q)
    return float(1 - special.ndtr(z - sigma))


# ---------------------------------------------------------------------------
# Compound Poisson / Cox moments
# ---------------------------------------------------------------------------


def daily_energy_moments(nbar: float, sigma: float, cv_rate: float, median: float = 1.0) -> dict:
    """Mean, variance and its aleatory/epistemic split for daily energy S."""
    m1, m2 = lognormal_moments(median, sigma)
    mean = nbar * m1
    var_aleatory = nbar * m2  # E[Var(S | xi)] = E[nbar xi m2]
    var_epistemic = (nbar * m1 * cv_rate) ** 2  # Var(E[S | xi]) = Var(nbar xi m1)
    var = var_aleatory + var_epistemic
    return dict(mean=mean, var=var, var_aleatory=var_aleatory, var_epistemic=var_epistemic,
                cv=np.sqrt(var) / mean, m1=m1, m2=m2)


def cv_daily_energy(nbar, sigma: float, cv_rate: float):
    """CV(S) = sqrt(e^{sigma^2}/nbar + CV_rate^2)."""
    return np.sqrt(np.exp(sigma**2) / np.asarray(nbar, dtype=float) + cv_rate**2)


def crossover_n(sigma: float, cv_rate: float) -> float:
    """nbar* = e^{sigma^2}/CV_rate^2: below it per-query noise dominates, above it the rate does."""
    return float(np.exp(sigma**2) / cv_rate**2)


def effective_load(nbar, sigma: float):
    """E_eff = nbar/e^{sigma^2}: per-query noise CV is 1/sqrt(E_eff), as for a Poisson load E_eff."""
    return np.asarray(nbar, dtype=float) / np.exp(sigma**2)


def concurrency(prompts_per_day: float, mean_service_s: float) -> float:
    """Little's law: requests in service R = arrival rate (1/s) x mean service time (s)."""
    return prompts_per_day / SECONDS_PER_DAY * mean_service_s


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def draw_rate_multipliers(n: int, cv_rate: float, rng: np.random.Generator) -> np.ndarray:
    """xi ~ Gamma(mean 1, CV cv_rate); xi = 1 when cv_rate = 0."""
    if cv_rate == 0:
        return np.ones(n)
    k = 1.0 / cv_rate**2
    return rng.gamma(k, 1.0 / k, size=n)


def simulate_daily_energy(nbar: float, sigma: float, cv_rate: float, n_days: int,
                          rng: np.random.Generator, median: float = 1.0,
                          max_draws_per_chunk: int = 20_000_000) -> dict:
    """Simulate n_days of (xi, N, S) exactly (no normal approximation), in memory-bounded chunks."""
    xi = draw_rate_multipliers(n_days, cv_rate, rng)
    N = rng.poisson(nbar * xi)
    S = np.zeros(n_days)
    log_m = np.log(median)
    start = 0
    while start < n_days:
        # take as many days as fit in the draw budget (at least one)
        cum = np.cumsum(N[start:])
        stop = start + max(1, int(np.searchsorted(cum, max_draws_per_chunk, side="right")))
        counts = N[start:stop]
        total = int(counts.sum())
        if total:
            x = np.exp(log_m + sigma * rng.standard_normal(total))
            day = np.repeat(np.arange(stop - start), counts)
            S[start:stop] = np.bincount(day, weights=x, minlength=stop - start)
        start = stop
    return dict(xi=xi, N=N, S=S)


def days_for_budget(nbar: float, budget: float = 2e8, lo: int = 200, hi: int = 20_000) -> int:
    """Number of simulated days so that about `budget` per-query draws are made."""
    return int(np.clip(budget / nbar, lo, hi))


def simulate_cv_curve(nbar_values, sigmas, cv_rate: float, seed: int = 20251011,
                      budget: float = 2e8, n_boot: int = 200) -> dict:
    """Sample CV of daily energy for each (sigma, nbar), with a bootstrap 90% interval."""
    rng = np.random.default_rng(seed)
    nbar_values = np.asarray(nbar_values, dtype=float)
    shape = (len(sigmas), nbar_values.size)
    cv_hat, lo, hi, days = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape, int)
    for i, s in enumerate(sigmas):
        for j, nb in enumerate(nbar_values):
            R = days_for_budget(nb, budget)
            S = simulate_daily_energy(nb, s, cv_rate, R, rng)["S"]
            cv_hat[i, j] = S.std(ddof=1) / S.mean()
            idx = rng.integers(0, R, size=(n_boot, R))
            Sb = S[idx]
            cvb = Sb.std(axis=1, ddof=1) / Sb.mean(axis=1)
            lo[i, j], hi[i, j] = np.quantile(cvb, [0.05, 0.95])
            days[i, j] = R
    return dict(nbar=nbar_values, sigmas=np.asarray(sigmas, float), cv_rate=cv_rate, cv_hat=cv_hat,
                cv_lo=lo, cv_hi=hi, n_days=days, seed=seed, budget=budget)


def rao_blackwell_demo(nbar: float, sigma: float, cv_rate: float, n_days: int,
                       rng: np.random.Generator, median: float = 1.0) -> dict:
    """Replace S by E[S | xi] = nbar xi m1: same mean, variance can only fall (by Var(S|xi))."""
    sim = simulate_daily_energy(nbar, sigma, cv_rate, n_days, rng, median)
    m1, _ = lognormal_moments(median, sigma)
    rb = nbar * sim["xi"] * m1
    return dict(mean_S=float(sim["S"].mean()), mean_RB=float(rb.mean()),
                var_S=float(sim["S"].var(ddof=1)), var_RB=float(rb.var(ddof=1)),
                theory=daily_energy_moments(nbar, sigma, cv_rate, median))


# ---------------------------------------------------------------------------
# Where AI sits on the paper's regime map
# ---------------------------------------------------------------------------


def ai_regime_numbers(nbar_range=(1e10, 2e10), sigma_range=(1.0, 3.0), prompts_per_day=2.5e9,
                      service_s=10.0, service_range_s=(2.0, 60.0),
                      cv_day_ahead=(0.05, 0.2), cv_2030=None) -> dict:
    """The two 'offered load' scales for AI and the per-query noise 1/sqrt(E) at each.

    Daily-energy scale: E_eff = nbar/e^{sigma^2}. Instantaneous concurrency: R = arrivals/s x E[S].
    cv_2030: (lo, hi) rate CV for the delivery-year horizon (from the PJM model), optional.
    """
    e_lo = float(effective_load(nbar_range[0], sigma_range[1]))
    e_hi = float(effective_load(nbar_range[1], sigma_range[0]))
    R = concurrency(prompts_per_day, service_s)
    R_lo, R_hi = concurrency(prompts_per_day, service_range_s[0]), concurrency(prompts_per_day, service_range_s[1])
    out = dict(
        assumptions=dict(nbar_per_day=list(nbar_range), sigma=list(sigma_range),
                         prompts_per_day_chatgpt=prompts_per_day, mean_service_s=service_s,
                         mean_service_range_s=list(service_range_s), cv_day_ahead=list(cv_day_ahead)),
        daily=dict(E_eff_lo=e_lo, E_eff_hi=e_hi, noise_hi=1 / np.sqrt(e_lo), noise_lo=1 / np.sqrt(e_hi)),
        concurrency=dict(arrivals_per_s=prompts_per_day / SECONDS_PER_DAY, R=R, R_lo=R_lo, R_hi=R_hi,
                         noise=1 / np.sqrt(R), noise_lo=1 / np.sqrt(R_hi), noise_hi=1 / np.sqrt(R_lo)),
    )
    # CV x sqrt(E): >> 1 means uncertainty-dominated (paper's Theorem 1, part a)
    out["ratio_day_ahead_vs_concurrency"] = [cv_day_ahead[0] * np.sqrt(R_lo), cv_day_ahead[1] * np.sqrt(R_hi)]
    out["ratio_day_ahead_vs_concurrency_central"] = [cv_day_ahead[0] * np.sqrt(R), cv_day_ahead[1] * np.sqrt(R)]
    if cv_2030 is not None:
        out["cv_2030"] = list(cv_2030)
        out["ratio_2030_vs_daily"] = [cv_2030[0] * np.sqrt(e_lo), cv_2030[1] * np.sqrt(e_hi)]
    return out


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """2.9e5 -> '2.9×10⁵' (figure labels)."""
    m, e = f"{x:.{digits}e}".split("e")
    if "." in m:
        m = m.rstrip("0").rstrip(".")
    return f"{m}×10{str(int(e)).translate(_SUPERSCRIPT)}"


def regime_boxes(ai: dict) -> dict:
    """Boxes and labels for the zoomed-out panel of the regime map (figure 2)."""
    c, d = ai["concurrency"], ai["daily"]
    cv_da, cv30 = ai["assumptions"]["cv_day_ahead"], ai["cv_2030"]
    return {
        "concurrency": (c["R_lo"], c["R_hi"], cv_da[0], cv_da[1]),
        "daily_2030": (d["E_eff_lo"], d["E_eff_hi"], cv30[0], cv30[1]),
        "labels": {
            "concurrency": f"AI today, day-ahead:\nR ≈ {sci(c['R'])} in service, "
                           f"CV {100 * cv_da[0]:.0f}–{100 * cv_da[1]:.0f}%",
            "daily_2030": f"AI in 2030, daily energy:\nE_eff ≈ {sci(d['E_eff_lo'], 0)}–{sci(d['E_eff_hi'], 0)}, "
                          f"CV ≈ {cv30[0]:.1f}–{cv30[1]:.1f}",
        },
    }
