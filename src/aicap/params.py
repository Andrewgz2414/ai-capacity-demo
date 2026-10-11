"""All inputs of the PJM load model (spec section 6), each with distribution, range and source.

Lognormal priors are specified by their 5th and 95th percentiles (P5-P95):
    mu = (ln lo + ln hi)/2,   s = (ln hi - ln lo)/(2 x 1.6449).
Uniform priors are specified by their bounds. All priors are independent (an assumption).

>>> EDIT HERE when presenter 1 hands over their numbers: the PRESENTER1 block below. <<<
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import special

PRIORS_VERSION = "1.0 (2026-10-11)"
Z95 = 1.6449  # standard-normal 95th percentile used to turn P5-P95 into a log-sd

# ---------------------------------------------------------------------------
# Presenter 1's numbers: PLACEHOLDERS until they hand over theirs (single place to edit)
# ---------------------------------------------------------------------------
PRESENTER1 = {
    # median Wh per chat query, lognormal P5-P95
    "M_c": (0.20, 0.40),
    # per-query log-sd sigma_c, uniform bounds
    "sigma_c": (0.7, 1.3),
    # reasoning energy multiplier r (energy per reasoning request / chat), lognormal P5-P95
    "r": (5.0, 30.0),
    # agentic energy multiplier a (energy per agentic task / chat), lognormal P5-P95
    "a": (30.0, 1000.0),
}
PRESENTER1_STATUS = "placeholder (spec section 6 values); replace with presenter 1's hand-over"


@dataclass(frozen=True)
class Prior:
    name: str
    symbol: str
    description: str
    dist: str  # "lognormal" (lo, hi = P5, P95) or "uniform" (lo, hi = bounds)
    lo: float
    hi: float
    unit: str
    group: str  # sensitivity group (spec section 6, optional Sobol part)
    source: str
    assumption: bool = False  # True where the range is judgment, not data

    # --- distribution helpers -------------------------------------------------
    @property
    def mu(self) -> float:
        return 0.5 * (np.log(self.lo) + np.log(self.hi)) if self.dist == "lognormal" else np.nan

    @property
    def s(self) -> float:
        return (np.log(self.hi) - np.log(self.lo)) / (2 * Z95) if self.dist == "lognormal" else np.nan

    @property
    def median(self) -> float:
        return float(np.exp(self.mu)) if self.dist == "lognormal" else 0.5 * (self.lo + self.hi)

    @property
    def mean(self) -> float:
        return float(np.exp(self.mu + self.s**2 / 2)) if self.dist == "lognormal" else 0.5 * (self.lo + self.hi)

    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        if self.dist == "lognormal":
            return np.exp(self.mu + self.s * special.ndtri(u))
        return self.lo + u * (self.hi - self.lo)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.dist == "lognormal":
            return np.exp(self.mu + self.s * rng.standard_normal(n))
        return rng.uniform(self.lo, self.hi, n)

    def scaled(self, factor: float) -> "Prior":
        """Shrink (factor < 1) the uncertainty, keeping the centre: lognormal log-sd x factor
        around the median; uniform half-width x factor around the midpoint."""
        if self.dist == "lognormal":
            m, half = self.mu, 0.5 * (np.log(self.hi) - np.log(self.lo)) * factor
            return replace(self, lo=float(np.exp(m - half)), hi=float(np.exp(m + half)))
        mid, half = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo) * factor
        return replace(self, lo=mid - half, hi=mid + half)


def _p(*args, **kw) -> Prior:
    return Prior(*args, **kw)


P1 = PRESENTER1
PRIORS: dict[str, Prior] = {p.name: p for p in [
    _p("N0", "N₀", "Global AI requests per day, 2025", "lognormal", 8e9, 5e10, "requests/day",
       "base volume",
       "ChatGPT 2.5 B prompts/day (TechCrunch, Jul 2025); Google ~480 T tokens/month (May 2025) at an "
       "ASSUMED 1-3 k tokens/request; other providers by judgment", True),
    _p("M_c", "M_c", "Median Wh per chat query (presenter 1)", "lognormal", *P1["M_c"], "Wh",
       "Wh per query",
       "Google 0.24 Wh (Elsworth et al. 2025); Altman 0.34 Wh; Oviedo et al. 0.31 Wh"),
    _p("sigma_c", "σ_c", "Per-query log-sd (presenter 1)", "uniform", *P1["sigma_c"], "-", "Wh per query",
       "Oviedo IQR 0.16-0.60 around 0.31 implies sigma ~ 0.98; +/-0.3 by judgment", True),
    _p("beta", "β", "Fleet overhead (network x daily load x redundancy)", "uniform", 1.00, 1.31, "-",
       "Wh per query",
       "Oviedo beta = 1.31 = 1.10 network x 1.08 daily load x 1.10 redundancy; 1.0 if the median "
       "already includes idle capacity"),
    _p("r", "r", "Reasoning energy multiplier (presenter 1)", "lognormal", *P1["r"], "x chat", "mix",
       "Oviedo: 13x median energy for 15x tokens; AI Energy Score v2: 30x average (GPU-only, non-production)"),
    _p("a", "a", "Agentic energy multiplier per task (presenter 1)", "lognormal", *P1["a"], "x chat", "mix",
       "Bai et al. 2026: ~1000x tokens, input-dominated, so the energy multiple is lower. ASSUMPTION", True),
    _p("w_R0", "w_R,0", "Reasoning share of requests, 2025", "uniform", 0.05, 0.15, "share", "mix",
       "Oviedo's 90/10 split"),
    _p("w_RT", "w_R,T", "Reasoning share of requests, delivery year", "uniform", 0.15, 0.50, "share", "mix",
       "Later share by judgment", True),
    _p("w_A0", "w_A,0", "Agentic share of requests, 2025", "uniform", 0.001, 0.005, "share", "mix",
       "ASSUMPTION", True),
    _p("w_AT", "w_A,T", "Agentic share of requests, delivery year", "lognormal", 0.005, 0.08, "share", "mix",
       "ASSUMPTION; widest prior", True),
    _p("g1", "g₁", "First-year volume growth factor", "lognormal", 1.8, 5.0, "x/yr", "growth",
       "ChatGPT 1 B -> 2.5 B/day (Dec 2024-Jul 2025); Google tokens 6.7x (May 2025-May 2026)"),
    _p("lam", "λ", "Growth decay rate (Gompertz)", "uniform", 0.2, 0.7, "1/yr", "growth",
       "Two-point fit to Google's token growth: ln(49) -> ln(6.7) gives ~0.7"),
    _p("h", "h", "First-year efficiency gain", "lognormal", 1.15, 2.2, "x/yr", "efficiency",
       "Oviedo 8-20x line-of-sight; Blackwell 1.5-2.5x per watt; AI Energy Score v2: most newer models "
       "not more efficient"),
    _p("tau", "τ", "Training-to-inference energy ratio", "lognormal", 0.25, 1.5, "-", "training",
       "Patterson et al. 2022 (~3/5 inference). VERIFY"),
    _p("s_US", "s_US", "US share of global AI load", "uniform", 0.40, 0.60, "share", "siting and peak",
       "IEA 2025: US 45% of 2024 data-centre electricity"),
    _p("s_PJM", "s_PJM", "PJM share of US data-centre load", "uniform", 0.30, 0.50, "share", "siting and peak",
       "PJM Nov 2025: PJM ~40% of US data centres in 2025"),
    _p("kappa", "κ", "Summer-peak coincidence factor", "uniform", 1.0, 1.3, "-", "siting and peak",
       "Oviedo S5: peak/average node power 1/(f + (1-f)/(1+a)) = 1.235; PJM's flat 70% utilization ~ 1.0"),
]}

GROUPS = ["growth", "mix", "base volume", "efficiency", "Wh per query", "training", "siting and peak"]

# Delivery years: horizon T in years from 2025 to the summer of the delivery year
# (PJM delivery years run June-May).
DELIVERY_YEARS = {
    2028: {"label": "2028/29", "T": 3, "summer": 2028},
    2030: {"label": "2030/31", "T": 5, "summer": 2030},
}
EFFICIENCY_STRUCTURES = ("matures", "compounding")  # default first

# Simulation settings
N_DRAWS = 200_000
SEED = 20251011

# Reference point for the Fermi ladder (spec section 6)
CHATGPT_PROMPTS_PER_DAY = 2.5e9  # TechCrunch, 21 Jul 2025
ALTMAN_WH = 0.34  # Wh per ChatGPT query (Altman)
PJM_DC_GROWTH_GW = 30.0  # PJM Load Analysis Subcommittee, 24 Nov 2025: up to ~30 GW of data-centre growth 2025-2030
COST_RATIOS = (0.05, 0.10, 0.25, 0.50)  # c_cap / c_short for the quantile-capacity reading


def priors_table() -> list[dict]:
    """One row per prior, for the guide and the notebook."""
    rows = []
    for p in PRIORS.values():
        rows.append(dict(name=p.name, symbol=p.symbol, description=p.description, dist=p.dist,
                         lo=p.lo, hi=p.hi, median=p.median, mean=p.mean,
                         log_sd=(p.s if p.dist == "lognormal" else None), unit=p.unit, group=p.group,
                         source=p.source, assumption=p.assumption))
    return rows


def scaled_priors(groups, factor: float, priors: dict | None = None) -> dict:
    """Priors with the uncertainty of the given groups scaled by `factor` (e.g. 0.5 = halved)."""
    priors = priors or PRIORS
    return {k: (p.scaled(factor) if p.group in groups else p) for k, p in priors.items()}
