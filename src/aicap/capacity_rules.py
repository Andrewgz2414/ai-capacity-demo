"""Three capacity rules for the Erlang-A model with a random arrival rate (spec section 3).

1. Optimal b*: integer minimiser of Pi(b). Pi is discretely convex in b (paper footnote,
   p.7; checked numerically in the tests), so b* is the smallest b whose forward
   difference Pi(b+1) - Pi(b) is >= 0. We find it by bisection on that difference
   (equivalent to an integer golden-section search, simpler, and exact for convex
   sequences); `brute_force_optimal` is the paper's own method (p.10) and is used to
   cross-check.
2. Newsvendor b-bar (paper eq. 3): ignores Poisson variability,
   b-bar = F^{-1}(1 - c/(theta (p + h/gamma))) / theta, rounded DOWN as in all of the
   paper's tables (43.75 -> 43, 87.5 -> 87, 37.5 -> 37).
3. Known-rate b_det: ignores rate uncertainty, the optimum if Lambda were exactly its
   mean: argmin_b C(b; E[Lambda]). Its square-root safety factor is
   beta = (b_det - E)/sqrt(E) with E = E[Lambda]/theta.

gap(rule) = Pi(rule)/Pi* - 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .erlang_a import SECTION4, CostFunction, Params, PointRate, RateDist


def newsvendor_capacity(rate: RateDist, params: Params = SECTION4, rounding: str = "floor"):
    """b-bar = (1/theta) Fbar^{-1}(c/(theta(p + h/gamma))) (eq. 3), rounded as in the paper.

    rounding: "floor" (paper), "nearest", "ceil" or "none" (real-valued).
    """
    alpha = params.fractile
    if alpha >= 1.0:  # capacity dearer than the penalty: install none (p.8)
        return 0
    x = float(rate.ppf(1.0 - alpha)) / params.theta  # Fbar^{-1}(alpha) = F^{-1}(1 - alpha)
    eps = 1e-9  # guard against 349.9999999 from floating point
    if rounding == "floor":
        return int(np.floor(x + eps))
    if rounding == "nearest":
        return int(np.floor(x + 0.5))
    if rounding == "ceil":
        return int(np.ceil(x - eps))
    if rounding == "none":
        return x
    raise ValueError(rounding)


def forward_difference(cost: CostFunction, b: int) -> float:
    return cost(b + 1) - cost(b)


def optimal_capacity(cost: CostFunction, lo: int, hi: int) -> int:
    """Smallest b >= 0 with Pi(b+1) - Pi(b) >= 0, starting from the bracket [lo, hi].

    The bracket is widened automatically until Delta(lo) < 0 <= Delta(hi)."""
    lo, hi = max(int(lo), 0), max(int(hi), int(lo) + 1)
    step = max(hi - lo, 4)
    while forward_difference(cost, hi) < 0:
        lo, hi, step = hi, hi + step, 2 * step
    step = max(hi - lo, 4)
    while lo > 0 and forward_difference(cost, lo) >= 0:
        hi, lo, step = lo, max(lo - step, 0), 2 * step
    if forward_difference(cost, lo) >= 0:  # lo == 0 and cost already increasing
        return 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if forward_difference(cost, mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def brute_force_optimal(cost: CostFunction, b_values) -> tuple[int, float, np.ndarray]:
    """Evaluate Pi on every b in `b_values` and return (argmin, min, all values)."""
    b_values = np.asarray(b_values, dtype=int)
    vals = np.array([cost(b) for b in b_values])
    i = int(np.argmin(vals))
    return int(b_values[i]), float(vals[i]), vals


def min_second_difference(values) -> float:
    """min_b [Pi(b+1) - 2 Pi(b) + Pi(b-1)]; >= 0 (up to rounding) means discretely convex."""
    v = np.asarray(values, dtype=float)
    return float(np.min(v[2:] - 2 * v[1:-1] + v[:-2]))


def known_rate_capacity(rate: RateDist, params: Params = SECTION4) -> int:
    """argmin_b C(b; E[Lambda]): the best capacity if the rate were known to equal its mean."""
    point = PointRate(rate.mean)
    E = rate.mean / params.theta
    cost = CostFunction(point, params)
    return optimal_capacity(cost, int(np.floor(E)), int(np.ceil(E + 3 * np.sqrt(E) + 3)))


def safety_factor(b: float, E: float) -> float:
    """beta in b = E + beta sqrt(E) (square-root staffing)."""
    return (b - E) / np.sqrt(E)


@dataclass
class RuleResult:
    label: str
    E: float  # offered load E[Lambda]/theta (Erlangs)
    cv: float  # CV of Lambda
    b_star: int
    Pi_star: float
    b_bar: int
    b_bar_real: float
    Pi_bar: float
    gap_bar: float
    b_det: int
    Pi_det: float
    gap_det: float
    beta_det: float
    n_evals: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_rules(rate: RateDist, params: Params = SECTION4, method: str = "auto",
                   n_nodes: int = 64, search: str = "bisect", **kw) -> RuleResult:
    """Compute b*, b-bar and b_det and their costs under the true Pi."""
    E = rate.mean / params.theta
    cost = CostFunction(rate, params, method=method, n_nodes=n_nodes, **kw)
    b_bar = newsvendor_capacity(rate, params)
    b_bar_real = newsvendor_capacity(rate, params, rounding="none")
    b_det = known_rate_capacity(rate, params)
    if search == "bisect":
        lo = max(0, min(b_bar, b_det) - 2)
        hi = max(b_bar, b_det) + 2
        b_star = optimal_capacity(cost, lo, hi)
    elif search == "brute":
        top = max(b_bar, b_det) + int(4 * np.sqrt(E + 1)) + 10
        b_star, _, _ = brute_force_optimal(cost, np.arange(0, top + 1))
    else:
        raise ValueError(search)
    Pi_star = cost(b_star)
    Pi_bar, Pi_det = cost(b_bar), cost(b_det)
    return RuleResult(
        label=rate.label, E=E, cv=rate.cv, b_star=b_star, Pi_star=Pi_star,
        b_bar=b_bar, b_bar_real=float(b_bar_real), Pi_bar=Pi_bar, gap_bar=Pi_bar / Pi_star - 1.0,
        b_det=b_det, Pi_det=Pi_det, gap_det=Pi_det / Pi_star - 1.0,
        beta_det=safety_factor(b_det, E), n_evals=cost.n_evals)


def fit_safety_factor(b_values, E_values) -> float:
    """Least-squares beta in b - E = beta sqrt(E) across several fixed-rate cases."""
    b, E = np.asarray(b_values, float), np.asarray(E_values, float)
    return float(np.sum((b - E) * np.sqrt(E)) / np.sum(E))


# ---------------------------------------------------------------------------
# The paper's published numbers (working-paper PDF of 10 May 2010) and reproduction
# ---------------------------------------------------------------------------

# (b*, Pi*, b-bar, Pi(b-bar), gap) as printed. Keys: (lo, hi) for U[lo, hi], float for fixed rates.
PAPER_TABLE2_3 = {  # Table 2 and Table 3, p.9
    (25, 50): (46, 17.34, 43, 17.49, 0.009),
    (50, 100): (89, 33.13, 87, 33.20, 0.002),
    (200, 400): (351, 127.08, 350, 127.08, 0.00006),
    37.5: (42, 15.94, 37, 17.67, 0.109),
    75.0: (83, 29.64, 75, 31.60, 0.066),
    300.0: (316, 109.01, 300, 112.4, 0.031),
}
PAPER_TABLE4 = {  # Table 4, p.13 (E = 150)
    (0, 300): (224, 88.34, 225, 88.34, 0.0),
    (125, 175): (165, 59.06, 162, 59.16, 0.002),
    (135, 165): (162, 57.40, 157, 57.78, 0.007),
    (140, 160): (162, 56.78, 155, 57.42, 0.011),
    (145, 155): (161, 56.40, 152, 57.73, 0.024),
    150.0: (161, 56.26, 150, 58.25, 0.035),
}
PAPER_TABLE7 = {  # Table 7, Appendix C.1, p.36 (rows not already in Table 2)
    (1, 2): (3, 1.43, 1, 5.26, 2.678),
    (2, 4): (5, 2.19, 3, 3.33, 0.521),
    (5, 10): (11, 4.32, 8, 5.01, 0.164),
    (10, 20): (20, 7.68, 17, 8.01, 0.043),
    (15, 30): (29, 10.95, 26, 11.12, 0.016),
    (20, 40): (37, 14.15, 35, 14.26, 0.007),
}


def rate_from_key(key) -> RateDist:
    from .erlang_a import UniformRate
    return UniformRate(*key) if isinstance(key, tuple) else PointRate(key)


def reproduction_rows(table: dict, params: Params = SECTION4, source: str = "") -> list[dict]:
    """Our three rules next to the paper's printed numbers, one dict per case."""
    rows = []
    for key, (pb, pP, pbb, pPb, pgap) in table.items():
        res = evaluate_rules(rate_from_key(key), params)
        rows.append(dict(
            case=res.label, source=source, E=res.E, cv=res.cv,
            b_star=res.b_star, Pi_star=res.Pi_star, b_bar=res.b_bar, Pi_bar=res.Pi_bar,
            gap_bar=res.gap_bar, b_det=res.b_det, Pi_det=res.Pi_det, gap_det=res.gap_det,
            beta_det=res.beta_det,
            paper_b_star=pb, paper_Pi_star=pP, paper_b_bar=pbb, paper_Pi_bar=pPb, paper_gap_bar=pgap,
            match_b=(res.b_star == pb and res.b_bar == pbb),
            max_abs_dPi=max(abs(res.Pi_star - pP), abs(res.Pi_bar - pPb))))
    return rows


# ---------------------------------------------------------------------------
# Regime map (spec section 4): which rule is cheaper over (offered load E, rate CV)
# ---------------------------------------------------------------------------

GAP_FLOOR = 1e-4  # 0.01%: gaps below this are treated as "practically optimal" in the log ratio


def regime_grid(E_values, cv_values, dist: str = "gamma", params: Params = SECTION4,
                progress: bool = False, **kw) -> dict:
    """Pi*, Pi(b-bar), Pi(b_det) over a grid of offered load E (Erlangs) and rate CV.

    dist: "gamma" or "lognormal" (same mean and CV). kw goes to the "split" integrator
    (nodes_per_panel, width_factor), so a refined rerun checks quadrature convergence.
    """
    from .erlang_a import GammaRate, LognormalRate
    make = {"gamma": GammaRate, "lognormal": LognormalRate}[dist]
    E_values, cv_values = np.asarray(E_values, float), np.asarray(cv_values, float)
    shape = (E_values.size, cv_values.size)
    keys = ["b_star", "Pi_star", "b_bar", "Pi_bar", "gap_bar", "b_det", "Pi_det", "gap_det", "n_evals"]
    out = {k: np.zeros(shape) for k in keys}
    for i, E in enumerate(E_values):
        for j, cv in enumerate(cv_values):
            res = evaluate_rules(make(E * params.theta, cv), params, method="split", **kw)
            for k in keys:
                out[k][i, j] = getattr(res, k)
        if progress:
            print(f"  {dist}: E = {E:8.1f} done", flush=True)
    out["E"], out["cv"], out["dist"] = E_values, cv_values, dist
    out["log_ratio"] = np.log10((out["gap_bar"] + GAP_FLOOR) / (out["gap_det"] + GAP_FLOOR))
    return out


def boundary_analysis(grid: dict) -> dict:
    """Where does the cheaper rule switch, and does CV = 1/sqrt(E) separate the regions?

    For each E, the crossing CV is found by linear interpolation of the log ratio in log CV.
    kappa = CV_cross * sqrt(E) (Theorem 1 predicts kappa = O(1), constant in E).
    """
    E, cv, lr = grid["E"], grid["cv"], grid["log_ratio"]
    kappas = []
    for i in range(E.size):
        row = lr[i]
        idx = np.flatnonzero((row[:-1] > 0) & (row[1:] <= 0))  # known-rate better -> newsvendor better
        if idx.size:
            j = idx[0]
            t = row[j] / (row[j] - row[j + 1])
            log_cv = np.log(cv[j]) + t * (np.log(cv[j + 1]) - np.log(cv[j]))
            kappas.append(float(np.exp(log_cv) * np.sqrt(E[i])))
        else:
            kappas.append(np.nan)
    kappas = np.array(kappas)
    EE, CC = np.meshgrid(E, cv, indexing="ij")
    above = CC > 1 / np.sqrt(EE)
    nv_better = lr < 0
    tie = np.isclose(grid["gap_bar"], grid["gap_det"])
    agree = np.where(tie, True, above == nv_better)
    kappa_med = float(np.nanmedian(kappas))
    above_k = CC > kappa_med / np.sqrt(EE)
    agree_k = np.where(tie, True, above_k == nv_better)
    return dict(kappa_by_E=kappas, kappa_median=kappa_med,
                kappa_min=float(np.nanmin(kappas)), kappa_max=float(np.nanmax(kappas)),
                frac_agree_theory_line=float(agree.mean()), frac_agree_fitted_line=float(agree_k.mean()),
                n_cells=int(agree.size), n_ties=int(tie.sum()))


REGIME_CACHE = None  # set lazily to outputs/cache/regime_grid.npz


def regime_cache_path():
    from . import OUTPUTS
    return OUTPUTS / "cache" / "regime_grid.npz"


def load_regime_grid(path=None) -> dict:
    """Load the cached regime map written by scripts/precompute_regime.py."""
    import json
    path = path or regime_cache_path()
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["meta"] = json.loads(str(data.pop("meta_json")))
    return data
