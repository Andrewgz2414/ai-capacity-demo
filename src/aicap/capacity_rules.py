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
