"""Erlang-A (M/M/b+M) queue with a random arrival rate.

Model of Bassamboo, Randhawa & Zeevi (2010), section 3 (p.7 of the working paper):
given Lambda = lam, the number in system N is a birth-death chain with birth rate lam
and death rate d(n) = min(n, b)*theta + (n - b)^+ * gamma. Capacity b is fixed before
Lambda is revealed, so the cost is averaged over Lambda (eq. 1):

    C(b; lam) = c*b + (h + p*gamma) * E_lam[Q],      Q = (N - b)^+
    Pi(b)     = E_Lambda[ C(b; Lambda) ].

Notation follows the paper: theta is the service rate (the spec calls it mu).

Two ways to integrate over Lambda are provided:
  * "gauss": Gauss quadrature for the rate distribution (Legendre for uniforms, as the
    spec asks; generalized Laguerre for gamma; Hermite for lognormal).
  * "split": the exact decomposition Pi(b) = Pibar(b) + (h + p*gamma) * E[D(Lambda, b)],
    where Pibar is the newsvendor cost (closed form) and
    D(lam, b) = E_lam[Q] - (lam - b*theta)^+ / gamma >= 0 is the "stochastic correction",
    which is negligible unless lam is within a few sqrt(b) of b*theta. Only that window
    needs quadrature, which makes wide rate distributions (regime map) cheap and accurate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, special, stats

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """Queue and cost parameters. Defaults are the paper's section 4.1 values (p.9)."""

    theta: float = 1.0  # service rate (mean service time 1)
    gamma: float = 3.0  # abandonment rate (mean patience 1/3)
    c: float = 1.0 / 3.0  # staffing cost per server per unit time
    h: float = 1.0  # holding cost per waiting customer per unit time
    p: float = 1.0  # penalty per abandonment

    @property
    def queue_weight(self) -> float:
        """Cost per unit of E[Q]: h + p*gamma (eq. 1)."""
        return self.h + self.p * self.gamma

    @property
    def underage(self) -> float:
        """Newsvendor price per unit of unserved rate: p + h/gamma (eq. 2)."""
        return self.p + self.h / self.gamma

    @property
    def fractile(self) -> float:
        """Critical fractile P(Lambda > theta*bbar) = c / (theta*(p + h/gamma)) (eq. 3)."""
        return self.c / (self.theta * self.underage)


SECTION4 = Params()

# ---------------------------------------------------------------------------
# Stationary law of the birth-death chain
# ---------------------------------------------------------------------------


def death_rates(n: np.ndarray, b: int, theta: float, gamma: float) -> np.ndarray:
    """d(n) = min(n, b)*theta + (n - b)^+ * gamma."""
    n = np.asarray(n, dtype=float)
    return np.minimum(n, b) * theta + np.maximum(n - b, 0.0) * gamma


def _n_max_guess(lam_max: float, b: int, theta: float, gamma: float) -> int:
    """Initial truncation level: mean of N plus 15 spreads."""
    mean_n = min(lam_max / theta, b) + max(lam_max - b * theta, 0.0) / gamma
    spread = np.sqrt(lam_max / min(theta, gamma) + 1.0)
    return int(np.ceil(mean_n + 15.0 * spread + 30.0))


def _check_b(b) -> int:
    if int(b) != b or b < 0:
        raise ValueError(f"b must be a non-negative integer, got {b!r}")
    return int(b)


@dataclass
class QueueMoments:
    """Stationary quantities of N, one entry per arrival rate."""

    lam: np.ndarray
    EQ: np.ndarray  # E[(N - b)^+], mean queue length
    EN: np.ndarray  # E[N]
    E_busy: np.ndarray  # E[min(N, b)], mean busy servers
    E_idle: np.ndarray  # E[(b - N)^+], mean idle servers
    P_wait: np.ndarray  # P(N >= b): an arrival must wait (PASTA)
    tail_mass: np.ndarray  # upper bound on the probability mass beyond the truncation
    tail_EQ: np.ndarray  # upper bound on the E[Q] contribution beyond the truncation
    n_max: int  # largest truncation level used


def queue_moments(lam, b, theta: float = 1.0, gamma: float = 3.0, tol: float = 1e-12,
                  chunk: int = 256) -> QueueMoments:
    """Exact stationary moments of the M/M/b+M chain for each arrival rate in `lam`.

    pi(n) is proportional to prod_{k<=n} lam/d(k); in log space
    log pi(n) = n*log(lam) - sum_{k<=n} log d(k) + const. The cumulative sum is shared by
    all arrival rates, so a batch of rates costs one outer product.

    Truncation: d(n) is non-decreasing, so for r = lam/d(n_max+1) < 1 the mass beyond
    n_max is at most pi(n_max)*r/(1-r) (geometric bound) and its E[Q] contribution at
    most pi(n_max)*[(n_max-b)^+ r/(1-r) + r/(1-r)^2]. n_max is enlarged until both bounds
    are below `tol`.
    """
    b = _check_b(b)
    if theta <= 0 or gamma <= 0:
        raise ValueError("theta and gamma must be positive")
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    if np.any(lam < 0) or not np.all(np.isfinite(lam)):
        raise ValueError("arrival rates must be finite and non-negative")

    m = lam.size
    EQ, EN, Eb, Ei, Pw = (np.zeros(m) for _ in range(5))
    tmass, tq = np.zeros(m), np.zeros(m)
    # lam == 0: N = 0 almost surely
    zero = lam == 0
    Ei[zero] = b
    Pw[zero] = 1.0 if b == 0 else 0.0

    idx_pos = np.flatnonzero(~zero)
    n_max_used = 0
    for start in range(0, idx_pos.size, chunk):
        idx = idx_pos[start:start + chunk]
        lc = lam[idx]
        n_max = _n_max_guess(lc.max(), b, theta, gamma)
        while True:
            n = np.arange(n_max + 1, dtype=float)
            d = death_rates(np.arange(1, n_max + 2), b, theta, gamma)  # d(1..n_max+1)
            log_d_cum = np.concatenate(([0.0], np.cumsum(np.log(d[:-1]))))  # sum_{k<=n} log d(k)
            logw = np.log(lc)[:, None] * n[None, :] - log_d_cum[None, :]
            logw -= special.logsumexp(logw, axis=1)[:, None]
            pi = np.exp(logw)
            r = lc / d[-1]  # lam / d(n_max + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                geo = np.where(r < 1, r / (1 - r), np.inf)
                geo2 = np.where(r < 1, r / (1 - r) ** 2, np.inf)
            mass_bound = pi[:, -1] * geo
            q_bound = pi[:, -1] * (max(n_max - b, 0) * geo + geo2)
            if np.all(mass_bound < tol) and np.all(q_bound < tol):
                break
            n_max = int(n_max * 1.5) + 50
        q = np.maximum(n - b, 0.0)
        busy = np.minimum(n, b)
        EQ[idx] = pi @ q
        EN[idx] = pi @ n
        Eb[idx] = pi @ busy
        Ei[idx] = b - Eb[idx]
        Pw[idx] = pi[:, b:].sum(axis=1) if b <= n_max else 0.0
        tmass[idx], tq[idx] = mass_bound, q_bound
        n_max_used = max(n_max_used, n_max)
    return QueueMoments(lam, EQ, EN, Eb, Ei, Pw, tmass, tq, n_max_used)


def stationary_distribution(lam: float, b: int, theta: float = 1.0, gamma: float = 3.0,
                            tol: float = 1e-12) -> np.ndarray:
    """pi(n), n = 0..n_max, for a single arrival rate (mainly for tests and plots)."""
    b = _check_b(b)
    if lam == 0:
        return np.array([1.0])
    n_max = _n_max_guess(lam, b, theta, gamma)
    while True:
        n = np.arange(n_max + 1, dtype=float)
        d = death_rates(np.arange(1, n_max + 2), b, theta, gamma)
        logw = n * np.log(lam) - np.concatenate(([0.0], np.cumsum(np.log(d[:-1]))))
        pi = np.exp(logw - special.logsumexp(logw))
        r = lam / d[-1]
        if r < 1 and pi[-1] * r / (1 - r) < tol:
            return pi
        n_max = int(n_max * 1.5) + 50


def expected_queue(lam, b, theta: float = 1.0, gamma: float = 3.0) -> np.ndarray | float:
    """E_lam[Q] for one or many arrival rates."""
    out = queue_moments(lam, b, theta, gamma).EQ
    return float(out[0]) if np.ndim(lam) == 0 else out


def cost_known_rate(b: int, lam: float, params: Params = SECTION4) -> float:
    """C(b; lam) = c*b + (h + p*gamma) * E_lam[Q]."""
    return params.c * b + params.queue_weight * expected_queue(lam, b, params.theta, params.gamma)


# ---------------------------------------------------------------------------
# Arrival-rate distributions
# ---------------------------------------------------------------------------


def gauss_gamma(shape: float, scale: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss rule for E[g(Lambda)], Lambda ~ Gamma(shape, scale) (generalized Laguerre).

    Golub-Welsch on the monic generalized-Laguerre Jacobi matrix (diagonal 2i + a + 1,
    off-diagonal sqrt(i(i + a)), a = shape - 1). Weights are normalised to sum to one,
    which avoids Gamma(shape) overflow for large shapes.
    """
    a = shape - 1.0
    i = np.arange(n, dtype=float)
    diag = 2 * i + a + 1
    off = np.sqrt(i[1:] * (i[1:] + a))
    x, v = linalg.eigh_tridiagonal(diag, off)
    w = v[0, :] ** 2
    return scale * x, w / w.sum()


class RateDist:
    """Interface for the arrival-rate distribution F."""

    mean: float
    sd: float
    label: str

    @property
    def cv(self) -> float:
        return self.sd / self.mean if self.mean > 0 else 0.0

    def ppf(self, u):  # F^{-1}
        raise NotImplementedError

    def partial_expectation(self, x: float) -> float:  # E[(Lambda - x)^+]
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def quadrature(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def core(self, eps: float = 1e-16) -> tuple[float, float]:
        """Interval outside which the density is negligible."""
        return float(self.ppf(eps)), float(self.ppf(1 - eps))

    # local scale on which the density varies (limits the panel width in "split")
    smooth_scale: float = np.inf


class PointRate(RateDist):
    """Known (deterministic) rate."""

    def __init__(self, lam: float):
        self.mean, self.sd, self.lam = float(lam), 0.0, float(lam)
        self.label = f"lambda = {lam:g}"

    def ppf(self, u):
        return np.full(np.shape(u), self.lam) if np.ndim(u) else self.lam

    def partial_expectation(self, x):
        return max(self.lam - x, 0.0)

    def quadrature(self, n):
        return np.array([self.lam]), np.array([1.0])

    def core(self, eps=1e-16):
        return self.lam, self.lam


class UniformRate(RateDist):
    def __init__(self, lo: float, hi: float):
        if not hi > lo >= 0:
            raise ValueError("need 0 <= lo < hi")
        self.lo, self.hi = float(lo), float(hi)
        self.mean = 0.5 * (lo + hi)
        self.sd = (hi - lo) / np.sqrt(12.0)
        self.label = f"U[{lo:g}, {hi:g}]"

    def ppf(self, u):
        return self.lo + np.asarray(u) * (self.hi - self.lo)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.lo) & (x <= self.hi), 1.0 / (self.hi - self.lo), 0.0)

    def partial_expectation(self, x):
        if x <= self.lo:
            return self.mean - x
        if x >= self.hi:
            return 0.0
        return (self.hi - x) ** 2 / (2.0 * (self.hi - self.lo))

    def quadrature(self, n):
        x, w = np.polynomial.legendre.leggauss(n)
        return 0.5 * (self.hi - self.lo) * x + 0.5 * (self.hi + self.lo), 0.5 * w

    def core(self, eps=1e-16):
        return self.lo, self.hi


class GammaRate(RateDist):
    """Gamma with given mean and CV (shape 1/CV^2)."""

    def __init__(self, mean: float, cv: float):
        self.mean, self.sd = float(mean), float(mean * cv)
        self.shape = 1.0 / cv**2
        self.scale = mean * cv**2
        self._dist = stats.gamma(self.shape, scale=self.scale)
        self.smooth_scale = self.sd
        self.label = f"Gamma(mean {mean:g}, CV {cv:g})"

    def ppf(self, u):
        return self._dist.ppf(u)

    def pdf(self, x):
        return self._dist.pdf(x)

    def partial_expectation(self, x):
        if x <= 0:
            return self.mean - x
        z = x / self.scale
        return self.mean * special.gammaincc(self.shape + 1, z) - x * special.gammaincc(self.shape, z)

    def quadrature(self, n):
        x, w = gauss_gamma(self.shape, self.scale, n)
        keep = w > 1e-300
        return x[keep], w[keep] / w[keep].sum()


class LognormalRate(RateDist):
    """Lognormal with given mean and CV."""

    def __init__(self, mean: float, cv: float):
        self.mean, self.sd = float(mean), float(mean * cv)
        self.s = np.sqrt(np.log1p(cv**2))
        self.mu = np.log(mean) - 0.5 * self.s**2
        self._dist = stats.lognorm(self.s, scale=np.exp(self.mu))
        # the density varies on scale ~ x*s; use the median as the reference point
        self.smooth_scale = min(self.sd, self.s * np.exp(self.mu))
        self.label = f"Lognormal(mean {mean:g}, CV {cv:g})"

    def ppf(self, u):
        return self._dist.ppf(u)

    def pdf(self, x):
        return self._dist.pdf(x)

    def partial_expectation(self, x):
        if x <= 0:
            return self.mean - x
        d1 = (self.mu + self.s**2 - np.log(x)) / self.s
        return self.mean * special.ndtr(d1) - x * special.ndtr(d1 - self.s)

    def quadrature(self, n):
        x, w = np.polynomial.hermite.hermgauss(n)
        return np.exp(self.mu + np.sqrt(2.0) * self.s * x), w / np.sqrt(np.pi)


# ---------------------------------------------------------------------------
# Expected cost Pi(b)
# ---------------------------------------------------------------------------


def newsvendor_cost(b: int, rate: RateDist, params: Params = SECTION4) -> float:
    """Pibar(b) = (p + h/gamma) E[Lambda - theta b]^+ + c b (eq. 2)."""
    return params.underage * rate.partial_expectation(params.theta * b) + params.c * b


def _correction_window(b: int, params: Params) -> tuple[float, float]:
    """Arrival rates outside [lo, hi] contribute D(lam, b) < ~1e-15."""
    bt = b * params.theta
    below = 12.0 * np.sqrt(max(bt, 1.0) * params.theta) + 10.0
    above = 12.0 * np.sqrt(max(bt, 1.0) * max(params.gamma, params.theta)) + 10.0
    return max(bt - below, 0.0), bt + above


def correction_D(lam, b: int, params: Params = SECTION4) -> np.ndarray:
    """D(lam, b) = E_lam[Q] - (lam - b theta)^+/gamma (>= 0; localised near lam = b theta).

    By flow balance lam = theta E[min(N, b)] + gamma E[Q], so for lam >= b theta
    D = (theta/gamma) E[(b - N)^+] (expected idle servers) and for lam < b theta D = E[Q].
    """
    qm = queue_moments(lam, b, params.theta, params.gamma)
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    above = lam >= b * params.theta
    return np.where(above, params.theta / params.gamma * qm.E_idle, qm.EQ)


def expected_correction(b: int, rate: RateDist, params: Params = SECTION4,
                        nodes_per_panel: int = 8, width_factor: float = 1.0) -> float:
    """E[D(Lambda, b)] by composite Gauss-Legendre on the window where D matters."""
    if isinstance(rate, PointRate):
        return float(correction_D(rate.lam, b, params)[0])
    w_lo, w_hi = _correction_window(b, params)
    c_lo, c_hi = rate.core()
    a, z = max(w_lo, c_lo), min(w_hi, c_hi)
    if z <= a:
        return 0.0
    scale = min(np.sqrt(max(b * params.theta, 1.0)), rate.smooth_scale) * width_factor
    # D has a kink at lam = b*theta (it subtracts (lam - b theta)^+/gamma), so that point
    # must be a panel edge; elsewhere D * pdf is smooth.
    bt = b * params.theta
    pieces = [(a, bt), (bt, z)] if a < bt < z else [(a, z)]
    edges = [a]
    for lo_, hi_ in pieces:
        n_pan = max(2, int(np.ceil((hi_ - lo_) / scale)))
        edges.extend(np.linspace(lo_, hi_, n_pan + 1)[1:])
    edges = np.asarray(edges)
    xg, wg = np.polynomial.legendre.leggauss(nodes_per_panel)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    lam = (mid[:, None] + half[:, None] * xg[None, :]).ravel()
    wts = (half[:, None] * wg[None, :]).ravel()
    return float(np.sum(wts * correction_D(lam, b, params) * rate.pdf(lam)))


def expected_cost(b: int, rate: RateDist, params: Params = SECTION4, method: str = "auto",
                  n_nodes: int = 64, **kw) -> float:
    """Pi(b) = E_Lambda[C(b; Lambda)].

    method="gauss": Gauss quadrature with `n_nodes` nodes for F (Legendre for uniforms).
    method="split": Pibar(b) + (h + p gamma) E[D(Lambda, b)] (closed form + local quadrature).
    method="auto": "gauss" for uniform and point rates, "split" otherwise.
    """
    b = _check_b(b)
    if isinstance(rate, PointRate):
        return cost_known_rate(b, rate.lam, params)
    if method == "auto":
        method = "gauss" if isinstance(rate, UniformRate) else "split"
    if method == "gauss":
        lam, w = rate.quadrature(n_nodes)
        eq = queue_moments(lam, b, params.theta, params.gamma).EQ
        return params.c * b + params.queue_weight * float(np.sum(w * eq))
    if method == "split":
        return newsvendor_cost(b, rate, params) + params.queue_weight * expected_correction(
            b, rate, params, **kw)
    raise ValueError(f"unknown method {method!r}")


class CostFunction:
    """Memoised b -> Pi(b) for one rate distribution (used by the capacity searches)."""

    def __init__(self, rate: RateDist, params: Params = SECTION4, method: str = "auto",
                 n_nodes: int = 64, **kw):
        self.rate, self.params, self.method, self.n_nodes, self.kw = rate, params, method, n_nodes, kw
        self.cache: dict[int, float] = {}

    def __call__(self, b: int) -> float:
        b = int(b)
        if b not in self.cache:
            self.cache[b] = expected_cost(b, self.rate, self.params, self.method, self.n_nodes, **self.kw)
        return self.cache[b]

    @property
    def n_evals(self) -> int:
        return len(self.cache)
