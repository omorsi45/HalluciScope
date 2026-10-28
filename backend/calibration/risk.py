"""Risk control by Learn-then-Test fixed-sequence testing (spec ruling R1).

The exchangeable unit is the source. The risk at threshold t is the mean over sources with at
least one bad unit of `L_i(t)`, the fraction of that source's bad units accepted at `t`, so each
source contributes one [0, 1]-valued loss. For each threshold the null "risk > alpha" gets the
Hoeffding-Bentkus p-value; walking a fixed grid from the strictest threshold down and stopping at
the first p-value above `delta` controls P(risk(tau) > alpha) <= delta with no multiplicity
correction. Verdict classes are always (supported, contradicted, insufficient).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# Fixed in advance: 0.95 down to 0.50 in steps of 0.01 (the start point must not depend on data).
GRID: tuple[float, ...] = tuple(round(0.95 - 0.01 * i, 2) for i in range(46))

# Guards float noise in n * r_hat (0.07 * 100 = 7.000000000000001) before the ceiling.
_CEIL_EPS = 1e-9


def binom_cdf(k: int, n: int, p: float) -> float:
    """Exact P(Binomial(n, p) <= k): log pmf terms from `lgamma`, summed by log-sum-exp."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_n_factorial = math.lgamma(n + 1)
    terms = [
        log_n_factorial - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(k + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top + math.log(sum(math.exp(term - top) for term in terms))))


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided upper bound: the largest p with `binom_cdf(k, n, p) >= delta`, by bisection."""
    if n == 0 or k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > 1e-12:
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) >= delta:
            lo = mid
        else:
            hi = mid
    return hi


def _kl(a: float, b: float) -> float:
    """h(a, b) = a ln(a/b) + (1-a) ln((1-a)/(1-b)), with 0 ln 0 = 0."""
    total = 0.0
    if a > 0.0:
        total += a * math.log(a / b)
    if a < 1.0:
        total += (1.0 - a) * math.log((1.0 - a) / (1.0 - b))
    return total


def hb_pvalue(r_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: mean of n [0, 1]-bounded losses > alpha, given `r_hat`."""
    if n == 0:
        return 1.0
    hoeffding = math.exp(-n * _kl(min(r_hat, alpha), alpha))
    bentkus = math.e * binom_cdf(math.ceil(n * r_hat - _CEIL_EPS), n, alpha)
    return min(1.0, hoeffding, bentkus)


def per_source_risk(
    scores: np.ndarray,
    bad: np.ndarray,
    groups: np.ndarray,
    t: float,
    accept_mask: np.ndarray | None = None,
) -> tuple[float, int]:
    """Mean over sources with a bad row of `L_i`, and the number of such sources.

    `L_i` is the fraction of the source's bad rows that are accepted: `scores >= t`, and also
    `accept_mask` when one is given. Bad rows the mask excludes stay in the denominator.
    """
    bad = np.asarray(bad, dtype=bool)
    if not bad.any():
        return 0.0, 0
    accepted = np.asarray(scores)[bad] >= t
    if accept_mask is not None:
        accepted &= np.asarray(accept_mask, dtype=bool)[bad]
    _, source_index = np.unique(np.asarray(groups)[bad], return_inverse=True)
    n_bad = np.bincount(source_index)
    n_accepted = np.bincount(source_index, weights=accepted.astype(float))
    return float(np.mean(n_accepted / n_bad)), len(n_bad)


def ltt_fixed_sequence(
    grid: Sequence[float],
    risk_at: Callable[[float], tuple[float, int]],
    alpha: float,
    delta: float,
) -> tuple[float | None, list[dict]]:
    """Walk `grid` in order and return the last threshold before the first p-value above `delta`.

    Returns `(None, trace)` when the first threshold already fails. The trace holds one
    `{"t", "r_hat", "n", "p"}` row per evaluated threshold, including the failing one.
    """
    tau: float | None = None
    trace: list[dict] = []
    for t in grid:
        r_hat, n = risk_at(t)
        p = hb_pvalue(r_hat, n, alpha)
        trace.append({"t": t, "r_hat": r_hat, "n": n, "p": p})
        if p > delta:
            break
        tau = t
    return tau, trace


@dataclass(frozen=True)
class ThresholdResult:
    tau_s: float | None
    tau_c: float | None
    n_sources_s: int
    n_sources_c: int
    trace_s: list[dict]
    trace_c: list[dict]


def select_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alpha_s: float = 0.10,
    alpha_c: float = 0.05,
    delta: float = 0.05,
    grid: Sequence[float] = GRID,
) -> ThresholdResult:
    """Pick tau_s for SUPPORTED, then tau_c for CONTRADICTED among rows not accepted as supported.

    SUPPORTED: score `probs[:, 0]`, bad rows are gold not supported. CONTRADICTED: score
    `probs[:, 1]`, bad rows are gold not contradicted, and a row counts as accepted only when
    `probs[:, 0] < tau_s`, so a row already accepted as supported is not also a contradiction.
    """
    scores_s = probs[:, 0]
    bad_s = y != 0
    tau_s, trace_s = ltt_fixed_sequence(
        grid, lambda t: per_source_risk(scores_s, bad_s, groups, t), alpha_s, delta
    )

    scores_c = probs[:, 1]
    bad_c = y != 1
    not_supported = np.ones(len(probs), dtype=bool) if tau_s is None else probs[:, 0] < tau_s
    tau_c, trace_c = ltt_fixed_sequence(
        grid,
        lambda t: per_source_risk(scores_c, bad_c, groups, t, accept_mask=not_supported),
        alpha_c,
        delta,
    )

    return ThresholdResult(
        tau_s=tau_s,
        tau_c=tau_c,
        n_sources_s=len(np.unique(groups[bad_s])),
        n_sources_c=len(np.unique(groups[bad_c])),
        trace_s=trace_s,
        trace_c=trace_c,
    )
