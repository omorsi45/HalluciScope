"""Cluster bootstrap by source (no metric libraries).

Rows that share a group (a RAGTruth source) are resampled together, so the interval reflects
the dependence between claims drawn from one source instead of treating each claim as independent.
"""

from collections.abc import Callable, Sequence

import numpy as np


def cluster_bootstrap(
    groups: Sequence[str],
    statistic: Callable[[np.ndarray], float | None],
    *,
    n_resamples: int = 1000,
    seed: int = 20261026,
    ci: float = 0.95,
) -> tuple[float | None, float | None, float | None]:
    """Return `(point, lo, hi)` for `statistic` over row indices, resampling whole groups.

    `point` is `statistic` over every row index. Each resample draws the unique groups (taken
    in sorted order, so row order does not matter) with replacement, concatenates the row
    indices of the drawn groups with multiplicity, and evaluates `statistic`. Resamples where
    it returns None are skipped. `lo` and `hi` are the `(1 - ci) / 2` and `1 - (1 - ci) / 2`
    percentiles (numpy's default linear method) of the remaining values, or None when every
    resample was None.
    """
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    # Row indices of each group in ascending order, found with one stable sort.
    order = np.argsort(inverse, kind="stable")
    rows = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    point = statistic(np.arange(len(groups)))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.integers(0, len(unique), size=len(unique))
        value = statistic(np.concatenate([rows[i] for i in drawn]))
        if value is not None:
            values.append(value)
    if not values:
        return point, None, None
    tail = (1 - ci) / 2
    lo, hi = np.percentile(values, [100 * tail, 100 * (1 - tail)])
    return point, float(lo), float(hi)
