"""Temperature scaling: one scalar T > 0 that minimizes the NLL of softmax(logits / T)."""

import math

import numpy as np

from backend.calibration.metrics import nll, softmax

INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0  # 1 / golden ratio


def fit_temperature(
    logits: np.ndarray, y: np.ndarray, *, lo: float = -3.0, hi: float = 3.0, tol: float = 1e-6
) -> float:
    """Golden-section search for `log T` in `[lo, hi]`, narrowed to width `tol`.

    The NLL as a function of `log T` has a single minimum, so the search is exact up to `tol`.
    If the NLL keeps falling (or rising) across the whole range, the answer sits at the matching
    boundary. Returns `T = exp(log T)`.
    """

    def loss(log_t: float) -> float:
        return nll(softmax(logits / math.exp(log_t)), y)

    a, b = lo, hi
    c = b - INV_PHI * (b - a)
    d = a + INV_PHI * (b - a)
    f_c, f_d = loss(c), loss(d)
    while b - a > tol:
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - INV_PHI * (b - a)
            f_c = loss(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + INV_PHI * (b - a)
            f_d = loss(d)
    return math.exp((a + b) / 2)
