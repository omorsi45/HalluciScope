"""Calibration metrics for verdict probabilities, in plain numpy.

Verdict classes are always (supported, contradicted, insufficient). Probability arrays have one
row per example and one column per class in that order, and `y` holds class indices.

Binned metrics share one rule: a confidence `c` falls in bin `min(int(c * n_bins), n_bins - 1)`,
so bins are [k/n, (k+1)/n) and the last bin also holds exactly 1.0.
"""

import numpy as np

N_BINS = 15


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax (over the last axis); the row max is subtracted so large logits are safe."""
    shifted = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _bin_stats(
    conf: np.ndarray, correct: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin example count, confidence sum, and correct-count sum."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    index = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    conf_sum = np.bincount(index, weights=conf, minlength=n_bins)
    correct_sum = np.bincount(index, weights=correct, minlength=n_bins)
    return counts, conf_sum, correct_sum


def _binned_gap(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """sum_b (n_b / n) * |mean_conf_b - acc_b|."""
    counts, conf_sum, correct_sum = _bin_stats(conf, correct, n_bins)
    # (n_b / n) * |conf_sum_b / n_b - correct_sum_b / n_b| simplifies to
    # |conf_sum_b - correct_sum_b| / n, which also leaves empty bins at exactly 0.
    return float(np.abs(conf_sum - correct_sum).sum() / counts.sum())


def ece_top_label(probs: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error of the predicted class: confidence is the row max."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    return _binned_gap(conf, correct, n_bins)


def classwise_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Mean over classes of the binned gap between `probs[:, c]` and the indicator `y == c`."""
    gaps = [_binned_gap(probs[:, c], y == c, n_bins) for c in range(probs.shape[1])]
    return float(np.mean(gaps))


def brier_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean over rows of the summed squared error against the one-hot label."""
    onehot = np.zeros_like(probs, dtype=float)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log probability of the true class, with probabilities floored at `eps`."""
    true_prob = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.maximum(true_prob, eps))))


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """One dict per bin with `lo`, `hi`, `count`, `mean_conf`, `accuracy`.

    Empty bins are kept so the list always has `n_bins` entries; their `mean_conf` and
    `accuracy` are None.
    """
    counts, conf_sum, correct_sum = _bin_stats(conf, correct, n_bins)
    bins: list[dict] = []
    for b in range(n_bins):
        count = int(counts[b])
        bins.append(
            {
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "count": count,
                "mean_conf": float(conf_sum[b] / count) if count else None,
                "accuracy": float(correct_sum[b] / count) if count else None,
            }
        )
    return bins


def risk_coverage(score: np.ndarray, gold_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective-prediction curve: accept examples from highest score down.

    Ties keep input order (stable sort). For `j = 1..n` accepted, coverage is `j / n` and risk is
    the fraction of the accepted examples whose `gold_ok` is False. Returns (coverage, risks).
    """
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    wrong = ~np.asarray(gold_ok, dtype=bool)[order]
    accepted = np.arange(1, len(order) + 1)
    return accepted / len(order), np.cumsum(wrong) / accepted


def aurc(risks: np.ndarray) -> float:
    """Area under the risk-coverage curve: the mean of the risk array."""
    return float(np.mean(risks))
