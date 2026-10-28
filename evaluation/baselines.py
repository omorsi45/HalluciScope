"""Baselines for the RAGTruth factuality comparison, scored on cached evidence signals.

The legacy baselines reuse EvidenceGraph's claim-level evidence (the cached signals, depth 8), so
they isolate the scoring method from the retrieval and unit changes. The shipped scorer mixed an
NLI term and a similarity term with weights 0.5 and 0.2 (plus self-consistency); without
self-consistency the two are renormalized to 5/7 and 2/7. A legacy score is a hallucination
score: low means supported.
"""

from collections.abc import Sequence
from typing import Literal

from backend.verify.types import EvidenceSignal
from evaluation.metrics.classification import detection_prf

LEGACY_NLI_W = 5 / 7
LEGACY_SIM_W = 2 / 7

BINARY_THRESHOLD = 0.5  # the shipped benchmark threshold: supported below it
TIER_SUPPORTED_BELOW = 0.2
TIER_CONTRADICTED_ABOVE = 0.6

DEFAULT_GRID: tuple[float, ...] = tuple(i / 100 for i in range(101))


def _max_cosine(signals: Sequence[EvidenceSignal]) -> float:
    """The best dense cosine clipped to [0, 1]; 0.0 without signals."""
    if not signals:
        return 0.0
    return min(1.0, max(0.0, max(s.dense_cos for s in signals)))


def legacy_as_shipped_score(signals: Sequence[EvidenceSignal]) -> float:
    """`5/7 * min_s(contra + 0.5 * entail) + 2/7 * (1 - clip(max dense_cos, 0, 1))`.

    The shipped code read the entailment column as neutral, so the middle term weights the
    entailment probability. 1.0 (fully unsupported) without signals.
    """
    if not signals:
        return 1.0
    nli_term = min(s.nli.contra + 0.5 * s.nli.entail for s in signals)
    return LEGACY_NLI_W * nli_term + LEGACY_SIM_W * (1.0 - _max_cosine(signals))


def legacy_fixed_score(signals: Sequence[EvidenceSignal]) -> float:
    """The shipped score with the label order fixed: `contra + 0.5 * neutral` per signal."""
    if not signals:
        return 1.0
    nli_term = min(s.nli.contra + 0.5 * s.nli.neutral for s in signals)
    return LEGACY_NLI_W * nli_term + LEGACY_SIM_W * (1.0 - _max_cosine(signals))


def legacy_predictions(scores: Sequence[float], mode: Literal["binary", "tiers"]) -> list[str]:
    """Verdicts from legacy scores.

    `binary`: supported below 0.5, else insufficient (for detection and false-supported metrics).
    `tiers`: supported below 0.2, contradicted above 0.6, else insufficient (for macro-F1).
    """
    if mode == "binary":
        return ["supported" if s < BINARY_THRESHOLD else "insufficient" for s in scores]
    if mode == "tiers":
        return [_tier(s) for s in scores]
    raise ValueError(f"mode must be 'binary' or 'tiers', got {mode!r}")


def _tier(score: float) -> str:
    if score < TIER_SUPPORTED_BELOW:
        return "supported"
    if score > TIER_CONTRADICTED_ABOVE:
        return "contradicted"
    return "insufficient"


def similarity_support(signals: Sequence[EvidenceSignal]) -> float:
    """The support from similarity alone: the best cosine clipped to [0, 1], 0.0 without signals."""
    return _max_cosine(signals)


def nli_support(signals: Sequence[EvidenceSignal]) -> float:
    """The support from NLI alone: the largest entailment probability, 0.0 without signals."""
    return max((s.nli.entail for s in signals), default=0.0)


def choose_threshold(
    support: Sequence[float],
    gold: Sequence[str],
    grid: Sequence[float] = DEFAULT_GRID,
) -> float:
    """The threshold with the best detection F1 when `supported` is predicted for `support >= t`.

    Detection treats not-supported as the positive class. Ties go to the smaller threshold. This
    is for the select role only; never call it with test labels.
    """
    if not grid:
        raise ValueError("grid must not be empty")
    best_threshold = 0.0
    best_f1 = -1.0
    for t in sorted(grid):
        pred = ["supported" if s >= t else "insufficient" for s in support]
        f1 = detection_prf(gold, pred)["f1"]
        if f1 > best_f1:
            best_threshold, best_f1 = t, f1
    return best_threshold
