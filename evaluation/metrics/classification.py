"""Original claim classification metrics for the RAGTruth numbers (no metric libraries).

Labels are the verdict classes in `CLASSES`. Hallucination detection treats "not supported"
(contradicted or insufficient) as the positive class, so a unit whose gold class is contradicted
and whose predicted class is insufficient is still a true positive. Every ratio with a zero
denominator is 0.0, except the two risk metrics, which are None when undefined.
"""

from collections.abc import Sequence

CLASSES = ("supported", "contradicted", "insufficient")
_SUPPORTED = "supported"


def _check_labels(labels: Sequence[str]) -> None:
    for label in labels:
        if label not in CLASSES:
            raise ValueError("label is not a verdict class")


def _check_pair(gold: Sequence[str], pred: Sequence[str]) -> None:
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    _check_labels(gold)
    _check_labels(pred)


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _detection(gold_positive: Sequence[bool], pred_positive: Sequence[bool]) -> dict:
    tp = fp = fn = tn = 0
    for g, p in zip(gold_positive, pred_positive, strict=True):
        if g and p:
            tp += 1
        elif p:
            fp += 1
        elif g:
            fn += 1
        else:
            tn += 1
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": _f1(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def detection_prf(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """Precision, recall and F1 for detecting unsupported units (positive = not supported)."""
    _check_pair(gold, pred)
    return _detection([g != _SUPPORTED for g in gold], [p != _SUPPORTED for p in pred])


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Mean per-class F1 over `CLASSES`, leaving out a class with no gold and no predicted unit."""
    _check_pair(gold, pred)
    scores = []
    for cls in CLASSES:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == cls and p != cls)
        if tp + fp + fn:
            scores.append(_f1(tp, fp, fn))
    return sum(scores) / len(scores) if scores else 0.0


def false_supported_rate(gold: Sequence[str], pred: Sequence[str]) -> float | None:
    """P(pred supported | gold not supported); None without any gold-not-supported unit."""
    _check_pair(gold, pred)
    n = sum(1 for g in gold if g != _SUPPORTED)
    if n == 0:
        return None
    k = sum(1 for g, p in zip(gold, pred, strict=True) if g != _SUPPORTED and p == _SUPPORTED)
    return k / n


def accepted_risk(gold: Sequence[str], pred: Sequence[str]) -> tuple[float | None, int, int]:
    """(k / n, k, n): of n predicted-supported units, k are gold not supported."""
    _check_pair(gold, pred)
    n = sum(1 for p in pred if p == _SUPPORTED)
    if n == 0:
        return None, 0, 0
    k = sum(1 for g, p in zip(gold, pred, strict=True) if p == _SUPPORTED and g != _SUPPORTED)
    return k / n, k, n


def supported_coverage(pred: Sequence[str]) -> float:
    """Fraction of units predicted supported."""
    _check_labels(pred)
    if not pred:
        return 0.0
    return sum(1 for p in pred if p == _SUPPORTED) / len(pred)


def confusion(gold: Sequence[str], pred: Sequence[str]) -> dict[str, dict[str, int]]:
    """Counts indexed `[gold][pred]`, with every class present in both levels."""
    _check_pair(gold, pred)
    table = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for g, p in zip(gold, pred, strict=True):
        table[g][p] += 1
    return table


def response_level(gold: Sequence[str], pred: Sequence[str], response_ids: Sequence[str]) -> dict:
    """`detection_prf` over responses: a response is positive when any of its units is."""
    _check_pair(gold, pred)
    if len(response_ids) != len(gold):
        raise ValueError("response_ids and gold differ in length")
    gold_positive: dict[str, bool] = {}
    pred_positive: dict[str, bool] = {}
    for response, g, p in zip(response_ids, gold, pred, strict=True):
        gold_positive[response] = gold_positive.get(response, False) or g != _SUPPORTED
        pred_positive[response] = pred_positive.get(response, False) or p != _SUPPORTED
    return _detection(list(gold_positive.values()), list(pred_positive.values()))
