import asyncio
import json
from pathlib import Path


def _auc_roc(y_true: list[int], y_scores: list[float]) -> float | None:
    """AUC-ROC via the rank-sum formula; None when only one class is present."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(y_scores)), key=lambda i: y_scores[i])
    ranks = [0.0] * len(y_scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_scores[order[j + 1]] == y_scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # average 1-based rank for tied scores
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, y_true, strict=True) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_metrics(y_true: list[int], y_pred: list[int], y_scores: list[float]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": _auc_roc(y_true, y_scores),
    }


def read_dataset(dataset_path: Path) -> list[dict]:
    """Blocking read of a JSONL dataset; call via asyncio.to_thread from coroutines."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path) as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, payload) -> None:
    """Blocking JSON write; call via asyncio.to_thread from coroutines."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


async def run_benchmark(dataset: str, output_dir: str, threshold: float = 0.5):
    from backend.cli.main import _build_pipeline

    dataset_path = Path(f"evaluation/datasets/{dataset}.jsonl")
    entries = await asyncio.to_thread(read_dataset, dataset_path)

    pipeline = _build_pipeline()
    output_path = Path(output_dir)
    await asyncio.to_thread(output_path.mkdir, parents=True, exist_ok=True)

    y_true, y_pred, y_scores = [], [], []
    results = []

    for entry in entries:
        result = await pipeline.analyze(
            document_text=entry["document"],
            question=entry["question"],
        )
        for i, sc in enumerate(result.scored_claims):
            if i < len(entry.get("claims", [])):
                y_true.append(int(entry["claims"][i]["hallucinated"]))
                y_pred.append(1 if sc.hallucination_score >= threshold else 0)
                y_scores.append(sc.hallucination_score)
        results.append({
            "question": entry["question"],
            "answer": result.answer,
            "overall_score": result.overall_score,
            "claims": [
                {"claim": sc.claim, "score": sc.hallucination_score, "tier": sc.tier.value}
                for sc in result.scored_claims
            ],
        })

    metrics = compute_metrics(y_true, y_pred, y_scores)

    await asyncio.to_thread(
        write_json,
        output_path / f"{dataset}_results.json",
        {"metrics": metrics, "y_true": y_true, "y_scores": y_scores, "details": results},
    )

    print(
        f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  "
        f"F1: {metrics['f1']:.4f}  AUC-ROC: {metrics['auc_roc']}"
    )
    return metrics
