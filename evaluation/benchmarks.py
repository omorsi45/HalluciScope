import json
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true: list[int], y_pred: list[int], y_scores: list[float]) -> dict:
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
    except ValueError:
        metrics["auc_roc"] = None
    return metrics


async def run_benchmark(dataset: str, output_dir: str, threshold: float = 0.5):
    from backend.cli.main import _build_pipeline

    dataset_path = Path(f"evaluation/datasets/{dataset}.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    pipeline = _build_pipeline()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_scores = [], [], []
    results = []

    with open(dataset_path) as f:
        for line in f:
            entry = json.loads(line)
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
                "claims": [{"claim": sc.claim, "score": sc.hallucination_score, "tier": sc.tier.value} for sc in result.scored_claims],
            })

    metrics = compute_metrics(y_true, y_pred, y_scores)

    with open(output_path / f"{dataset}_results.json", "w") as f:
        json.dump({"metrics": metrics, "y_true": y_true, "y_scores": y_scores, "details": results}, f, indent=2)

    print(f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}  AUC-ROC: {metrics['auc_roc']}")
    return metrics
