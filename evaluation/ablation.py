import json
from pathlib import Path

from evaluation.benchmarks import compute_metrics

CONFIGURATIONS = [
    {"name": "NLI only",                "nli": 1.0, "consistency": 0.0, "similarity": 0.0},
    {"name": "Self-consistency only",   "nli": 0.0, "consistency": 1.0, "similarity": 0.0},
    {"name": "Semantic similarity only","nli": 0.0, "consistency": 0.0, "similarity": 1.0},
    {"name": "NLI + Self-consistency",  "nli": 0.6, "consistency": 0.4, "similarity": 0.0},
    {"name": "NLI + Similarity",        "nli": 0.7, "consistency": 0.0, "similarity": 0.3},
    {"name": "Consistency + Similarity", "nli": 0.0, "consistency": 0.6, "similarity": 0.4},
    {"name": "Full ensemble",           "nli": 0.5, "consistency": 0.3, "similarity": 0.2},
]


async def run_ablation(dataset: str, output_dir: str, threshold: float = 0.5):
    from backend.config import Settings
    from backend.models.loader import get_embedding_model, get_nli_model
    from backend.core.chunker import Chunker
    from backend.core.verifiers.nli import NLIVerifier
    from backend.core.verifiers.similarity import SimilarityVerifier
    from backend.core.verifiers.consistency import ConsistencyVerifier
    from backend.core.ensemble import EnsembleScorer
    from backend.core.pipeline import Pipeline

    dataset_path = Path(f"evaluation/datasets/{dataset}.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    settings = Settings()
    embedding_model = get_embedding_model()
    tokenizer, nli_model = get_nli_model()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    for config in CONFIGURATIONS:
        print(f"\nRunning: {config['name']}...")
        ensemble = EnsembleScorer(nli_weight=config["nli"], consistency_weight=config["consistency"], similarity_weight=config["similarity"])
        pipeline = Pipeline(
            settings=settings,
            chunker=Chunker(embedding_model=embedding_model, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, top_k=settings.top_k_chunks),
            nli_verifier=NLIVerifier(tokenizer=tokenizer, model=nli_model),
            similarity_verifier=SimilarityVerifier(embedding_model=embedding_model),
            consistency_verifier=ConsistencyVerifier(embedding_model=embedding_model, base_url=settings.ollama_base_url, model=settings.ollama_model, n_samples=settings.consistency_samples, temperature=settings.consistency_temperature, similarity_threshold=settings.consistency_similarity_threshold),
            ensemble=ensemble,
        )
        y_true, y_pred, y_scores = [], [], []
        with open(dataset_path) as f:
            for line in f:
                entry = json.loads(line)
                result = await pipeline.analyze(document_text=entry["document"], question=entry["question"])
                for i, sc in enumerate(result.scored_claims):
                    if i < len(entry.get("claims", [])):
                        y_true.append(int(entry["claims"][i]["hallucinated"]))
                        y_pred.append(1 if sc.hallucination_score >= threshold else 0)
                        y_scores.append(sc.hallucination_score)
        metrics = compute_metrics(y_true, y_pred, y_scores)
        all_results.append({"config": config["name"], **metrics})
        print(f"  F1: {metrics['f1']:.4f}")

    single_f1s = [r["f1"] for r in all_results[:3]]
    best_single = max(single_f1s)
    for r in all_results:
        r["delta_vs_best_single"] = r["f1"] - best_single

    with open(output_path / f"{dataset}_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'Configuration':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Delta':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['config']:<30} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['delta_vs_best_single']:>+10.4f}")
    return all_results
