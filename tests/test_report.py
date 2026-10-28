"""Markdown rendering of retrieval benchmark results."""

import json

from evaluation import report
from evaluation.report import render_retrieval


def system(ndcg, recall10, recall100, mrr, p50, p95) -> dict:
    return {
        "ndcg@10": ndcg,
        "recall@10": recall10,
        "recall@100": recall100,
        "mrr@10": mrr,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
    }


def results_for(dataset: str = "scifact", **env_overrides) -> dict:
    return {
        "dataset": dataset,
        "split": "test",
        "n_queries": 300,
        "n_docs": 5183,
        "n_passages": 7001,
        "n_skipped_docs": 0,
        "systems": {
            "bm25": system(0.612345, 0.7, 0.9, 0.55, 12.34, 45.61),
            "hybrid": system(0.7, 0.8, 0.95, 0.65, 30.04, 80.0),
        },
        "checks": {"acl_leaks": 0, "bm25_parity_max_abs_diff": 3.5e-15},
        "params": {
            "k1": 0.9,
            "b": 0.4,
            "rrf_k": 60,
            "candidates": 300,
            "rerank_depth": 50,
            "ef_search": 300,
            "passage_target_chars": 800,
        },
        "models": {"embedder": "BAAI/bge-small-en-v1.5", "reranker": "cross-encoder/rerank-x"},
        "env": {
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "git_dirty": False,
            "cpu": "Example CPU 9000",
            "ram_gb": 16.0,
            "python": "3.14.9",
            "torch": "2.14.1+cpu",
            "postgres": "17.11",
            "pgvector": "0.8.7",
            **env_overrides,
        },
        "started_at": "2026-10-27T12:00:00+00:00",
        "finished_at": "2026-10-27T12:10:00+00:00",
        "ingest_seconds": 61.5,
        "runtime_seconds": 600.0,
    }


def test_render_retrieval_golden():
    text = render_retrieval([results_for()])

    assert "| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |" in text
    assert "| bm25 | 0.6123 | 0.7000 | 0.9000 | 0.5500 | 12.3 | 45.6 |" in text
    assert "| hybrid | 0.7000 | 0.8000 | 0.9500 | 0.6500 | 30.0 | 80.0 |" in text
    assert "0123456789abcdef0123456789abcdef01234567" in text
    assert "Example CPU 9000" in text
    assert "scifact" in text
    assert "acl_leaks" in text
    assert "bm25_parity_max_abs_diff" in text
    assert "no tuning on test qrels" in text.lower()
    assert "\u2014" not in text


def test_render_retrieval_one_section_per_dataset_in_input_order():
    text = render_retrieval([results_for("scifact"), results_for("nfcorpus")])

    assert text.index("scifact") < text.index("nfcorpus")
    assert text.count("| System | nDCG@10 |") == 2


def test_render_retrieval_marks_unknown_git_state_instead_of_crashing():
    text = render_retrieval([results_for(git_commit=None, git_dirty=None)])

    assert "unknown" in text
    assert "None" not in text


def test_report_cli_writes_the_rendered_file(tmp_path):
    source = tmp_path / "retrieval_scifact.json"
    source.write_text(json.dumps(results_for()), encoding="utf-8")
    out = tmp_path / "docs" / "retrieval.md"

    exit_code = report.main(["retrieval", str(source), "--out", str(out)])

    assert exit_code == 0
    written = out.read_text(encoding="utf-8")
    assert "| bm25 | 0.6123 | 0.7000 | 0.9000 | 0.5500 | 12.3 | 45.6 |" in written
    assert "\r" not in written
