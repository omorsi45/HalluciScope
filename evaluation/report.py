"""Render benchmark result files as markdown for ``docs/results/``.

    python -m evaluation.report retrieval evaluation/results/retrieval_*.json \
        --out docs/results/retrieval.md

Every number in the output is read from a results file written by ``evaluation.run_retrieval``;
nothing is computed or filled in here beyond rounding.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_TABLE_HEADER = (
    "| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
)

_METHODOLOGY = (
    "Methodology: BM25 parameters were fixed before evaluation. Passages are sentence-aligned "
    "and every system collapses its passage ranking to documents with MaxP (a document ranks "
    "by its best passage). There was no tuning on test qrels."
)


def _text(value: Any, unit: str = "") -> str:
    return "unknown" if value is None else f"{value}{unit}"


def _commit_line(env: Mapping[str, Any]) -> str:
    dirty = env.get("git_dirty")
    state = "unknown" if dirty is None else ("dirty" if dirty else "clean")
    return f"- Commit: {_text(env.get('git_commit'))} (working tree: {state})"


def _section(result: Mapping[str, Any]) -> str:
    env = result["env"]
    params = result["params"]
    models = result["models"]
    checks = result["checks"]
    lines = [
        f"## {result['dataset']} ({result['split']} split)",
        "",
        f"- Queries: {result['n_queries']}, documents: {result['n_docs']}, "
        f"passages: {result['n_passages']}, documents skipped as empty: "
        f"{result['n_skipped_docs']}",
        f"- Run: {result['started_at']} to {result['finished_at']} "
        f"(ingest {result['ingest_seconds']:.1f} s, total {result['runtime_seconds']:.1f} s)",
        _commit_line(env),
        f"- CPU: {_text(env.get('cpu'))}, RAM: {_text(env.get('ram_gb'), ' GB')}",
        f"- Python: {_text(env.get('python'))}, torch: {_text(env.get('torch'))}, "
        f"PostgreSQL: {_text(env.get('postgres'))}, pgvector: {_text(env.get('pgvector'))}",
        f"- Models: embedder {models['embedder']}, reranker {models['reranker']}",
        f"- Parameters: BM25 k1 = {params['k1']}, b = {params['b']}; RRF k = {params['rrf_k']}; "
        f"candidates = {params['candidates']}; rerank depth = {params['rerank_depth']} documents; "
        f"HNSW ef_search = {params['ef_search']}; passage target = "
        f"{params['passage_target_chars']} characters",
        "",
        *_TABLE_HEADER,
    ]
    for name, metrics in result["systems"].items():
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.4f} | {metrics['recall@10']:.4f} "
            f"| {metrics['recall@100']:.4f} | {metrics['mrr@10']:.4f} "
            f"| {metrics['latency_ms_p50']:.1f} | {metrics['latency_ms_p95']:.1f} |"
        )
    lines += [
        "",
        "Checks:",
        "",
        f"- acl_leaks: {checks['acl_leaks']} (must be 0)",
        f"- bm25_parity_max_abs_diff: {checks['bm25_parity_max_abs_diff']:.3e} (must be < 1e-9)",
        "",
        _METHODOLOGY,
    ]
    return "\n".join(lines)


def render_retrieval(results: Sequence[Mapping[str, Any]]) -> str:
    """One section per results mapping, in the order given, as a markdown document."""
    sections = [_section(result) for result in results]
    return "# Retrieval benchmark\n\n" + "\n\n".join(sections) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.report")
    commands = parser.add_subparsers(dest="command", required=True)
    retrieval = commands.add_parser("retrieval", help="render retrieval benchmark results")
    retrieval.add_argument("results", nargs="+", type=Path, help="results JSON files")
    retrieval.add_argument("--out", type=Path, default=Path("docs/results/retrieval.md"))
    args = parser.parse_args(argv)

    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the committed file uses LF whatever the platform.
    args.out.write_text(render_retrieval(results), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
