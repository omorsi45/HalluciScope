"""Retrieval benchmark: six systems on a BEIR dataset, scored with the original metrics.

    python -m evaluation.run_retrieval --dataset scifact
    python -m evaluation.report retrieval evaluation/results/retrieval_scifact.json

Every system calls ``PostgresRetriever.search`` for the top 300 passages and collapses them to
documents with MaxP (R10a). ``hybrid_rerank`` then reranks the passages of the top 50 documents.
All parameters are fixed here before any evaluation; nothing is tuned on the qrels.

Two checks are recorded in the output and enforced by the exit code: ``acl_leaks`` (a second
collection of the same tenant holds copies of the first 100 documents; no search may return one)
and ``bm25_parity_max_abs_diff`` (SQL BM25 against the Python reference).
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from backend.config import Settings
from backend.db.pool import Database
from backend.models.loader import load_embedder, load_reranker
from backend.models.protocols import Embedder, Reranker
from backend.retrieval.bm25 import bm25_reference, query_terms
from backend.retrieval.dense import EF_SEARCH_MAX, EF_SEARCH_MIN
from backend.retrieval.rerank import rerank_hits
from backend.retrieval.service import PostgresRetriever
from backend.retrieval.types import PassageHit, SearchMode, SearchScope
from backend.retrieval.visible import passage_filter
from backend.services import Services
from evaluation.beir import BeirData, ensure_bench_scope, ingest_beir, load_beir
from evaluation.data import fetch
from evaluation.metrics.retrieval import evaluate_run, maxp

SYSTEMS = ("fts", "bm25", "dense", "dense_exact", "hybrid", "hybrid_rerank")
# system -> (retriever mode, exact dense scan). hybrid_rerank retrieves like hybrid.
_SEARCH_ARGS: dict[str, tuple[SearchMode, bool]] = {
    "fts": ("fts", False),
    "bm25": ("bm25", False),
    "dense": ("dense", False),
    "dense_exact": ("dense", True),
    "hybrid": ("hybrid", False),
    "hybrid_rerank": ("hybrid", False),
}

SPLIT = "test"
TENANT_NAME = "bench"
DEFAULT_DATABASE_URL = "postgresql://eg_app@localhost:54329/evidencegraph_bench"

K = 300  # passages requested per query (the retriever maximum)
CANDIDATES = 300  # passages each hybrid component fetches
RERANK_DEPTH = 50  # documents whose passages the reranker scores
K1 = 0.9
B = 0.4
RRF_K = 60
EF_SEARCH = min(EF_SEARCH_MAX, max(EF_SEARCH_MIN, K))  # what dense search sets for limit K
PASSAGE_TARGET_CHARS = 800  # the passage builder default; recorded, not applied here

DISTRACTOR_DOCS = 100
DISTRACTOR_PREFIX = "distractor-"
PARITY_QUERIES = 20
PARITY_TOP = 10
PARITY_TOLERANCE = 1e-9

_REPO_ROOT = Path(__file__).resolve().parent.parent


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: the value at 1-based rank ``ceil(p * n / 100)`` once sorted.

    There is no interpolation, so the result is always one of the observed values.
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(p * len(ordered) / 100))
    return ordered[rank - 1]


def split_leaks(
    hits: Sequence[PassageHit], names: Mapping[UUID, str], leaked: Collection[UUID]
) -> tuple[list[PassageHit], int]:
    """The hits that belong to the corpus (in order) and the number that came from ``leaked``.

    ``names`` maps the corpus documents to their BEIR ids, ``leaked`` holds the distractor
    documents. A hit in neither means the benchmark collection is not the corpus alone.
    """
    clean: list[PassageHit] = []
    leaks = 0
    for hit in hits:
        if hit.document_id in leaked:
            leaks += 1
        elif hit.document_id in names:
            clean.append(hit)
        else:
            raise RuntimeError("search returned a document outside the benchmark corpus")
    return clean, leaks


def document_ranking(hits: Sequence[PassageHit], names: Mapping[UUID, str]) -> list[str]:
    """BEIR ids of the hits' documents, ordered by each document's best passage (MaxP)."""
    passage_to_doc = {str(hit.passage_id): names[hit.document_id] for hit in hits}
    return maxp([str(hit.passage_id) for hit in hits], passage_to_doc)


async def rerank_documents(
    reranker: Reranker,
    query: str,
    hits: Sequence[PassageHit],
    names: Mapping[UUID, str],
    depth: int = RERANK_DEPTH,
) -> list[str]:
    """Rerank the passages of the top ``depth`` MaxP documents; the other documents follow.

    Every retrieved passage of the top documents is scored, each document takes its best
    passage's score, and the remaining documents keep their original order after them.
    """
    ranking = document_ranking(hits, names)
    top = set(ranking[:depth])
    pool = [hit for hit in hits if names[hit.document_id] in top]
    reranked = await rerank_hits(reranker, query, pool, len(pool))
    return document_ranking(reranked, names) + ranking[depth:]


def _run(command: Sequence[str], cwd: Path | None = None) -> str | None:
    """Stripped stdout of a command, or None when it cannot run or exits non-zero."""
    try:
        done = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_info(repo: Path) -> tuple[str | None, bool | None]:
    """(HEAD commit, working tree dirty) of ``repo``; both None when git cannot tell (R-B2).

    That covers no git executable and a directory that is not a repository, which is what a
    WSL process sees in a Windows-created worktree.
    """
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    if not commit:
        return None, None
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    return commit, None if status is None else bool(status)


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _cpu() -> str | None:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    for line in _read_text("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return None


def _ram_gb() -> float | None:
    """Installed memory in GiB, one decimal."""
    if sys.platform == "darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        return round(int(raw) / 2**30, 1) if raw and raw.isdigit() else None
    for line in _read_text("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            # "MemTotal:  15939260 kB" (kibibytes).
            return round(int(fields[1]) / 2**20, 1) if len(fields) > 1 else None
    return None


def _torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.__version__


async def collect_env(db: Database) -> dict[str, Any]:
    """The machine and software this run measured; never raises for a missing source (R-B2)."""
    commit, dirty = git_info(_REPO_ROOT)
    async with db.system_tx() as conn:
        cur = await conn.execute("show server_version")
        postgres = (await cur.fetchone())["server_version"]
        cur = await conn.execute("select extversion from pg_extension where extname = 'vector'")
        row = await cur.fetchone()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "cpu": _cpu(),
        "ram_gb": _ram_gb(),
        "python": sys.version,
        "torch": _torch_version(),
        "postgres": postgres,
        "pgvector": None if row is None else row["extversion"],
    }


async def _count_passages(db: Database, scope: SearchScope) -> int:
    visible, params = passage_filter(scope, None)
    async with db.tenant_tx(scope.tenant_id) as conn:
        cur = await conn.execute(
            "select count(*) as n from passages p "
            f"join documents d on d.id = p.document_id where {visible}",
            params,
        )
        return (await cur.fetchone())["n"]


async def bm25_parity(
    db: Database, retriever: PostgresRetriever, scope: SearchScope, queries: Sequence[str]
) -> float:
    """Largest absolute gap between SQL BM25 and ``bm25_reference`` over the top 10 per query.

    The reference sees every visible passage's ``passage_terms`` and token count. Each SQL hit
    is compared with the reference score of the same passage, and the SQL scores with the
    reference's ranked top 10 (so a passage the SQL missed also shows up). Infinite when the
    SQL returns a passage or a number of rows the reference does not.
    """
    visible, params = passage_filter(scope, None)
    async with db.tenant_tx(scope.tenant_id) as conn:
        cur = await conn.execute(
            "select p.id, p.token_count from passages p "
            f"join documents d on d.id = p.document_id where {visible}",
            params,
        )
        doc_lens = {str(row["id"]): row["token_count"] for row in await cur.fetchall()}
        cur = await conn.execute(
            "select t.passage_id, t.term, t.tf from passage_terms t "
            "join passages p on p.id = t.passage_id "
            f"join documents d on d.id = p.document_id where {visible}",
            params,
        )
        term_freqs: dict[str, dict[str, int]] = {}
        for row in await cur.fetchall():
            term_freqs.setdefault(str(row["passage_id"]), {})[row["term"]] = row["tf"]
        terms = [await query_terms(conn, query) for query in queries]

    worst = 0.0
    for query, query_term_list in zip(queries, terms, strict=True):
        hits = await retriever.search(
            query, scope, k=PARITY_TOP, mode="bm25", rerank=False, candidates=CANDIDATES
        )
        reference = bm25_reference(query_term_list, term_freqs, doc_lens, K1, B)
        expected = sorted(reference.values(), reverse=True)[:PARITY_TOP]
        if len(hits) != len(expected):
            return math.inf
        for hit, ranked in zip(hits, expected, strict=True):
            same_passage = reference.get(str(hit.passage_id))
            if same_passage is None:
                return math.inf
            worst = max(
                worst,
                abs(hit.scores["bm25"] - same_passage),
                abs(hit.scores["bm25"] - ranked),
            )
    return worst


async def run_system(
    name: str,
    retriever: PostgresRetriever,
    reranker: Reranker,
    scope: SearchScope,
    queries: Mapping[str, str],
    names: Mapping[UUID, str],
    leaked: Collection[UUID],
) -> tuple[dict[str, list[str]], list[float], int]:
    """Run every query on one system: (document rankings, latencies in ms, leaked hits).

    The latency of a query is the wall time of the search, plus the reranking for
    ``hybrid_rerank``; bookkeeping after it is not timed.
    """
    mode, exact = _SEARCH_ARGS[name]
    rankings: dict[str, list[str]] = {}
    latencies_ms: list[float] = []
    leaks = 0
    for query_id, text in queries.items():
        started = time.perf_counter()
        hits = await retriever.search(
            text, scope, k=K, mode=mode, rerank=False, candidates=CANDIDATES, exact=exact
        )
        clean, found = split_leaks(hits, names, leaked)
        if name == "hybrid_rerank":
            rankings[query_id] = await rerank_documents(reranker, text, clean, names)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
        else:
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            rankings[query_id] = document_ranking(clean, names)
        leaks += found
    return rankings, latencies_ms, leaks


def _load_models(kind: str, settings: Settings) -> tuple[Embedder, Reranker]:
    if kind == "fake":
        # The deterministic stand-ins of the test suite: for smoke runs, never for results.
        from tests.fakes import HashEmbedder, KeywordReranker

        return HashEmbedder(), KeywordReranker()
    return load_embedder(settings), load_reranker(settings)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _systems(text: str) -> tuple[str, ...]:
    chosen = [name.strip() for name in text.split(",") if name.strip()]
    unknown = sorted(set(chosen) - set(SYSTEMS))
    if unknown or not chosen:
        raise argparse.ArgumentTypeError(f"systems must be among {','.join(SYSTEMS)}")
    return tuple(name for name in SYSTEMS if name in chosen)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run_retrieval")
    parser.add_argument("--dataset", required=True, choices=["scifact", "nfcorpus"])
    parser.add_argument(
        "--database-url",
        help="default: $HALLUCISCOPE_BENCH_DATABASE_URL, else " + DEFAULT_DATABASE_URL,
    )
    parser.add_argument(
        "--systems", type=_systems, default=SYSTEMS, help="comma separated, default: all"
    )
    parser.add_argument("--limit-queries", type=_positive_int, help="use the first N queries")
    parser.add_argument("--embedder", choices=["real", "fake"], default="real")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="folder holding <dataset>/ in BEIR layout; default: the pinned, verified download",
    )
    parser.add_argument(
        "--out", type=Path, help="default: evaluation/results/retrieval_<dataset>.json"
    )
    args = parser.parse_args(argv)
    if args.database_url is None:
        args.database_url = os.environ.get("HALLUCISCOPE_BENCH_DATABASE_URL", DEFAULT_DATABASE_URL)
    if args.out is None:
        args.out = Path("evaluation/results") / f"retrieval_{args.dataset}.json"
    return args


def _load_dataset(args: argparse.Namespace) -> BeirData:
    root = args.data_root if args.data_root is not None else fetch(args.dataset)
    return load_beir(root, args.dataset, SPLIT)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Ingest the corpus (when missing), run every requested system, write and return results."""
    started_at = datetime.now(UTC)
    clock = time.perf_counter()
    data = _load_dataset(args)
    # Only queries with a relevant document are scored (and so only they are run).
    query_ids = [q for q in data.queries if any(grade > 0 for grade in data.qrels[q].values())]
    if args.limit_queries is not None:
        query_ids = query_ids[: args.limit_queries]
    queries = {q: data.queries[q] for q in query_ids}
    qrels = {q: data.qrels[q] for q in query_ids}

    settings = Settings()
    embedder, reranker = _load_models(args.embedder, settings)
    db = Database(args.database_url, max_size=4)
    await db.open()
    try:
        services = Services(settings, db, embedder, reranker)
        tenant_id, collection_id = await ensure_bench_scope(db, TENANT_NAME, args.dataset)
        _, distractor_id = await ensure_bench_scope(db, TENANT_NAME, f"{args.dataset}-distractor")
        ingest_started = time.perf_counter()
        corpus = await ingest_beir(services, data, tenant_id=tenant_id, collection_id=collection_id)
        distractor = await ingest_beir(
            services,
            data,
            tenant_id=tenant_id,
            collection_id=distractor_id,
            doc_ids=list(data.corpus)[:DISTRACTOR_DOCS],
            filename_prefix=DISTRACTOR_PREFIX,
        )
        ingest_seconds = time.perf_counter() - ingest_started

        # The scope holds the benchmark collection only; the distractor stays unreadable.
        scope = SearchScope(tenant_id, (collection_id,))
        names = {document_id: beir_id for beir_id, document_id in corpus.doc_map.items()}
        leaked = set(distractor.doc_map.values())
        retriever = PostgresRetriever(db, embedder, reranker, k1=K1, b=B, rrf_k=RRF_K)

        systems: dict[str, dict[str, float]] = {}
        acl_leaks = 0
        for name in args.systems:
            rankings, latencies_ms, leaks = await run_system(
                name, retriever, reranker, scope, queries, names, leaked
            )
            acl_leaks += leaks
            scores = evaluate_run(rankings, qrels)
            systems[name] = {
                "ndcg@10": scores["ndcg@10"],
                "recall@10": scores["recall@10"],
                "recall@100": scores["recall@100"],
                "mrr@10": scores["mrr@10"],
                "latency_ms_p50": percentile(latencies_ms, 50),
                "latency_ms_p95": percentile(latencies_ms, 95),
            }

        parity = await bm25_parity(db, retriever, scope, list(queries.values())[:PARITY_QUERIES])
        n_passages = await _count_passages(db, scope)
        env = await collect_env(db)
    finally:
        await db.close()

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "split": SPLIT,
        "n_queries": len(queries),
        "n_docs": len(corpus.doc_map),
        "n_passages": n_passages,
        "n_skipped_docs": corpus.n_skipped_empty,
        "systems": systems,
        "checks": {"acl_leaks": acl_leaks, "bm25_parity_max_abs_diff": parity},
        "params": {
            "k1": K1,
            "b": B,
            "rrf_k": RRF_K,
            "candidates": CANDIDATES,
            "rerank_depth": RERANK_DEPTH,
            "ef_search": EF_SEARCH,
            "passage_target_chars": PASSAGE_TARGET_CHARS,
        },
        "models": {"embedder": embedder.model_id, "reranker": reranker.model_id},
        "env": env,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ingest_seconds": ingest_seconds,
        "runtime_seconds": time.perf_counter() - clock,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark; exit 1 when a check fails (the results file is written first)."""
    args = parse_args(argv)
    result = asyncio.run(run(args))
    print(f"wrote {args.out}")
    checks = result["checks"]
    failed = False
    if checks["acl_leaks"] > 0:
        print(f"FAILED: acl_leaks = {checks['acl_leaks']}, must be 0", file=sys.stderr)
        failed = True
    if checks["bm25_parity_max_abs_diff"] >= PARITY_TOLERANCE:
        print(
            f"FAILED: bm25_parity_max_abs_diff = {checks['bm25_parity_max_abs_diff']:.3e}, "
            f"must be below {PARITY_TOLERANCE:.0e}",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
