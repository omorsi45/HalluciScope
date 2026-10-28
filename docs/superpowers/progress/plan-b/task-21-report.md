# Task 21 report: BEIR retrieval benchmark on real data

Status: DONE_WITH_CONCERNS

Commit: f68c2ac docs: add measured BEIR retrieval results (branch feature/evidencegraph, not pushed). Files: docs/results/retrieval.md, evaluation/manifests/beir.json. No code changed.

## Runtimes

- scifact run: wall clock 728 s (stopwatch around the wsl call, includes model load); runner-reported runtime_seconds and ingest_seconds are in the tables below. Exit 0.
- nfcorpus run: wall clock 668 s. Exit 0.
- Full suite: 1016 passed, 2 deselected, 2 warnings in 141.33 s (wall 145 s, exit 0). The 2 deselected are the 'models' marker (addopts -m 'not models' in pyproject.toml). The 2 warnings are httpx ResourceWarnings from the existing suite. Log: .local/task21-pytest.log
- ruff check .: All checks passed! (exit 0)

## Checks (from the results JSON)

scifact:
```json
  "checks": {
    "acl_leaks": 0,
    "bm25_parity_max_abs_diff": 7.105427357601002e-15
  }
```

nfcorpus:
```json
  "checks": {
    "acl_leaks": 0,
    "bm25_parity_max_abs_diff": 3.552713678800501e-15
  }
```

Expected gates: acl_leaks == 0, bm25_parity_max_abs_diff < 1e-9, env.git_commit 40 hex chars, env.git_dirty false. All met on both datasets (see env block below). BM25 sanity thresholds (scifact >= 0.55, nfcorpus >= 0.25): met, see tables.

## Env block (from the results JSON)

scifact:
```json
  "env": {
    "git_commit": "05a6b403e187f3401f34bad4235b79a6f8cff433",
    "git_dirty": false,
    "cpu": "AMD Ryzen 9 7900X 12-Core Processor",
    "ram_gb": 15.2,
    "python": "3.14.9 (main, Oct 21 2026, 17:48:02) [Clang 23.1.3 ]",
    "torch": "2.14.1+cpu",
    "postgres": "17.11 (Debian 17.11-1.pgdg12+2)",
    "pgvector": "0.8.7"
  }
```

nfcorpus:
```json
  "env": {
    "git_commit": "05a6b403e187f3401f34bad4235b79a6f8cff433",
    "git_dirty": false,
    "cpu": "AMD Ryzen 9 7900X 12-Core Processor",
    "ram_gb": 15.2,
    "python": "3.14.9 (main, Oct 21 2026, 17:48:02) [Clang 23.1.3 ]",
    "torch": "2.14.1+cpu",
    "postgres": "17.11 (Debian 17.11-1.pgdg12+2)",
    "pgvector": "0.8.7"
  }
```

## Rendered results (verbatim copy of docs/results/retrieval.md)

### Retrieval benchmark

#### scifact (test split)

- Queries: 300, documents: 5183, passages: 14741, documents skipped as empty: 0
- Run: 2026-10-28T02:58:18+00:00 to 2026-10-28T03:10:23+00:00 (ingest 286.5 s, total 725.7 s)
- Commit: 05a6b403e187f3401f34bad4235b79a6f8cff433 (working tree: clean)
- CPU: AMD Ryzen 9 7900X 12-Core Processor, RAM: 15.2 GB
- Python: 3.14.9 (main, Oct 21 2026, 17:48:02) [Clang 23.1.3 ], torch: 2.14.1+cpu, PostgreSQL: 17.11 (Debian 17.11-1.pgdg12+2), pgvector: 0.8.7
- Models: embedder BAAI/bge-small-en-v1.5, reranker cross-encoder/ms-marco-MiniLM-L6-v2
- Parameters: BM25 k1 = 0.9, b = 0.4; RRF k = 60; candidates = 300; rerank depth = 50 documents; HNSW ef_search = 300; passage target = 800 characters

| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fts | 0.4398 | 0.6037 | 0.7911 | 0.3914 | 68.0 | 169.1 |
| bm25 | 0.6686 | 0.7850 | 0.9129 | 0.6363 | 26.2 | 36.2 |
| dense | 0.7105 | 0.8307 | 0.9600 | 0.6802 | 21.8 | 24.3 |
| dense_exact | 0.7104 | 0.8307 | 0.9600 | 0.6801 | 63.6 | 66.5 |
| hybrid | 0.7230 | 0.8489 | 0.9650 | 0.6887 | 43.1 | 52.6 |
| hybrid_rerank | 0.7006 | 0.8319 | 0.9650 | 0.6669 | 1124.8 | 1747.7 |

Checks:

- acl_leaks: 0 (must be 0)
- bm25_parity_max_abs_diff: 7.105e-15 (must be < 1e-9)

Methodology: BM25 parameters were fixed before evaluation. Passages are sentence-aligned and every system collapses its passage ranking to documents with MaxP (a document ranks by its best passage). There was no tuning on test qrels.

#### nfcorpus (test split)

- Queries: 323, documents: 3633, passages: 10964, documents skipped as empty: 0
- Run: 2026-10-28T03:10:38+00:00 to 2026-10-28T03:21:43+00:00 (ingest 211.4 s, total 665.7 s)
- Commit: 05a6b403e187f3401f34bad4235b79a6f8cff433 (working tree: clean)
- CPU: AMD Ryzen 9 7900X 12-Core Processor, RAM: 15.2 GB
- Python: 3.14.9 (main, Oct 21 2026, 17:48:02) [Clang 23.1.3 ], torch: 2.14.1+cpu, PostgreSQL: 17.11 (Debian 17.11-1.pgdg12+2), pgvector: 0.8.7
- Models: embedder BAAI/bge-small-en-v1.5, reranker cross-encoder/ms-marco-MiniLM-L6-v2
- Parameters: BM25 k1 = 0.9, b = 0.4; RRF k = 60; candidates = 300; rerank depth = 50 documents; HNSW ef_search = 300; passage target = 800 characters

| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fts | 0.2534 | 0.1304 | 0.2254 | 0.4125 | 25.1 | 41.9 |
| bm25 | 0.3211 | 0.1578 | 0.2462 | 0.5174 | 26.3 | 47.2 |
| dense | 0.3488 | 0.1629 | 0.3081 | 0.5455 | 23.3 | 68.1 |
| dense_exact | 0.3500 | 0.1645 | 0.3093 | 0.5455 | 61.0 | 85.8 |
| hybrid | 0.3596 | 0.1757 | 0.3143 | 0.5629 | 45.9 | 59.5 |
| hybrid_rerank | 0.3504 | 0.1663 | 0.3143 | 0.5580 | 1115.8 | 1664.9 |

Checks:

- acl_leaks: 0 (must be 0)
- bm25_parity_max_abs_diff: 3.553e-15 (must be < 1e-9)

Methodology: BM25 parameters were fixed before evaluation. Passages are sentence-aligned and every system collapses its passage ranking to documents with MaxP (a document ranks by its best passage). There was no tuning on test qrels.

## Manifest (verbatim copy of evaluation/manifests/beir.json)

```json
{
  "nfcorpus": {
    "fetched_at": "2026-10-28T02:32:17+00:00",
    "n_docs": 3633,
    "n_passages": 10964,
    "n_queries": 323,
    "sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    "split": "test",
    "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
  },
  "scifact": {
    "fetched_at": "2026-10-28T02:32:15+00:00",
    "n_docs": 5183,
    "n_passages": 14741,
    "n_queries": 300,
    "sha256": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    "split": "test",
    "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
  }
}
```

fetched_at is the UTC modification time of the downloaded zip under evaluation/data/ (set when the controller fetched it; my 'fetch' call was a verified no-op).  Generator: .local/make_beir_manifest.py (uncommitted).

## Process notes

- Ran fetch, both benchmarks, render, manifest through .local/wsl-git.sh; tests and ruff through .local/wsl.sh; status and commit with Windows git. Logs: .local/bench-scifact.log, .local/bench-nfcorpus.log (the leading 'NativeCommandError' lines in them are PowerShell 5.1 wrapping WSL stderr progress bars, not a failure; both runs exited 0 and wrote their JSON).
- The manifest script needed PYTHONPATH=. because running a script file puts its own directory (.local), not the cwd, on sys.path: run as 'wsl-git.sh env PYTHONPATH=. timeout 120 python .local/make_beir_manifest.py'. No repo code was touched.
- git status before commit showed only the two new files; evaluation/results/ and evaluation/data/ are gitignored, .local/ is in .git/info/exclude. Both files verified LF-only, no em dashes, trailing newline. Commit hook (gitleaks) passed.

## Concerns

- hybrid_rerank nDCG@10 is lower than hybrid on both datasets (see tables: scifact hybrid 0.7230 vs hybrid_rerank 0.7006; nfcorpus hybrid 0.3596 vs hybrid_rerank 0.3504). The cross-encoder (ms-marco-MiniLM-L6-v2, rerank depth 50) does not help in this setup and costs about 1.1 s p50 per query on CPU. Published as measured, no tuning. The controller may want to note this in the writeup.
- The fts system is far below bm25 on scifact (0.4398 vs 0.6686). I did not investigate the cause; it is not a gate in the brief.
- The env block reports RAM 15.2 GB, while the machine rules list about 31 GB usable. I did not check the WSL memory limit, so the gap is unexplained.
- Committed docs/results/retrieval.md was not hand edited; its Methodology line says no tuning on test qrels, which holds because this task changed no parameters.
