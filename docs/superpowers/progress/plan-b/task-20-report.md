# Task 20 report: retrieval benchmark runner and report renderer

Status: DONE_WITH_CONCERNS (concerns below are about running Task 21, not about the code)

Commit: `05a6b40 feat: add retrieval benchmark runner and report` on `feature/evidencegraph` (not pushed).
Pre-commit hooks ran (gitleaks: no leaks found).

## What I built

- `evaluation/run_retrieval.py`: `python -m evaluation.run_retrieval --dataset {scifact,nfcorpus} [--database-url] [--systems] [--limit-queries] [--embedder real|fake] [--data-root] [--out]`.
  - `parse_args(argv)`, `async run(args) -> dict` (also writes the JSON to `--out`), `main(argv) -> int` (exit 1 when `acl_leaks > 0` or `bm25_parity_max_abs_diff >= 1e-9`; the JSON is written before the exit code is decided).
  - Six systems exactly as in the brief: every system calls `PostgresRetriever.search(query, scope, k=300, candidates=300, rerank=False, mode=..., exact=...)` and collapses with `maxp`. `hybrid_rerank` keeps the top 50 MaxP documents, reranks every retrieved passage of those documents (`rerank_hits`), scores a document by its best passage, and appends the remaining hybrid documents in their original order.
  - Latency per query: `perf_counter` wall time of search (plus reranking for `hybrid_rerank`). p50/p95 use the nearest-rank method (`rank = ceil(p * n / 100)` on the sorted values, no interpolation); stated in the `percentile` docstring.
  - Checks: `acl_leaks` (distractor collection with copies of the first 100 corpus documents, filenames `distractor-<id>.txt`, never in the search scope; every hit from it in any system is counted) and `bm25_parity_max_abs_diff` (first 20 queries, SQL BM25 top 10 against `bm25_reference` over all visible passages' `passage_terms` and token counts; both per-passage and rank-wise comparisons; infinite if the SQL returns a passage or row count the reference does not).
  - Env per R-B2: `git` from PATH with `cwd` = repo root (null/null when git cannot run or the directory is not a repository); CPU and RAM from `sysctl` on macOS, `/proc/cpuinfo` (`model name`) and `/proc/meminfo` (`MemTotal`) on Linux; `python` = `sys.version`; `torch` imported inside the function; `postgres` = `show server_version`; `pgvector` = `pg_extension.extversion`.
  - Params recorded: `k1 0.9, b 0.4, rrf_k 60, candidates 300, rerank_depth 50, ef_search 300, passage_target_chars 800`. The retriever is constructed with these same `K1/B/RRF_K` constants, and `ef_search` is derived from the dense module's own clamp for `limit = 300`.
  - `--embedder fake` imports `HashEmbedder` and `KeywordReranker` from `tests.fakes` (lazy import, smoke runs only).
- `evaluation/report.py`: `render_retrieval(results) -> str` (one section per dataset in input order: environment and parameter lines, the table `| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |`, metrics to 4 decimals and latency to 1, the checks, the fixed methodology note) and `python -m evaluation.report retrieval <results.json>... --out docs/results/retrieval.md` (LF output). Null env values render as `unknown`.
- `tests/test_report.py` (new, 4 tests) and `tests/test_beir.py` (+14 test cases: smoke run and supporting tests).

## Files changed

- Created: `evaluation/run_retrieval.py`, `evaluation/report.py`, `tests/test_report.py`
- Modified: `tests/test_beir.py` (imports + appended tests; existing tests untouched)

## RED evidence

Command: `wsl.sh timeout 300 python -m pytest tests/test_report.py tests/test_beir.py -q` before any implementation:

```
ERROR collecting tests/test_report.py
tests/test_report.py:5: in <module>
    from evaluation import report
E   ImportError: cannot import name 'report' from 'evaluation'
ERROR collecting tests/test_beir.py
tests/test_beir.py:16: in <module>
    from evaluation import beir, data, run_retrieval
E   ImportError: cannot import name 'run_retrieval' from 'evaluation'
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
2 errors in 0.99s
```

Both fail because the feature does not exist (nothing slow or dangerous ran).

## GREEN evidence

- `tests/test_report.py tests/test_beir.py`: `44 passed in 3.42s` (includes the `@pg` smoke tests; none skipped).
- Manual end-to-end run of `run_retrieval.main` on a scratch database with the tiny dataset (twice, second run reuses the ingested corpus): exit code 0 both times, all six systems present, `acl_leaks 0`, `bm25_parity_max_abs_diff 0.0`, then `evaluation.report retrieval` rendered a correct markdown file. Scratch database and temp files removed.
- Mutation checks (each reverted afterwards by restoring a byte copy of the file):
  - reference BM25 computed with k1 = 2.0: `test_bm25_parity_flags_a_retriever_with_other_parameters` fails
  - `top = set(ranking)` (rerank depth ignored): `test_rerank_documents_reranks_only_the_top_documents` fails
  - distractor not ingested (`doc_ids=[]`): `test_runner_smoke` fails on the collection row assertion
  - document scored by its first passage: `test_rerank_documents_reranks_only_the_top_documents` fails
- Full suite: `wsl.sh timeout 900 python -m pytest -q` -> `1016 passed, 2 deselected, 2 warnings in 136.74s`. The 2 deselected are the `models` marker; the 2 warnings are the pre-existing `coroutine ... never awaited` warnings in `tests/test_generator.py` (not touched by this task).
- Lint: `ruff check .` -> `All checks passed!` (run again after the last edit, before the commit). `ruff format` was applied to my two Python files only; the repo as a whole is not ruff-format clean (30 older files), and I did not touch them.

## Tests added (what each one catches)

- `test_render_retrieval_golden`, `..._one_section_per_dataset_in_input_order`, `..._marks_unknown_git_state_instead_of_crashing`, `test_report_cli_writes_the_rendered_file`
- `test_percentile_is_nearest_rank`, `test_split_leaks_*` (2), `test_document_ranking_orders_documents_by_their_best_passage`, `test_rerank_documents_*` (2: depth respected; best-passage scoring)
- `test_git_info_is_null_outside_a_repository` (R-B2), `test_main_exits_nonzero_when_a_check_fails` (5 cases incl. the `>= 1e-9` boundary)
- `@pg test_run_system_counts_distractor_hits_only_when_the_collection_is_in_scope` (a real leak is counted when the distractor is in scope, zero otherwise)
- `@pg test_bm25_parity_flags_a_retriever_with_other_parameters`
- `@pg test_runner_smoke` (every system present, checks, metric ranges, params, models, env keys, JSON file equals the return value, and the distractor collection holds 3 `distractor-` documents)

## Decisions (brief ambiguities, spec/rulings applied)

1. Distractor collection name is `<dataset>-distractor` (not the bare `distractor`): a bare name shared by two datasets would collide on `distractor-<id>.txt` filenames across datasets. The prefix and the "first 100 corpus documents" are as in the brief.
2. `--data-root` (smoke-test hook): given, the BEIR folder is read from `<data-root>/<dataset>` as is. Not given, `evaluation.data.fetch(dataset)` is used, which verifies the pinned sha256 (and downloads if missing).
3. Scored and run queries are those with at least one grade > 0 (the metrics module skips the others anyway); `--limit-queries N` takes the first N of them in file order. `n_queries` is that count.
4. A hit from a document that is in neither the corpus nor the distractor raises `RuntimeError` (a polluted benchmark collection must not silently produce numbers).
5. Leaked hits are dropped from the ranking and counted; the run fails by exit code anyway.
6. The parity check runs regardless of `--systems`; `acl_leaks` covers the systems that ran.
7. `n_docs` = documents of this corpus in the collection (ingested + already present); `n_passages` = current passages of non-deleted documents in the benchmark collection.

## Concerns for Task 21 (controller)

- Git info will be null when the runner runs under WSL in this worktree (gitdir is a Windows path), so the rendered report says "Commit: unknown". If the published report must carry a commit hash, run the benchmark from a checkout where `git rev-parse HEAD` works.
- `ram_gb` under WSL is the VM's `MemTotal` (15.2 GiB on this machine), not the host's installed RAM (about 32 GB). CPU is the host's model name. Decide which to publish.
- The default database `postgresql://eg_app@localhost:54329/evidencegraph_bench` must exist, be migrated and owned/granted for `eg_app`; the runner does not create or migrate it. Use a fresh database for the published run: `ingest_seconds` only measures this run's ingestion (about 0 when the corpus is already present).
- Real models (`--embedder real`) were not run here: unit tests never download models, so the `load_embedder`/`load_reranker` path and real-data latency are untested in this task. No warm-up query is run, so the first query includes any model warm-up (this affects the maximum, only marginally p95).
- If the parity check ever fails with a missing/extra passage, `bm25_parity_max_abs_diff` is `Infinity` in the JSON (valid for Python's `json`, not strict JSON).
- Timing includes the query embedding (it happens inside `search`) and, for `hybrid_rerank`, the reranker call on a worker thread.
