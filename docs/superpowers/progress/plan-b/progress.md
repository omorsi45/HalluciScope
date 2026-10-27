# SDD ledger: plan docs/superpowers/plans/2026-10-26-evidencegraph-b-ingest-retrieval.md

Spec: docs/superpowers/specs/2026-10-26-evidencegraph-design.md (reachable; R1-R10 override earlier sections).
Branch: feature/evidencegraph. Start commit: 536b023 (Plan A complete, incl. final fix wave 15bc7d6..536b023).
Process (Omar, 2026-10-27): controller plans and verifies; one claude-sonnet-5.5 implementer per task, sequential; no per-task reviewer, no Opus. Controller acceptance check per task: RED then GREEN in the report, full suite and ruff pass (controller re-runs), commit with the plan's message exists, diff stat touches only the task's files. One whole-branch review and one fix wave at the end of Plan D.
Briefs: feature worktree .superpowers/sdd/plan-b/task-N-brief.md (Global Constraints + Review Focus + task). Reports: this folder, task-N-report.md.

## Environment (Windows PC, 2026-10-27)
- Repo at C:/Users/Omar Morsi/Desktop/Projects/HalluciScope; worktrees .worktrees/evidencegraph and .worktrees/notes. .git/info/exclude is shared, so new notes files under docs/superpowers need `git add -f`.
- Postgres: Docker container eg-pg (pgvector/pgvector:pg17; PostgreSQL 17.11, pgvector 0.8.7, citext 1.6), localhost:54329, trust. Dev db evidencegraph with eg_owner/eg_app created by hand (mirrors scripts/dev-postgres.sh).
- Python runs in WSL2 Ubuntu 24.04 via .local/wsl.sh (venv ~/venvs/evidencegraph, Python 3.14.9, torch 2.14.1+cpu, same pins otherwise). Native Windows Python 3.14.7 fails the suite: psycopg refuses ProactorEventLoop (239 errors) and the worker uses loop.add_signal_handler (POSIX only). Ruling: run Python only in WSL; do not change the code for native Windows. Cost if wrong: a Windows developer cannot run tests natively (documented in Plan D operations docs).
- Baseline in WSL: 446 passed in 61 s, ruff clean, migrate applied [1, 2] on the fresh dev db.
- Node v22.23.2 installed (Plan D needs 24+). Ollama 0.34.2 installed, not running. Docker 29.8.0 running.

## Pre-flight scan

| Rows | Produces / consumes | Finding |
|---|---|---|
| T1 self | types, errors, normalize, assembler; tests list exact cases | Consistent. |
| T1 -> T2, T3 | Sentence, PassageDraft, PageSpan, SectionSpan | Consistent. |
| T2 self | abbreviation and initial cases vs rule 3 (uppercase/digit after) | Checked each case by hand: "p.m. on", "St. today", "fig. 2", "e.g. case" behave as the expected counts. Consistent. |
| T3 self | overlap test (10 x 100 chars, target 350 -> 3 per passage, overlap 1); no-overlap test (700-char sentences) | Consistent with the overlap rule. |
| T4 -> T5, T6 | decode_utf8 reused by csv; parsers/__init__ empty in T4, dispatch in T6 | Consistent. |
| T6 self | parse_child imports only backend.ingest, stdlib, pypdf; legacy core/document.py delegates | Consistent. |
| T7 self | eg_token_count in generated column; triggers; deferred Plan A indexes | Consistent. Migration 0003 numbering: Plan A ends at 0002. |
| T7 -> T9 | create_version inserts into the tables and relies on the slice trigger | Consistent. |
| T8 -> T9, T10, T13-T17 | Embedder/Reranker protocols, Services, HashEmbedder, KeywordReranker | Consistent. |
| T9 -> T10, T11, T13 | repo functions, create_version, VersionResult | Consistent. |
| T10 -> T11-T17 | services fixture, drain helper, build_handlers(services) | Consistent. T10 changes cli worker wiring: existing test_run_worker_* tests must stay green without loading real models (dispatch note). |
| T11 self | collections DELETE 409 uses count_live_documents (T9) | Consistent. |
| T12 self | diff route in documents.py (T11) | Consistent. Spec 15 also says passages matched by text_sha256 are unchanged; sentence-level SequenceMatcher yields the same equal hunks for identical text. Consistent in effect. |
| T13 -> T14, T16 | visible_version_ids, passage_filter, seed_docs | Consistent. |
| T15 -> T16 | rrf_fuse, rerank_hits | Consistent. |
| T16 -> T17, T20 | PostgresRetriever.search signature incl. exact | Consistent. |
| T17 vs spec R4 | plan uses one RLS table search_requests (query + hits); R4 names search_results(hits) | Ruling R-B1 below. |
| T17 vs spec 16 | spec 16 lets only bm25/fts run in the API; R7 lets every mode without rerank run there | R7 overrides 16; plan follows R7. Consistent. |
| T17 self | startup lifespan loads the embedder | Existing startup tests must not download models (dispatch note). |
| T18 -> T20 | evaluate_run, maxp | Consistent. |
| T19 -> T20, Plan C | REGISTRY incl. RAGTruth files pinned at commit c103204b | Consistent; hash step needs network. |
| T20 self | env via /usr/bin/git and sysctl (macOS only) | Ruling R-B2 below. |
| T21 self | Postgres steps use createdb/psql and dev-postgres.sh (macOS) | Ruling R-B3 below. |
| Plan-mandated defects | none found (every listed test asserts behavior) | Clean. |

Rulings:
- Ruling R-B1: Task 17 keeps the plan's single RLS table `search_requests` (query, params, status, hits) instead of R4's `search_results(hits)`. Why: the worker must read the query from somewhere other than the job payload, R4's intent (ids-only payloads, RLS storage, 1 hour expiry) is fully met, and one row per request is simpler. Cost if wrong: a table rename and a split migration.
- Ruling R-B2: Task 20 collects env portably: git from PATH (not /usr/bin/git); CPU and RAM from sysctl on macOS and /proc/cpuinfo plus /proc/meminfo on Linux. Why: benchmarks here run in WSL Linux. Cost if wrong: none (macOS path unchanged).
- Ruling R-B3: Task 21 on this machine creates the bench database with `docker exec eg-pg ...` and runs every command through .local/wsl.sh; the results state "WSL2 Ubuntu on AMD Ryzen 9 7900X, CPU only" and the RAM visible to WSL. The spec's M1 references describe the original machine; scale and hardware are reported as measured. Cost if wrong: none to correctness.

## Progress
