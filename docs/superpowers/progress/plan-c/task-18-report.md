# Plan C Task 18 report: quick check, legacy API adapters, `halluciscope check`

Status: DONE_WITH_CONCERNS (see Concerns)
Commit: `2ae414f feat: serve quick checks and the legacy API through the new engine` on `feature/evidencegraph` (not pushed)

## What was built

- `backend/config.py`: `quick_wait_s: float = 120.0`.
- `backend/ingest/intake.py`: `intake_upload(..., enqueue_job: bool = True)`; `False` records the upload row and queues no `ingest_upload`.
- `backend/ingest/handlers.py`: `process_upload(services, tenant_id, upload_id) -> VersionResult | None` extracted from the job. Idempotent (None for a missing or already final upload). `handle_ingest_upload` is now a one-line wrapper. `_append_version` and `_fail_permanently` take `services`/`db` instead of a `JobContext`; the reverify job's `request_id` comes from `request_id_var` (the worker binds it to `job.request_id`, so it is the same value).
- `backend/db/repos/collections.py`: `insert_collection(..., *, kind="normal")`.
- `backend/analysis/service.py`: `create_analysis_request(..., enqueue=True)`; `require_capacity(conn, principal, settings)` (the 429 check, shared with the quick path); a `quick` analysis calls `process_upload` inside the `retrieving` stage and then follows the ask path (retrieve, generate, extract, verify). `PermanentJobError` from the inline ingestion becomes the analysis `error_code`.
- `backend/analysis/quick.py` (new): `create_quick_analysis(db, principal, *, data, filename, question, settings, enqueue=True) -> UUID` exactly as the brief: `require_capacity` first, then collection `quick-<uuid4 hex>` (kind `quick`) plus an `admin` grant, then `intake_upload(collection_kinds=("quick",), enqueue_job=False)`, then `create_analysis_request(kind="quick", ..., options={"upload_id", "extraction": "llm_atomic"}, collection_kinds=("quick",), enqueue=enqueue)`.
- `backend/api/routes/legacy.py` (new): `POST /api/analyze`, `GET /api/analyses?limit=`, `POST /api/feedback`, the pure mapping `analyze_response(...)`, and `DeprecationHeadersMiddleware` (adds `Deprecation: true` and `Link: </api/v1/ask>; rel="successor-version"` to every response of the three paths, errors included).
- `backend/api/schemas.py`: legacy shapes with `id`/`analysis_id` as `str`; request fields reuse the v1 validators (new `DocumentText` alias in `schemas_v1.py`: 1..1,000,000 chars, no NUL, no lone surrogates; `AskQuestion` 1..2000; `FeedbackNote`).
- `backend/api/app.py`: legacy router mounted at `/api`, middleware added, old `analysis` and `feedback` routers removed. `GET /health` untouched.
- `backend/api/startup.py`: legacy pipeline, chunker, verifiers and SQLite repository wiring removed. It now opens the database, the auth service, the embedder and the shared HTTP client.
- `backend/services.py`: `build_services(settings, db, http)` (calibrator file check, embedder, reranker, NLI, parity check, `OllamaClient`), `load_calibrator`, `CalibratorRejected(field)`. The worker (`_run_worker`) now calls `build_services`.
- `backend/cli/main.py`: `halluciscope check --doc PATH --question TEXT [--verbose]` runs in-process (`create_quick_analysis(enqueue=False)` then `run_analysis`), prints `Question:`, `Answer:` and a `Claim | Verdict | P(supported)` table (verbose adds an `Evidence` column: `<title> v<version> p.<page>` and the span text of the top evidence item). Exit 1 with `error: <code>` on stderr for any failure. The CLI principal is tenant `cli`, user `cli@localhost` (created on first use, `disabled_at` set, password hash `!` that verifies against nothing), role `owner`, `auth="system"`.
- Deleted with `git rm`: `backend/api/routes/analysis.py`, `backend/api/routes/feedback.py`, `tests/test_api.py`.
- Tests: new `tests/test_legacy_quick.py` (29 tests), `tests/test_cli.py` rewritten (9 tests), `tests/test_api_startup.py` updated to the new startup (6 tests).

## RED evidence

Command (tests written first, nothing implemented):
`python -m pytest tests/test_legacy_quick.py tests/test_cli.py tests/test_api_startup.py -q`

Result: `38 failed, 4 passed in 7.39s` (the 4 passes were the pre-existing startup tests of the old wiring). Reasons seen in the log:
- every `/api/analyze`, `/api/analyses`, `/api/feedback` test: `assert 500 == 200` / `500 == 401` (the old route read `app.state.pipeline`, which is absent), no `Deprecation` headers;
- `tests/test_cli.py`: `AttributeError: module 'backend.cli.main' has no attribute 'build_services'`;
- direct-call tests: `ModuleNotFoundError: backend.analysis.quick`, `ImportError` for `process_upload` and `analyze_response`;
- `test_startup_wires_neither_the_legacy_pipeline_nor_the_sqlite_repository` (temporary RED for the startup change): failed because `app.state.pipeline` existed.

No RED run loaded a model, reached Ollama or hung; the longest run took 7 s.

## GREEN evidence

- Focused: `tests/test_legacy_quick.py tests/test_cli.py tests/test_api_startup.py`: `43 passed in 21.26s` (first GREEN); after adding the member test `test_quick_check_works_for_a_member_without_collection_grants`: all pass.
- Mutation check: changing the quick grant from `admin` to `read` makes the member test fail with 403 `insufficient_permission`; reverted afterwards.
- Existing worker/ingest tests (`test_worker.py`, `test_ingest_upload.py`, `test_run_analysis.py`, `test_analysis_routes.py`) passed unchanged, including the `_run_worker` harness tests that patch `backend.models.loader.*` and `backend.llm.client.OllamaClient`.
- Full suite on the final tree (before commit): `1469 passed, 3 deselected, 2 warnings in 263.57s`. The 2 warnings are the existing `tests/test_generator.py` `AsyncMock` warnings in `backend/core/generator.py` (legacy core, untouched). The 3 deselected are `models`-marked tests (`addopts = -m 'not models'`).
- `ruff check .`: `All checks passed!`
- Commit hook (gitleaks): no leaks found.
- No em dash in any file I touched (grep for U+2014 only hits the untouched legacy `backend/core/generator.py`).

## Decisions

1. **Quick runs like an ask.** Per the brief, `process_upload` first, then the ask path. The inline ingestion sits inside the `retrieving` stage, so no stage name was added and a poll shows `retrieving` while the document is indexed.
2. **Ingestion failures keep their code.** An unreadable document (`no_text`, `bad_pdf`, ...) fails the analysis with that ingest code (the `error_code` column is free text); the legacy route maps it to 500 as the brief's "500 otherwise" says. `RetryableJobError("embed_failed")` becomes `internal_error` (LLM-side work is never retried).
3. **Crash-recovery guard.** `process_upload` returns None for a final upload, so a quick run after an earlier attempt that failed the upload (then died before failing the analysis) would have answered from an empty collection. `_ingest_quick_upload` therefore reads the upload status and raises `PermanentJobError(upload.error_code)` for a failed one. Covered by `test_quick_analysis_with_a_failed_upload_fails_with_the_upload_code`.
4. **`Deprecation`/`Link` on every response** (401, 422, 429, 413, 504 included) through a small ASGI middleware scoped to the three exact paths; a route-level header would miss framework-generated errors.
5. **`create_quick_analysis` cleans up after a refused upload** (415, 413, ...): the still empty hidden collection is deleted (`test_create_quick_analysis_leaves_no_collection_behind_when_the_file_is_refused`).
6. **Name clash:** the new `enqueue` keyword shadows the `enqueue` function in `analysis/service.py`, so the module is now imported as `from backend.jobs import queue` and called as `queue.enqueue`.
7. **`GET /api/analyses`** lists the caller's succeeded analyses of any kind, newest first, `limit` 1..200 (default 50). Failed or queued analyses have no score, so they are not listed. `overall_score` is the mean of `1 - p_supported` over the claims (0.0 without claims), computed in one SQL aggregate.
8. **Feedback:** an unparseable `analysis_id`, an analysis the caller may not see (other user or tenant) and an unknown `claim_index` all answer 404 `claim_not_found`; a second answer for the same claim replaces the first (existing upsert).
9. **Calibrator rejection logging stays in the CLI.** `build_services` raises `CalibratorRejected(field)`; `_run_worker` and `check` log `calibrator artifact rejected: <field>` under the `backend.cli.main` logger and exit 1, so the existing worker tests needed no change.
10. **CLI details:** untrusted text (answer, claims, evidence) is printed as Rich `Text`, so `[/oops]` in model output cannot raise a markup error (`test_cli_check_prints_text_that_looks_like_markup`); `--doc` is validated by Typer (`exists=True`, exit 2 for a missing file); the file is read in the sync command because ruff `ASYNC240` forbids blocking pathlib calls in coroutines; services are built before the first write, so a rejected artifact stores nothing.
11. **`tests/test_api_startup.py`:** the old fixture patched legacy symbols that no longer exist. The "startup failure" test now fails the auth service instead of `get_embedding_model`, the repo-close test became an http-client-close test, and `test_startup_opens_no_sqlite_database` checks that no SQLite file is created.
12. **Left in place on purpose:** `_build_pipeline` in `backend/cli/main.py` (the legacy `evaluate` command, via `evaluation/benchmarks.py`, imports it) and the legacy core modules and their unit tests (Task 19).

## Concerns

- The legacy endpoints now require a signed-in session (CSRF applies to cookie sessions) and return string ids. `frontend/src/api/client.ts` and the quick-check page still use the old unauthenticated, integer-id contract until Plan D.
- Not exercised: the real `build_services` (real models, real Ollama) and a real `halluciscope check` run. Tests replace the loaders and the LLM by design; the worker harness tests cover the `build_services` wiring with fakes.
- A legacy client that pastes a whitespace-only document gets a 500 `no_text` (the brief's status table has no 4xx for ingest failures).
- Two requests racing past the active-analysis limit can leave one hidden quick collection with an unprocessed upload (harmless, never listed).
- `AGENTS.md`, `backend/AGENTS.md` and the README still describe the old SQLite/legacy layout (not part of this task).
