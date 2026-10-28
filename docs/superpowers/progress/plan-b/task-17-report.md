# Plan B Task 17: POST /api/v1/search and the `search` job

Status: DONE (one decision to flag, see "Decisions" 1 and 2)

Commit: `ca8eb87 feat: add search endpoint with worker-side reranking` on `feature/evidencegraph` (not pushed). Gitleaks pre-commit hook ran clean; no hook was skipped.

## What was built

- `POST /api/v1/search` (`backend/api/routes/search.py`), body `SearchIn`, response `SearchOut`.
  - Scope: one `tenant_tx` runs `readable_collection_ids(conn, principal, collection_ids)` (kind `normal`); when `as_of` is given the same transaction also reads `select now()` from Postgres and that becomes `known_at`. Live search sends no `as_of`/`known_at`.
  - Without `rerank`: `PostgresRetriever(db, app.state.embedder, None).search(..., rerank=False)` in the API process. No job, no `search_requests` row.
  - With `rerank`: one `tenant_tx` inserts a `search_requests` row (`query`, `params` = `{k, mode, collection_ids, as_of, known_at}`) and enqueues `search` with payload exactly `{"search_request_id": "<uuid>"}`, priority 10, and the request's `request_id`. The route then polls the row every 100 ms, each poll in its own short transaction, up to `settings.search_wait_s`. `done` -> read `hits`, delete the row, respond 200. `failed` -> delete row, 502 `search_failed`. Deadline -> delete row, 504 `search_timeout`.
- `backend/retrieval/handlers.py`: `handle_search(ctx)` and the failure hook `fail_search(ctx, error_code)`.
  - Handler loads the row in `tenant_tx`; a missing or non-`queued` row returns. It rebuilds the `SearchScope` from `params`, runs `PostgresRetriever(services.db, services.embedder, services.reranker).search(..., rerank=True)`, stores `hits` (list of `PassageHitOut` JSON) with status `done` and `completed_at`, then deletes this tenant's `search_requests` rows older than 1 hour (RLS confines the delete). Any `Exception` marks the row `failed` / `search_failed`, logs a constant message (class and frames only), and raises `PermanentJobError("search_failed")`.
  - Hook sets the row `failed` with the given code only while it is `queued`; idempotent; wraps everything in `try/except Exception` and logs a constant message.
- `backend/api/schemas_v1.py`: `SearchQuery` (strip, 1..2000, surrogate and NUL checks), `SearchIn`, `PassageHitOut` (`from_attributes`, mirrors `PassageHit`), `SearchOut`.
- `backend/jobs/registry.py`: `"search"` added to `build_handlers` and `build_failure_handlers`.
- `backend/api/startup.py`: lifespan sets `app.state.embedder = load_embedder(settings)` right after `app.state.auth`, inside the `try` so a load failure still closes the pool.
- `backend/config.py`: `search_wait_s: float = 15.0`. `backend/api/app.py`: router included under `/api/v1`.

## Files changed

Created: `backend/api/routes/search.py`, `backend/retrieval/handlers.py`, `tests/test_search_route.py` (LF).
Modified: `backend/api/app.py`, `backend/api/schemas_v1.py`, `backend/api/startup.py`, `backend/config.py`, `backend/jobs/registry.py`, `tests/test_api_startup.py`, `tests/test_worker.py` (CRLF preserved; `git ls-files --eol` shows `i/lf w/crlf`).

## Tests

`tests/test_search_route.py`, 24 test functions (38 cases after parametrization; 42 with `tests/test_api_startup.py`), all `pg`:

Brief's eight: `test_bm25_search_returns_hits_with_text`, `test_hybrid_without_rerank_runs_in_api` (no worker, asserts no job and no request row), `test_rerank_through_worker` (background `drain` every 50 ms; 200, `reranked` true, every hit has `scores["rerank"]`, top rerank 3.0, request row gone, job succeeded), `test_rerank_timeout_504` (`search_wait_s=0.2`), `test_job_payload_has_only_id` (keys exactly `{"search_request_id"}`, priority 10, query token appears in no `jobs` row), `test_unreadable_collection_404` (rerank false and true; hidden and nonexistent id, no job created), `test_query_validation` (10 parametrized bodies: empty, blank, 2001 chars, NUL, lone surrogate, k 0 and 51, bad mode, bad collection id, naive as_of), `test_as_of_search_returns_old_version` (old, current, before-first-version).

Added: `test_default_scope_is_the_readable_collections`, `test_query_is_stripped_before_the_length_check`, `test_search_requires_a_signed_in_caller` (401), `test_rerank_request_row_holds_query_and_params` (reads the queued row: query, user, exact params), `test_rerank_honours_as_of` (params round trip through the worker), `test_queued_rerank_ignores_versions_created_after_the_request` (known_at is pinned at request time), `test_timed_out_request_leaves_nothing_for_the_worker`, `test_route_reports_502_when_the_job_fails`, `test_job_stores_hits_and_marks_request_done`, `test_job_failure_fails_the_request_without_retry` (job `failed`, attempts 1, `last_error` `search_failed`; neither `SECRET-TEXT` nor the query in logs), `test_job_skips_a_request_that_is_not_queued`, `test_job_deletes_old_requests_of_its_own_tenant_only`, and the hook tests: `test_failure_hook_fails_a_queued_request`, `..._leaves_a_terminal_request_alone[done|failed]`, `..._is_idempotent`, `..._never_raises[no_request_id|no_tenant|database_down|bad_request_id]` (no `SECRET-TEXT` in logs).

`tests/test_api_startup.py`: the `patched` fixture now also patches `startup.load_embedder` (no lifespan test can load a real model); new `test_startup_loads_the_search_embedder` and `test_embedder_load_failure_closes_pool`.
`tests/test_worker.py`: registry test now expects `{"ingest_upload", "search"}` for handlers and hooks; the import-path test also imports `backend.retrieval.handlers`, `backend.api.app`, `backend.api.routes.search` and still asserts torch, sentence_transformers and transformers stay unloaded.

### RED

Command (WSL wrapper, 300 s):
`pytest tests/test_search_route.py tests/test_api_startup.py tests/test_worker.py::test_registries_register_the_ingest_and_search_jobs_with_their_failure_hooks tests/test_worker.py::test_worker_import_paths_do_not_load_ml_libraries -q`

Result: `36 failed, 7 errors in 19.05s` (nothing hung; every dependency was absent so the run failed fast). Reasons, from `--tb=line`:

- route missing: `AssertionError: {"type":"/errors/not_found",...,"code":"not_found"}`, `assert 404 == 422` (all ten validation cases), `assert 404 == 401`, `assert 'not_found' == 'collection_not_found'`, `KeyError: 'hits'`, `assert 404 == 502`
- job not registered: `KeyError: 'search'` (hook tests), `assert 0 == 1` (drain ran no job)
- settings field missing: `AttributeError: Settings(...) has no attribute 'search_wait_s'` (timeout, payload, timed-out tests)
- startup: `AttributeError: <module 'backend.api.startup'> has no attribute 'load_embedder'` (all four startup tests via the fixture)
- `test_registries_...`: handler set was `{"ingest_upload"}`; `test_worker_import_paths_...`: `CalledProcessError` because `backend.retrieval.handlers` did not exist.

### GREEN

Same command after implementing: `43 passed in 15.34s`. After adding `test_queued_rerank_ignores_versions_created_after_the_request` and formatting: `tests/test_search_route.py tests/test_api_startup.py` run four times in a row: `42 passed` each time (14 to 15 s), no flakes.

### Mutation checks (temporary edits, restored and verified byte-identical)

| Mutation | Test that failed |
|---|---|
| `known_at` not pinned (`None`) | `test_queued_rerank_ignores_versions_created_after_the_request` |
| no row delete on the last look (timeout) | `test_timed_out_request_leaves_nothing_for_the_worker` |
| job priority 0 | `test_job_payload_has_only_id` (`assert 0 == 10`) |
| hook ignores terminal status | 3 hook tests (both terminal cases, idempotency) |
| handler reruns non-queued row | `test_job_skips_a_request_that_is_not_queued` |
| handler re-raises instead of `PermanentJobError` | `test_job_failure_fails_the_request_without_retry` (`queued` instead of `failed`) |
| query text added to the error log | same test (`'zebrafish' not in logs`) |
| sweep statement removed | `test_job_deletes_old_requests_of_its_own_tenant_only` |

### Full suite and lint

- `timeout 900 python -m pytest -q`: `941 passed, 2 deselected, 2 warnings in 149.08s`. The two warnings are the existing `tests/test_generator.py` un-awaited coroutine warnings; the 2 deselected come from the repo's existing pytest config.
- `ruff check .`: `All checks passed!`.
- `ruff format` applied to my two new files only. `ruff format --check` on `tests/test_worker.py` reports a pre-existing double blank line near line 1075 (not mine, left alone).
- No em dashes in any changed file; new files are LF, edited files kept CRLF.

## Decisions

1. `as_of` is `AwareDatetime`, not bare `datetime` as the brief's signature says. A naive timestamp would be read in the database session time zone; `documents.py` already uses `AwareDatetime` for `effective_from`. Naive `as_of` is a 422 (covered by `test_query_validation[naive_as_of]`). Aware values behave exactly as `datetime`.
2. The route deletes the `search_requests` row on `failed` and on timeout, not only after `done` (brief: delete after a `done` read). Reason: after a 504 the row would sit queued, a worker would later run a full rerank nobody waits for, and the query text would linger until some later search job sweeps. With the row gone the late job is a no-op (handler returns on a missing row). Covered by `test_timed_out_request_leaves_nothing_for_the_worker`. If the controller wants rows kept for the 1 hour sweep, remove the `or last_look` and the `failed` delete in `_read_request`; behavior otherwise unchanged.
3. `params["collection_ids"]` stores the readable ids resolved at request time (never `null`/"all"), because the worker has no principal and must not widen scope. `as_of` and `known_at` are ISO 8601 strings, parsed with `datetime.fromisoformat` in the handler.
4. `known_at` comes from the database (`select now()` in the scope transaction), not the Python clock, so it compares against `created_at` (stamped by Postgres `now()`) with no clock skew. Verified by the pinning test above.
5. `PassageHitOut` has `from_attributes=True` so both the route and handler build it directly from `PassageHit` dataclasses; the worker stores `model_dump(mode="json")` and the route re-validates the stored JSON.
6. Only the existing legacy imports in `backend/api/startup.py` pull torch (verified: `import backend.api.startup` already loaded torch before this task via `backend.core`). The new modules do not: `backend.api.app`, `backend.api.routes.search`, `backend.retrieval.handlers` import no torch, and the extended import-path test asserts that.

## Concerns

- A client that disconnects mid-wait leaves its `search_requests` row (and a queued job) until the next successful search job in that tenant sweeps rows older than 1 hour. No cancellation cleanup was added (not in the brief).
- If no worker with a reranker is running, `rerank=true` takes the full `search_wait_s` (15 s) before a 504; by design per the brief.
- Existing job `default_timeout_s` (300 s) applies to `search`; no per-kind timeout was added.
