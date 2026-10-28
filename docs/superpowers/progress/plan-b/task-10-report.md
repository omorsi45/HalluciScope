# Plan B, Task 10 report: upload intake, ingest_upload job, failure hook, worker services

Status: DONE_WITH_CONCERNS (concerns are design notes, nothing blocking)

Commit: `197f11f feat: add upload intake and ingestion job` on `feature/evidencegraph` (not pushed).

## What was built

- `backend/ingest/intake.py`: `IntakeResult` and `intake_upload(...)` with the extra
  `collection_kinds: Sequence[str] = ("normal",)` passed through to `require_collection(..., kinds=...)`.
  One `tenant_tx`: permission check, blob, document (new document path), upload row, job, audit.
  - Exactly one of `collection_id` / `document_id`, else `ValueError`.
  - Filename sanitized (control characters removed, basename on `/` and `\`, max 255 characters, extension kept).
  - `mime_for` failure -> `ApiError(415, "unsupported_type")`; size over `settings.max_upload_bytes` -> `413 too_large`;
    `effective_from` later than now + 60 s -> `422 effective_from_in_future`; default now.
  - Title defaults to the filename stem, capped at 300 characters.
  - Existing document: missing, soft-deleted, or caller has no grant (or wrong collection kind) -> `404 document_not_found`;
    read-only grant -> `403 insufficient_permission`. New document without grant -> `404 collection_not_found`.
  - Same `blob_sha256` as the current version -> upload row `duplicate` with `version_id` = current version, no job.
  - Otherwise upload `queued` plus job `ingest_upload`: payload `{"upload_id": "<uuid>"}`, `tenant_id`, priority 0,
    `idempotency_key = "upload:<uuid>"`, `request_id` from `request_id_var`.
  - Audit `document_uploaded`: target = document id, details = `{"upload_id": ...}` (ids only).
- `backend/ingest/handlers.py`:
  - `handle_ingest_upload(ctx)`: loads the upload in `tenant_tx`; returns when it is missing or not `queued`/`processing`;
    sets `processing`; loads the blob; parses with `parse_with_timeout` outside any transaction; `IngestError` -> upload
    `failed` + `error_code`, then `PermanentJobError(code)`. Then in a new `tenant_tx`: locks the upload row
    (`for update`, a second worker waits and sees a final status), `create_version`, sets `status`/`version_id`/
    `stats {"n_passages","n_reused","n_new"}`, and for `created_version` with a previous version enqueues
    `reverify_document` (payload = 3 ids, priority 5, key `reverify:<new_version_id>`), all atomically.
    `IngestError` there is handled the same way; a plain `LookupError` (missing or soft-deleted document) fails with
    `document_deleted`; any error raised by the embedder becomes `RetryableJobError("embed_failed")`.
  - `fail_ingest_upload(ctx, error_code)`: the failure hook. Sets the upload `failed` with that code only while
    `status in ('queued','processing')`, so `created_version`, `duplicate` and `failed` are untouched. Idempotent,
    catches `Exception` and logs a constant message (class and frames only), never raises.
  - Logging: context var `tenant_id` bound as `str`; one `upload ingested` info line with whitelisted fields only
    (`outcome`, `n_passages`, `n_reused`, `n_new`, `version_number`). No filenames, text or exception messages.
- `backend/jobs/registry.py`: `build_handlers` -> `{"ingest_upload": handle_ingest_upload}`;
  `build_failure_handlers` -> `{"ingest_upload": fail_ingest_upload}`.
- `backend/cli/main.py` `_run_worker`: `services = Services(settings, db, load_embedder(settings), load_reranker(settings))`
  built after `db.open()` and the signal handlers, shared by every slot. Loader imports are function-local, so
  importing the CLI, registry, handlers and intake loads no torch (tested in a subprocess).
- `tests/pg_fixtures.py`: fixture `services` (`Services(Settings(), db, HashEmbedder(), KeywordReranker())`) and
  `async def drain(db, services, kinds=None) -> int`; both exported in `__all__`.

Files changed (7): `backend/ingest/intake.py` (new), `backend/ingest/handlers.py` (new), `backend/jobs/registry.py`,
`backend/cli/main.py`, `tests/test_ingest_upload.py` (new, 49 test cases), `tests/pg_fixtures.py`, `tests/test_worker.py`.
`git diff --cached --stat`: 7 files, 1267 insertions, 9 deletions (no whole-file line-ending rewrites of the CRLF files).

## RED evidence

Baseline before any change: `pytest tests/test_worker.py tests/test_cli.py tests/test_cli_module.py` -> `54 passed`.

1. New test module (feature modules did not exist yet):

```
wsl.sh timeout 300 python -m pytest tests/test_ingest_upload.py -q -x
tests/test_ingest_upload.py:18: in <module>
    from backend.ingest import handlers, intake
E   ImportError: cannot import name 'handlers' from 'backend.ingest'
ERROR tests/test_ingest_upload.py
1 error in 0.62s        exit=2
```

2. Worker tests (updated/new, real assertion failures):

```
wsl.sh timeout 300 python -m pytest tests/test_worker.py -q
FAILED test_registries_register_the_ingest_upload_job_and_its_failure_hook
    assert set(build_handlers(None)) == {"ingest_upload"}  ->  AssertionError: assert set() == {'ingest_upload'}
FAILED test_worker_import_paths_do_not_load_ml_libraries
    CalledProcessError: ... import backend.ingest.handlers, backend.ingest.intake ... returned non-zero exit status 1
FAILED test_run_worker_builds_one_services_container_from_the_loaded_models
    assert isinstance(first, Services)  ->  assert False  where False = isinstance(None, Services)
FAILED test_run_worker_closes_the_database_when_a_model_fails_to_load
    AssertionError: Regex pattern did not match.   (the loader is never called, the run only ends via the 10 s guard)
4 failed, 50 passed in 17.47s
```

(The first draft of the last test used a bare `pytest.raises(OSError)`, which the 10 s guard's `TimeoutError`
satisfies because `TimeoutError` is an `OSError`; I tightened it with `match="model files missing"` before this run.)

A RED run never loaded a model, parsed a huge file or touched the network: the harness patches
`backend.models.loader.load_embedder` / `load_reranker`, the `test_ingest_upload.py` embedder is `HashEmbedder`, and
`drain` bounds every `run_once` at 60 s and the loop at 100 jobs.

## GREEN evidence

```
wsl.sh timeout 300 python -m pytest tests/test_ingest_upload.py tests/test_worker.py -q
1 failed, 102 passed          # test_logs_never_carry_filenames_or_text: my test setup bug (see below)
```

Cause: the third upload in that test reused text already embedded, so `_reusable_embeddings` skipped the failing
embedder and a second `upload ingested` line appeared. Fixed the test data (distinct text), not the code. Then:

```
wsl.sh timeout 300 python -m pytest tests/test_ingest_upload.py -q
49 passed in 16.91s
```

### Mutation checks (tests are not vacuous)

Each production mutation was applied to the working copy, the relevant tests run, and the file restored:

| Mutation | Result |
|---|---|
| hook updates regardless of terminal status | 4 failed (`test_failure_hook_leaves_a_terminal_upload_alone` x3, `test_failure_hook_is_idempotent`) |
| `require_collection(..., kinds=("normal",))` instead of `collection_kinds` | `test_collection_kinds_are_passed_to_the_policy` failed |
| never mark `duplicate` in intake | `test_same_bytes_duplicate_without_job` failed |
| enqueue `reverify_document` unconditionally | `test_new_version_enqueues_reverify` failed |
| embedder errors not mapped to `embed_failed` | `test_embedding_failure_retries_then_fails_the_upload` failed (`internal_error` != `embed_failed`) |

### Brief tests and where they are

`test_upload_then_drain_creates_version`, `test_upload_is_atomic_with_job`, `test_same_bytes_duplicate_without_job`,
`test_resaved_same_text_duplicate`, `test_new_version_enqueues_reverify`, `test_empty_and_image_only_fail_no_text`
(empty, whitespace-only, `make_pdf([""])`; each: upload `failed`, `no_text`, job `failed`, `attempts == 1`),
`test_unsupported_extension_415`, `test_too_large_413`, `test_effective_from_future_422`,
`test_member_without_write_404_or_403`, `test_filename_sanitized`, `test_handler_idempotent`.

Additional tests: failure hook (open upload -> failed, terminal untouched, idempotent, never raises in 4 failure modes,
job dying through exhausted retries, and a real `Worker.run` lease expiry leaving `failed/lease_expired`), embed failure
retry (`queued` + upload still `processing` until the third attempt, then `dead` + upload `failed/embed_failed`),
`document_deleted`, `effective_from_out_of_order` (rolls back, no version, no reverify job), `bad_encoding`,
`unsupported_type` (fake PDF), limit boundary (exactly the limit accepted), clock skew boundary (+30 s accepted),
exactly-one-target `ValueError`, deleted/missing document 404, `collection_kinds` pass-through (both paths),
title default/cap, job shape + `request_id`, audit ids only, `str` context var, unknown upload id ignored,
log hygiene (no filename, text or exception text in any log line).

In `tests/test_worker.py`: `_Harness` now fakes `load_embedder` / `load_reranker` and records the `Services` each
`Worker` receives; new tests assert one shared `Services` (settings, the opened db, the loaded embedder and reranker),
that a model load failure closes the database, and that the import paths load no torch/sentence-transformers/transformers.

## Full suite and lint

```
wsl.sh timeout 900 python -m pytest -q
796 passed, 2 deselected, 2 warnings in 92.27s     exit=0
```

The 2 deselected are the `models` marker tests (default `-m 'not models'`). The 2 warnings are pre-existing
(`tests/test_generator.py`, un-awaited AsyncMock coroutine); nothing else failed, nothing skipped by me.

```
wsl.sh timeout 120 ruff check .
All checks passed!     exit=0
```

## Decisions (brief or interface gaps)

1. `drain` passes `failure_handlers=build_failure_handlers(services)` to the `Worker`; the brief's snippet only passed
   handlers. Without it a test could never observe the hook the way production wires it, and requirement (a) asks for
   hook tests. Hooks run after `PermanentJobError` too, which also exercises their idempotency in the normal path.
2. The existing `test_build_handlers_is_empty_in_plan_a` asserted `{}` and is now false by design; replaced by
   `test_registries_register_the_ingest_upload_job_and_its_failure_hook`.
3. Filename over 255 characters keeps its extension (the stem is cut), so a long but valid name is not rejected as
   `unsupported_type`.
4. `LookupError -> document_deleted` is matched on the exact type, so `KeyError`/`IndexError` bugs inside
   `create_version` are not misreported as a deleted document (they retry as `internal_error`).
5. A missing upload row makes the handler return (nothing to do, job succeeds); the intake writes upload and job in one
   transaction, so this only guards against manual deletion.
6. The hook and the handler share one `_set_failed` statement guarded by `status = any(queued, processing)`.
7. Audit target is the document (`document_uploaded`), the upload id is in `details`.
8. Duplicate uploads store zeroed `stats` in the handler path (resaved same text); the intake-side duplicate keeps `stats = {}`.
9. `test_*` for the RED of the new module is an `ImportError` collection error (module missing), not a skeleton
   with `NotImplementedError`; the mutation table above is the evidence that the tests can fail on behavior.

## Concerns

- Model loading in `_run_worker` is synchronous (`load_embedder`, `load_reranker`), blocking the event loop for the
  load time (seconds, first run downloads) before any slot starts. No job is running at that point, so it is harmless
  today; wrap in `asyncio.to_thread` if startup responsiveness ever matters.
- `Worker` is built with `default_timeout_s=300` and no per-kind timeout, so `ingest_upload` of a very large document
  on a CPU embedder could end as `job_timeout` and retry. Plan B Task 11+ may want `timeouts_s={"ingest_upload": ...}`.
- `intake_upload` expects a timezone-aware `effective_from` (a naive datetime raises `TypeError` on comparison); the
  API schema in the route task should enforce that.
- Cancelling the handler mid-parse relies on `parse_with_timeout` killing its child (already in place, Task 5).
