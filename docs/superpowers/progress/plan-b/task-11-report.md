# Task 11 report: document, version, and upload routes

Status: DONE_WITH_CONCERNS (one brief expectation is unreachable with the current schema, see Decision 1)

Commit: `1d04692 feat: add document, version, and upload routes` (branch `feature/evidencegraph`, on top of `197f11f`). Not pushed.

## What was built

All routes live under `/api/v1`, cookie or API key auth, in `backend/api/routes/documents.py`:

| Route | Permission | Result |
|---|---|---|
| `POST /collections/{collection_id}/documents` (multipart `file`, optional `title`, `effective_from`) | write | 202 `UploadAcceptedOut(document_id, upload_id, status)` |
| `POST /documents/{document_id}/versions` (multipart `file`, optional `effective_from`) | write | 202 `UploadAcceptedOut` |
| `GET /uploads/{upload_id}` | read | `UploadOut` |
| `GET /documents?collection_id=&limit=&cursor=` | read | `Page[DocumentOut]`; without `collection_id` all readable `normal` collections |
| `GET /documents/{document_id}` | read | `DocumentDetailOut` (with `versions: list[VersionSummaryOut]`) |
| `DELETE /documents/{document_id}` | write | 204, soft delete, audit `document_deleted` (ids only) |
| `GET /documents/{document_id}/versions/{n}` | read | `VersionOut` |
| `GET /documents/{document_id}/versions/{n}/text?start=&end=` | read | `VersionTextOut`; default `end = min(start + 100_000, char_count)`; `start < 0`, `end > char_count`, `end < start`, or `end - start > 100_000` gives 422 `bad_range` |
| `GET /documents/{document_id}/versions/{n}/passages` | read | `list[PassageBoundaryOut]` (no passage text) |

Uploads go through `intake_upload` (dedupe, limits, job in one place); the route only reads the file bytes and passes `request.app.state.settings`. Missing, soft-deleted, unreadable, cross-tenant, or non-`normal`-kind documents and uploads are 404 `document_not_found` / `upload_not_found`; a missing version number is 404 `version_not_found`; a reader who lacks write gets 403 `insufficient_permission`.

## Files changed

- Created `backend/api/routes/documents.py` (the routes above).
- Created `tests/test_document_routes.py` (22 tests, all `pg`).
- Modified `backend/api/schemas_v1.py`: `Title` string type (stripped, 1 to 300 chars, no lone surrogates, no NUL) plus `UploadAcceptedOut`, `UploadOut`, `DocumentOut`, `VersionSummaryOut`, `DocumentDetailOut`, `VersionOut`, `VersionTextOut`, `PassageBoundaryOut`.
- Modified `backend/api/app.py`: registers the documents router in v1.
- Modified `backend/api/routes/collections.py`: DELETE maps the foreign key violation to 409 `collection_not_empty` (Decision 1).
- Modified `backend/db/repos/documents.py`: `list_versions` and `get_version` now also return `n_passages` (count of passages) and `is_current` (version id equals `documents.current_version_id`), computed in the same SQL statement as the row; new `get_upload` (upload joined to its live document and version number). Existing keys are unchanged; the Task 9 repo test still passes.

## RED evidence

Tests were written first (21 tests at that point). Command:

`wsl ... wsl.sh timeout 300 python -m pytest tests/test_document_routes.py -q --no-header -p no:cacheprovider --tb=line`

Result: `20 failed, 1 passed in 6.18s`. Every failure was the missing route, e.g.

```
tests/test_document_routes.py:68: AssertionError: {"type":"/errors/not_found","title":"Not Found","status":404,"detail":"Not Found","code":"not_found"}
    assert 404 == 202
```

The one pass is `test_empty_collection_can_still_be_deleted`, a regression guard for existing behavior.

Failures found after the routes existed (each fixed by a test-first step, see Decisions):

- `test_title_with_nul_character_is_422`: `assert (500, 'internal_error') == (422, 'validation_error')`; server log: `psycopg.DataError: PostgreSQL text fields cannot contain NUL (0x00) bytes`.
- `test_collection_with_documents_cannot_be_deleted` (brief version, second half): `assert 500 == 204`; server log: `psycopg.errors.ForeignKeyViolation: update or delete on table "collections" violates foreign key constraint "documents_tenant_id_collection_id_fkey" on table "documents"`. After I changed the test to the real behavior (Decision 1) it failed with `assert (500, 'internal_error') == (409, 'collection_not_empty')` before the fix.
- `test_routes_require_a_session_and_csrf`: first draft failed because `client3` still held Carol's session cookie (test bug, fixed with `client3.cookies.clear()`); no production change.

## GREEN evidence

- `tests/test_document_routes.py`: `22 passed in 23.51s`.
- Together with the neighbours: `tests/test_document_routes.py tests/test_collections.py tests/test_ingest_versions.py`: `50 passed in 25.51s`.
- Mutation check: removing the permission check from `_load_document` made `test_unreadable_documents_are_404[bob]`, `test_read_only_member_cannot_upload`, and `test_non_normal_collections_are_not_reachable` fail (3 failed, 19 passed); restored afterwards.
- Full suite (`timeout 900 python -m pytest -q`): `818 passed, 2 deselected, 2 warnings in 117.44s`, exit 0. The 2 warnings are the existing `tests/test_generator.py` "coroutine ... never awaited" RuntimeWarnings; the 2 deselected are the `models` marker tests.
- Lint (`timeout 120 ruff check .`): `All checks passed!`. New files are also `ruff format` clean.
- Pre-commit hook (gitleaks) ran: `no leaks found`. No em dashes in any changed file.

## Tests (tests/test_document_routes.py)

Brief tests: `test_upload_drain_and_read_back`, `test_text_range_validation`, `test_passage_boundaries_cover_text`, `test_new_version_route_and_listing`, `test_cross_tenant_404s` (as `test_unreadable_documents_are_404[carol|bob]`, covering a cross-tenant user and a same-tenant member without a grant, with response bodies identical to those for random ids so there is no existence oracle), `test_read_only_member_cannot_upload`, `test_soft_deleted_document_404`, `test_collection_with_documents_cannot_be_deleted`.

Added: text range defaults and empty slice; code point offsets with combining accents, emoji, CJK; duplicate re-upload gives status `duplicate` and queues nothing; `effective_from` (naive, garbage, future, valid); 415/413/missing file/long title/blank title; NUL in title; empty collection deletable; missing version numbers (0, 2, -1, huge) give 404 `version_not_found` on all three version routes; listing pagination, bad cursors, permission filtering; `quick` collection documents unreachable; 401 without a session and 403 `csrf_failed` without a token; API key upload and read; no `SECRET` filename or text in logs.

## Decisions

1. Collection delete (the concern). The brief says DELETE of a collection with live documents is 409 `collection_not_empty` and "succeeds after the document is deleted". The second half cannot hold: `documents` has `foreign key (tenant_id, collection_id) references collections (tenant_id, id)` with no cascade, and a soft-deleted document keeps its row, versions, and passages (spec section 8: "versions and passages stay so past analyses keep resolvable provenance", hard purge out of scope; the immutability triggers reject deleting versions and passages, so even a cascade could not purge them). So deleting a collection that ever held a document fails with a foreign key violation (a 500 before my change). I map that violation to 409 `collection_not_empty` (detail: "A collection that holds documents, including deleted ones whose history is kept, cannot be deleted"), which covers live documents and soft-deleted history. The test asserts 409 while the document is live and 409 again after it is soft-deleted, and that an empty collection still returns 204. I did not touch migrations or immutability triggers. This needs a controller ruling if "delete a collection that held documents" must work: options are archiving/soft-deleting collections, or a purge design. As built, such a collection can never be removed.
2. `n_passages` and `is_current` come from SQL in the repo (same statement as the version row), per the notes, instead of being computed in Python from separate reads.
3. Upload settings come from `request.app.state.settings` (set by `create_app`; the production app factory does not set `app.state.services`). I did not set `api_app.state.services` in tests because nothing in these routes reads it.
4. `title` is `Title` (stripped, 1 to 300 chars): longer than 300 is 422 `validation_error` at the route, although `intake_upload` itself truncates silently for direct callers. An empty form value is treated as absent by FastAPI, so the filename stem is used. NUL is rejected because Postgres text cannot hold it (RED above).
5. `effective_from` is `AwareDatetime`: a naive value would have hit `TypeError` inside `intake_upload` (comparison with an aware `now`) and become a 500; it is now 422 `validation_error`. Future values still give 422 `effective_from_in_future` from intake.
6. `GET /uploads/{id}` for an upload whose document was soft-deleted is 404 `upload_not_found` (brief: deleted documents and uploads are 404). The repo query filters on `d.deleted_at is null`.
7. Text and passages routes find the version through `list_versions` (no text column), so a text slice never loads the whole text; `GET .../versions/{n}` uses `get_version` (which does read the text column and discards it, bounded by the 2,000,000 character limit).
8. Response models: `UploadAcceptedOut` (with `status` of `queued` or `duplicate`) is used for both upload routes as the brief says; the spec's endpoint list shows `{document_id, upload_id}` and `{upload_id}`, the brief's richer shape is a superset.

## Concerns

- Decision 1 (collection that held documents can never be deleted; 409 instead of the brief's 204 after soft delete).
- Flaky observation, not reproduced: in the first full-file run, `test_listing_paginates_and_filters_by_permission` once failed because the returned order was the reverse of upload order. Diagnosis from the assertion output: the three documents came back in ascending creation order, which fits equal or inverted `created_at` values (a clock step or coarse clock in the Docker/WSL VM, since `created_at` is `now()` and ties break by `id`). It passed 19 of 19 later runs (6 isolated, 8 in subsets, 5 full file). I could not confirm the cause. I changed the test to compare against the database's own `order by created_at desc, id desc` instead of assuming creation order, so it checks pagination (no gaps or duplicates, right order contract) without depending on the clock. If this shows up elsewhere, other tests that assume `created_at` ordering across separate transactions may be affected.
- Pre-existing, not changed: the other shared request strings (`Name`, `Description`, API key names, and so on) do not reject NUL, so a JSON `"\u0000"` in a collection name or description still reaches Postgres and returns 500. Only `Title` has the NUL check. A one-line fix in the shared validator would cover all of them.
- `GET /documents/{id}/versions/{n}` transfers the version text column from Postgres only to drop it (Decision 7); fine at current limits, but a dedicated repo query would avoid it if that matters later.
- The working tree was clean at `197f11f` when I started and is clean after the commit; the report file is not committed.
