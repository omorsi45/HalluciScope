# Plan B Task 9 report: version creation with dedupe and incremental embeddings

Status: DONE
Commit: `0fd6ef1 feat: add version creation with dedupe and embedding reuse` (branch `feature/evidencegraph`, not pushed)

## What I built

- `backend/ingest/service.py`: `VersionResult` and `create_version`, following the brief's step order. It locks the document row `for update`, then dedupes on `sha256(text)` against the current version only (R5). Then it enforces `effective_from` ordering, runs `build_passages`, applies the passage limits, and reuses embeddings. It embeds the rest in `embed_batch_size` batches via `asyncio.to_thread`, then inserts the version and passages, demotes the previous version's passages, and sets `documents.current_version_id`.
- `MAX_PASSAGES = 5_000` is a module-level constant read at call time, so tests monkeypatch `backend.ingest.service.MAX_PASSAGES`.
- `backend/db/repos/documents.py`: `insert_blob`, `get_blob`, `insert_document`, `get_document`, `list_documents`, `list_versions`, `get_version`, `get_version_text`, `list_passages`, `soft_delete_document`, `count_live_documents`. SQL is parameterized throughout.
- `tests/test_ingest_versions.py`: 16 pg tests (14 functions, 2 parametrized).

## Files changed (3 new, committed by name)

- backend/db/repos/documents.py
- backend/ingest/service.py
- tests/test_ingest_versions.py

## RED evidence

Command: `wsl.sh timeout 300 python -m pytest tests/test_ingest_versions.py -q -x`, run before any production file existed.

```
ERROR collecting tests/test_ingest_versions.py
tests/test_ingest_versions.py:12: in <module>
    from backend.db.repos import documents, tenants
E   ImportError: cannot import name 'documents' from 'backend.db.repos'
1 error in 0.76s        exit=2
```

It fails because the feature is missing, not because of a typo. Nothing expensive was triggered (no model, no network; the fake embedder is the only embedder).

Mutation checks after GREEN, to confirm the tests detect real defects. I reverted both mutations.

- Embedding reuse lookup returning `{}`: `test_changed_version_reuses_unchanged_passages` and `test_revert_creates_new_version` failed (`assert 5 == 0` on `n_new`).
- Skipping the demotion of the previous version's passages: `test_changed_version_reuses_unchanged_passages` failed (`assert True is False` on the old version's `is_current`).

## GREEN evidence

Focused run, `pytest tests/test_ingest_versions.py -v`:

```
test_first_version_creates_passages PASSED
test_page_map_is_stored_as_triples PASSED
test_identical_text_is_duplicate PASSED
test_changed_version_reuses_unchanged_passages PASSED
test_reused_embedding_equals_original PASSED
test_revert_creates_new_version PASSED
test_effective_from_out_of_order PASSED
test_unicode_offsets_end_to_end[precomposed] PASSED
test_unicode_offsets_end_to_end[combining] PASSED
test_too_many_passages PASSED
test_no_text[empty] PASSED
test_no_text[whitespace] PASSED
test_missing_or_deleted_document_raises_lookup_error PASSED
test_reuse_does_not_cross_tenants PASSED
test_embeddings_are_batched PASSED
test_documents_repo_round_trip PASSED
16 passed in 1.47s
```

Full suite, `timeout 900 python -m pytest -q`, run before the commit with the final code:

```
744 passed, 2 deselected, 2 warnings in 124.74s (0:02:04)    exit=0
```

The 2 warnings are the existing `RuntimeWarning` in `tests/test_generator.py` (unawaited AsyncMock), unrelated to this task. The 2 deselected are the default-deselected `models` tests.

## Lint

- `ruff check .` printed `All checks passed!` (exit 0).
- `ruff format --check` on my 3 files: `3 files already formatted`.
- `ruff format --check .` over the whole repo reports 28 files that would be reformatted. None are mine, and I did not touch them. This is pre-existing and out of scope.
- My files are LF with no CR bytes and no em dash (checked on the bytes). The commit prints the expected LF to CRLF autocrlf warning.
- The commit's secret scan found no leaks.

## Brief coverage (Review Focus 1 and 2)

- Review Focus 1: `test_identical_text_is_duplicate` asserts status `duplicate`, same `version_id`, no new version or passage rows, and no embedder call.
- Review Focus 2: `test_unicode_offsets_end_to_end` runs on the brief's literal text and on a decomposed-combining-accent variant. Two sections put the second passage after the emoji and CJK, so its `start_char` is not 0. Every passage insert passes the slice trigger. `get_version_text(version_id, start_char, end_char)` equals both the stored passage text and `text[start:end]`.
- `test_reuse_does_not_cross_tenants`: tenant B has its own tenant row and collection (`seed` helper, tenants repo plus a direct insert, like the existing pg tests). The same text embeds anew (`n_reused == 0`, the spy call count grows by 5).

## Decisions (brief silent or ambiguous)

1. Duplicate result: `previous_version_id=None`, `n_passages = n_reused = n_new = 0`. Nothing was processed, and it keeps `n_new + n_reused == n_passages`. The caller reads the current version's `version_id` and `version_number` from the result.
2. `n_new` is the number of texts actually embedded, de-duplicated by `text_sha256`. `n_reused = n_passages - n_new`. An identical passage text repeated inside one version is embedded once and counts as reused. This keeps `n_new == len(embedder texts)` exactly, as the brief's test wording requires.
3. Check order follows the brief: lock, duplicate, `effective_from` order, build, `too_many_passages`, `no_text`. An equal `effective_from` is allowed (only earlier is rejected). `effective_from` must be tz-aware, as it is for timestamptz columns.
4. `list_documents(conn, collection_ids, page)` takes `PageParams` (imported under `TYPE_CHECKING` only, so the repo layer has no runtime import of the API layer). It returns up to `limit + 1` rows, the same convention as `list_collections`. The cursor key is cast `::uuid`, so the route must validate it first with `decode_uuid_cursor`.
5. `get_document` and `list_documents` share one column set: `id, collection_id, title, filename, mime_type, current_version_id, created_by, created_at, deleted_at, current_version_number, version_count`. `version_count` is the number of versions.
6. `list_versions` returns every column except `text`, `page_map`, and `sections`. `get_version` returns all of those plus `text`, `page_map`, and `sections`.
7. `get_version_text` raises `LookupError` for a missing version id instead of returning an empty string, so a bad id cannot look like an empty slice. Route code looks up the version first.
8. `list_passages` returns exactly the boundary columns from the plan (`id, ordinal, start_char, end_char, page_start, page_end, section`), without `is_current`.
9. I added tests beyond the brief's list for behavior the brief specifies but does not name: page_map and metadata storage, reused vector equality, `no_text` (empty and whitespace), `LookupError` for a missing or soft-deleted document, batching by `embed_batch_size`, and the repo round trip. They are in the same file.
10. `INGEST_PASSAGES` is incremented at the end of `create_version`. If the caller later rolls back, the counter is not decremented (counters cannot be).

## Concerns

- Nothing blocking. `create_version` is not atomic against a caller rollback for the metric only (decision 10).
- The `previous_version_id` and zero-count shape for `duplicate` (decision 1) is my choice. Task 10's stats/handler code should read `version_id` and `version_number` from the result, not the counts, when the status is `duplicate`.
- Pre-existing, out of scope: 28 repo files fail `ruff format --check`.
