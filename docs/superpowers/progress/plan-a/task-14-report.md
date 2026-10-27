# Task 14 report

Status: DONE

## Built
- backend/authz/policy.py: PERMISSION_RANK, effective_permission, require_collection (returns row + permission), readable_collection_ids (signatures per brief).
- backend/api/pagination.py: encode/decode_cursor, page_params, PageParams, Page[T] (Generic, because ruff target is py311), decode_uuid_cursor (non-UUID key -> 400 bad_cursor).
- backend/db/repos/collections.py, backend/api/routes/collections.py (GET/POST/GET id/PATCH/DELETE/GET grants/PUT grants), schemas in schemas_v1.py, router mounted in app.py.
- tests/test_collections.py (9 tests), client3 fixture in tests/pg_fixtures.py (+ __all__).

## RED
`pytest tests/test_collections.py -q`: ModuleNotFoundError: No module named 'backend.authz' (collection error).

## GREEN
Focused: 9 passed. Full suite: 274 passed. `ruff check .`: All checks passed.

## Decisions
- 409 detection checks constraint name collections_tenant_id_name_key; other unique violations re-raise.
- Audit details are counts only: grants_changed {grant_count}, collection_updated {fields_changed}; created/deleted have none.
- PUT grants order: admin check (404/403), duplicate check (422 validation_error), membership count (422 not_a_member, no ids listed).
- PATCH with no fields is a no-op returning the current row, no audit.
- readable_collection_ids with explicit ids returns them deduplicated in request order.
- switch-tenant returns MeOut without csrf_token, so Bob keeps his registration csrf token in tests.

## Concerns
- `ruff format` on the whole repo reformats untouched pre-existing files; I reverted that and formatted only my files.
