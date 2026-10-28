# Plan B Task 7 report: documents, versions, passages schema (migration 0003)

Status: DONE_WITH_CONCERNS (concerns are minor, see the end)
Commit: `40e2d2e feat: add versioned document schema with integrity triggers` on `feature/evidencegraph` (not pushed)

## What was built

- `backend/db/migrations/0003_documents.sql` (new): `eg_token_count(tsvector)`, tables `blobs`, `documents`, `document_versions`, `passages`, `passage_terms`, `uploads`, `search_requests`, all with the 0002 RLS pattern (enable, force, `tenant_isolation` policy with identical USING and WITH CHECK). Triggers: `document_versions_immutable`, `passages_slice_check` (BEFORE INSERT), `passages_immutable` (BEFORE UPDATE OR DELETE), `passages_fill_terms` (AFTER INSERT). Indexes from the brief (HNSW m=16 ef_construction=64 cosine, GIN on tsv, btree indexes) plus the four Plan A deferred indexes.
- `backend/db/pool.py`: `AsyncConnectionPool(configure=_configure)`; `_configure` awaits `pgvector.psycopg.register_vector_async(conn)`.
- `tests/test_schema_documents.py` (new): 13 tests (all `pg`).
- Not touched: `tests/pg_fixtures.py` (the truncate list already covers every non-migration table, as the brief said), `tests/test_rls.py` (`GLOBAL_TABLES` unchanged).

## Grants

Same mechanism as 0001/0002: no explicit grants in the migration. `alter default privileges for role eg_owner ... grant select, insert, update, delete on tables to eg_app` from 0001 applies to every table 0003 creates (migrations run as eg_owner). Verified by the tests, which run as `eg_app` through the `db` fixture.

The `passages_fill_terms` trigger function is `security invoker` (default), so it writes `passage_terms` as `eg_app` under forced RLS. The inserted rows carry `new.tenant_id`, which equals the session tenant, so the WITH CHECK passes. `test_terms_and_token_count_filled` proves it works as `eg_app`.

## Tenant integrity (binding resolution applied)

All tenant-to-tenant foreign keys include `tenant_id`; global references (`tenants`, `users`) stay single-column.

| Child | Foreign key |
|---|---|
| documents | `(tenant_id, collection_id) -> collections (tenant_id, id)` |
| documents | `(tenant_id, current_version_id) -> document_versions (tenant_id, id)` `deferrable initially deferred` (named `documents_current_version_fk`, added after both tables exist) |
| document_versions | `(tenant_id, document_id) -> documents (tenant_id, id)` |
| passages | `(tenant_id, collection_id) -> collections (tenant_id, id)` |
| passages | `(tenant_id, version_id, document_id) -> document_versions (tenant_id, id, document_id)` |
| passage_terms | `(tenant_id, passage_id) -> passages (tenant_id, id)` |
| uploads | `(tenant_id, document_id) -> documents`, `(tenant_id, blob_sha256) -> blobs (tenant_id, sha256)`, `(tenant_id, version_id) -> document_versions (tenant_id, id)` |

Referenced-table uniques added: `documents unique (tenant_id, id)`; `document_versions unique (tenant_id, id)` and `unique (tenant_id, id, document_id)`; `passages unique (tenant_id, id)` and `unique (tenant_id, id, version_id)`. `uploads.version_id` is nullable, so the composite FK is not checked while it is null (MATCH SIMPLE), as intended.

## RED evidence

Command (tests written first, no implementation):
`wsl ... wsl.sh timeout 300 python -m pytest tests/test_schema_documents.py -q --tb=line`

```
FAILED test_passage_must_match_slice            UndefinedTable: relation "documents" does not exist
FAILED test_passage_hash_must_match_text
FAILED test_unicode_slice_matches
FAILED test_terms_and_token_count_filled
FAILED test_versions_immutable
FAILED test_passage_content_immutable_but_flags_mutable
FAILED test_passage_delete_rejected
FAILED test_passage_version_document_mismatch_rejected
FAILED test_row_pointing_at_other_tenants_collection_rejected[document]
FAILED test_row_pointing_at_other_tenants_collection_rejected[passage]
FAILED test_new_tables_have_forced_rls          assert {'blobs', 'document_versions', ...} <= dict_keys([... 'collections'])
FAILED test_embedding_roundtrip
FAILED test_plan_a_deferred_indexes_exist       missing [('memberships', '(user_id)'), ...]
13 failed in 0.84s
```
All 13 failed because the feature is missing (no tables, no indexes), not because of typos.

Intermediate step (migration added, pool unchanged): `9 failed, 4 passed`. Every failure was `psycopg.ProgrammingError: cannot adapt type 'ndarray' using placeholder '%s'`, i.e. exactly what the pool `configure` change fixes. The 4 passes were the tests that need no vector parameter (versions immutable, document cross-tenant FK, forced RLS, deferred indexes).

## GREEN evidence

After the pool change, one test failed: `test_embedding_roundtrip` with `AttributeError: 'Vector' object has no attribute 'dtype'`. Root cause: pgvector 0.5.1's psycopg loader returns a `pgvector.Vector` object, not an ndarray (read `pgvector/psycopg/vector.py` in the venv). This was a wrong assumption in my test, not a defect in the schema or pool. The test now reads `["embedding"].to_numpy()` and still asserts dtype float32, shape (384,), and exact equality with the numpy vector written. Downstream tasks must call `.to_numpy()` or `.to_list()` on vector columns they read.

```
tests/test_schema_documents.py -v : 13 passed in 1.12s
```

Mutation check (cross-tenant test guards the real behavior): temporarily changed `documents` and `passages` collection FKs to single-column `(collection_id) references collections (id)`; both parametrized cases failed with `DID NOT RAISE ForeignKeyViolation`. Reverted (grep confirms both FKs are composite again, and the full suite below ran after the revert).

## Full suite and lint

- Full suite (final, after last edit): `715 passed, 2 warnings in 74.10s` (`timeout 900 python -m pytest -q`). Warnings are the existing `tests/test_generator.py` un-awaited coroutine RuntimeWarning, unrelated.
- `ruff check .`: `All checks passed!`
- `ruff format --check` on my two Python files: already formatted. (`ruff format --check .` reports 29 pre-existing files that would be reformatted; not touched.)
- Commit-time gitleaks hook: no leaks found.
- No em dashes in any changed file; new files are LF; `pool.py` stays CRLF in the working copy (0 LF-only lines).

## Decisions and deviations from the brief text

1. Tenant-qualified keys per the binding resolution (table above). The sketch's `unique (id, collection_id)` on documents is written `unique (tenant_id, id, collection_id)`; no foreign key uses it yet, it is there so a later composite reference can include `tenant_id`. The sketch's `unique (id, document_id)` and `unique (id, version_id)` became `(tenant_id, id, document_id)` and `(tenant_id, id, version_id)`.
2. `documents_current_version_fk` is `deferrable initially deferred`. Violations surface at commit; `set constraints documents_current_version_fk immediate` makes them immediate.
3. `passages_slice_check` returns the row untouched when the version is not found for that tenant, so the composite FK reports a missing or foreign version as `ForeignKeyViolation` instead of a generic RaiseException. That is what makes the version/document mismatch test raise `ForeignKeyViolation` as required.
4. `passages_immutable` raises the same message `passage content is immutable` for DELETE (the brief says only that DELETE raises). It compares every column except `is_current` and `collection_id` (and the generated `tsv`/`token_count`, which derive from `text`).
5. Test helper signature: `insert_version(db, text, *, tenant_id=None)` takes the `Database`, not a `conn`, because RLS needs a separate tenant-scoped transaction per tenant (tenant row via `system_tx`, the rest via `tenant_tx`).
6. Tests beyond the brief's eight: `test_passage_hash_must_match_text` (the sha256 clause of the trigger), `test_passage_delete_rejected`, `test_plan_a_deferred_indexes_exist`, and the requested cross-tenant test (`test_row_pointing_at_other_tenants_collection_rejected`, parametrized over a document and a passage pointing at another tenant's collection).
7. `test_new_tables_have_forced_rls` imports `GLOBAL_TABLES` and `TENANT_POLICY` from `tests/test_rls.py` and asserts the seven new tables exist, are not global, have enable+force RLS, and carry the exact `tenant_isolation` policy. The existing generic checks in `tests/test_rls.py` also cover them automatically (they iterate all non-global tables) and still pass.

## Concerns

- `register_vector_async` raises `ProgrammingError('vector type not found in the database')` when the `vector` extension is absent, so `Database.open()` now requires a migrated (or at least extension-bearing) database. Test fixtures create the extension first; production code paths must run `migrate` before opening the pool. Comment added in `pool.py`.
- The slice trigger trusts `document_versions.char_count` for the `end_char <= char_count` clause, per the brief; nothing constrains `char_count = char_length(text)`. Substring equality already bounds the slice to the real text, so this only matters if a writer stores a wrong `char_count`. Not added, to stay within the brief.
- pgvector 0.5.1 returns `Vector` objects when reading embeddings (see GREEN evidence); later tasks that read the column need `.to_numpy()`.
