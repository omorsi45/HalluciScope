# Task 7 report: collections with forced row-level security

Status: DONE

## Built
- `backend/db/migrations/0002_collections.sql`: `collections` and `collection_grants` exactly as the brief specifies, each with `enable` + `force row level security` and a `tenant_isolation` policy (using + with check on `nullif(current_setting('app.tenant_id', true), '')::uuid`).
- `tests/test_rls.py`: `GLOBAL_TABLES` as listed in the brief and four `pg` tests:
  - `test_collection_visible_only_in_own_tenant`
  - `test_no_tenant_context_sees_no_rows`
  - `test_insert_for_other_tenant_rejected`
  - `test_every_tenant_table_has_forced_rls`

## RED
`.venv/bin/python -m pytest tests/test_rls.py -q` before the migration: `4 failed`.
- three tests: `psycopg.errors.UndefinedTable: relation "collections" does not exist`
- forced-RLS test: `assert {'collection_grants', 'collections'} <= dict_keys([])`

## GREEN
- `tests/test_rls.py`: `4 passed in 0.65s`
- full suite: `70 passed, 2 warnings`
- `.venv/bin/ruff check .`: `All checks passed!`
- no em dashes in the new files.

## Decisions
- The forced-RLS test asserts `{"collections", "collection_grants"} <= checked tables` before the per-table check, so it cannot pass vacuously (RED output shows it fails with empty set).
- Insert-for-other-tenant raises `InsufficientPrivilege` (SQLSTATE 42501), confirmed by the test passing. eg_app has full DML on the table, so the only source of 42501 is the RLS WITH CHECK violation.
- Tests create tenants via `system_tx()` (tenants has no RLS). `tests/test_schema_global.py` has its own `GLOBAL_TABLES` copy from Task 6; left untouched (scope).
- No grants added to the migration: 0001's default privileges for eg_owner already give eg_app DML on new tables.

## Concerns
- Two pre-existing `RuntimeWarning`s from `tests/test_generator.py` (un-awaited AsyncMock); unrelated to this task.
- Duplicate `GLOBAL_TABLES` definitions (test_schema_global.py and test_rls.py) could drift; the brief mandates the definition in test_rls.py.


## Fix round 1

Finding: `collection_grants` was not bound to its tenant (cross-tenant collection FK accepted, existence oracle, cascade across tenants). Implemented the controller ruling.

### Changes
- `backend/db/migrations/0002_collections.sql` (edited in place):
  - `collections`: added `unique (tenant_id, id)`.
  - `collection_grants`: dropped the single-column FKs on `collection_id` and `user_id`; added `foreign key (tenant_id, collection_id) references collections (tenant_id, id) on delete cascade` and `foreign key (tenant_id, user_id) references memberships (tenant_id, user_id) on delete cascade`. Primary key, RLS and policy unchanged.
- `tests/test_rls.py`: three new `pg` tests plus helpers `_create_member`, `_create_collection`, `_INSERT_GRANT`:
  - `test_grant_for_other_tenants_collection_rejected` (ForeignKeyViolation)
  - `test_grant_for_non_member_rejected` (ForeignKeyViolation; user is a member of tenant B only)
  - `test_removing_membership_removes_grants` (grant count 0 after deleting the membership)

### RED (tests written first, migration untouched)
`.venv/bin/python -m pytest tests/test_rls.py -q` -> `3 failed, 4 passed`
- `Failed: DID NOT RAISE ForeignKeyViolation` (cross-tenant collection)
- `Failed: DID NOT RAISE ForeignKeyViolation` (non-member)
- `assert 1 == 0` (grant survived membership deletion)

### GREEN
- `.venv/bin/python -m pytest tests/test_rls.py -q` -> `7 passed in 0.37s`
- `.venv/bin/python -m pytest -q` -> `73 passed, 2 warnings` (same 2 pre-existing test_generator warnings)
- `.venv/bin/ruff check .` -> `All checks passed!`
- No em dashes in the edited files.

### Dev database check
`psql -h localhost -p 54329 -U eg_owner -d evidencegraph -Atc "select version from schema_migrations"` returned no rows (`select count(*) from schema_migrations` = 0; `to_regclass('public.collections')` is null). 0002 was never applied there, so no drop/recreate was needed. Postgres left running.
