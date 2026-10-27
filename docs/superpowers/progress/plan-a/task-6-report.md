# Task 6 report

Status: DONE. Commit d16fc58 `feat: add global schema and tenant-scoped transactions`.

## Built
- `backend/db/migrations/0001_global.sql`: tenants, users (citext email unique), memberships, sessions, api_keys, auth_throttle, jobs (+ `jobs_claim_idx`), workers, audit_log (+ tenant/at/id index), grants to eg_app, default privileges, revoke write on schema_migrations. Spec section 8 columns and check constraints, plus the dispatch's NOT NULL/default/FK choices.
- `backend/db/pool.py`: `Database` (AsyncConnectionPool, open=False, autocommit, dict_row), `tenant_tx`, `system_tx`; each transaction's first statement is `select set_config('app.tenant_id', %s, true)`.
- `tests/pg_fixtures.py`: session `pg_urls`, function `db` (truncate as eg_owner, all public tables except schema_migrations, restart identity cascade); added to `__all__`. Existing helpers unchanged.
- `tests/test_schema_global.py`: 6 pg tests from the brief.
- `git rm backend/db/migrations/.gitkeep`. `pyproject.toml` already had the session loop scopes, so unchanged.

## RED
- First run: `ModuleNotFoundError: No module named 'backend.db.pool'` (collection error).
- After pool.py only: `fixture 'pg_urls' not found`, 6 errors.
- After fixtures, before migration: 3 failed (`relation "jobs" does not exist`, tables missing, users insert), 3 passed. The 3 passes (no write on schema_migrations, no bypassrls, GUC leak) hold trivially without the schema; they gain meaning once the grants/tables exist.

## GREEN
- `pytest tests/test_schema_global.py -q`: 6 passed.
- Full suite: 66 passed, 2 warnings (pre-existing `AsyncMock` never-awaited warning in test_generator.py). `ruff check .`: clean (one SIM300 in my test fixed first).

## Decisions
- Extensions: kept `create extension if not exists citext/vector` in the migration. Verified it does not raise for the non-superuser eg_owner when the superuser already created them (migration applied cleanly via `migrate()` as eg_owner in `pg_urls`). The comment in the SQL records that extensions and roles are created beforehand.
- `tenant_tx`/`system_tx` are plain methods returning an `AbstractAsyncContextManager[AsyncConnection]` over a shared private `_tx` asynccontextmanager.
- `Database.open()` uses `wait=True` so connection failure surfaces at open.
- Tests use the `db` fixture (so truncation runs) rather than raw URLs, except the leak test which needs `max_size=1` and builds its own Database from `pg_urls`.

## Concerns
None.
