# Task 5 report

Status: DONE. Commit e76ec7f `feat: add migration runner and local Postgres setup`.

## Built
- `scripts/dev-postgres.sh` (`start`/`stop`): reuses `.local/pgdata` (initdb only if missing), starts via `pg_ctl` only when `pg_ctl status` says not running, waits with `pg_isready`, creates roles `eg_owner`/`eg_app` (LOGIN NOSUPERUSER NOBYPASSRLS) and DB `evidencegraph` only if missing, creates `vector` and `citext` as postgres, prints both URLs. Run twice in a row: second run succeeds.
- `backend/db/migrate.py`: `MIGRATIONS_DIR`, `Migration`, `MigrationError`, `MigrationChecksumError`, `discover`, `migrate`, `pending` per brief. Checksums of all applied migrations verified before applying; each pending file runs in its own transaction after `pg_advisory_xact_lock(727274)`.
- `Settings`: 10 new fields with the brief's defaults.
- `halluciscope migrate` CLI command (`asyncio.run`, prints `applied: [..]` or `up to date`).
- `tests/pg_fixtures.py` (ADMIN_URL, role_url, ensure_roles, create_database, drop_database, fresh_db_url), re-exported in `tests/conftest.py`.
- `backend/db/migrations/.gitkeep`.
- `pyproject.toml`: `asyncio_default_fixture_loop_scope` and `asyncio_default_test_loop_scope` = "session" (R-A1).

## RED
`.venv/bin/python -m pytest tests/test_migrate.py -q` -> `ModuleNotFoundError: No module named 'backend.db.migrate'`, collection error.

## GREEN
- `tests/test_migrate.py`: 6 passed.
- Full suite: 59 passed, 2 warnings (pre-existing un-awaited AsyncMock warning in test_generator).
- `ruff check .`: All checks passed.
- Script verification: `psql -U eg_owner -d evidencegraph -Atc "select extname from pg_extension order by 1"` -> `citext`, `plpgsql`, `vector`.
- `halluciscope migrate` against evidencegraph DB -> `up to date`; `--help` lists `migrate`.
- Unreachable server (`HALLUCISCOPE_TEST_ADMIN_URL=...:1`) -> `Failed: Postgres not reachable at postgresql://postgres@localhost:1/postgres. Run scripts/dev-postgres.sh start` (fails, never skips).
- No leftover `eg_tmp_*` databases after the run.

## Decisions
- `pyproject.toml` committed though the brief's commit list omits it; the brief's own R-A1 requirement modifies it.
- `conftest.py` star import placed at top with `# noqa: F403` plus reason comment; `pg_fixtures.__all__` limits exported names (ruff RUF022-clean).
- `migrate` re-reads applied versions inside the advisory-locked transaction so concurrent runners do not double-apply.
- Migration SQL executed with `conn.execute(text)` without params (simple query protocol), so multi-statement files work.
- Script sets `PGOPTIONS=-c client_min_messages=warning` for admin psql to suppress "already exists" notices.
- `.venv/bin/halluciscope` entry point needs `PYTHONPATH=.` (package not installed editable here); pre-existing, not changed.

## Concerns
None. Cluster left running on port 54329.


## Fix round 1

Findings addressed: 1 (concurrent bootstrap race), 2 (fixture db leak), 3 (unquoted socket dir).

### Changes
- `backend/db/migrate.py`: new `_bootstrap(conn)` runs `pg_advisory_xact_lock(727274)`, then `create table if not exists schema_migrations`, then reads applied rows, all inside one `conn.transaction()`. `migrate()` and `pending()` both use it (no more unlocked create/verify).
- `tests/pg_fixtures.py`: `create_database(name)` moved inside the `try` of `fresh_db_url`, so `drop_database` (`drop ... if exists`) always runs.
- `scripts/dev-postgres.sh`: `-o "-p $PGPORT -k '$ROOT/.local'"` (socket dir quoted for the `/bin/sh -c` that pg_ctl uses).
- `tests/test_migrate.py`: added `test_concurrent_migrate_on_fresh_database` (pg). Uses `fresh_db_url` plus 4 extra `eg_tmp_*` databases made via `create_database`/`drop_database` (dropped in `finally`); on each of the 5 databases runs two `migrate()` calls with `asyncio.gather(..., return_exceptions=True)`, asserts no exception and `sorted(results) == [[], [1, 2]]`.
- Finding 2 test skipped (optional): the fix is a one-line reorder in a pytest fixture, which is awkward to drive from inside the same run.

### RED (before fix, test added first)
`.venv/bin/python -m pytest tests/test_migrate.py -q`
```
AssertionError: [[1, 2], UniqueViolation('duplicate key value violates unique constraint "pg_type_typname_nsp_index"
  DETAIL:  Key (typname, typnamespace)=(schema_migrations, 2200) already exists.')]
FAILED tests/test_migrate.py::test_concurrent_migrate_on_fresh_database
1 failed, 6 passed in 1.22s
```

### GREEN (after fix)
- `.venv/bin/python -m pytest tests/test_migrate.py -q` run 3 times: `7 passed` each (1.31s, 1.48s, 1.05s).
- `.venv/bin/python -m pytest -q`: `60 passed, 2 warnings in 2.69s` (same pre-existing warnings).
- `.venv/bin/ruff check .`: `All checks passed!`
- `bash -n scripts/dev-postgres.sh`: syntax OK.
- `bash scripts/dev-postgres.sh start` (cluster already running): exit 0, printed both URLs.
- No leftover `eg_tmp_*` databases afterwards (psql listing empty).
- Not verified: start from a repo path containing a space (only syntax and the normal-path run were checked).
