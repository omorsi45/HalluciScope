# Task 18 report: health, readiness, metrics

Status: DONE (with one environment note). Commit: f11d939 `feat: add health, readiness, and metrics endpoints`.

## Built
- `backend/api/routes/health.py` (v1 router, no auth on health):
  - `GET /health/live` -> `{"status": "ok"}`.
  - `GET /health/ready` -> status/db/migrations/worker/llm; 200 when db and migrations ok, else 503. Each check catches `Exception`, logs `logger.exception("readiness check failed", extra={"stage": ...})`, and reports only the failure state (db "error", migrations "pending", worker "stale", llm "unavailable"). Worker freshness is computed in SQL (`max(heartbeat_at) > now() - 60s`, null -> stale). LLM probe: `app.state.http.get(f"{ollama_base_url}/api/version", timeout=2.0)`, ok on 200.
  - `GET /metrics`: 404 `not_found` if token unset/blank; 401 `unauthenticated` (with `WWW-Authenticate: Bearer`) unless full Authorization header equals `Bearer <token>` via `hmac.compare_digest` on bytes; else `REGISTRY.render()` as `text/plain; version=0.0.4` after `JOBS_QUEUED.set(count_queued)`.
- `backend/db/migrate.py`: `applied_versions(conn) -> set[int]`; `to_regclass` check first, empty set when `schema_migrations` is missing; works with both tuple and dict row factories.
- `backend/api/app.py`: health router included in v1 (legacy `GET /health` untouched).
- `backend/api/startup.py`: `app.state.http = http_client` (same single client as `http_client`, closed once in existing shutdown).
- Tests: `tests/test_health_metrics.py` (13 tests: the 7 from the brief plus old-heartbeat stale, db-failure 503 with no error text and stage log, blank-token 404, bare/missing Authorization 401, `applied_versions` with and without the table); `tests/test_api_startup.py` asserts `app.state.http is app.state.http_client`.

## RED
`pytest tests/test_health_metrics.py -q` before implementation: `8 failed, 2 passed` (the 2 passes are the two 404 tests, which pass trivially because the route did not exist; failures were 404s on live/ready/metrics and ImportError for `backend.api.routes.health`). The later-added `applied_versions` and startup assertions were added after the first RED run and verified green with the implementation present (not separately run RED).

## GREEN / gate
- `pytest tests/test_health_metrics.py tests/test_api_startup.py -q`: 13 passed.
- Full suite: `402 passed, 2 warnings in 37.73s` (391 before + 11 new).
- `ruff check .`: All checks passed! (changed files also pass `ruff format --check`; 23 other files were already unformatted before this task, not touched).
- `halluciscope migrate` on dev DB: first run `applied: [1, 2]`, second run `up to date`.

## Decisions
- Startup sets `app.state.http` as an alias of the existing `http_client`/`app.state.http_client` rather than a second client (per dispatch).
- When the db check fails, migrations/worker are reported `pending`/`stale` (cannot be verified); status 503.
- `WWW-Authenticate: Bearer` added to the 401, a standard addition.

## Concerns
- Environment: the `.pth` files in `.venv.nosync` carry the macOS `hidden` flag (iCloud workaround), so Python skips them and the editable install is not on `sys.path`. `.venv/bin/halluciscope migrate` fails with `No module named 'backend'` unless run with `PYTHONPATH=$PWD`; with it, it works as above. pytest is unaffected (rootdir on path). Not changed; fix with `chflags nohidden .venv.nosync/lib/python3.14/site-packages/*.pth`.
- Pre-push gitleaks hook ran on commit; no leaks.


## Fix round 1

Scope: findings round 1 (`task-18-findings-r1.md`), controller rulings applied as written.

### Changes
- `backend/api/routes/health.py`: the three DB reads (`select 1`, `applied_versions`/`discover`, worker freshness) now run in ONE `system_tx` under `asyncio.timeout(2.0)`, first statement `set local statement_timeout = '1500ms'`. Any failure or the deadline logs `logger.exception(..., stage="db")` once and reports db "error", migrations "pending", worker "stale" (503); no further DB calls. LLM probe wrapped in `asyncio.timeout(2.0)` (httpx `timeout=2.0` kept, per phase) and logs `logger.warning("readiness check failed", extra={"stage": "llm"})` with no traceback. `/metrics`: a failing `count_queued` logs a WARNING (`metrics queue depth unavailable`, stage `metrics_queue`) and the scrape still renders. Removed `_check_db/_check_migrations/_check_worker/_failed`.
- `backend/cli/main.py`: added `if __name__ == "__main__": app()`.
- Tests added in `tests/test_health_metrics.py`: hanging DB gives 503 within 3 s (`asyncio.wait_for`), one shared bounded transaction whose first statement mentions `statement_timeout`, dependent checks skipped on DB failure (exactly one `system_tx` call), slow-drip LLM (async handler sleeping 10 s) unavailable within 3 s (`asyncio.wait_for`), LLM 500 unavailable, LLM failure logs WARNING with no `exc_info`, metrics renders when `count_queued` fails with the warning log; WWW-Authenticate `Bearer` assertion added to the 401 test. New `tests/test_cli_module.py`: subprocess `[sys.executable, "-m", "backend.cli.main", "--help"]`, cwd repo root, exit 0 and "migrate" in stdout.

### RED
`perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_health_metrics.py tests/test_cli_module.py -q --no-header` before implementation: `7 failed, 12 passed`.
- hanging DB: `TimeoutError` from the `wait_for(…, 3.0)` bound (fails fast, no hang)
- shared transaction: `assert 3 == 1` (three transactions)
- dependent checks skipped: `assert 3 == 1` (three `system_tx` calls)
- slow LLM: `TimeoutError` from the 3 s bound
- LLM warning: `assert 40 == 30` (logged at ERROR)
- metrics queue failure: `assert 500 == 200`
- CLI module: `assert 'migrate' in ''` (exit 0, empty stdout, since nothing invoked `app()`)
Already passing before the change (they lock existing behavior): LLM 500 unavailable, WWW-Authenticate header on the 401.

### GREEN / gate
- `perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_health_metrics.py tests/test_api_startup.py tests/test_cli_module.py -q --no-header`: `21 passed in 12.88s`.
- `perl -e 'alarm 600; exec @ARGV' .venv/bin/python -m pytest -q --no-header`: `410 passed, 2 warnings in 41.59s`.
- `perl -e 'alarm 120; exec @ARGV' .venv/bin/ruff check .`: `All checks passed!`
- `perl -e 'alarm 120; exec @ARGV' .venv/bin/python -m backend.cli.main migrate`: `up to date`.
- `ruff format --check` on the changed files: health.py and both test files formatted; `backend/cli/main.py` was already unformatted before this round (the untouched `evaluate` command), left alone.

### Decisions
- On DB failure the migrations/worker fields are "pending"/"stale" (unverifiable), as ruled.
- The `statement_timeout` value is a module constant interpolated into the SQL, not user input.
