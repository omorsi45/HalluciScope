# Task 16 report: Postgres job queue

Status: DONE. Commit: e44f8a4 `feat: add Postgres job queue with leases and retries`.

## Built
`backend/jobs/queue.py` with `Job`, `enqueue`, `claim`, `heartbeat`, `complete`, `fail`, `release`, `requeue_stale`, `count_queued`. All take the caller's connection and never open a transaction. No migration changes.

Files: `backend/jobs/__init__.py` (empty), `backend/jobs/queue.py`, `tests/test_jobs_queue.py` (40 test cases, all `pg`).

## RED
`perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_jobs_queue.py -q`
```
ERROR collecting tests/test_jobs_queue.py
E   ModuleNotFoundError: No module named 'backend.jobs'
1 error in 0.12s
```

## GREEN
Focused: `40 passed in 1.34s` (also clean with `-W error`).
Full suite: `338 passed, 2 warnings in 33.22s` (298 before + 40 new). The 2 warnings are pre-existing (`tests/test_generator.py`, un-awaited AsyncMock coroutine), not from this task.
Lint: `.venv/bin/ruff check .` -> `All checks passed!`. No em dashes in new files. gitleaks pre-commit hook: no leaks.

## Decisions
- Enqueue: `insert ... on conflict (idempotency_key) do nothing returning id`, then select by key when nothing returned. `run_after` via `coalesce(%s::timestamptz, now())`. Payload via `Jsonb`.
- Payload validation (R4) runs before any SQL. Errors name the key only; non-str keys give a message with no key repr. Flat lists only (nested list/dict rejected), max 1,000 items, str max 100 chars, floats must be finite. bool/int/None allowed.
- `error_code` validated with `re.fullmatch` (not `$`, which would accept a trailing newline); a test covers `"ok\n"`. Validation happens before the DB, so an invalid code leaves the job running.
- `fail`: single UPDATE with CASE; backoff `now() + make_interval(secs => 5 * power(4, attempts - 1))` in SQL. Clears `locked_by/locked_at/heartbeat_at` on every outcome. Returns new status or None when lease not held.
- `requeue_stale`: counts both requeued and dead rows; requeued jobs set `run_after = now()` and keep `last_error`; dead ones get `lease_expired`. Attempt is not decremented.
- `release`: attempts - 1, status queued, lock cleared, `run_after` untouched.
- `count_queued`: counts every `status = 'queued'` row including ones delayed by `run_after` (the brief does not specify; queue depth reading).
- `claim` uses `kind = any(%s::text[])` so an empty kinds list returns None cleanly.
- Beyond the brief's list, added tests: enqueue inside `tenant_tx`, stored options, concurrent same-key enqueue (5 connections), payload acceptance boundaries, wrong-worker complete/fail/release, mutations on non-running jobs, fresh leases untouched by `requeue_stale`, `run_after` delay on enqueue. A `pool5` fixture (5-connection Database, depends on `db` so tables are emptied afterwards) serves the concurrency tests.
- Tests compare `run_after` to the DB's `now()` in SQL, never Python time.

## Concerns
- `test_concurrent_claims_never_duplicate` would still pass if SKIP LOCKED were dropped (blocked claimers return fewer rows but the loop continues), so it guards against duplicate claims, not lock contention. Matches the brief's Review Focus 4 shape.
