# Task 17 report: worker loop, handler registry, `halluciscope worker`

## Built
- `backend/jobs/worker.py`: `PermanentJobError`, `RetryableJobError`, `JobContext`, `Handler`, `default_worker_id()`, `Worker` (`run_once`, `run`).
- `backend/jobs/registry.py`: `build_handlers(services)` returning `{}`.
- `backend/cli/main.py`: `worker` command (`_run_worker`): configure_logging, `Database`, metrics server on 127.0.0.1, SIGTERM/SIGINT wired to `stop`, `worker_concurrency` slotted Workers via `asyncio.gather`, metrics server and database closed on exit.
- `tests/test_worker.py`: 20 tests (all brief tests plus extras: context, retryable, dead on last attempt, default timeout, grace-period finish, idle wake on stop, jobs enqueued while idle, stale requeue by `run()`, log-line fields, registry empty, CLI `--help`).

## RED
`pytest tests/test_worker.py -q` -> collection error `ModuleNotFoundError: No module named 'backend.jobs.registry'` (1 error, 0.27 s).

## GREEN
- `tests/test_worker.py`: `20 passed in 2.11s`.
- Full suite (`perl alarm 600`): `358 passed, 2 warnings` (338 prior + 20). The 2 warnings are pre-existing `RuntimeWarning` in `tests/test_generator.py` (unawaited AsyncMock).
- `ruff check .`: All checks passed.
- `PYTHONPATH=. .venv/bin/halluciscope worker --help` exits 0 and `CliRunner` test passes. Note: the bare `.venv/bin/halluciscope` script fails with `ModuleNotFoundError: backend` when `PYTHONPATH` is unset (the package is not installed into the venv; pre-existing, affects every command).
- Manual smoke (not a test, bounded by perl alarm, throwaway migrated DB, dropped afterwards): `halluciscope worker` with concurrency 2 registered two `workers` rows `<host>:<pid>:<uuid6>:0/1`, exited 0 in 0.5 s on SIGTERM, rows deleted.

## Decisions
- `Worker.run_once()` uses no stop event; `run()` uses a private `_process(job, stop)` that races handler vs stop, waits `shutdown_grace_s`, cancels, awaits, then `release`.
- `workers` row refresh and `requeue_stale` (first call immediately, then every 30 s) run as background tasks for the life of `run()`, so the row keeps refreshing while a job runs.
- Log messages are constant; extras limited to `worker_id`, `kind`, `attempt`, `outcome`; `job_id` bound as `str`. Handler failure uses `logger.exception("job handler error")`. The test captures logs through the production `JsonFormatter` (caplog's default formatter would print the exception message).
- If `complete`/`fail` returns "lease not held", log a warning `job lease lost` and do not count the metric (no outcome fits).
- `run()` loop guards each iteration with `except Exception: logger.exception("worker loop error")` plus an idle wait, so a transient DB error does not kill the worker.
- CLI sets the pool `max_size=max(10, 4 * worker_concurrency)` (each slot can hold a handler, heartbeat and housekeeping connection) so job heartbeats are not starved.

## Concerns
- Error codes passed to the two exception types are not validated; a code outside `[a-z0-9_]{1,64}` makes `queue.fail` raise `ValueError`, which the loop guard logs and the job then recovers through stale requeue. Plan B handlers must use valid codes.
- A heartbeat that finds the lease lost only logs and stops; it does not cancel the handler.


## Fix round 1

### Changes
- `backend/jobs/queue.py`: public `is_valid_error_code(code) -> bool` (also False for non-str); `fail` uses it.
- `backend/jobs/worker.py`:
  - Handler-supplied codes are normalized centrally (`_normalize_code`): invalid becomes `internal_error`, permanent/retryable kept. Constructors stay permissive.
  - A handler that raises `CancelledError` while not shutting down is an `internal_error` retry (backoff), not a release, so no hot loop.
  - After the grace period the cancelled handler is awaited for another `shutdown_grace_s`; if it still runs, log `job handler did not stop` and return without release (stale requeue recovers it). `_process` no longer awaits the stuck task.
  - Job timeout vs inner `TimeoutError` distinguished by `asyncio.timeout(...).expired()`; inner one is `internal_error` with `logger.exception`.
  - `kind`/`attempt`/`worker_id` on `job handler error`, `job lease lost` (finish, release and heartbeat paths) and heartbeat error lines; `error_code` on failure outcome lines; `job released` logged only when `release` returned True (otherwise `job lease lost`).
  - `request_id=job.request_id` bound together with `job_id` for the whole job.
- `backend/cli/main.py`: `worker` exits 2 with a message if `worker_concurrency < 1`. `_run_worker`: `db.open`, metrics server start and signal handlers are inside `try/finally`; slots run as tasks, on first slot exception `stop` is set and all other slots are awaited before the metrics server and pool are closed, then the first error is re-raised.

### Tests added (tests/test_worker.py, tests/test_jobs_queue.py)
`is_valid_error_code` accept/reject (12 params); invalid code on `PermanentJobError` (4 params, status failed, `internal_error`, not running, JOBS failed +1); invalid code on `RetryableJobError`; `run()` leaves no running job after an invalid code; self-cancelling handler -> queued, attempts 1, no "job released"; `run()` does not hot loop on it (1 call); handler that swallows cancellation is abandoned, not released (run returns, job still `running`, log line present); inner `TimeoutError` -> `internal_error`; log fields on handler error, lease lost (finish and heartbeat), `error_code` on outcome; "released" not logged when lease lost; run survives a raising `claim`; `request_id` bound and not leaked to the next job; CLI concurrency < 1 exits 2 without starting; `_run_worker` waits for sibling slots before closing; `_run_worker` closes the database when the metrics server fails. All worker-loop tests are bounded by `asyncio.wait_for`.

### Commands
- RED: `perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_worker.py tests/test_jobs_queue.py -q --no-header -p no:cacheprovider`
- GREEN: same command.
- Full: `perl -e 'alarm 600; exec @ARGV' .venv/bin/python -m pytest -q --no-header -p no:cacheprovider`
- Lint: `perl -e 'alarm 120; exec @ARGV' .venv/bin/ruff check .`

### RED (before implementation)
`30 failed, 62 passed in 51.75s`. Failure reasons: `AttributeError: module 'backend.jobs.queue' has no attribute 'is_valid_error_code'` (12); `ValueError: error_code must match [a-z0-9_]{1,64}` raised through `queue.fail` (5, run_once raises; run() loop logs "worker loop error" and the job stays running: "condition not met in time"); `assert 408 == 1` (self-cancelling handler re-run 408 times in 0.4 s); abandoned handler: `wait_for` timed out (run() hung awaiting the stuck task); `'job_timeout' == 'internal_error'`; `KeyError: 'request_id'` / `[None, None] == ['req-1', None]`; missing `kind`/`error_code` on log lines; `'job released'` logged though release returned False; CLI `0 == 2` exit code; `_run_worker`: `'slot1.stopped' not in [...]` and `['db.open'] != ['db.open', 'db.close']`. After adding the heartbeat test: `KeyError: 'kind'`.
Already-passing characterization tests (no RED needed, behavior existed): `test_run_survives_a_failing_queue_call`, finish-path `test_lease_lost_log_line_carries_job_context`.

### GREEN
- `tests/test_worker.py tests/test_jobs_queue.py`: `93 passed in 6.38s` (after the final lint-only edit; `93 passed in 4.95s` before it).
- Full suite: `391 passed, 2 warnings in 81.01s` (the 2 pre-existing `RuntimeWarning`s in `tests/test_generator.py`). Run before two formatting-only edits (one docstring, one parametrize line wrap), after which the two focused files were re-run green.
- `ruff check .`: `All checks passed!`

### Decisions / notes
- Abandoned handler: the job stays `running` until `requeue_stale`; the heartbeat task is cancelled, so the lease goes stale after `stale_after_s`.
- A handler that swallows the shutdown cancel and completes normally within the second grace window is still treated as cancelled-by-shutdown and released (same as before).
- Remaining concern: an abandoned handler task keeps running in the process until exit.
