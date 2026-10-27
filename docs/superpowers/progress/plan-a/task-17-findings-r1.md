# Task 17 open findings (round 1), from the review, with controller rulings

1. (Important) worker.py:33,41,188: handler error codes are not validated. PermanentJobError("Bad-Input") makes queue.fail raise ValueError before any SQL; in run() the loop guard logs, the job stays running until stale requeue (about 120 to 150 s), a permanent failure is retried up to max_attempts and ends dead/lease_expired instead of failed, and JOBS never counts it; in run_once() the ValueError reaches the caller.
   Controller ruling: normalize in the worker (`code if valid else "internal_error"`, keeping permanent), using a public helper `is_valid_error_code(code: str) -> bool` exported from backend/jobs/queue.py (the same pattern queue.fail uses). Also validate in the PermanentJobError and RetryableJobError constructors? No: keep constructors permissive and normalize centrally, so a bad code never crashes the raise site. Test: a handler raising PermanentJobError("Bad-Input") ends failed with last_error internal_error after run_once, and the job is not running.

Same round (cheap, same lines, controller rulings):
- A handler that raises CancelledError while the worker is not shutting down is treated as internal_error retry (not released), so it cannot hot-loop. Release only when shutting down. Test it.
- After the grace period, wait for the cancelled handler with a bound (shutdown_grace_s again); if it still has not finished, log "job handler did not stop" and return without release (stale requeue recovers it).
- CLI: if any slot's run() raises, set stop and wait for the other slots before closing the pool and metrics server; move db.open, metrics server start, and signal handler setup inside try/finally so each opened resource is closed. worker_concurrency below 1 is a startup error (exit code 2 with a message).
- Distinguish the job timeout from TimeoutError raised inside the handler using the asyncio.timeout context manager's expired() result; an inner TimeoutError is internal_error and is logged with logger.exception.
- Add kind and attempt to the "job handler error" and "job lease lost" log lines, and error_code to outcome lines; log "released" only when queue.release returned True.
- Test: the run() loop survives a queue call that raises (monkeypatch claim to raise once) and processes the next job.
- Spec 18 requires request ids propagated into worker logs: bind request_id=job.request_id (a str or None) together with job_id for the duration of the job. Test: a job enqueued with request_id "req-1" produces worker log lines carrying request_id "req-1".
