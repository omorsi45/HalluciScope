# Task 18 open findings (round 1), from the review, with controller rulings

1. (Important) Readiness has no end-to-end deadline and can hang (backend/api/routes/health.py:30-61, 64-73, 82-86). DB checks use the shared pool (30 s acquire default, no statement or connect timeout); query execution is unbounded; ready() runs all three DB checks even after the DB check fails; probes can starve the 10-connection pool; the LLM httpx timeout=2.0 is per phase, so slow-drip responses are unbounded.
   Controller ruling: run the three DB reads (select 1, applied versions, worker freshness) in one system_tx under `async with asyncio.timeout(2.0)`, with `set local statement_timeout = '1500ms'` as the first statement; skip the migrations and worker checks when the database check fails (report migrations "pending" and worker "stale" in that case); wrap the LLM probe in `asyncio.timeout(2.0)`. Tests: a system_tx that sleeps forever gives 503 within 3 s; an async MockTransport handler that sleeps 10 s gives llm "unavailable" within 3 s; an LLM 500 gives "unavailable".

Same round (cheap, controller rulings):
- The llm stage logs at WARNING without frames (logger.warning("readiness check failed", extra={"stage": "llm"})); db stages stay logger.exception.
- /metrics: if count_queued fails, log a warning with stage "metrics_queue" and still render (the gauge keeps its previous value).
- Assert the WWW-Authenticate header on the 401 test.
- CLI: add `if __name__ == "__main__": app()` to backend/cli/main.py so `.venv/bin/python -m backend.cli.main <command>` works from the worktree root (local environment ruling: the macOS hidden flag makes Python skip .pth files, so the installed entry point is unreliable on this machine). Test: subprocess `[sys.executable, "-m", "backend.cli.main", "--help"]` with cwd at the repo root exits 0 and prints "migrate".
