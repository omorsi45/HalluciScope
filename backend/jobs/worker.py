"""Background worker: claims jobs, keeps their lease alive and records the outcome.

One Worker runs one job at a time (the lease is fenced by worker id), so concurrency is
several Worker instances with distinct ids. Handler errors are reduced to short error
codes; exception messages are never logged or stored because they may carry document text.
"""

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from backend.db.pool import Database
from backend.jobs import queue
from backend.jobs.queue import Job
from backend.obs.logging import bind
from backend.obs.metrics import JOB_DURATION, JOBS

logger = logging.getLogger(__name__)

REQUEUE_INTERVAL_S = 30.0

_DONE = "done"
_CANCELLED_BY_SHUTDOWN = "cancelled_by_shutdown"
_ABANDONED = "abandoned"


def _normalize_code(code: str) -> str:
    """Keep a valid handler error code; anything queue.fail would reject becomes internal_error."""
    return code if queue.is_valid_error_code(code) else "internal_error"


class PermanentJobError(Exception):
    """The job can never succeed; mark it failed without retrying."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RetryableJobError(Exception):
    """A transient failure; retry with backoff until attempts run out."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class JobContext:
    job: Job
    db: Database
    worker_id: str
    services: Any


Handler = Callable[[JobContext], Awaitable[None]]
# Called once with the final error code when a job fails permanently or runs out of attempts.
FailureHandler = Callable[[JobContext, str], Awaitable[None]]


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:6]}"


class Worker:
    def __init__(
        self,
        db: Database,
        handlers: Mapping[str, Handler],
        *,
        failure_handlers: Mapping[str, FailureHandler] | None = None,
        services: Any = None,
        worker_id: str | None = None,
        poll_interval_s: float = 1.0,
        heartbeat_s: float = 15.0,
        stale_after_s: float = 120.0,
        timeouts_s: Mapping[str, float] | None = None,
        default_timeout_s: float = 300.0,
        shutdown_grace_s: float = 30.0,
    ) -> None:
        self._db = db
        self._handlers = dict(handlers)
        self._failure_handlers = dict(failure_handlers or {})
        self._kinds = sorted(self._handlers)
        self._services = services
        self.worker_id = worker_id or default_worker_id()
        self._poll_interval_s = poll_interval_s
        self._heartbeat_s = heartbeat_s
        self._stale_after_s = stale_after_s
        self._timeouts_s = dict(timeouts_s or {})
        self._default_timeout_s = default_timeout_s
        self._shutdown_grace_s = shutdown_grace_s

    async def run_once(self) -> bool:
        """Claim and run at most one job of a registered kind; True if one ran."""
        job = await self._claim()
        if job is None:
            return False
        await self._process(job, None)
        return True

    async def run(self, stop: asyncio.Event) -> None:
        await self._register()
        background = [
            asyncio.create_task(self._refresh_worker_row()),
            asyncio.create_task(self._requeue_stale_loop()),
        ]
        try:
            while not stop.is_set():
                try:
                    job = await self._claim()
                    if job is None:
                        await self._idle(stop)
                    else:
                        await self._process(job, stop)
                except Exception:
                    logger.exception("worker loop error", extra={"worker_id": self.worker_id})
                    await self._idle(stop)
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await self._deregister()

    async def _idle(self, stop: asyncio.Event) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), self._poll_interval_s)

    async def _claim(self) -> Job | None:
        async with self._db.system_tx() as conn:
            return await queue.claim(conn, self.worker_id, self._kinds)

    async def _process(self, job: Job, stop: asyncio.Event | None) -> None:
        started = time.monotonic()
        with bind(job_id=str(job.id), request_id=job.request_id):
            handler_task = asyncio.create_task(self._invoke(job))
            heartbeat_task = asyncio.create_task(self._heartbeat_job(job))
            state = _DONE
            try:
                state = await self._await_handler(handler_task, stop)
            finally:
                heartbeat_task.cancel()
                waiting: list[asyncio.Future] = [heartbeat_task]
                if state != _ABANDONED:
                    if not handler_task.done():
                        handler_task.cancel()
                    waiting.append(handler_task)
                await asyncio.gather(*waiting, return_exceptions=True)
            if state == _ABANDONED:
                # Releasing would let another worker run the job beside the stuck handler;
                # the lease goes stale and requeue_stale recovers it.
                logger.error("job handler did not stop", extra=self._extra(job))
                return
            if state == _CANCELLED_BY_SHUTDOWN:
                await self._release(job)
                return
            JOB_DURATION.labels(kind=job.kind).observe(time.monotonic() - started)
            if handler_task.cancelled():
                # Cancelled from inside the handler while we were not shutting down: retry it
                # with backoff instead of releasing, which would hot-loop on the same job.
                await self._finish(job, ("internal_error", False))
            else:
                await self._finish(job, handler_task.result())

    async def _await_handler(self, handler_task: asyncio.Task, stop: asyncio.Event | None) -> str:
        """Wait for the handler; return _DONE, _CANCELLED_BY_SHUTDOWN or _ABANDONED."""
        if stop is None:
            await asyncio.wait({handler_task})
            return _DONE
        stop_task = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({handler_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_task.cancel()
        if handler_task.done():
            return _DONE
        await asyncio.wait({handler_task}, timeout=self._shutdown_grace_s)
        if handler_task.done():
            return _DONE
        handler_task.cancel()
        await asyncio.wait({handler_task}, timeout=self._shutdown_grace_s)
        return _CANCELLED_BY_SHUTDOWN if handler_task.done() else _ABANDONED

    async def _invoke(self, job: Job) -> tuple[str, bool] | None:
        """Run the handler; return None on success or (error_code, permanent)."""
        ctx = JobContext(job=job, db=self._db, worker_id=self.worker_id, services=self._services)
        timeout = asyncio.timeout(self._timeouts_s.get(job.kind, self._default_timeout_s))
        try:
            async with timeout:
                await self._handlers[job.kind](ctx)
        except PermanentJobError as exc:
            return _normalize_code(exc.code), True
        except RetryableJobError as exc:
            return _normalize_code(exc.code), False
        except TimeoutError:
            if timeout.expired():
                return "job_timeout", False
            # A TimeoutError the handler raised itself (for example from a client call).
            logger.exception("job handler error", extra=self._extra(job))
            return "internal_error", False
        except Exception:
            logger.exception("job handler error", extra=self._extra(job))
            return "internal_error", False
        return None

    def _extra(self, job: Job) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "kind": job.kind, "attempt": job.attempts}

    async def _finish(self, job: Job, result: tuple[str, bool] | None) -> None:
        status = None
        async with self._db.system_tx() as conn:
            if result is None:
                held = await queue.complete(conn, job.id, self.worker_id)
                outcome = "succeeded"
            else:
                code, permanent = result
                status = await queue.fail(conn, job.id, self.worker_id, code, permanent=permanent)
                held = status is not None
                outcome = {"failed": "failed", "queued": "retry", "dead": "dead"}.get(status or "")
        if not held:
            logger.warning("job lease lost", extra=self._extra(job))
            return
        JOBS.labels(kind=job.kind, outcome=outcome).inc()
        extra = {**self._extra(job), "outcome": outcome}
        if result is not None:
            extra["error_code"] = result[0]
        logger.info("job finished", extra=extra)
        if result is not None and status in ("failed", "dead"):
            await self._run_failure_hook(job, result[0])

    async def _run_failure_hook(self, job: Job, error_code: str) -> None:
        """Let the job's kind settle its domain rows; a hook error is logged, never raised."""
        hook = self._failure_handlers.get(job.kind)
        if hook is None:
            return
        ctx = JobContext(job=job, db=self._db, worker_id=self.worker_id, services=self._services)
        try:
            await hook(ctx, error_code)
        except Exception:
            logger.exception("job failure hook error", extra=self._extra(job))

    async def _release(self, job: Job) -> None:
        async with self._db.system_tx() as conn:
            released = await queue.release(conn, job.id, self.worker_id)
        if released:
            logger.info("job released", extra={**self._extra(job), "outcome": "released"})
        else:
            logger.warning("job lease lost", extra=self._extra(job))

    async def _heartbeat_job(self, job: Job) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_s)
            try:
                async with self._db.system_tx() as conn:
                    held = await queue.heartbeat(conn, job.id, self.worker_id)
            except Exception:
                logger.exception("job heartbeat error", extra=self._extra(job))
                continue
            if not held:
                logger.warning("job lease lost", extra=self._extra(job))
                return

    async def _register(self) -> None:
        async with self._db.system_tx() as conn:
            await conn.execute(
                "insert into workers (id, kinds) values (%s, %s) "
                "on conflict (id) do update set kinds = excluded.kinds, heartbeat_at = now()",
                (self.worker_id, self._kinds),
            )

    async def _deregister(self) -> None:
        async with self._db.system_tx() as conn:
            await conn.execute("delete from workers where id = %s", (self.worker_id,))

    async def _refresh_worker_row(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_s)
            try:
                await self._register()
            except Exception:
                logger.exception("worker heartbeat error", extra={"worker_id": self.worker_id})

    async def _requeue_stale_loop(self) -> None:
        while True:
            try:
                async with self._db.system_tx() as conn:
                    requeued, dead = await queue.requeue_stale_detailed(
                        conn, self._stale_after_s
                    )
                if requeued:
                    logger.info("stale jobs requeued", extra={"worker_id": self.worker_id})
                for job in dead:
                    await self._run_failure_hook(job, "lease_expired")
            except Exception:
                logger.exception("requeue error", extra={"worker_id": self.worker_id})
            await asyncio.sleep(REQUEUE_INTERVAL_S)
