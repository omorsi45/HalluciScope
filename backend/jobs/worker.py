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


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:6]}"


class Worker:
    def __init__(
        self,
        db: Database,
        handlers: Mapping[str, Handler],
        *,
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
        with bind(job_id=str(job.id)):
            handler_task = asyncio.create_task(self._invoke(job))
            heartbeat_task = asyncio.create_task(self._heartbeat_job(job))
            shutting_down = False
            try:
                shutting_down = await self._await_handler(handler_task, stop)
            finally:
                heartbeat_task.cancel()
                if not handler_task.done():
                    handler_task.cancel()
                await asyncio.gather(heartbeat_task, handler_task, return_exceptions=True)
            if shutting_down or handler_task.cancelled():
                await self._release(job)
                return
            JOB_DURATION.labels(kind=job.kind).observe(time.monotonic() - started)
            await self._finish(job, handler_task.result())

    async def _await_handler(self, handler_task: asyncio.Task, stop: asyncio.Event | None) -> bool:
        """Wait for the handler; True if shutdown cut it short and it was cancelled."""
        if stop is None:
            await asyncio.wait({handler_task})
            return False
        stop_task = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({handler_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_task.cancel()
        if handler_task.done():
            return False
        await asyncio.wait({handler_task}, timeout=self._shutdown_grace_s)
        if handler_task.done():
            return False
        handler_task.cancel()
        await asyncio.wait({handler_task})
        return True

    async def _invoke(self, job: Job) -> tuple[str, bool] | None:
        """Run the handler; return None on success or (error_code, permanent)."""
        ctx = JobContext(job=job, db=self._db, worker_id=self.worker_id, services=self._services)
        timeout_s = self._timeouts_s.get(job.kind, self._default_timeout_s)
        try:
            async with asyncio.timeout(timeout_s):
                await self._handlers[job.kind](ctx)
        except PermanentJobError as exc:
            return exc.code, True
        except RetryableJobError as exc:
            return exc.code, False
        except TimeoutError:
            return "job_timeout", False
        except Exception:
            logger.exception("job handler error")
            return "internal_error", False
        return None

    async def _finish(self, job: Job, result: tuple[str, bool] | None) -> None:
        async with self._db.system_tx() as conn:
            if result is None:
                held = await queue.complete(conn, job.id, self.worker_id)
                outcome = "succeeded"
            else:
                code, permanent = result
                status = await queue.fail(conn, job.id, self.worker_id, code, permanent=permanent)
                held = status is not None
                outcome = {"failed": "failed", "queued": "retry", "dead": "dead"}.get(status or "")
        extra = {
            "worker_id": self.worker_id,
            "kind": job.kind,
            "attempt": job.attempts,
        }
        if not held:
            logger.warning("job lease lost", extra=extra)
            return
        JOBS.labels(kind=job.kind, outcome=outcome).inc()
        logger.info("job finished", extra={**extra, "outcome": outcome})

    async def _release(self, job: Job) -> None:
        async with self._db.system_tx() as conn:
            await queue.release(conn, job.id, self.worker_id)
        logger.info(
            "job released",
            extra={
                "worker_id": self.worker_id,
                "kind": job.kind,
                "attempt": job.attempts,
                "outcome": "released",
            },
        )

    async def _heartbeat_job(self, job: Job) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_s)
            try:
                async with self._db.system_tx() as conn:
                    held = await queue.heartbeat(conn, job.id, self.worker_id)
            except Exception:
                logger.exception("job heartbeat error")
                continue
            if not held:
                logger.warning("job lease lost", extra={"worker_id": self.worker_id})
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
                    requeued = await queue.requeue_stale(conn, self._stale_after_s)
                if requeued:
                    logger.info("stale jobs requeued", extra={"worker_id": self.worker_id})
            except Exception:
                logger.exception("requeue error", extra={"worker_id": self.worker_id})
            await asyncio.sleep(REQUEUE_INTERVAL_S)
