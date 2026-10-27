"""Liveness, readiness and the token-protected metrics endpoint under /api/v1."""

import asyncio
import hmac
import logging
from typing import NamedTuple

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from backend.api.errors import ApiError
from backend.auth.deps import get_db
from backend.db.migrate import applied_versions, discover
from backend.db.pool import Database
from backend.jobs.queue import count_queued
from backend.obs.metrics import JOBS_QUEUED, REGISTRY

logger = logging.getLogger(__name__)

router = APIRouter()

_WORKER_FRESH_SECONDS = 60
_READY_DEADLINE_SECONDS = 2.0
_STATEMENT_TIMEOUT = "1500ms"
_LLM_TIMEOUT_SECONDS = 2.0
_METRICS_CONTENT_TYPE = "text/plain; version=0.0.4"


class _DbReadiness(NamedTuple):
    db: bool
    migrations: bool
    worker: bool


async def _check_database(db: Database) -> _DbReadiness:
    """Run every database check in one bounded transaction.

    A failure at any point (including the deadline) reports the database as down and the
    checks that depend on it as unverified, so a hung database cannot stall readiness.
    """
    try:
        async with asyncio.timeout(_READY_DEADLINE_SECONDS):
            async with db.system_tx() as conn:
                await conn.execute(f"set local statement_timeout = '{_STATEMENT_TIMEOUT}'")
                await conn.execute("select 1")
                applied = await applied_versions(conn)
                migrations_ok = all(m.version in applied for m in discover())
                cur = await conn.execute(
                    "select coalesce(max(heartbeat_at) > now() - make_interval(secs => %s), false)"
                    " as ok from workers",
                    (_WORKER_FRESH_SECONDS,),
                )
                worker_ok = bool((await cur.fetchone())["ok"])
        return _DbReadiness(True, migrations_ok, worker_ok)
    except Exception:
        logger.exception("readiness check failed", extra={"stage": "db"})
        return _DbReadiness(False, False, False)


async def _check_llm(request: Request) -> bool:
    try:
        base = request.app.state.settings.ollama_base_url.rstrip("/")
        # httpx timeouts apply per phase; the outer deadline bounds the whole probe.
        async with asyncio.timeout(_LLM_TIMEOUT_SECONDS):
            response = await request.app.state.http.get(
                f"{base}/api/version", timeout=_LLM_TIMEOUT_SECONDS
            )
        return response.status_code == 200
    except Exception:
        # Ollama being down is routine: no traceback, never the exception text.
        logger.warning("readiness check failed", extra={"stage": "llm"})
        return False


@router.get("/health/live")
async def live():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request, db: Database = Depends(get_db)):
    checks = await _check_database(db)
    llm_ok = await _check_llm(request)
    serving = checks.db and checks.migrations
    body = {
        "status": "ok" if serving else "unavailable",
        "db": "ok" if checks.db else "error",
        "migrations": "ok" if checks.migrations else "pending",
        "worker": "ok" if checks.worker else "stale",
        "llm": "ok" if llm_ok else "unavailable",
    }
    return JSONResponse(body, status_code=200 if serving else 503)


@router.get("/metrics")
async def metrics(request: Request, db: Database = Depends(get_db)):
    token = request.app.state.settings.metrics_token
    # An empty or whitespace-only token means "not configured", never "open".
    if token is None or not token.strip():
        raise ApiError(404, "not_found", "Not found")
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
        raise ApiError(
            401,
            "unauthenticated",
            "Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        async with db.system_tx() as conn:
            JOBS_QUEUED.set(await count_queued(conn))
    except Exception:
        # The gauge keeps its previous value; the scrape itself still succeeds.
        logger.warning("metrics queue depth unavailable", extra={"stage": "metrics_queue"})
    return Response(REGISTRY.render(), media_type=_METRICS_CONTENT_TYPE)
