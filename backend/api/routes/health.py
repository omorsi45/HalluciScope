"""Liveness, readiness and the token-protected metrics endpoint under /api/v1."""

import hmac
import logging

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
_LLM_TIMEOUT_SECONDS = 2.0
_METRICS_CONTENT_TYPE = "text/plain; version=0.0.4"


def _failed(stage: str) -> None:
    """Log a failed readiness check by stage only; the body never carries error text."""
    logger.exception("readiness check failed", extra={"stage": stage})


async def _check_db(db: Database) -> bool:
    try:
        async with db.system_tx() as conn:
            await conn.execute("select 1")
        return True
    except Exception:
        _failed("db")
        return False


async def _check_migrations(db: Database) -> bool:
    try:
        async with db.system_tx() as conn:
            applied = await applied_versions(conn)
        return all(m.version in applied for m in discover())
    except Exception:
        _failed("migrations")
        return False


async def _check_worker(db: Database) -> bool:
    try:
        async with db.system_tx() as conn:
            cur = await conn.execute(
                "select coalesce(max(heartbeat_at) > now() - make_interval(secs => %s), false)"
                " as ok from workers",
                (_WORKER_FRESH_SECONDS,),
            )
            return bool((await cur.fetchone())["ok"])
    except Exception:
        _failed("worker")
        return False


async def _check_llm(request: Request) -> bool:
    try:
        base = request.app.state.settings.ollama_base_url.rstrip("/")
        response = await request.app.state.http.get(
            f"{base}/api/version", timeout=_LLM_TIMEOUT_SECONDS
        )
        return response.status_code == 200
    except Exception:
        _failed("llm")
        return False


@router.get("/health/live")
async def live():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request, db: Database = Depends(get_db)):
    db_ok = await _check_db(db)
    migrations_ok = await _check_migrations(db)
    worker_ok = await _check_worker(db)
    llm_ok = await _check_llm(request)
    serving = db_ok and migrations_ok
    body = {
        "status": "ok" if serving else "unavailable",
        "db": "ok" if db_ok else "error",
        "migrations": "ok" if migrations_ok else "pending",
        "worker": "ok" if worker_ok else "stale",
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
    async with db.system_tx() as conn:
        JOBS_QUEUED.set(await count_queued(conn))
    return Response(REGISTRY.render(), media_type=_METRICS_CONTENT_TYPE)
