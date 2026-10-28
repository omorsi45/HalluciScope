"""Quick checks: one document and one question, answered and verified through the same engine.

The document is stored like any other, in a hidden per-tenant collection of kind `quick` (R4),
so the evidence links keep their foreign keys. Such a collection never shows up in listings or
default scopes; only the code below, which passes `kinds=("quick",)` everywhere, reaches it.

The upload is not queued as an `ingest_upload` job. The `run_analysis` job ingests it inline
(`process_upload`), so one job covers the whole check.
"""

import logging
from uuid import UUID, uuid4

from backend.analysis.service import create_analysis_request, require_capacity
from backend.auth.service import Principal
from backend.config import Settings
from backend.db.pool import Database
from backend.db.repos import collections
from backend.ingest.intake import intake_upload

logger = logging.getLogger(__name__)

QUICK_KINDS = ("quick",)


async def create_quick_analysis(
    db: Database,
    principal: Principal,
    *,
    data: bytes,
    filename: str,
    question: str,
    settings: Settings,
    enqueue: bool = True,
) -> UUID:
    """Store the document in a new hidden collection and a queued `quick` analysis of it.

    `enqueue=False` queues no job at all, for a caller that runs the analysis in-process.
    Raises the intake errors (415 `unsupported_type`, 413 `too_large`, ...) and a 429
    `too_many_analyses`. The limit is checked before anything is stored.
    """
    async with db.tenant_tx(principal.tenant_id) as conn:
        await require_capacity(conn, principal, settings)

    async with db.tenant_tx(principal.tenant_id) as conn:
        collection = await collections.insert_collection(
            conn,
            principal.tenant_id,
            f"quick-{uuid4().hex}",
            "",
            principal.user_id,
            kind="quick",
        )
        await collections.insert_grant(
            conn, principal.tenant_id, collection["id"], principal.user_id, "admin"
        )
    try:
        intake = await intake_upload(
            db,
            principal,
            data=data,
            filename=filename,
            collection_id=collection["id"],
            settings=settings,
            collection_kinds=QUICK_KINDS,
            enqueue_job=False,
        )
    except Exception:
        await _discard_collection(db, principal.tenant_id, collection["id"])
        raise
    return await create_analysis_request(
        db,
        principal,
        kind="quick",
        question=question,
        answer=None,
        collection_ids=[collection["id"]],
        as_of=None,
        options={"upload_id": str(intake.upload_id), "extraction": "llm_atomic"},
        settings=settings,
        collection_kinds=QUICK_KINDS,
        enqueue=enqueue,
    )


async def _discard_collection(db: Database, tenant_id: UUID, collection_id: UUID) -> None:
    """Remove the still empty collection after a refused upload; never hides the refusal."""
    try:
        async with db.tenant_tx(tenant_id) as conn:
            await collections.delete_collection(conn, collection_id)
    except Exception:
        logger.exception("quick collection cleanup failed")
