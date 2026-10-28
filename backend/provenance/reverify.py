"""Re-verification of live claims (R3), on a source change and on request.

When a document gets a new current version, `claims_to_recheck` picks the claims of live
analyses that the change can touch, and `recheck_claims` verifies them again against the current
documents. Each re-verification appends a `claim_checks` row; the stored claim and its verdict
are never rewritten, so the history stays.

The re-check runs as the analysis's creator, with their access as of now (R8): the stored
collection ids intersected with what they can read today. A creator who is gone gets nothing
written. Verdicts come from the same `VerificationEngine` as the analysis itself. Claim text and
passage text never reach a log line.
"""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection

from backend.auth.service import principal_for_user
from backend.authz.policy import readable_subset
from backend.calibration.artifact import CalibratorMismatch
from backend.db.repos import analyses
from backend.jobs.worker import JobContext, PermanentJobError
from backend.models.protocols import Reranker
from backend.obs.logging import bind
from backend.retrieval.types import SearchScope
from backend.services import Services
from backend.verify.engine import VerificationEngine
from backend.verify.types import Claim, ClaimVerdict

logger = logging.getLogger(__name__)

Trigger = Literal["evidence_changed", "new_passage", "thin_evidence", "manual"]

# One row per claim of a live, finished analysis over the collection. `lowest_rerank` ignores
# null scores and is null when the claim has none; `evidence_changed` is true when an evidence
# link points into the old version at a passage whose text is not in the new version.
_CLAIMS_SQL = """
select c.id as claim_id, c.analysis_id, c.text,
    (select count(*) from evidence_links e where e.claim_id = c.id)::int as n_links,
    (select min(e.rerank_score) from evidence_links e where e.claim_id = c.id) as lowest_rerank,
    exists (
        select 1 from evidence_links e
        join passages p on p.id = e.passage_id and p.version_id = e.version_id
        where e.claim_id = c.id and e.version_id = %(old_version)s
        and not (p.text_sha256 = any(%(new_hashes)s::bytea[]))
    ) as evidence_changed
from claims c join analyses a on a.id = c.analysis_id
where a.status = 'succeeded' and a.as_of is null and %(collection)s = any(a.collection_ids)
order by a.created_at, a.id, c.ordinal
"""


@dataclass(frozen=True)
class RecheckCandidate:
    claim_id: UUID
    analysis_id: UUID
    trigger: Trigger


async def claims_to_recheck(
    conn: AsyncConnection,
    reranker: Reranker,
    *,
    document_id: UUID,
    old_version_id: UUID,
    new_version_id: UUID,
    evidence_k: int,
) -> list[RecheckCandidate]:
    """The claims a new current version of the document can have changed (R3), with the reason.

    Only live (no `as_of`), succeeded analyses whose collections include the document's are
    considered, ordered by analysis then claim ordinal. A claim is a candidate when, in this
    order of precedence:

    - `evidence_changed`: an evidence link points into the old version at a passage whose text
      hash is absent from the new version;
    - `thin_evidence`: the new version has changed passages (hash absent from the old version)
      and the claim has fewer than `evidence_k` evidence links;
    - `new_passage`: the new version has changed passages and the best rerank logit of the claim
      against them is at least the claim's lowest stored rerank score (a claim whose links all
      have no rerank score always qualifies).
    """
    cur = await conn.execute("select collection_id from documents where id = %s", (document_id,))
    document = await cur.fetchone()
    if document is None:
        return []
    old_hashes = {bytes(h) for h in await _passage_hashes(conn, old_version_id)}
    new_passages = await _passages(conn, new_version_id)
    new_hashes = [bytes(row["text_sha256"]) for row in new_passages]
    changed_texts = list(
        dict.fromkeys(
            row["text"] for row in new_passages if bytes(row["text_sha256"]) not in old_hashes
        )
    )

    cur = await conn.execute(
        _CLAIMS_SQL,
        {
            "old_version": old_version_id,
            "new_hashes": new_hashes,
            "collection": document["collection_id"],
        },
    )
    found: list[RecheckCandidate] = []
    for row in await cur.fetchall():
        trigger: Trigger | None = None
        if row["evidence_changed"]:
            trigger = "evidence_changed"
        elif changed_texts:
            if row["n_links"] < evidence_k:
                trigger = "thin_evidence"
            elif await _reranks_into_evidence(reranker, row, changed_texts):
                trigger = "new_passage"
        if trigger is not None:
            found.append(RecheckCandidate(row["claim_id"], row["analysis_id"], trigger))
    return found


async def _passage_hashes(conn: AsyncConnection, version_id: UUID) -> list[bytes]:
    cur = await conn.execute(
        "select distinct text_sha256 from passages where version_id = %s", (version_id,)
    )
    return [row["text_sha256"] for row in await cur.fetchall()]


async def _passages(conn: AsyncConnection, version_id: UUID) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "select text_sha256, text from passages where version_id = %s order by ordinal",
        (version_id,),
    )
    return await cur.fetchall()


async def _reranks_into_evidence(
    reranker: Reranker, claim: Mapping[str, Any], changed_texts: Sequence[str]
) -> bool:
    """True when the best changed passage scores at least the claim's weakest stored evidence."""
    lowest = claim["lowest_rerank"]
    if lowest is None:
        return True
    # CPU-bound model call: off the event loop, one call per claim.
    scores = await asyncio.to_thread(reranker.score, claim["text"], list(changed_texts))
    return max(scores) >= lowest


def _build_engine(services: Services) -> VerificationEngine:
    """The engine over the worker's models, or a permanent job error naming the calibrator."""
    if services.calibrator is None:
        raise PermanentJobError("calibrator_missing")
    try:
        return VerificationEngine(
            services.retriever(),
            services.reranker,
            services.nli,
            services.embedder,
            services.calibrator,
        )
    except CalibratorMismatch:
        raise PermanentJobError("calibrator_mismatch") from None


async def recheck_claims(
    services: Services,
    tenant_id: UUID,
    analysis_id: UUID,
    claim_ids: Sequence[UUID],
    reason: Literal["source_changed", "manual"],
) -> int:
    """Verify the claims again against the current documents and append a check per claim.

    The scope is the analysis's stored collections that its creator can read now (R8). Returns
    the number of checks written: 0 when the analysis is gone, is not a finished live analysis,
    or its creator is no longer an enabled member of the tenant.
    """
    if not claim_ids:
        return 0
    db = services.db
    wanted = set(claim_ids)
    async with db.tenant_tx(tenant_id) as conn:
        analysis = await analyses.get_analysis(conn, analysis_id)
        if analysis is None or analysis["status"] != "succeeded" or analysis["as_of"] is not None:
            return 0
        rows = [c for c in await analyses.get_claims(conn, analysis_id) if c["id"] in wanted]
    if not rows:
        return 0
    principal = await principal_for_user(db, tenant_id, analysis["user_id"])
    if principal is None:
        return 0
    with bind(tenant_id=str(tenant_id), user_id=str(analysis["user_id"])):
        async with db.tenant_tx(tenant_id) as conn:
            readable = await readable_subset(conn, principal, analysis["collection_ids"])
        scope = SearchScope(
            tenant_id, tuple(cid for cid in analysis["collection_ids"] if cid in readable)
        )
        claims = [
            Claim(
                ordinal=row["ordinal"],
                text=row["text"],
                answer_start=row["answer_start"],
                answer_end=row["answer_end"],
                method=row["extraction_method"],
            )
            for row in rows
        ]
        # No transaction is open while the models run.
        verdicts = await _build_engine(services).verify(claims, scope)

        async with db.tenant_tx(tenant_id) as conn:
            latest = await analyses.latest_checks(conn, [row["id"] for row in rows])
            for row, verdict in zip(rows, verdicts, strict=True):
                previous = latest[row["id"]]["verdict"] if row["id"] in latest else row["verdict"]
                await analyses.insert_claim_check(
                    conn,
                    tenant_id=tenant_id,
                    claim_id=row["id"],
                    reason=reason,
                    verdict=verdict.verdict.value,
                    p_supported=verdict.probs["supported"],
                    p_contradicted=verdict.probs["contradicted"],
                    p_insufficient=verdict.probs["insufficient"],
                    evidence=_evidence_json(verdict),
                    checked_version_ids=list(
                        dict.fromkeys(item.hit.version_id for item in verdict.evidence)
                    ),
                    changed=verdict.verdict.value != previous,
                )
    logger.info("claims rechecked", extra={"outcome": reason, "n_claims": len(verdicts)})
    return len(verdicts)


def _evidence_json(verdict: ClaimVerdict) -> list[dict[str, Any]]:
    """The evidence a check was reached on: ids, relation, rank, span and NLI, never text."""
    return [
        {
            "passage_id": str(item.hit.passage_id),
            "version_id": str(item.hit.version_id),
            "relation": item.relation,
            "rank": rank,
            "span_start": item.span_start,
            "span_end": item.span_end,
            "nli": {
                "entail": item.nli.entail,
                "contra": item.nli.contra,
                "neutral": item.nli.neutral,
            },
        }
        for rank, item in enumerate(verdict.evidence, start=1)
    ]


async def handle_reverify_document(ctx: JobContext) -> None:
    """Re-verify the claims a new current version of a document can have changed."""
    services = ctx.services
    tenant_id = ctx.job.tenant_id
    payload = ctx.job.payload
    # Fail before any work when the calibrator is missing or was trained on other models.
    _build_engine(services)
    with bind(tenant_id=str(tenant_id)):
        async with services.db.tenant_tx(tenant_id) as conn:
            found = await claims_to_recheck(
                conn,
                services.reranker,
                document_id=UUID(payload["document_id"]),
                old_version_id=UUID(payload["old_version_id"]),
                new_version_id=UUID(payload["new_version_id"]),
                evidence_k=services.calibrator.evidence_k,
            )
        written = 0
        # The candidates come ordered by analysis, so each analysis is one run.
        for analysis_id, group in groupby(found, key=lambda candidate: candidate.analysis_id):
            written += await recheck_claims(
                services,
                tenant_id,
                analysis_id,
                [candidate.claim_id for candidate in group],
                "source_changed",
            )
        logger.info("document reverified", extra={"n_claims": written})


async def handle_reverify_analysis(ctx: JobContext) -> None:
    """Re-verify every claim of one analysis on request."""
    services = ctx.services
    tenant_id = ctx.job.tenant_id
    analysis_id = UUID(ctx.job.payload["analysis_id"])
    _build_engine(services)
    with bind(tenant_id=str(tenant_id)):
        async with services.db.tenant_tx(tenant_id) as conn:
            claims = await analyses.get_claims(conn, analysis_id)
        await recheck_claims(services, tenant_id, analysis_id, [c["id"] for c in claims], "manual")


async def log_reverify_failure(ctx: JobContext, error_code: str) -> None:
    """Failure hook for both re-verification jobs: there is no row to settle, so it only logs.

    Each analysis is rechecked in one transaction, so a dead job leaves whole analyses
    rechecked or untouched. Never raises: the worker must be able to settle other jobs.
    """
    try:
        tenant_id = None if ctx.job.tenant_id is None else str(ctx.job.tenant_id)
        with bind(tenant_id=tenant_id, job_id=str(ctx.job.id)):
            logger.warning(
                "reverify job failed", extra={"kind": ctx.job.kind, "error_code": error_code}
            )
    except Exception:
        logger.exception("reverify failure hook error")
