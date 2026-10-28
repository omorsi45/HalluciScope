"""Analyses: the request that stores one and queues its job, and the run that answers or verifies.

`create_analysis_request` stores the analysis and its job in one transaction. The job payload
is the analysis id; the question and answer live only in the RLS table `analyses`.

`run_analysis` is the worker side. An ask analysis retrieves passages, has the LLM answer from
them and verifies the answer; a verify analysis verifies the stored answer. A quick analysis
first ingests its own upload inline (`process_upload`), then runs like an ask over that one
hidden collection. Verdicts come only from the NLI features and the calibrator
(`VerificationEngine`); the LLM writes the answer and splits it into claims, nothing more.
An ask analysis may also resample its answer (`options.consistency_samples`) for a diagnostic
stored beside each claim; it runs inside the verifying stage and never changes a verdict.
Every failure is reduced to a stable error code: exception messages and text never reach a log
line or a row.
"""

import asyncio
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection

from backend.api.errors import ApiError
from backend.auth.service import Principal
from backend.authz.policy import readable_collection_ids
from backend.calibration.artifact import CalibratorMismatch
from backend.config import Settings
from backend.db.pool import Database
from backend.db.repos import analyses
from backend.generate.answer import CitationRef, generate_answer, parse_citations
from backend.ingest.handlers import process_upload
from backend.jobs import queue
from backend.jobs.worker import PermanentJobError
from backend.llm.client import LLMClient, LLMFailure
from backend.obs.logging import bind, request_id_var
from backend.retrieval.types import PassageHit, SearchMode, SearchScope
from backend.services import Services
from backend.verify.claims import llm_atomic_claims, sentence_claims
from backend.verify.consistency import consistency_diagnostic
from backend.verify.engine import VerificationEngine
from backend.verify.types import Claim, ClaimVerdict

logger = logging.getLogger(__name__)

STAGES = ("retrieving", "generating", "extracting", "verifying", "persisting")

ASK_K = 8
ASK_CANDIDATES = 50
JOB_PRIORITY = 10

_TERMINAL_STATUSES = ("succeeded", "failed")
_INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class CitationRecord(CitationRef):
    """A parsed citation plus the exact (passage, version) it resolved to.

    Both ids are None for an `unknown_marker`, and set together for a `valid` citation.
    """

    passage_id: UUID | None
    version_id: UUID | None


class _RetrievalFailed(Exception):
    """Stands in for whatever the retriever raised, so no message travels with it."""


class _GuardedRetriever:
    """Reports every retriever failure as `_RetrievalFailed`, whoever called it."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def search(
        self,
        query: str,
        scope: SearchScope,
        *,
        k: int = 10,
        mode: SearchMode = "hybrid",
        rerank: bool = True,
        candidates: int = 50,
        exact: bool = False,
    ) -> list[PassageHit]:
        try:
            return await self._inner.search(
                query,
                scope,
                k=k,
                mode=mode,
                rerank=rerank,
                candidates=candidates,
                exact=exact,
            )
        except Exception:
            raise _RetrievalFailed from None


class _Stages:
    """Commits each stage as the current one (so a poll shows progress) and times it."""

    def __init__(self, db: Database, tenant_id: UUID, analysis_id: UUID) -> None:
        self._db = db
        self._tenant_id = tenant_id
        self._analysis_id = analysis_id
        self.timings: dict[str, float] = {}

    @asynccontextmanager
    async def enter(self, stage: str) -> AsyncIterator[None]:
        async with self._db.tenant_tx(self._tenant_id) as conn:
            await analyses.set_progress(conn, self._analysis_id, stage)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage] = round(time.perf_counter() - started, 3)


async def require_capacity(conn: AsyncConnection, principal: Principal, settings: Settings) -> None:
    """Raise a 429 `too_many_analyses` when the user already has the maximum queued or running."""
    if await analyses.count_active(conn, principal.user_id) >= settings.max_active_analyses:
        raise ApiError(
            429,
            "too_many_analyses",
            "Too many analyses in progress",
            "Wait for a running analysis to finish.",
        )


async def create_analysis_request(
    db: Database,
    principal: Principal,
    *,
    kind: Literal["ask", "verify", "quick"],
    question: str | None,
    answer: str | None,
    collection_ids: Sequence[UUID] | None,
    as_of: datetime | None,
    options: Mapping[str, Any],
    settings: Settings,
    collection_kinds: Sequence[str] = ("normal",),
    enqueue: bool = True,
) -> UUID:
    """Store a queued analysis and its `run_analysis` job; return the analysis id.

    `enqueue=False` stores the analysis without a job, for a caller that runs it in-process.
    Raises a 429 `too_many_analyses` when the user already has `max_active_analyses` queued or
    running, and a 404 `collection_not_found` for a collection they cannot read.
    """
    async with db.tenant_tx(principal.tenant_id) as conn:
        await require_capacity(conn, principal, settings)
        readable = await readable_collection_ids(
            conn, principal, collection_ids, kinds=collection_kinds
        )
        # The database clock stamps created_at, so it is what known_at must be compared to.
        cur = await conn.execute("select now() as now")
        known_at = (await cur.fetchone())["now"]
        analysis_id = await analyses.create_analysis(
            conn,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            kind=kind,
            question=question,
            answer=answer,
            collection_ids=readable,
            as_of=as_of,
            known_at=known_at,
            options=dict(options),
        )
        if enqueue:
            await queue.enqueue(
                conn,
                "run_analysis",
                {"analysis_id": str(analysis_id)},
                tenant_id=principal.tenant_id,
                priority=JOB_PRIORITY,
                idempotency_key=f"analysis:{analysis_id}",
                request_id=request_id_var.get(),
            )
    return analysis_id


def resolve_citations(
    refs: Sequence[CitationRef], context: Sequence[PassageHit]
) -> list[CitationRecord]:
    """Tie each valid citation to the exact (passage, version) it numbered in the context."""
    records = []
    for ref in refs:
        hit = context[ref.number - 1] if ref.status == "valid" else None
        records.append(
            CitationRecord(
                number=ref.number,
                marker_start=ref.marker_start,
                marker_end=ref.marker_end,
                sentence_start=ref.sentence_start,
                sentence_end=ref.sentence_end,
                status=ref.status,
                passage_id=None if hit is None else hit.passage_id,
                version_id=None if hit is None else hit.version_id,
            )
        )
    return records


def _pinned(claims: Sequence[Claim], context: Sequence[PassageHit]) -> dict[int, list[PassageHit]]:
    """The context passages each claim's sentence cited, by claim ordinal."""
    pinned = {}
    for claim in claims:
        hits = [
            context[number - 1] for number in claim.cited_numbers if 1 <= number <= len(context)
        ]
        if hits:
            pinned[claim.ordinal] = hits
    return pinned


async def _extract_claims(
    llm: LLMClient | None, answer: str, options: Mapping[str, Any]
) -> list[Claim]:
    if options.get("extraction", "llm_atomic") != "llm_atomic":
        return sentence_claims(answer, method="sentence")
    if llm is None:
        return sentence_claims(answer, method="sentence_fallback")
    return await llm_atomic_claims(llm, answer)


async def _with_consistency(
    services: Services,
    question: str,
    context: Sequence[PassageHit],
    verdicts: Sequence[ClaimVerdict],
    n_samples: int,
) -> list[ClaimVerdict]:
    """The verdicts with the self-consistency diagnostic added; nothing else changes."""
    entries = await consistency_diagnostic(
        services.llm,
        services.embedder,
        question,
        context,
        [verdict.claim for verdict in verdicts],
        n_samples=n_samples,
    )
    return [
        replace(verdict, diagnostics={**verdict.diagnostics, "consistency": entry})
        for verdict, entry in zip(verdicts, entries, strict=True)
    ]


async def _ingest_quick_upload(
    services: Services, tenant_id: UUID, analysis: Mapping[str, Any]
) -> None:
    """Parse and index the quick analysis's own upload, so the retrieval below can see it.

    Raises `PermanentJobError(code)` with the ingestion error code when the document cannot be
    read, including when an earlier attempt already failed the upload.
    """
    upload_id = UUID(analysis["options"]["upload_id"])
    if await process_upload(services, tenant_id, upload_id) is not None:
        return
    # Already final: a finished upload is fine, but a failed one must not be answered from an
    # empty collection.
    async with services.db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select status, error_code from uploads where id = %s", (upload_id,)
        )
        upload = await cur.fetchone()
    if upload is None:
        raise PermanentJobError(_INTERNAL_ERROR)
    if upload["status"] == "failed":
        raise PermanentJobError(upload["error_code"] or _INTERNAL_ERROR)


async def _execute(services: Services, tenant_id: UUID, analysis: Mapping[str, Any]) -> None:
    """Run every stage and store the results; raises on any failure."""
    db = services.db
    calibrator = services.calibrator
    analysis_id = analysis["id"]
    as_of = analysis["as_of"]
    scope = SearchScope(
        tenant_id,
        tuple(analysis["collection_ids"]),
        as_of,
        analysis["known_at"] if as_of is not None else None,
    )
    retriever = _GuardedRetriever(services.retriever())
    # Refuses models other than the ones the calibrator was trained with, before any work.
    engine = VerificationEngine(
        retriever, services.reranker, services.nli, services.embedder, calibrator
    )
    stages = _Stages(db, tenant_id, analysis_id)
    started = time.perf_counter()

    async with db.tenant_tx(tenant_id) as conn:
        await analyses.set_status(conn, analysis_id, "running")

    context: Sequence[PassageHit] = ()
    if analysis["kind"] in ("ask", "quick"):
        question = analysis["question"]
        async with stages.enter("retrieving"):
            if analysis["kind"] == "quick":
                await _ingest_quick_upload(services, tenant_id, analysis)
            hits = await retriever.search(
                question, scope, k=ASK_K, rerank=True, candidates=ASK_CANDIDATES
            )
        async with stages.enter("generating"):
            generated = await generate_answer(services.llm, question, hits)
        answer, refs, context = generated.text, generated.citations, generated.context
    else:
        answer = analysis["answer"]
        refs = parse_citations(answer, 0)

    async with stages.enter("extracting"):
        claims = await _extract_claims(services.llm, answer, analysis["options"])
    async with stages.enter("verifying"):
        verdicts = await engine.verify(claims, scope, _pinned(claims, context))
        samples = analysis["options"].get("consistency_samples", 0)
        if analysis["kind"] == "ask" and samples > 0:
            verdicts = await _with_consistency(
                services, analysis["question"], context, verdicts, samples
            )

    async with stages.enter("persisting"):
        citations = resolve_citations(refs, context)
        counts = Counter(verdict.verdict.value for verdict in verdicts)
        summary = {
            "n_claims": len(verdicts),
            "supported": counts["supported"],
            "contradicted": counts["contradicted"],
            "insufficient": counts["insufficient"],
            "n_citations": len(citations),
            "n_invalid_citations": sum(1 for record in citations if record.status != "valid"),
            "extraction": dict(Counter(verdict.claim.method for verdict in verdicts)),
        }
        config = {
            "llm": None if services.llm is None else services.llm.model_id,
            "nli": services.nli.model_id,
            "embedder": services.embedder.model_id,
            "reranker": services.reranker.model_id,
            "calibrator_id": calibrator.artifact_id,
            "evidence_k": calibrator.evidence_k,
            "candidates": calibrator.candidates,
            "premise_mode": calibrator.premise_mode,
        }
    # The insert is stamped with these timings, so "persisting" covers the work before it.
    timings = {**stages.timings, "total": round(time.perf_counter() - started, 3)}
    async with db.tenant_tx(tenant_id) as conn:
        await analyses.save_results(
            conn,
            analysis_id,
            answer=answer,
            verdicts=verdicts,
            citations=citations,
            config=config,
            timings=timings,
            summary=summary,
        )
        # The last stage is done: "done" becomes current and every real stage is in stages_done.
        await analyses.set_progress(conn, analysis_id, "done")
    logger.info(
        "analysis finished",
        extra={"kind": analysis["kind"], "outcome": "succeeded", "n_claims": len(verdicts)},
    )


async def _attempt(services: Services, tenant_id: UUID, analysis: Mapping[str, Any]) -> str | None:
    """Run the analysis inside its time limit; None on success, else the error code."""
    if services.calibrator is None:
        return "calibrator_missing"
    timeout = asyncio.timeout(services.settings.analysis_timeout_s)
    try:
        async with timeout:
            await _execute(services, tenant_id, analysis)
    except TimeoutError:
        if timeout.expired():
            return "analysis_timeout"
        logger.exception("analysis failed")
        return _INTERNAL_ERROR
    except LLMFailure as exc:
        return exc.code
    except PermanentJobError as exc:
        # Only a quick analysis's inline ingestion raises this: its code is the analysis's.
        return exc.code
    except CalibratorMismatch:
        return "calibrator_mismatch"
    except _RetrievalFailed:
        return "retrieval_failed"
    except Exception:
        logger.exception("analysis failed")
        return _INTERNAL_ERROR
    return None


async def run_analysis(services: Services, tenant_id: UUID, analysis_id: UUID) -> None:
    """Run one analysis to a final status.

    A final analysis is left alone. On failure the analysis is marked `failed` with the error
    code and `PermanentJobError(code)` is raised: LLM work is never retried automatically.
    """
    async with services.db.tenant_tx(tenant_id) as conn:
        analysis = await analyses.get_analysis(conn, analysis_id)
    if analysis is None or analysis["status"] in _TERMINAL_STATUSES:
        return
    with bind(tenant_id=str(tenant_id), user_id=str(analysis["user_id"])):
        code = await _attempt(services, tenant_id, analysis)
        if code is None:
            return
        if code != _INTERNAL_ERROR:  # internal errors were logged with their traceback
            logger.warning("analysis failed", extra={"kind": analysis["kind"], "error_code": code})
        async with services.db.tenant_tx(tenant_id) as conn:
            await analyses.set_status(conn, analysis_id, "failed", error_code=code)
    raise PermanentJobError(code)
