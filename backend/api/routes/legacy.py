"""The deprecated /api routes (analyze, analyses, feedback) as adapters over the new engine.

`POST /api/analyze` runs a quick analysis (``backend.analysis.quick``) and waits for the worker,
then answers in the old shape. Verdicts and probabilities are the engine's; nothing is
recomputed here. Every response of these routes carries `Deprecation` and a `Link` to the
successor, errors included (``DeprecationHeadersMiddleware``). Document text, questions and
answers are never logged, and a failed analysis is reported by its error code alone.
"""

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.analysis.quick import create_quick_analysis
from backend.api.errors import ApiError
from backend.api.schemas import (
    AnalysisListItem,
    AnalyzeRequest,
    AnalyzeResponse,
    ClaimResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from backend.auth.deps import current_principal, get_db
from backend.auth.service import Principal
from backend.authz.policy import require_analysis
from backend.db.pool import Database
from backend.db.repos import analyses

PASTED_FILENAME = "pasted.txt"
POLL_INTERVAL_S = 0.25

_DEPRECATED_PATHS = frozenset({"/api/analyze", "/api/analyses", "/api/feedback"})
_SUCCESSOR_LINK = '</api/v1/ask>; rel="successor-version"'
_FINAL_STATUSES = ("succeeded", "failed")
# The old tiers for the verdicts, in the vocabulary of the old UI.
_TIERS = {"supported": "supported", "contradicted": "hallucinated", "insufficient": "uncertain"}
_FAILURE_STATUS = {"llm_unavailable": 503, "llm_timeout": 504, "analysis_timeout": 504}

# One row per succeeded analysis of the user, scored as the mean over its claims.
_LIST_SQL = (
    "select a.id, a.question, a.answer, a.created_at, "
    "coalesce(avg(1 - c.p_supported), 0.0) as overall_score "
    "from analyses a left join claims c on c.analysis_id = a.id "
    "where a.user_id = %s and a.status = 'succeeded' "
    "group by a.id order by a.created_at desc, a.id desc limit %s"
)

router = APIRouter()


class DeprecationHeadersMiddleware:
    """Mark every response of the legacy routes as deprecated, the error responses included."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in _DEPRECATED_PATHS:
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Deprecation"] = "true"
                headers["Link"] = _SUCCESSOR_LINK
            await send(message)

        await self.app(scope, receive, send_wrapper)


def analyze_response(
    analysis: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> AnalyzeResponse:
    """The old response for a succeeded analysis.

    `claims` are in ordinal order. `evidence` rows carry `claim_id`, the NLI scores and the
    passage `text`, and are in rank order within each claim.
    """
    by_claim: dict[Any, list[Mapping[str, Any]]] = {}
    for item in evidence:
        by_claim.setdefault(item["claim_id"], []).append(item)

    scored = [_scored_claim(claim, by_claim.get(claim["id"], [])) for claim in claims]
    # Distinct passage texts, claim by claim, in the order they were first seen.
    chunks = dict.fromkeys(
        item["text"] for claim in claims for item in by_claim.get(claim["id"], [])
    )
    scores = [claim.hallucination_score for claim in scored]
    return AnalyzeResponse(
        id=str(analysis["id"]),
        question=analysis["question"],
        answer=analysis["answer"],
        scored_claims=scored,
        retrieved_chunks=list(chunks),
        overall_score=sum(scores) / len(scores) if scores else 0.0,
    )


def _scored_claim(claim: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> ClaimResponse:
    # The best entailment decides which chunk is shown; a tie goes to the better ranked item.
    best = max(evidence, key=lambda item: item["nli_entail"], default=None)
    if best is None:
        nli = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0, "matched_chunk": ""}
    else:
        nli = {
            "entailment": best["nli_entail"],
            "contradiction": best["nli_contra"],
            "neutral": best["nli_neutral"],
            "matched_chunk": best["text"],
        }
    return ClaimResponse(
        claim=claim["text"],
        hallucination_score=1 - claim["p_supported"],
        tier=_TIERS[claim["verdict"]],
        verifier_details={
            "nli": nli,
            "calibrated": {
                "p_supported": claim["p_supported"],
                "p_contradicted": claim["p_contradicted"],
                "p_insufficient": claim["p_insufficient"],
                "verdict": claim["verdict"],
            },
        },
    )


async def _wait_until_final(
    db: Database, principal: Principal, analysis_id: UUID, wait_s: float
) -> Mapping[str, Any]:
    """Poll the analysis until it is final or `wait_s` has passed; returns its last state."""
    deadline = time.monotonic() + wait_s
    while True:
        async with db.tenant_tx(principal.tenant_id) as conn:
            analysis = await analyses.get_analysis(conn, analysis_id)
        remaining = deadline - time.monotonic()
        if analysis["status"] in _FINAL_STATUSES or remaining <= 0:
            return analysis
        await asyncio.sleep(min(POLL_INTERVAL_S, remaining))


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> AnalyzeResponse:
    settings = request.app.state.settings
    analysis_id = await create_quick_analysis(
        db,
        principal,
        data=body.document_text.encode(),
        filename=PASTED_FILENAME,
        question=body.question,
        settings=settings,
    )
    analysis = await _wait_until_final(db, principal, analysis_id, settings.quick_wait_s)
    if analysis["status"] == "failed":
        code = analysis["error_code"] or "internal_error"
        raise ApiError(_FAILURE_STATUS.get(code, 500), code, "Analysis failed")
    if analysis["status"] != "succeeded":
        raise ApiError(
            504,
            "analysis_pending",
            "Analysis still running",
            "The analysis has not finished yet. Follow the Location header to read it later.",
            headers={"Location": f"/api/v1/analyses/{analysis_id}"},
        )
    async with db.tenant_tx(principal.tenant_id) as conn:
        claims = await analyses.get_claims(conn, analysis_id)
        evidence = await analyses.get_evidence(conn, [claim["id"] for claim in claims])
    return analyze_response(analysis, claims, evidence)


@router.get("/analyses", response_model=list[AnalysisListItem])
async def list_analyses(
    limit: int = Query(50, ge=1, le=200),
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> list[AnalysisListItem]:
    async with db.tenant_tx(principal.tenant_id) as conn:
        cur = await conn.execute(_LIST_SQL, (principal.user_id, limit))
        rows = await cur.fetchall()
    return [
        AnalysisListItem(
            id=str(row["id"]),
            question=row["question"] or "",
            answer=row["answer"] or "",
            overall_score=row["overall_score"],
            created_at=row["created_at"].isoformat(),
        )
        for row in rows
    ]


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> FeedbackResponse:
    try:
        analysis_id = UUID(body.analysis_id)
    except ValueError:
        raise _claim_not_found() from None
    async with db.tenant_tx(principal.tenant_id) as conn:
        try:
            await require_analysis(conn, principal, analysis_id)
        except ApiError:
            # The caller learns only that there is no such claim for them.
            raise _claim_not_found() from None
        cur = await conn.execute(
            "select id, tenant_id from claims where analysis_id = %s and ordinal = %s",
            (analysis_id, body.claim_index),
        )
        claim = await cur.fetchone()
        if claim is None:
            raise _claim_not_found()
        feedback_id = await analyses.upsert_feedback(
            conn,
            tenant_id=claim["tenant_id"],
            claim_id=claim["id"],
            user_id=principal.user_id,
            label="correct" if body.is_correct else "incorrect",
            note=body.note,
        )
    return FeedbackResponse(id=str(feedback_id))


def _claim_not_found() -> ApiError:
    return ApiError(404, "claim_not_found", "Claim not found")
