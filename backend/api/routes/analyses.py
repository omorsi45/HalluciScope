"""Ask, verify, and analysis read routes under /api/v1, plus claim feedback and claim checks.

An analysis is visible to its creator and to tenant owners and admins (R8). A claim is visible
exactly when its analysis is. Reading one withholds the source material (passage text, spans,
document and version details) of every collection the viewer can no longer read, and marks
each such evidence item and citation `redacted`. Question and answer text is never logged here.
"""

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from psycopg import AsyncConnection

from backend.analysis.service import create_analysis_request
from backend.api.errors import ApiError
from backend.api.pagination import (
    Page,
    PageParams,
    decode_uuid_cursor,
    encode_cursor,
    page_params,
)
from backend.api.schemas_v1 import (
    AnalysisAcceptedOut,
    AnalysisOut,
    AnalysisSummaryOut,
    AskIn,
    CitationOut,
    ClaimCheckOut,
    ClaimOut,
    EvidenceOut,
    FeedbackIn,
    FeedbackOut,
    GraphOut,
    NLIOut,
    VerifyIn,
)
from backend.auth.deps import current_principal, get_db
from backend.auth.service import Principal
from backend.authz.policy import is_tenant_admin, readable_subset, require_analysis
from backend.db.pool import Database
from backend.db.repos import analyses
from backend.provenance.graph import build_graph, passage_collection_ids

router = APIRouter()

QUESTION_PREVIEW_CHARS = 200


@router.post("/ask", response_model=AnalysisAcceptedOut, status_code=202)
async def ask(
    request: Request,
    body: AskIn,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> AnalysisAcceptedOut:
    analysis_id = await create_analysis_request(
        db,
        principal,
        kind="ask",
        question=body.question,
        answer=None,
        collection_ids=body.collection_ids,
        as_of=body.as_of,
        options=body.options.model_dump(),
        settings=request.app.state.settings,
    )
    return AnalysisAcceptedOut(analysis_id=analysis_id, status="queued")


@router.post("/verify", response_model=AnalysisAcceptedOut, status_code=202)
async def verify(
    request: Request,
    body: VerifyIn,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> AnalysisAcceptedOut:
    analysis_id = await create_analysis_request(
        db,
        principal,
        kind="verify",
        question=body.question or None,
        answer=body.answer,
        collection_ids=body.collection_ids,
        as_of=body.as_of,
        options=body.options.model_dump(),
        settings=request.app.state.settings,
    )
    return AnalysisAcceptedOut(analysis_id=analysis_id, status="queued")


@router.get("/analyses", response_model=Page[AnalysisSummaryOut])
async def list_analyses(
    mine: bool = True,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
    page: PageParams = Depends(page_params),
) -> Page[AnalysisSummaryOut]:
    if not mine and not is_tenant_admin(principal):
        raise ApiError(403, "forbidden", "Insufficient role")
    if page.after:
        # The repository binds the key as a UUID: reject anything else as a bad cursor first.
        decode_uuid_cursor(page.after)
    async with db.tenant_tx(principal.tenant_id) as conn:
        rows = await analyses.list_analyses(
            conn, user_id=principal.user_id if mine else None, page=page
        )
        has_more = len(rows) > page.limit
        rows = rows[: page.limit]
        changed = await _with_changed_checks(conn, [row["id"] for row in rows])
    next_cursor = encode_cursor(rows[-1]["created_at"], str(rows[-1]["id"])) if has_more else None
    return Page[AnalysisSummaryOut](
        items=[_summary_out(row, row["id"] in changed) for row in rows], next_cursor=next_cursor
    )


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut)
async def get_analysis(
    analysis_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> AnalysisOut:
    async with db.tenant_tx(principal.tenant_id) as conn:
        row = await require_analysis(conn, principal, analysis_id)
        claims = await analyses.get_claims(conn, analysis_id)
        claim_ids = [claim["id"] for claim in claims]
        evidence = await analyses.get_evidence(conn, claim_ids)
        checks = await analyses.latest_checks(conn, claim_ids)
        citations = await analyses.get_citations(conn, analysis_id)
        passages = await _citation_passages(conn, citations)
        collection_ids = {item["collection_id"] for item in evidence}
        collection_ids |= {passage["collection_id"] for passage in passages.values()}
        readable = await readable_subset(conn, principal, collection_ids)
        # Only what the viewer may read is fetched, so a withheld span never leaves the database.
        span_texts = await _span_texts(
            conn, [item for item in evidence if item["collection_id"] in readable]
        )

    by_claim: dict[UUID, list[EvidenceOut]] = {}
    for item in evidence:
        by_claim.setdefault(item["claim_id"], []).append(
            _evidence_out(item, item["collection_id"] in readable, span_texts.get(item["id"]))
        )
    return AnalysisOut(
        id=row["id"],
        kind=row["kind"],
        question=row["question"],
        answer=row["answer"],
        status=row["status"],
        error_code=row["error_code"],
        progress=row["progress"],
        live=row["as_of"] is None,
        as_of=row["as_of"],
        known_at=row["known_at"],
        collection_ids=list(row["collection_ids"]),
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        config=row["config"],
        timings=row["timings"],
        summary=row["summary"],
        claims=[
            _claim_out(claim, by_claim.get(claim["id"], []), checks.get(claim["id"]))
            for claim in claims
        ],
        citations=[
            _citation_out(
                citation, passages.get((citation["passage_id"], citation["version_id"])), readable
            )
            for citation in citations
        ],
    )


@router.get("/analyses/{analysis_id}/graph", response_model=GraphOut)
async def get_analysis_graph(
    analysis_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> dict[str, list[dict[str, Any]]]:
    async with db.tenant_tx(principal.tenant_id) as conn:
        await require_analysis(conn, principal, analysis_id)
        collection_ids = await passage_collection_ids(conn, analysis_id)
        readable = await readable_subset(conn, principal, collection_ids)
        return await build_graph(conn, analysis_id, readable)


@router.post("/claims/{claim_id}/feedback", response_model=FeedbackOut, status_code=201)
async def post_feedback(
    claim_id: UUID,
    body: FeedbackIn,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> FeedbackOut:
    async with db.tenant_tx(principal.tenant_id) as conn:
        claim = await _require_claim(conn, principal, claim_id)
        feedback_id = await analyses.upsert_feedback(
            conn,
            tenant_id=claim["tenant_id"],
            claim_id=claim_id,
            user_id=principal.user_id,
            label=body.label,
            note=body.note,
        )
    return FeedbackOut(id=feedback_id)


@router.get("/claims/{claim_id}/checks", response_model=list[ClaimCheckOut])
async def list_claim_checks(
    claim_id: UUID,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> list[ClaimCheckOut]:
    async with db.tenant_tx(principal.tenant_id) as conn:
        await _require_claim(conn, principal, claim_id)
        cur = await conn.execute(
            "select id, reason, verdict, p_supported, p_contradicted, p_insufficient, changed, "
            "checked_version_ids, created_at from claim_checks where claim_id = %s "
            "order by created_at desc, id desc",
            (claim_id,),
        )
        rows = await cur.fetchall()
    return [_check_out(row) for row in rows]


async def _require_claim(
    conn: AsyncConnection, principal: Principal, claim_id: UUID
) -> dict[str, Any]:
    """The claim row when its analysis is visible to the caller, else 404 ``claim_not_found``."""
    cur = await conn.execute(
        "select id, tenant_id, analysis_id from claims where id = %s", (claim_id,)
    )
    claim = await cur.fetchone()
    if claim is None:
        raise ApiError(404, "claim_not_found", "Claim not found")
    try:
        await require_analysis(conn, principal, claim["analysis_id"])
    except ApiError as exc:
        # The analysis rule decides; the caller learns only that the claim is not theirs.
        raise ApiError(404, "claim_not_found", "Claim not found") from exc
    return claim


def _summary_out(row: Mapping[str, Any], changed: bool) -> AnalysisSummaryOut:
    question = row["question"]
    return AnalysisSummaryOut(
        id=row["id"],
        kind=row["kind"],
        question_preview=None if question is None else question[:QUESTION_PREVIEW_CHARS],
        status=row["status"],
        error_code=row["error_code"],
        live=row["as_of"] is None,
        as_of=row["as_of"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        summary=row["summary"],
        changed_since=changed,
    )


def _probs(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "supported": row["p_supported"],
        "contradicted": row["p_contradicted"],
        "insufficient": row["p_insufficient"],
    }


def _claim_out(
    claim: Mapping[str, Any], evidence: list[EvidenceOut], check: Mapping[str, Any] | None
) -> ClaimOut:
    relations = {item.relation for item in evidence}
    return ClaimOut(
        id=claim["id"],
        ordinal=claim["ordinal"],
        text=claim["text"],
        answer_start=claim["answer_start"],
        answer_end=claim["answer_end"],
        extraction_method=claim["extraction_method"],
        verdict=claim["verdict"],
        probs=_probs(claim),
        features=claim["features"],
        diagnostics=claim["diagnostics"],
        # Judged on every item, redacted or not: a withheld source still disagrees.
        conflict={"supports", "contradicts"} <= relations,
        evidence=evidence,
        latest_check=None if check is None else _check_out(check),
    )


def _check_out(check: Mapping[str, Any]) -> ClaimCheckOut:
    return ClaimCheckOut(
        id=check["id"],
        reason=check["reason"],
        verdict=check["verdict"],
        probs=_probs(check),
        changed=check["changed"],
        checked_version_ids=check["checked_version_ids"],
        created_at=check["created_at"],
    )


def _evidence_out(item: Mapping[str, Any], readable: bool, span_text: str | None) -> EvidenceOut:
    """The evidence item; a collection the viewer cannot read leaves only ids and numbers."""

    def shown(column: str) -> Any:
        return item[column] if readable else None

    return EvidenceOut(
        passage_id=item["passage_id"],
        version_id=item["version_id"],
        document_id=shown("document_id"),
        document_title=shown("document_title"),
        version_number=shown("version_number"),
        effective_from=shown("effective_from"),
        page_start=shown("page_start"),
        page_end=shown("page_end"),
        relation=item["relation"],
        rank=item["rank"],
        span_start=item["span_start"],
        span_end=item["span_end"],
        nli=NLIOut(
            entail=item["nli_entail"], contra=item["nli_contra"], neutral=item["nli_neutral"]
        ),
        rerank_score=item["rerank_score"],
        text=shown("text"),
        span_text=span_text if readable else None,
        passage_is_current=shown("passage_is_current"),
        redacted=not readable,
    )


def _citation_out(
    citation: Mapping[str, Any], passage: Mapping[str, Any] | None, readable: set[UUID]
) -> CitationOut:
    # An unknown marker has no passage, so there is nothing to withhold.
    redacted = passage is not None and passage["collection_id"] not in readable
    shown = passage is not None and not redacted
    return CitationOut(
        number=citation["number"],
        marker_start=citation["marker_start"],
        marker_end=citation["marker_end"],
        sentence_start=citation["sentence_start"],
        sentence_end=citation["sentence_end"],
        status=citation["status"],
        passage_id=citation["passage_id"],
        version_id=citation["version_id"],
        document_title=passage["document_title"] if shown else None,
        version_number=passage["version_number"] if shown else None,
        redacted=redacted,
    )


async def _with_changed_checks(conn: AsyncConnection, analysis_ids: Sequence[UUID]) -> set[UUID]:
    """The analyses among ``analysis_ids`` with a claim re-check that changed a verdict."""
    cur = await conn.execute(
        "select distinct c.analysis_id from claims c "
        "join claim_checks k on k.claim_id = c.id "
        "where c.analysis_id = any(%s::uuid[]) and k.changed",
        (list(analysis_ids),),
    )
    return {row["analysis_id"] for row in await cur.fetchall()}


async def _citation_passages(
    conn: AsyncConnection, citations: Sequence[Mapping[str, Any]]
) -> dict[tuple[UUID, UUID], dict[str, Any]]:
    """Collection, document title, and version number of each cited (passage, version) pair."""
    passage_ids = [c["passage_id"] for c in citations if c["passage_id"] is not None]
    cur = await conn.execute(
        "select p.id as passage_id, p.version_id, p.collection_id, d.title as document_title, "
        "v.version_number "
        "from passages p "
        "join document_versions v on v.id = p.version_id "
        "join documents d on d.id = v.document_id "
        "where p.id = any(%s::uuid[])",
        (passage_ids,),
    )
    return {(row["passage_id"], row["version_id"]): row for row in await cur.fetchall()}


async def _span_texts(
    conn: AsyncConnection, evidence: Sequence[Mapping[str, Any]]
) -> dict[UUID, str]:
    """The version text at each item's span, by evidence link id, sliced in the database."""
    if not evidence:
        return {}
    cur = await conn.execute(
        "select t.link_id, substr(v.text, t.span_start + 1, t.span_end - t.span_start) as piece "
        "from unnest(%s::uuid[], %s::uuid[], %s::int[], %s::int[]) "
        "as t(link_id, version_id, span_start, span_end) "
        "join document_versions v on v.id = t.version_id",
        (
            [item["id"] for item in evidence],
            [item["version_id"] for item in evidence],
            [item["span_start"] for item in evidence],
            [item["span_end"] for item in evidence],
        ),
    )
    return {row["link_id"]: row["piece"] for row in await cur.fetchall()}
