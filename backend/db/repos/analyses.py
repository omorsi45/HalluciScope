"""Analyses repository: analyses, claims, evidence links, citations, re-checks, and feedback.

Connections come from ``tenant_tx``. ``save_results`` writes inside the caller's transaction, so
the claims, their evidence, the citations, and the ``succeeded`` status commit together.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

if TYPE_CHECKING:
    from datetime import datetime

    from backend.analysis.service import CitationRecord
    from backend.api.pagination import PageParams
    from backend.verify.types import ClaimVerdict

_TERMINAL_STATUSES = ("succeeded", "failed")

_ANALYSIS_COLUMNS = (
    "id, user_id, kind, question, answer, collection_ids, as_of, known_at, status, progress, "
    "error_code, options, config, timings, summary, created_at, completed_at"
)

_CLAIM_COLUMNS = (
    "id, analysis_id, ordinal, text, answer_start, answer_end, extraction_method, verdict, "
    "p_supported, p_contradicted, p_insufficient, features, diagnostics, created_at"
)

# Each link joins the exact (passage, version) pair it stores, so the text, document, and
# version shown are the ones the verdict was reached on, even after newer versions exist.
_EVIDENCE_SQL = (
    "select e.id, e.claim_id, e.passage_id, e.version_id, e.relation, e.rank, e.span_start, "
    "e.span_end, e.nli_entail, e.nli_contra, e.nli_neutral, e.rerank_score, e.dense_cos, "
    "p.collection_id, p.text, p.start_char, p.end_char, p.page_start, p.page_end, "
    "v.version_number, v.effective_from, v.document_id, d.title as document_title, "
    "p.is_current as passage_is_current, p.text_sha256 "
    "from evidence_links e "
    "join passages p on p.id = e.passage_id and p.version_id = e.version_id "
    "join document_versions v on v.id = e.version_id "
    "join documents d on d.id = v.document_id "
    "where e.claim_id = any(%s::uuid[]) "
    "order by e.claim_id, e.rank"
)

_CITATION_COLUMNS = (
    "id, analysis_id, number, marker_start, marker_end, sentence_start, sentence_end, "
    "passage_id, version_id, status"
)

_CHECK_COLUMNS = frozenset(
    {
        "tenant_id",
        "claim_id",
        "reason",
        "verdict",
        "p_supported",
        "p_contradicted",
        "p_insufficient",
        "evidence",
        "checked_version_ids",
        "changed",
    }
)


async def create_analysis(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    user_id: UUID,
    kind: str,
    question: str | None,
    answer: str | None,
    collection_ids: Sequence[UUID],
    as_of: "datetime | None",
    known_at: "datetime",
    options: dict[str, Any],
) -> UUID:
    cur = await conn.execute(
        "insert into analyses (tenant_id, user_id, kind, question, answer, collection_ids, "
        "as_of, known_at, options) values (%s, %s, %s, %s, %s, %s::uuid[], %s, %s, %s) "
        "returning id",
        (
            tenant_id,
            user_id,
            kind,
            question,
            answer,
            list(collection_ids),
            as_of,
            known_at,
            Jsonb(options),
        ),
    )
    return (await cur.fetchone())["id"]


async def count_active(conn: AsyncConnection, user_id: UUID) -> int:
    """Analyses of the user that are queued or running."""
    cur = await conn.execute(
        "select count(*) as n from analyses where user_id = %s and status in ('queued', 'running')",
        (user_id,),
    )
    return (await cur.fetchone())["n"]


async def get_analysis(conn: AsyncConnection, analysis_id: UUID) -> dict[str, Any] | None:
    cur = await conn.execute(
        f"select {_ANALYSIS_COLUMNS} from analyses where id = %s", (analysis_id,)
    )
    return await cur.fetchone()


async def list_analyses(
    conn: AsyncConnection, *, user_id: UUID | None, page: "PageParams"
) -> list[dict[str, Any]]:
    """Analyses newest first; ``user_id=None`` lists the whole tenant.

    Returns up to ``page.limit + 1`` rows so the caller can tell whether another page exists.
    The cursor key must already be validated as a UUID (``decode_uuid_cursor``).
    """
    conditions: list[str] = []
    params: list[Any] = []
    if user_id is not None:
        conditions.append("user_id = %s")
        params.append(user_id)
    if page.after:
        ts, key = page.after
        conditions.append("(created_at, id) < (%s, %s::uuid)")
        params.extend((ts, str(key)))
    where_sql = f"where {' and '.join(conditions)} " if conditions else ""
    params.append(page.limit + 1)
    cur = await conn.execute(
        f"select {_ANALYSIS_COLUMNS} from analyses {where_sql}"
        "order by created_at desc, id desc limit %s",
        params,
    )
    return await cur.fetchall()


async def set_status(
    conn: AsyncConnection, analysis_id: UUID, status: str, *, error_code: str | None = None
) -> None:
    """Set the status; a terminal status (succeeded, failed) also stamps ``completed_at``."""
    await conn.execute(
        "update analyses set status = %s, error_code = %s, "
        "completed_at = case when %s then now() end where id = %s",
        (status, error_code, status in _TERMINAL_STATUSES, analysis_id),
    )


async def set_progress(conn: AsyncConnection, analysis_id: UUID, stage: str) -> None:
    """Make ``stage`` the current stage and move the previous one into ``stages_done``."""
    await conn.execute(
        "update analyses set progress = jsonb_build_object("
        "'stage', %(stage)s::text, "
        "'stages_done', coalesce(progress -> 'stages_done', '[]'::jsonb) || "
        "case when progress ->> 'stage' is not null and progress ->> 'stage' <> %(stage)s::text "
        "then jsonb_build_array(progress -> 'stage') else '[]'::jsonb end) "
        "where id = %(id)s",
        {"stage": stage, "id": analysis_id},
    )


async def save_results(
    conn: AsyncConnection,
    analysis_id: UUID,
    *,
    answer: str,
    verdicts: Sequence["ClaimVerdict"],
    citations: Sequence["CitationRecord"],
    config: dict[str, Any],
    timings: dict[str, Any],
    summary: dict[str, Any],
) -> list[UUID]:
    """Store the claims, evidence links, and citations and mark the analysis succeeded.

    Runs in the caller's transaction. Returns the claim ids in ordinal order. A claim without
    evidence is stored as is. Raises ``LookupError`` when the analysis is not visible.
    """
    cur = await conn.execute(
        "update analyses set status = 'succeeded', answer = %s, config = %s, timings = %s, "
        "summary = %s, completed_at = now() where id = %s returning tenant_id",
        (answer, Jsonb(config), Jsonb(timings), Jsonb(summary), analysis_id),
    )
    analysis = await cur.fetchone()
    if analysis is None:
        raise LookupError("analysis not found")
    tenant_id = analysis["tenant_id"]

    claim_ids: list[UUID] = []
    link_rows: list[tuple[Any, ...]] = []
    for verdict in sorted(verdicts, key=lambda item: item.claim.ordinal):
        claim = verdict.claim
        cur = await conn.execute(
            "insert into claims (tenant_id, analysis_id, ordinal, text, answer_start, "
            "answer_end, extraction_method, verdict, p_supported, p_contradicted, "
            "p_insufficient, features, diagnostics) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) returning id",
            (
                tenant_id,
                analysis_id,
                claim.ordinal,
                claim.text,
                claim.answer_start,
                claim.answer_end,
                claim.method,
                verdict.verdict.value,
                verdict.probs["supported"],
                verdict.probs["contradicted"],
                verdict.probs["insufficient"],
                Jsonb(verdict.features),
                Jsonb(verdict.diagnostics),
            ),
        )
        claim_id = (await cur.fetchone())["id"]
        claim_ids.append(claim_id)
        link_rows.extend(
            (
                tenant_id,
                claim_id,
                item.hit.passage_id,
                item.hit.version_id,
                item.relation,
                rank,
                item.span_start,
                item.span_end,
                item.nli.entail,
                item.nli.contra,
                item.nli.neutral,
                item.rerank,
                item.dense_cos,
            )
            for rank, item in enumerate(verdict.evidence, start=1)
        )

    async with conn.cursor() as bulk:
        if link_rows:
            await bulk.executemany(
                "insert into evidence_links (tenant_id, claim_id, passage_id, version_id, "
                "relation, rank, span_start, span_end, nli_entail, nli_contra, nli_neutral, "
                "rerank_score, dense_cos) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                link_rows,
            )
        if citations:
            await bulk.executemany(
                "insert into citations (tenant_id, analysis_id, number, marker_start, "
                "marker_end, sentence_start, sentence_end, passage_id, version_id, status) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        tenant_id,
                        analysis_id,
                        citation.number,
                        citation.marker_start,
                        citation.marker_end,
                        citation.sentence_start,
                        citation.sentence_end,
                        citation.passage_id,
                        citation.version_id,
                        citation.status,
                    )
                    for citation in citations
                ],
            )
    return claim_ids


async def get_claims(conn: AsyncConnection, analysis_id: UUID) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"select {_CLAIM_COLUMNS} from claims where analysis_id = %s order by ordinal",
        (analysis_id,),
    )
    return await cur.fetchall()


async def get_evidence(conn: AsyncConnection, claim_ids: Sequence[UUID]) -> list[dict[str, Any]]:
    """Evidence links of the claims, by claim then rank, with their passage, version, document."""
    cur = await conn.execute(_EVIDENCE_SQL, (list(claim_ids),))
    return await cur.fetchall()


async def get_citations(conn: AsyncConnection, analysis_id: UUID) -> list[dict[str, Any]]:
    cur = await conn.execute(
        f"select {_CITATION_COLUMNS} from citations where analysis_id = %s "
        "order by marker_start, number, id",
        (analysis_id,),
    )
    return await cur.fetchall()


async def insert_claim_check(conn: AsyncConnection, **fields: Any) -> UUID:
    """Append a re-verification row. ``fields`` are the ``claim_checks`` columns to set."""
    unexpected = fields.keys() - _CHECK_COLUMNS
    if unexpected:
        raise TypeError(f"unexpected claim check fields: {sorted(unexpected)}")
    if "evidence" in fields:
        fields["evidence"] = Jsonb(fields["evidence"])
    # Column names come from the whitelist above; values are bound parameters.
    columns = sorted(fields)
    placeholders = ", ".join(["%s"] * len(columns))
    cur = await conn.execute(
        f"insert into claim_checks ({', '.join(columns)}) values ({placeholders}) returning id",
        [fields[column] for column in columns],
    )
    return (await cur.fetchone())["id"]


async def latest_checks(
    conn: AsyncConnection, claim_ids: Sequence[UUID]
) -> dict[UUID, dict[str, Any]]:
    """The newest ``claim_checks`` row of each claim that has one."""
    cur = await conn.execute(
        "select distinct on (claim_id) id, claim_id, reason, verdict, p_supported, "
        "p_contradicted, p_insufficient, evidence, checked_version_ids, changed, created_at "
        "from claim_checks where claim_id = any(%s::uuid[]) "
        "order by claim_id, created_at desc, id desc",
        (list(claim_ids),),
    )
    return {row["claim_id"]: row for row in await cur.fetchall()}


async def upsert_feedback(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    claim_id: UUID,
    user_id: UUID,
    label: str,
    note: str | None,
) -> UUID:
    """One feedback row per user per claim: a repeat call replaces the label and note."""
    cur = await conn.execute(
        "insert into feedback (tenant_id, claim_id, user_id, label, note) "
        "values (%s, %s, %s, %s, %s) "
        "on conflict (claim_id, user_id) do update "
        "set label = excluded.label, note = excluded.note, updated_at = now() "
        "returning id",
        (tenant_id, claim_id, user_id, label, note),
    )
    return (await cur.fetchone())["id"]
