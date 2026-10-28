"""Schema and repository tests for analyses, claims, evidence links, citations, and feedback."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
import pytest
from psycopg.types.json import Jsonb

from backend.analysis.service import CitationRecord
from backend.api.pagination import PageParams
from backend.db.repos import analyses
from backend.retrieval.types import PassageHit
from backend.verify.types import Claim, ClaimVerdict, EvidenceItem, NLIResult, Verdict
from tests.test_rls import GLOBAL_TABLES, TENANT_POLICY
from tests.test_schema_documents import (
    Seeded,
    create_tenant,
    insert_passage,
    insert_version,
    sha256,
)

pytestmark = pytest.mark.pg

NEW_TABLES = {"analyses", "claims", "evidence_links", "citations", "claim_checks", "feedback"}

TEXT = "Refunds take 30 days. Exchanges take 14 days."
FIRST = TEXT[:21]  # "Refunds take 30 days."
SECOND = TEXT[22:]  # "Exchanges take 14 days."


@dataclass(frozen=True)
class World:
    """One tenant with a document version, two passages of it, and a user."""

    seeded: Seeded
    user_id: uuid.UUID
    passages: tuple[uuid.UUID, uuid.UUID]

    @property
    def tenant_id(self) -> uuid.UUID:
        return self.seeded.tenant_id

    @property
    def version_id(self) -> uuid.UUID:
        return self.seeded.version_id


async def create_user(db, email: str) -> uuid.UUID:
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "insert into users (email, password_hash, display_name) values (%s, 'x', %s) "
            "returning id",
            (email, email),
        )
        return (await cur.fetchone())["id"]


async def make_world(db, name: str = "acme") -> World:
    tenant_id = await create_tenant(db, name)
    seeded = await insert_version(db, TEXT, tenant_id=tenant_id)
    user_id = await create_user(db, f"{name}-{uuid.uuid4().hex[:8]}@example.com")
    async with db.tenant_tx(tenant_id) as conn:
        first = await insert_passage(conn, seeded, FIRST, 0, len(FIRST))
        second = await insert_passage(conn, seeded, SECOND, 22, len(TEXT), ordinal=1)
    return World(seeded, user_id, (first, second))


async def insert_analysis_row(conn, world: World) -> uuid.UUID:
    cur = await conn.execute(
        "insert into analyses (tenant_id, user_id, kind, collection_ids, known_at) "
        "values (%s, %s, 'ask', %s, now()) returning id",
        (world.tenant_id, world.user_id, [world.seeded.collection_id]),
    )
    return (await cur.fetchone())["id"]


async def insert_claim_row(
    conn, tenant_id: uuid.UUID, analysis_id: uuid.UUID, *, ordinal: int = 0
) -> uuid.UUID:
    cur = await conn.execute(
        "insert into claims (tenant_id, analysis_id, ordinal, text, answer_start, answer_end, "
        "extraction_method, verdict, p_supported, p_contradicted, p_insufficient, features) "
        "values (%s, %s, %s, %s, 0, 21, 'sentence', 'supported', 0.8, 0.1, 0.1, %s) returning id",
        (tenant_id, analysis_id, ordinal, FIRST, Jsonb({})),
    )
    return (await cur.fetchone())["id"]


async def insert_link_row(
    conn,
    tenant_id: uuid.UUID,
    claim_id: uuid.UUID,
    passage_id: uuid.UUID,
    version_id: uuid.UUID,
    *,
    span: tuple[int, int] = (0, 7),
) -> uuid.UUID:
    cur = await conn.execute(
        "insert into evidence_links (tenant_id, claim_id, passage_id, version_id, relation, "
        "rank, span_start, span_end, nli_entail, nli_contra, nli_neutral, rerank_score, "
        "dense_cos) values (%s, %s, %s, %s, 'supports', 1, %s, %s, 0.9, 0.05, 0.05, null, 0.7) "
        "returning id",
        (tenant_id, claim_id, passage_id, version_id, span[0], span[1]),
    )
    return (await cur.fetchone())["id"]


async def insert_citation_row(
    conn,
    tenant_id: uuid.UUID,
    analysis_id: uuid.UUID,
    *,
    status: str,
    passage_id: uuid.UUID | None = None,
    version_id: uuid.UUID | None = None,
) -> uuid.UUID:
    cur = await conn.execute(
        "insert into citations (tenant_id, analysis_id, number, marker_start, marker_end, "
        "sentence_start, sentence_end, passage_id, version_id, status) "
        "values (%s, %s, 1, 22, 26, 0, 21, %s, %s, %s) returning id",
        (tenant_id, analysis_id, passage_id, version_id, status),
    )
    return (await cur.fetchone())["id"]


async def insert_check_row(conn, tenant_id: uuid.UUID, claim_id: uuid.UUID) -> uuid.UUID:
    cur = await conn.execute(
        "insert into claim_checks (tenant_id, claim_id, reason, verdict, p_supported, "
        "p_contradicted, p_insufficient, checked_version_ids, changed) "
        "values (%s, %s, 'source_changed', 'insufficient', 0.1, 0.1, 0.8, %s, true) returning id",
        (tenant_id, claim_id, []),
    )
    return (await cur.fetchone())["id"]


async def insert_feedback_row(
    conn, tenant_id: uuid.UUID, claim_id: uuid.UUID, user_id: uuid.UUID
) -> uuid.UUID:
    cur = await conn.execute(
        "insert into feedback (tenant_id, claim_id, user_id, label) "
        "values (%s, %s, %s, 'correct') returning id",
        (tenant_id, claim_id, user_id),
    )
    return (await cur.fetchone())["id"]


@dataclass(frozen=True)
class Parents:
    """A world with one analysis and one claim, the parents other rows point at."""

    world: World
    analysis_id: uuid.UUID
    claim_id: uuid.UUID

    @property
    def tenant_id(self) -> uuid.UUID:
        return self.world.tenant_id


async def make_parents(db, name: str) -> Parents:
    world = await make_world(db, name)
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await insert_analysis_row(conn, world)
        claim_id = await insert_claim_row(conn, world.tenant_id, analysis_id)
    return Parents(world, analysis_id, claim_id)


async def test_new_tables_forced_rls(db):
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select c.relname, c.relrowsecurity, c.relforcerowsecurity "
            "from pg_class c join pg_namespace n on n.oid = c.relnamespace "
            "where c.relkind in ('r', 'p') and n.nspname = 'public'"
        )
        tables = {row["relname"]: row for row in await cur.fetchall()}
        cur = await conn.execute(
            "select tablename, qual, with_check from pg_policies "
            "where schemaname = 'public' and policyname = 'tenant_isolation'"
        )
        policies = {row["tablename"]: row for row in await cur.fetchall()}

    assert tables.keys() >= NEW_TABLES
    assert NEW_TABLES.isdisjoint(GLOBAL_TABLES)
    unprotected = sorted(
        name
        for name in NEW_TABLES
        if not (tables[name]["relrowsecurity"] and tables[name]["relforcerowsecurity"])
    )
    assert unprotected == []
    assert sorted(NEW_TABLES - policies.keys()) == []
    wrong_policy = sorted(
        name
        for name in NEW_TABLES
        if (policies[name]["qual"], policies[name]["with_check"]) != (TENANT_POLICY, TENANT_POLICY)
    )
    assert wrong_policy == []


async def test_same_tenant_rows_accepted(db):
    """Control for the cross-tenant tests: the same inserts succeed inside one tenant."""
    parents = await make_parents(db, "acme")
    world = parents.world
    async with db.tenant_tx(world.tenant_id) as conn:
        await insert_link_row(
            conn, world.tenant_id, parents.claim_id, world.passages[0], world.version_id
        )
        await insert_citation_row(
            conn,
            world.tenant_id,
            parents.analysis_id,
            status="valid",
            passage_id=world.passages[0],
            version_id=world.version_id,
        )
        await insert_check_row(conn, world.tenant_id, parents.claim_id)
        await insert_feedback_row(conn, world.tenant_id, parents.claim_id, world.user_id)


async def test_evidence_link_version_must_match_passage(db):
    world = await make_world(db)
    other_version = await insert_version(db, TEXT, tenant_id=world.tenant_id)
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await insert_analysis_row(conn, world)
        claim_id = await insert_claim_row(conn, world.tenant_id, analysis_id)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        async with db.tenant_tx(world.tenant_id) as conn:
            await insert_link_row(
                conn, world.tenant_id, claim_id, world.passages[0], other_version.version_id
            )


async def test_evidence_span_must_be_non_empty(db):
    parents = await make_parents(db, "acme")
    world = parents.world
    with pytest.raises(psycopg.errors.CheckViolation):
        async with db.tenant_tx(world.tenant_id) as conn:
            await insert_link_row(
                conn,
                world.tenant_id,
                parents.claim_id,
                world.passages[0],
                world.version_id,
                span=(7, 7),
            )


async def test_citation_valid_with_passage_and_version(db):
    parents = await make_parents(db, "acme")
    world = parents.world
    async with db.tenant_tx(world.tenant_id) as conn:
        await insert_citation_row(
            conn,
            world.tenant_id,
            parents.analysis_id,
            status="valid",
            passage_id=world.passages[0],
            version_id=world.version_id,
        )


async def test_citation_unknown_marker_without_passage(db):
    parents = await make_parents(db, "acme")
    async with db.tenant_tx(parents.tenant_id) as conn:
        await insert_citation_row(
            conn, parents.tenant_id, parents.analysis_id, status="unknown_marker"
        )


async def test_citation_valid_without_passage_rejected(db):
    parents = await make_parents(db, "acme")
    with pytest.raises(psycopg.errors.CheckViolation):
        async with db.tenant_tx(parents.tenant_id) as conn:
            await insert_citation_row(conn, parents.tenant_id, parents.analysis_id, status="valid")


async def test_citation_valid_without_version_rejected(db):
    parents = await make_parents(db, "acme")
    with pytest.raises(psycopg.errors.CheckViolation):
        async with db.tenant_tx(parents.tenant_id) as conn:
            await insert_citation_row(
                conn,
                parents.tenant_id,
                parents.analysis_id,
                status="valid",
                passage_id=parents.world.passages[0],
            )


async def test_citation_unknown_marker_with_passage_rejected(db):
    parents = await make_parents(db, "acme")
    world = parents.world
    with pytest.raises(psycopg.errors.CheckViolation):
        async with db.tenant_tx(world.tenant_id) as conn:
            await insert_citation_row(
                conn,
                world.tenant_id,
                parents.analysis_id,
                status="unknown_marker",
                passage_id=world.passages[0],
                version_id=world.version_id,
            )


async def test_claim_checks_append_only(db):
    parents = await make_parents(db, "acme")
    async with db.tenant_tx(parents.tenant_id) as conn:
        check_id = await insert_check_row(conn, parents.tenant_id, parents.claim_id)
    with pytest.raises(psycopg.errors.RaiseException, match="claim_checks are append-only"):
        async with db.tenant_tx(parents.tenant_id) as conn:
            await conn.execute("update claim_checks set changed = false where id = %s", (check_id,))


async def test_deleting_analysis_removes_its_rows(db):
    parents = await make_parents(db, "acme")
    world = parents.world
    async with db.tenant_tx(world.tenant_id) as conn:
        await insert_link_row(
            conn, world.tenant_id, parents.claim_id, world.passages[0], world.version_id
        )
        await insert_citation_row(
            conn,
            world.tenant_id,
            parents.analysis_id,
            status="valid",
            passage_id=world.passages[0],
            version_id=world.version_id,
        )
        await insert_check_row(conn, world.tenant_id, parents.claim_id)
        await insert_feedback_row(conn, world.tenant_id, parents.claim_id, world.user_id)
        await conn.execute("delete from analyses where id = %s", (parents.analysis_id,))
        counts = {}
        for table in ("claims", "evidence_links", "citations", "claim_checks", "feedback"):
            cur = await conn.execute(f"select count(*) as n from {table}")  # constant names
            counts[table] = (await cur.fetchone())["n"]
    assert counts == {
        "claims": 0,
        "evidence_links": 0,
        "citations": 0,
        "claim_checks": 0,
        "feedback": 0,
    }


async def _claim_to_other_analysis(conn, mine: Parents, theirs: Parents):
    await insert_claim_row(conn, mine.tenant_id, theirs.analysis_id, ordinal=1)


async def _citation_to_other_analysis(conn, mine: Parents, theirs: Parents):
    await insert_citation_row(conn, mine.tenant_id, theirs.analysis_id, status="unknown_marker")


async def _link_to_other_claim(conn, mine: Parents, theirs: Parents):
    await insert_link_row(
        conn, mine.tenant_id, theirs.claim_id, mine.world.passages[0], mine.world.version_id
    )


async def _link_to_other_passage(conn, mine: Parents, theirs: Parents):
    await insert_link_row(
        conn, mine.tenant_id, mine.claim_id, theirs.world.passages[0], theirs.world.version_id
    )


async def _citation_to_other_passage(conn, mine: Parents, theirs: Parents):
    await insert_citation_row(
        conn,
        mine.tenant_id,
        mine.analysis_id,
        status="valid",
        passage_id=theirs.world.passages[0],
        version_id=theirs.world.version_id,
    )


async def _check_to_other_claim(conn, mine: Parents, theirs: Parents):
    await insert_check_row(conn, mine.tenant_id, theirs.claim_id)


async def _feedback_to_other_claim(conn, mine: Parents, theirs: Parents):
    await insert_feedback_row(conn, mine.tenant_id, theirs.claim_id, mine.world.user_id)


CROSS_TENANT_CASES = {
    "claim_to_analysis": _claim_to_other_analysis,
    "citation_to_analysis": _citation_to_other_analysis,
    "evidence_link_to_claim": _link_to_other_claim,
    "evidence_link_to_passage": _link_to_other_passage,
    "citation_to_passage": _citation_to_other_passage,
    "claim_check_to_claim": _check_to_other_claim,
    "feedback_to_claim": _feedback_to_other_claim,
}


@pytest.mark.parametrize("case", sorted(CROSS_TENANT_CASES))
async def test_row_pointing_at_other_tenants_parent_rejected(db, case):
    theirs = await make_parents(db, "globex")
    mine = await make_parents(db, "acme")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        async with db.tenant_tx(mine.tenant_id) as conn:
            await CROSS_TENANT_CASES[case](conn, mine, theirs)


# Repository tests

ANSWER = "Refunds take 30 days. [P1]"


async def new_analysis(
    conn, world: World, *, user_id: uuid.UUID | None = None, kind: str = "ask"
) -> uuid.UUID:
    return await analyses.create_analysis(
        conn,
        tenant_id=world.tenant_id,
        user_id=user_id or world.user_id,
        kind=kind,
        question="How long do refunds take?",
        answer=None,
        collection_ids=[world.seeded.collection_id],
        as_of=None,
        known_at=datetime.now(UTC),
        options={"extraction": "sentence"},
    )


def passage_hit(world: World, index: int) -> PassageHit:
    text, start = ((FIRST, 0), (SECOND, 22))[index]
    return PassageHit(
        passage_id=world.passages[index],
        document_id=world.seeded.document_id,
        version_id=world.version_id,
        version_number=1,
        effective_from=datetime.now(UTC),
        document_title="Policy",
        text=text,
        start_char=start,
        end_char=start + len(text),
        page_start=None,
        page_end=None,
        section=None,
        scores={},
        rank=index + 1,
    )


def evidence_item(
    world: World,
    index: int,
    *,
    relation: str,
    span: tuple[int, int],
    rerank: float | None,
    dense_cos: float,
) -> EvidenceItem:
    return EvidenceItem(
        hit=passage_hit(world, index),
        nli=NLIResult(entail=0.6, contra=0.3, neutral=0.1),
        relation=relation,
        span_start=span[0],
        span_end=span[1],
        dense_cos=dense_cos,
        rerank=rerank,
    )


def claim_verdict(
    ordinal: int, *, evidence: tuple[EvidenceItem, ...] = (), verdict: Verdict = Verdict.SUPPORTED
) -> ClaimVerdict:
    return ClaimVerdict(
        claim=Claim(
            ordinal=ordinal,
            text=f"claim {ordinal}",
            answer_start=ordinal * 25,
            answer_end=ordinal * 25 + 21,
            method="sentence",
            cited_numbers=(1,),
        ),
        verdict=verdict,
        probs={"supported": 0.7, "contradicted": 0.2, "insufficient": 0.1},
        features={"max_entail": 0.9},
        evidence=evidence,
        signals=(),
        diagnostics={"note": "hand built"},
    )


async def save(conn, analysis_id, verdicts, citations=()):
    return await analyses.save_results(
        conn,
        analysis_id,
        answer=ANSWER,
        verdicts=verdicts,
        citations=citations,
        config={"nli_model": "fake-nli"},
        timings={"total_ms": 12.5},
        summary={"supported": len(verdicts)},
    )


async def test_create_and_get_analysis(db):
    world = await make_world(db)
    as_of = datetime(2026, 9, 1, tzinfo=UTC)
    known_at = datetime(2026, 10, 1, tzinfo=UTC)
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await analyses.create_analysis(
            conn,
            tenant_id=world.tenant_id,
            user_id=world.user_id,
            kind="verify",
            question=None,
            answer=ANSWER,
            collection_ids=[world.seeded.collection_id],
            as_of=as_of,
            known_at=known_at,
            options={"extraction": "sentence"},
        )
        row = await analyses.get_analysis(conn, analysis_id)
    assert row is not None
    assert {
        key: row[key]
        for key in (
            "id",
            "user_id",
            "kind",
            "question",
            "answer",
            "collection_ids",
            "as_of",
            "known_at",
            "status",
            "progress",
            "error_code",
            "options",
            "completed_at",
        )
    } == {
        "id": analysis_id,
        "user_id": world.user_id,
        "kind": "verify",
        "question": None,
        "answer": ANSWER,
        "collection_ids": [world.seeded.collection_id],
        "as_of": as_of,
        "known_at": known_at,
        "status": "queued",
        "progress": {},
        "error_code": None,
        "options": {"extraction": "sentence"},
        "completed_at": None,
    }


async def test_get_analysis_missing_or_other_tenant_is_none(db):
    theirs = await make_world(db, "globex")
    mine = await make_world(db, "acme")
    async with db.tenant_tx(theirs.tenant_id) as conn:
        analysis_id = await new_analysis(conn, theirs)
    async with db.tenant_tx(mine.tenant_id) as conn:
        assert await analyses.get_analysis(conn, analysis_id) is None
        assert await analyses.get_analysis(conn, uuid.uuid4()) is None


async def test_count_active(db):
    world = await make_world(db)
    other_user = await create_user(db, "other@example.com")
    async with db.tenant_tx(world.tenant_id) as conn:
        queued = await new_analysis(conn, world)
        running = await new_analysis(conn, world)
        succeeded = await new_analysis(conn, world)
        failed = await new_analysis(conn, world)
        await new_analysis(conn, world, user_id=other_user)
        await analyses.set_status(conn, running, "running")
        await analyses.set_status(conn, succeeded, "succeeded")
        await analyses.set_status(conn, failed, "failed", error_code="internal_error")
        assert await analyses.count_active(conn, world.user_id) == 2
        await analyses.set_status(conn, queued, "failed", error_code="analysis_timeout")
        assert await analyses.count_active(conn, world.user_id) == 1


async def test_set_status_sets_completed_at_only_when_terminal(db):
    world = await make_world(db)
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await new_analysis(conn, world)
        await analyses.set_status(conn, analysis_id, "running")
        running = await analyses.get_analysis(conn, analysis_id)
        await analyses.set_status(conn, analysis_id, "failed", error_code="llm_unavailable")
        failed = await analyses.get_analysis(conn, analysis_id)
    assert (running["status"], running["completed_at"], running["error_code"]) == (
        "running",
        None,
        None,
    )
    assert (failed["status"], failed["error_code"]) == ("failed", "llm_unavailable")
    assert failed["completed_at"] is not None


async def test_set_progress_appends_finished_stages(db):
    world = await make_world(db)
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await new_analysis(conn, world)
        await analyses.set_progress(conn, analysis_id, "retrieve")
        first = (await analyses.get_analysis(conn, analysis_id))["progress"]
        await analyses.set_progress(conn, analysis_id, "verify")
        await analyses.set_progress(conn, analysis_id, "verify")
        second = (await analyses.get_analysis(conn, analysis_id))["progress"]
        await analyses.set_progress(conn, analysis_id, "persist")
        third = (await analyses.get_analysis(conn, analysis_id))["progress"]
    assert first == {"stage": "retrieve", "stages_done": []}
    assert second == {"stage": "verify", "stages_done": ["retrieve"]}
    assert third == {"stage": "persist", "stages_done": ["retrieve", "verify"]}


async def test_list_analyses_scopes_and_pages(db):
    world = await make_world(db)
    other_user = await create_user(db, "other@example.com")
    mine = []
    for _ in range(3):
        async with db.tenant_tx(world.tenant_id) as conn:  # own transaction: distinct created_at
            mine.append(await new_analysis(conn, world))
    async with db.tenant_tx(world.tenant_id) as conn:
        theirs = await new_analysis(conn, world, user_id=other_user)
        first = await analyses.list_analyses(
            conn, user_id=world.user_id, page=PageParams(limit=2, after=None)
        )
        everyone = await analyses.list_analyses(
            conn, user_id=None, page=PageParams(limit=10, after=None)
        )
        boundary = first[1]
        rest = await analyses.list_analyses(
            conn,
            user_id=world.user_id,
            page=PageParams(limit=2, after=(boundary["created_at"], str(boundary["id"]))),
        )
    # limit + 1 rows come back so the caller can tell whether another page exists.
    assert [row["id"] for row in first] == mine[::-1]
    assert {row["id"] for row in everyone} == {*mine, theirs}
    assert [row["id"] for row in rest] == [mine[0]]


async def test_save_results_roundtrip(db):
    world = await make_world(db)
    second_start = TEXT.index("14 days")
    verdict = claim_verdict(
        0,
        evidence=(
            evidence_item(world, 0, relation="supports", span=(0, 7), rerank=2.5, dense_cos=0.75),
            evidence_item(
                world,
                1,
                relation="related",
                span=(second_start, second_start + len("14 days")),
                rerank=None,
                dense_cos=0.5,
            ),
        ),
    )
    citation = CitationRecord(
        number=1,
        marker_start=22,
        marker_end=26,
        sentence_start=0,
        sentence_end=21,
        status="valid",
        passage_id=world.passages[0],
        version_id=world.version_id,
    )
    async with db.tenant_tx(world.tenant_id) as conn:
        await conn.execute(
            "update passages set is_current = true where id = %s", (world.passages[0],)
        )
        analysis_id = await new_analysis(conn, world)
        claim_ids = await save(conn, analysis_id, [verdict], [citation])
        analysis = await analyses.get_analysis(conn, analysis_id)
        claims = await analyses.get_claims(conn, analysis_id)
        evidence = await analyses.get_evidence(conn, claim_ids)
        citations = await analyses.get_citations(conn, analysis_id)
        cur = await conn.execute(
            "select effective_from from document_versions where id = %s", (world.version_id,)
        )
        effective_from = (await cur.fetchone())["effective_from"]

    assert len(claim_ids) == 1
    assert analysis["status"] == "succeeded"
    assert analysis["completed_at"] is not None
    assert analysis["answer"] == ANSWER
    assert analysis["config"] == {"nli_model": "fake-nli"}
    assert analysis["timings"] == {"total_ms": 12.5}
    assert analysis["summary"] == {"supported": 1}

    assert len(claims) == 1
    claim = claims[0]
    assert claim["id"] == claim_ids[0]
    assert (claim["analysis_id"], claim["ordinal"], claim["text"]) == (analysis_id, 0, "claim 0")
    assert (claim["answer_start"], claim["answer_end"]) == (0, 21)
    assert (claim["extraction_method"], claim["verdict"]) == ("sentence", "supported")
    assert (claim["p_supported"], claim["p_contradicted"], claim["p_insufficient"]) == (
        0.7,
        0.2,
        0.1,
    )
    assert claim["features"] == {"max_entail": 0.9}
    assert claim["diagnostics"] == {"note": "hand built"}

    assert [row["rank"] for row in evidence] == [1, 2]
    first, second = evidence
    assert {
        key: first[key]
        for key in (
            "claim_id",
            "passage_id",
            "version_id",
            "relation",
            "span_start",
            "span_end",
            "nli_entail",
            "nli_contra",
            "nli_neutral",
            "rerank_score",
            "dense_cos",
        )
    } == {
        "claim_id": claim_ids[0],
        "passage_id": world.passages[0],
        "version_id": world.version_id,
        "relation": "supports",
        "span_start": 0,
        "span_end": 7,
        "nli_entail": 0.6,
        "nli_contra": 0.3,
        "nli_neutral": 0.1,
        "rerank_score": 2.5,
        "dense_cos": 0.75,
    }
    assert (second["relation"], second["rerank_score"], second["dense_cos"]) == (
        "related",
        None,
        0.5,
    )
    assert (second["span_start"], second["span_end"]) == (second_start, second_start + 7)
    # Joined passage, version, and document fields.
    assert {
        key: first[key]
        for key in (
            "collection_id",
            "text",
            "start_char",
            "end_char",
            "page_start",
            "page_end",
            "version_number",
            "effective_from",
            "document_id",
            "document_title",
            "passage_is_current",
            "text_sha256",
        )
    } == {
        "collection_id": world.seeded.collection_id,
        "text": FIRST,
        "start_char": 0,
        "end_char": 21,
        "page_start": None,
        "page_end": None,
        "version_number": 1,
        "effective_from": effective_from,
        "document_id": world.seeded.document_id,
        "document_title": "Policy",
        "passage_is_current": True,
        "text_sha256": sha256(FIRST),
    }
    assert (second["text"], second["passage_is_current"]) == (SECOND, False)

    assert len(citations) == 1
    assert {
        key: citations[0][key]
        for key in (
            "analysis_id",
            "number",
            "marker_start",
            "marker_end",
            "sentence_start",
            "sentence_end",
            "passage_id",
            "version_id",
            "status",
        )
    } == {
        "analysis_id": analysis_id,
        "number": 1,
        "marker_start": 22,
        "marker_end": 26,
        "sentence_start": 0,
        "sentence_end": 21,
        "passage_id": world.passages[0],
        "version_id": world.version_id,
        "status": "valid",
    }


async def test_save_results_returns_claim_ids_in_ordinal_order(db):
    world = await make_world(db)
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await new_analysis(conn, world)
        claim_ids = await save(
            conn,
            analysis_id,
            [
                claim_verdict(1, verdict=Verdict.INSUFFICIENT),
                claim_verdict(0),
            ],
        )
        claims = await analyses.get_claims(conn, analysis_id)
        evidence = await analyses.get_evidence(conn, claim_ids)
    assert [claim["id"] for claim in claims] == claim_ids
    assert [(claim["ordinal"], claim["verdict"]) for claim in claims] == [
        (0, "supported"),
        (1, "insufficient"),
    ]
    assert evidence == []  # a claim without evidence is stored, not an error


async def test_save_results_keeps_unknown_marker_citation_without_passage(db):
    world = await make_world(db)
    unknown = CitationRecord(
        number=9,
        marker_start=22,
        marker_end=26,
        sentence_start=0,
        sentence_end=21,
        status="unknown_marker",
        passage_id=None,
        version_id=None,
    )
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await new_analysis(conn, world)
        await save(conn, analysis_id, [claim_verdict(0)], [unknown])
        citations = await analyses.get_citations(conn, analysis_id)
    assert [
        (row["number"], row["status"], row["passage_id"], row["version_id"]) for row in citations
    ] == [(9, "unknown_marker", None, None)]


async def test_save_results_unknown_analysis_raises(db):
    world = await make_world(db)
    with pytest.raises(LookupError):
        async with db.tenant_tx(world.tenant_id) as conn:
            await save(conn, uuid.uuid4(), [claim_verdict(0)])


async def test_latest_checks_returns_newest_per_claim(db):
    world = await make_world(db)
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await new_analysis(conn, world)
        checked, unchecked = await save(conn, analysis_id, [claim_verdict(0), claim_verdict(1)])
        await analyses.insert_claim_check(
            conn,
            tenant_id=world.tenant_id,
            claim_id=checked,
            reason="source_changed",
            verdict="supported",
            p_supported=0.9,
            p_contradicted=0.05,
            p_insufficient=0.05,
            evidence=[],
            checked_version_ids=[world.version_id],
            changed=False,
        )
        newest = await analyses.insert_claim_check(
            conn,
            tenant_id=world.tenant_id,
            claim_id=checked,
            reason="manual",
            verdict="contradicted",
            p_supported=0.1,
            p_contradicted=0.8,
            p_insufficient=0.1,
            evidence=[{"passage_id": str(world.passages[0])}],
            checked_version_ids=[world.version_id],
            changed=True,
        )
        latest = await analyses.latest_checks(conn, [checked, unchecked])
        nothing = await analyses.latest_checks(conn, [])
    assert set(latest) == {checked}
    row = latest[checked]
    assert row["id"] == newest
    assert (row["reason"], row["verdict"], row["changed"]) == ("manual", "contradicted", True)
    assert (row["p_supported"], row["p_contradicted"], row["p_insufficient"]) == (0.1, 0.8, 0.1)
    assert row["evidence"] == [{"passage_id": str(world.passages[0])}]
    assert row["checked_version_ids"] == [world.version_id]
    assert nothing == {}


async def test_insert_claim_check_rejects_unknown_field(db):
    world = await make_world(db)
    with pytest.raises(TypeError, match="unexpected claim check fields"):
        async with db.tenant_tx(world.tenant_id) as conn:
            await analyses.insert_claim_check(conn, tenant_id=world.tenant_id, verdict_text="x")


async def test_feedback_upsert(db):
    world = await make_world(db)
    other_user = await create_user(db, "other@example.com")
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await new_analysis(conn, world)
        (claim_id,) = await save(conn, analysis_id, [claim_verdict(0)])
        first = await analyses.upsert_feedback(
            conn,
            tenant_id=world.tenant_id,
            claim_id=claim_id,
            user_id=world.user_id,
            label="correct",
            note=None,
        )
    async with db.tenant_tx(world.tenant_id) as conn:
        second = await analyses.upsert_feedback(
            conn,
            tenant_id=world.tenant_id,
            claim_id=claim_id,
            user_id=world.user_id,
            label="incorrect",
            note="Wrong number",
        )
        other = await analyses.upsert_feedback(
            conn,
            tenant_id=world.tenant_id,
            claim_id=claim_id,
            user_id=other_user,
            label="correct",
            note=None,
        )
        cur = await conn.execute(
            "select user_id, label, note, created_at, updated_at from feedback where claim_id = %s",
            (claim_id,),
        )
        rows = {row["user_id"]: row for row in await cur.fetchall()}
    assert second == first
    assert other != first
    assert set(rows) == {world.user_id, other_user}
    mine = rows[world.user_id]
    assert (mine["label"], mine["note"]) == ("incorrect", "Wrong number")
    assert mine["updated_at"] > mine["created_at"]
    assert (rows[other_user]["label"], rows[other_user]["note"]) == ("correct", None)


async def test_feedback_note_limited_to_1000_characters(db):
    world = await make_world(db)
    async with db.tenant_tx(world.tenant_id) as conn:
        analysis_id = await new_analysis(conn, world)
        (claim_id,) = await save(conn, analysis_id, [claim_verdict(0)])
        await analyses.upsert_feedback(
            conn,
            tenant_id=world.tenant_id,
            claim_id=claim_id,
            user_id=world.user_id,
            label="correct",
            note="x" * 1000,
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        async with db.tenant_tx(world.tenant_id) as conn:
            await analyses.upsert_feedback(
                conn,
                tenant_id=world.tenant_id,
                claim_id=claim_id,
                user_id=world.user_id,
                label="correct",
                note="x" * 1001,
            )
