"""Passage provenance: resolve_passage, validate_analysis, and GET /api/v1/passages/{id}."""

import hashlib
from dataclasses import FrozenInstanceError, dataclass, fields
from datetime import UTC, datetime
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from backend.db.repos import documents, tenants
from backend.ingest.parsers.text import parse_text
from backend.ingest.service import create_version
from backend.provenance.citations import CitationReport, LinkProblem, validate_analysis
from backend.provenance.resolve import ResolvedPassage, resolve_passage
from tests.pg_fixtures import seed_docs
from tests.test_analysis_routes import (
    REFUND_QUESTION,
    REFUNDS,
    SHIPPING_REFUNDS,
    accepted_id,
    background_worker,
    claims_json,
    collection_with_docs,
    grant_read,
    post_ask,
    wait_done,
)
from tests.test_collections import _create, _team

pytestmark = pytest.mark.pg

PASSAGES = "/api/v1/passages"
CONTEXT_CHARS = 300
PASSAGE_KEYS = {
    "passage_id",
    "version_id",
    "version_number",
    "effective_from",
    "document_id",
    "document_title",
    "text",
    "start_char",
    "end_char",
    "page_start",
    "page_end",
    "context_before",
    "context_after",
    "is_current_version",
    "current_equivalent_passage_id",
    "superseded_by_version",
    "collection_id",
    "ordinal",
    "section",
}
EXCHANGES = "Exchanges take 14 days."
# [P1] is the refunds passage; [P9] has no context passage behind it.
ANSWER = "Refunds take 30 days [P1, P9]."


def sentence(tag: str) -> str:
    """About 500 characters with non-ASCII characters, so the passage builder gives every
    sentence its own passage and code point offsets differ from byte offsets."""
    words = " ".join(f"{tag}{i}" for i in range(60))
    return f"Section {tag} \U0001f600 caf\u00e9 notes {words}."


ALPHA, BETA, GAMMA, DELTA, BETA_REVISED = (
    sentence(tag) for tag in ("alpha", "beta", "gamma", "delta", "beta2")
)
V1_TEXT = " ".join([ALPHA, BETA, GAMMA])
# Alpha and gamma are unchanged but shift by one passage; beta is rewritten.
V2_TEXT = " ".join([DELTA, ALPHA, BETA_REVISED, GAMMA])


@dataclass(frozen=True)
class Doc:
    tenant_id: UUID
    collection_id: UUID
    document_id: UUID
    versions: list[list[dict]]  # the stored passages of version 1, 2, ... in ordinal order


async def passages_of(db, tenant_id: UUID, document_id: UUID, version_number: int) -> list[dict]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select p.id, p.version_id, p.ordinal, p.start_char, p.end_char, p.page_start, "
            "p.page_end, p.section, p.collection_id, p.text, v.text as version_text, "
            "v.effective_from, v.version_number "
            "from passages p join document_versions v on v.id = p.version_id "
            "where v.document_id = %s and v.version_number = %s order by p.ordinal",
            (document_id, version_number),
        )
        return await cur.fetchall()


async def add_version(db, services, tenant_id: UUID, document_id: UUID, text: str) -> None:
    data = text.encode()
    async with db.tenant_tx(tenant_id) as conn:
        await create_version(
            conn,
            services.embedder,
            tenant_id=tenant_id,
            document_id=document_id,
            parsed=parse_text(data),
            blob_sha256=hashlib.sha256(data).digest(),
            effective_from=datetime.now(UTC),
            created_by=None,
            settings=services.settings,
        )


async def build_doc(client, client2, client3, db, services, *texts: str) -> Doc:
    """One document in Alice's collection with a version per text."""
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection_id = UUID((await _create(client, alice))["id"])
    (document_id,) = await seed_docs(db, services, tenant_id, collection_id, texts[:1])
    for text in texts[1:]:
        await add_version(db, services, tenant_id, document_id, text)
    versions = [
        await passages_of(db, tenant_id, document_id, number) for number in range(1, len(texts) + 1)
    ]
    return Doc(tenant_id, collection_id, document_id, versions)


@pytest.fixture
async def single(client, client2, client3, db, services) -> Doc:
    doc = await build_doc(client, client2, client3, db, services, V1_TEXT)
    assert [p["text"] for p in doc.versions[0]] == [ALPHA, BETA, GAMMA]
    return doc


@pytest.fixture
async def versioned(client, client2, client3, db, services) -> Doc:
    doc = await build_doc(client, client2, client3, db, services, V1_TEXT, V2_TEXT)
    assert [p["text"] for p in doc.versions[0]] == [ALPHA, BETA, GAMMA]
    assert [p["text"] for p in doc.versions[1]] == [DELTA, ALPHA, BETA_REVISED, GAMMA]
    return doc


@dataclass(frozen=True)
class Ran:
    alice: dict
    bob: dict
    carol: dict
    tenant_id: UUID
    collection_id: UUID
    analysis_id: UUID
    body: dict  # the analysis as Alice reads it

    def evidence(self) -> list[dict]:
        return [item for claim in self.body["claims"] for item in claim["evidence"]]

    def cited_passage_id(self, number: int) -> UUID:
        (citation,) = [c for c in self.body["citations"] if c["number"] == number]
        return UUID(citation["passage_id"])


@pytest.fixture
async def ran(client, client2, client3, db, services) -> Ran:
    """An ask analysis over two documents: one claim, two evidence links, citations P1 and P9."""
    alice, bob, carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(
        client, alice, db, services, [SHIPPING_REFUNDS, EXCHANGES]
    )
    services.llm.responses = [ANSWER, claims_json((0, REFUNDS))]
    accepted = await post_ask(client, alice, REFUND_QUESTION, collection_ids=[collection["id"]])
    analysis_id = await accepted_id(accepted)
    async with background_worker(db, services):
        body = await wait_done(client, analysis_id)
    assert body["status"] == "succeeded", body
    return Ran(
        alice,
        bob,
        carol,
        UUID(alice["tenant"]["id"]),
        UUID(collection["id"]),
        UUID(analysis_id),
        body,
    )


async def validate(db, ran: Ran) -> CitationReport:
    async with db.tenant_tx(ran.tenant_id) as conn:
        return await validate_analysis(conn, ran.analysis_id)


async def fetch(db, tenant_id: UUID, query: str, params: tuple = ()) -> list[dict]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def corrupt(pg_urls, statement: str, params: tuple) -> None:
    """Break stored data the way a faulty repair could.

    Triggers and foreign keys do not fire for a superuser whose `session_replication_role` is
    `replica`, which is the only way to store rows the schema otherwise refuses.
    """
    url = make_conninfo(pg_urls["owner"], user="postgres")
    async with await psycopg.AsyncConnection.connect(
        url, autocommit=True, connect_timeout=5
    ) as conn:
        await conn.execute("set session_replication_role = replica")
        await conn.execute(statement, params)


# --- resolve_passage ----------------------------------------------------------------------------


async def test_resolve_current_passage(db, single):
    stored = single.versions[0][1]
    text = stored["version_text"]
    start, end = stored["start_char"], stored["end_char"]

    async with db.tenant_tx(single.tenant_id) as conn:
        resolved = await resolve_passage(conn, stored["id"])

    assert {field.name for field in fields(ResolvedPassage)} == PASSAGE_KEYS
    assert resolved == ResolvedPassage(
        passage_id=stored["id"],
        version_id=stored["version_id"],
        version_number=1,
        effective_from=stored["effective_from"],
        document_id=single.document_id,
        document_title="Document 1",
        text=BETA,
        start_char=start,
        end_char=end,
        page_start=stored["page_start"],
        page_end=stored["page_end"],
        context_before=text[start - CONTEXT_CHARS : start],
        context_after=text[end : end + CONTEXT_CHARS],
        is_current_version=True,
        current_equivalent_passage_id=stored["id"],
        superseded_by_version=None,
        collection_id=single.collection_id,
        ordinal=1,
        section=stored["section"],
    )
    assert text[start:end] == resolved.text
    assert (len(resolved.context_before), len(resolved.context_after)) == (300, 300)
    with pytest.raises(FrozenInstanceError):
        resolved.text = "changed"  # type: ignore[misc]


async def test_resolve_context_is_clipped_at_text_edges(db, single):
    first, *_, last = single.versions[0]
    async with db.tenant_tx(single.tenant_id) as conn:
        at_start = await resolve_passage(conn, first["id"])
        at_end = await resolve_passage(conn, last["id"])

    assert (first["start_char"], at_start.context_before) == (0, "")
    assert at_start.context_after == first["version_text"][first["end_char"] :][:CONTEXT_CHARS]
    assert at_end.context_after == ""
    assert at_end.context_before == last["version_text"][: last["start_char"]][-CONTEXT_CHARS:]
    assert len(at_end.context_before) == CONTEXT_CHARS


async def test_resolve_superseded_with_equivalent(db, versioned):
    v1, v2 = versioned.versions
    async with db.tenant_tx(versioned.tenant_id) as conn:
        alpha = await resolve_passage(conn, v1[0]["id"])
        gamma = await resolve_passage(conn, v1[2]["id"])
        current = await resolve_passage(conn, v2[1]["id"])

    # The unchanged paragraphs sit one passage further in version 2: matched by text hash.
    assert (alpha.version_number, alpha.is_current_version, alpha.superseded_by_version) == (
        1,
        False,
        2,
    )
    assert alpha.current_equivalent_passage_id == v2[1]["id"]
    assert (gamma.superseded_by_version, gamma.current_equivalent_passage_id) == (2, v2[3]["id"])
    # The text and context stay those of version 1, whatever the current version says.
    assert alpha.text == ALPHA
    assert gamma.context_before == v1[2]["version_text"][: v1[2]["start_char"]][-CONTEXT_CHARS:]
    assert (current.version_number, current.is_current_version) == (2, True)
    assert (current.superseded_by_version, current.current_equivalent_passage_id) == (
        None,
        v2[1]["id"],
    )


async def test_resolve_changed_passage_has_no_equivalent(db, versioned):
    v1, _v2 = versioned.versions
    async with db.tenant_tx(versioned.tenant_id) as conn:
        beta = await resolve_passage(conn, v1[1]["id"])

    assert beta.text == BETA
    assert (beta.is_current_version, beta.superseded_by_version) == (False, 2)
    assert beta.current_equivalent_passage_id is None


async def test_resolve_equivalent_prefers_lowest_ordinal(client, client2, client3, db, services):
    doc = await build_doc(
        client, client2, client3, db, services, V1_TEXT, " ".join([DELTA, ALPHA, ALPHA, GAMMA])
    )
    v1, v2 = doc.versions
    assert [p["text"] for p in v2] == [DELTA, ALPHA, ALPHA, GAMMA]
    async with db.tenant_tx(doc.tenant_id) as conn:
        old = await resolve_passage(conn, v1[0]["id"])
        second_copy = await resolve_passage(conn, v2[2]["id"])

    # Two passages of the current version hold the text: the first one stands in for the old one.
    assert old.current_equivalent_passage_id == v2[1]["id"]
    # A current passage is its own equivalent, even when an earlier passage has the same text.
    assert second_copy.current_equivalent_passage_id == v2[2]["id"]


async def test_resolve_deleted_document_has_no_equivalent(db, versioned):
    v1, v2 = versioned.versions
    async with db.tenant_tx(versioned.tenant_id) as conn:
        assert await documents.soft_delete_document(conn, versioned.document_id)
    async with db.tenant_tx(versioned.tenant_id) as conn:
        old = await resolve_passage(conn, v1[0]["id"])
        current = await resolve_passage(conn, v2[1]["id"])

    # The passage still resolves, because past analyses cite it, but nothing is current.
    assert (old.text, old.current_equivalent_passage_id) == (ALPHA, None)
    assert (current.text, current.current_equivalent_passage_id) == (ALPHA, None)


async def test_resolve_unknown_or_foreign_passage_is_none(db, single):
    stored = single.versions[0][0]
    async with db.tenant_tx(single.tenant_id) as conn:
        missing = await resolve_passage(conn, uuid4())
    async with db.system_tx() as conn:
        other_tenant = (await tenants.insert_tenant(conn, "other"))["id"]
    async with db.tenant_tx(other_tenant) as conn:
        foreign = await resolve_passage(conn, stored["id"])

    assert missing is None
    assert foreign is None


# --- GET /passages/{id} -------------------------------------------------------------------------


async def test_passages_route_requires_read(client, client2, client3, db, ran):
    passage_id = ran.cited_passage_id(1)
    await grant_read(client, ran.alice, str(ran.collection_id), ran.bob)

    granted = await client2.get(f"{PASSAGES}/{passage_id}")
    await grant_read(client, ran.alice, str(ran.collection_id))  # Bob loses read access
    revoked = await client2.get(f"{PASSAGES}/{passage_id}")
    owner = await client.get(f"{PASSAGES}/{passage_id}")
    foreign = await client3.get(f"{PASSAGES}/{passage_id}")
    unknown = await client.get(f"{PASSAGES}/{uuid4()}")

    assert granted.status_code == 200, granted.text
    assert granted.json() == owner.json()
    for hidden in (revoked, foreign, unknown):
        assert hidden.status_code == 404
        assert hidden.json()["code"] == "passage_not_found"
    assert owner.status_code == 200
    body = owner.json()
    assert set(body) == PASSAGE_KEYS
    (evidence,) = [e for e in ran.evidence() if e["passage_id"] == str(passage_id)]
    assert body["passage_id"] == str(passage_id)
    assert body["text"] == evidence["text"] == SHIPPING_REFUNDS
    assert (body["version_number"], body["document_title"]) == (1, "Document 1")
    assert body["version_id"] == evidence["version_id"]
    assert body["document_id"] == evidence["document_id"]
    assert body["collection_id"] == str(ran.collection_id)
    assert (body["is_current_version"], body["superseded_by_version"]) == (True, None)
    assert body["current_equivalent_passage_id"] == str(passage_id)
    # The passage is the whole document, so there is no context on either side.
    assert (body["start_char"], body["end_char"]) == (0, len(SHIPPING_REFUNDS))
    assert body["context_before"] == body["context_after"] == ""


# --- validate_analysis --------------------------------------------------------------------------


async def test_validate_clean_analysis(db, ran):
    links = await fetch(db, ran.tenant_id, "select id from evidence_links")

    report = await validate(db, ran)

    assert len(links) == len(ran.evidence()) == 2
    assert report == CitationReport(
        n_links=2,
        n_valid_links=2,
        n_citations=2,
        n_valid_citations=1,
        n_unknown_markers=1,
        problems=(),
    )


@pytest.mark.parametrize(
    "span",
    [lambda start, end: (end - 2, end + 8), lambda start, end: (start - 8, start + 2)],
    ids=["past_the_end", "before_the_start"],
)
async def test_validate_detects_outside_span(db, ran, span):
    before = await validate(db, ran)
    (anchor,) = await fetch(
        db,
        ran.tenant_id,
        "select e.claim_id, e.passage_id, e.version_id, p.start_char, p.end_char "
        "from evidence_links e join passages p on p.id = e.passage_id where e.passage_id = %s",
        (ran.cited_passage_id(1),),
    )
    span_start, span_end = span(anchor["start_char"], anchor["end_char"])
    async with db.tenant_tx(ran.tenant_id) as conn:
        cur = await conn.execute(
            "insert into evidence_links (tenant_id, claim_id, passage_id, version_id, relation, "
            "rank, span_start, span_end) values (%s, %s, %s, %s, 'related', 99, %s, %s) "
            "returning id",
            (
                ran.tenant_id,
                anchor["claim_id"],
                anchor["passage_id"],
                anchor["version_id"],
                span_start,
                span_end,
            ),
        )
        link_id = (await cur.fetchone())["id"]

    report = await validate(db, ran)

    assert before.problems == ()
    assert report == CitationReport(
        n_links=before.n_links + 1,
        n_valid_links=before.n_valid_links,
        n_citations=before.n_citations,
        n_valid_citations=before.n_valid_citations,
        n_unknown_markers=before.n_unknown_markers,
        problems=(LinkProblem("span_outside_passage", link_id),),
    )


async def test_validate_detects_span_text_mismatch(db, ran, pg_urls):
    passage_id = ran.cited_passage_id(1)
    (link,) = await fetch(
        db, ran.tenant_id, "select id from evidence_links where passage_id = %s", (passage_id,)
    )
    await corrupt(
        pg_urls, "update passages set text = repeat('x', length(text)) where id = %s", (passage_id,)
    )

    report = await validate(db, ran)

    assert report.problems == (LinkProblem("span_text_mismatch", link["id"]),)
    assert (report.n_links, report.n_valid_links) == (2, 1)
    # The citation only needs the passage to belong to its version, which it still does.
    assert (report.n_citations, report.n_valid_citations) == (2, 1)


async def test_validate_detects_missing_passage(db, ran, pg_urls):
    passage_id = ran.cited_passage_id(1)
    (link,) = await fetch(
        db, ran.tenant_id, "select id from evidence_links where passage_id = %s", (passage_id,)
    )
    (citation,) = await fetch(
        db,
        ran.tenant_id,
        "select id from citations where analysis_id = %s and number = 1",
        (ran.analysis_id,),
    )
    await corrupt(pg_urls, "delete from passages where id = %s", (passage_id,))

    report = await validate(db, ran)

    assert set(report.problems) == {
        LinkProblem("missing_passage", link["id"]),
        LinkProblem("missing_passage", citation["id"]),
    }
    assert len(report.problems) == 2
    assert (report.n_links, report.n_valid_links) == (2, 1)
    assert (report.n_citations, report.n_valid_citations, report.n_unknown_markers) == (2, 0, 1)


async def add_second_version(db, services, ran: Ran, passage_id: UUID) -> UUID:
    """Give the document of the cited passage a version 2 and return its id."""
    (evidence,) = [e for e in ran.evidence() if e["passage_id"] == str(passage_id)]
    document_id = UUID(evidence["document_id"])
    await add_version(db, services, ran.tenant_id, document_id, SHIPPING_REFUNDS + " Extra.")
    (other,) = await fetch(
        db,
        ran.tenant_id,
        "select id from document_versions where document_id = %s and version_number = 2",
        (document_id,),
    )
    return other["id"]


async def test_validate_detects_link_naming_the_wrong_version(db, services, ran, pg_urls):
    passage_id = ran.cited_passage_id(1)
    (link,) = await fetch(
        db, ran.tenant_id, "select id from evidence_links where passage_id = %s", (passage_id,)
    )
    other_version = await add_second_version(db, services, ran, passage_id)
    await corrupt(
        pg_urls,
        "update evidence_links set version_id = %s where id = %s",
        (other_version, link["id"]),
    )

    report = await validate(db, ran)

    # No passage row matches the (passage, version) pair the link stores.
    assert report.problems == (LinkProblem("missing_passage", link["id"]),)
    assert (report.n_links, report.n_valid_links) == (2, 1)


async def test_validate_detects_citation_version_mismatch(db, services, ran, pg_urls):
    passage_id = ran.cited_passage_id(1)
    other_version = await add_second_version(db, services, ran, passage_id)
    (citation,) = await fetch(
        db,
        ran.tenant_id,
        "select id from citations where analysis_id = %s and number = 1",
        (ran.analysis_id,),
    )
    await corrupt(
        pg_urls,
        "update citations set version_id = %s where id = %s",
        (other_version, citation["id"]),
    )

    report = await validate(db, ran)

    assert report.problems == (LinkProblem("citation_version_mismatch", citation["id"]),)
    assert (report.n_links, report.n_valid_links) == (2, 2)
    assert (report.n_citations, report.n_valid_citations, report.n_unknown_markers) == (2, 0, 1)
