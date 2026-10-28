"""POST /api/v1/ask, POST /api/v1/verify, GET /api/v1/analyses: visibility (R8) and redaction."""

import asyncio
import contextlib
import json
import time
from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
import pytest_asyncio

from backend.api.pagination import encode_cursor
from backend.db.repos import analyses, tenants
from tests.helpers.auth import csrf, signup
from tests.pg_fixtures import drain, seed_docs
from tests.test_collections import _create, _put_grants, _team

pytestmark = pytest.mark.pg

ASK = "/api/v1/ask"
VERIFY = "/api/v1/verify"
ANALYSES = "/api/v1/analyses"

REFUNDS = "Refunds take 30 days."
REFUNDS_45 = "Refunds take 45 days."
EXCHANGES = "Exchanges take 14 days."
# The refund claim is the second sentence, so its span is a slice in the middle of the text.
SHIPPING_REFUNDS = "Shipping is free for orders over fifty dollars. Refunds take 30 days."
# Shares two tokens with the refunds passages and one with the exchanges passage, so the
# refunds passage is [P1].
REFUND_QUESTION = "How long do refunds take?"
TWO_FACTS = "Refunds take 30 days. Exchanges take 14 days."

ANALYSIS_KEYS = {
    "id",
    "kind",
    "question",
    "answer",
    "status",
    "error_code",
    "progress",
    "live",
    "as_of",
    "known_at",
    "collection_ids",
    "created_at",
    "completed_at",
    "config",
    "timings",
    "summary",
    "claims",
    "citations",
}
SUMMARY_KEYS = {
    "id",
    "kind",
    "question_preview",
    "status",
    "error_code",
    "live",
    "as_of",
    "created_at",
    "completed_at",
    "summary",
    "changed_since",
}
CLAIM_KEYS = {
    "id",
    "ordinal",
    "text",
    "answer_start",
    "answer_end",
    "extraction_method",
    "verdict",
    "probs",
    "features",
    "diagnostics",
    "conflict",
    "evidence",
    "latest_check",
}
EVIDENCE_KEYS = {
    "passage_id",
    "version_id",
    "document_id",
    "document_title",
    "version_number",
    "effective_from",
    "page_start",
    "page_end",
    "relation",
    "rank",
    "span_start",
    "span_end",
    "nli",
    "rerank_score",
    "text",
    "span_text",
    "passage_is_current",
    "redacted",
}
CITATION_KEYS = {
    "number",
    "marker_start",
    "marker_end",
    "sentence_start",
    "sentence_end",
    "status",
    "passage_id",
    "version_id",
    "document_title",
    "version_number",
    "redacted",
}
CHECK_KEYS = {
    "id",
    "reason",
    "verdict",
    "probs",
    "changed",
    "checked_version_ids",
    "created_at",
}
# What a redacted evidence item and a redacted citation lose.
EVIDENCE_NULLED = (
    "text",
    "span_text",
    "document_title",
    "document_id",
    "version_number",
    "effective_from",
    "page_start",
    "page_end",
    "passage_is_current",
)
CITATION_NULLED = ("document_title", "version_number")


@pytest.fixture(autouse=True)
def _shared_services(api_app, services):
    """The routes and `drain` run on the same fakes."""
    api_app.state.services = services


@pytest_asyncio.fixture(loop_scope="session")
async def client4(api_app):
    """Fourth client with its own cookie jar, for Dana."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://test"
    ) as http_client:
        yield http_client


@contextlib.asynccontextmanager
async def background_worker(db, services):
    """Drain the queue every 50 ms while the body runs, like a worker next to the API."""

    async def loop():
        while True:
            await drain(db, services)
            await asyncio.sleep(0.05)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def claims_json(*claims: tuple[int, str]) -> str:
    return json.dumps({"claims": [{"sentence": i, "text": text} for i, text in claims]})


async def dana_joins(client4, alice, db, role: str = "member") -> dict:
    """Dana joins Alice's workspace with the role, without any collection grant."""
    dana = await signup(client4, email="dana@example.com", display_name="Dana")
    async with db.system_tx() as conn:
        await tenants.add_membership(conn, alice["tenant"]["id"], dana["user"]["id"], role)
    switched = await client4.post(
        "/api/v1/auth/switch-tenant",
        json={"tenant_id": alice["tenant"]["id"]},
        headers=csrf(dana),
    )
    assert switched.status_code == 200, switched.text
    return {**dana, "tenant": switched.json()["tenant"]}


async def grant_read(client, alice, collection_id, *readers) -> None:
    """Replace the grants of the collection: Alice as admin plus a read grant per reader."""
    grants = [{"user_id": alice["user"]["id"], "permission": "admin"}]
    grants += [{"user_id": reader["user"]["id"], "permission": "read"} for reader in readers]
    response = await _put_grants(client, alice, collection_id, grants)
    assert response.status_code == 200, response.text


async def collection_with_docs(client, alice, db, services, texts, name="Policies") -> dict:
    collection = await _create(client, alice, name)
    await seed_docs(db, services, UUID(alice["tenant"]["id"]), UUID(collection["id"]), texts)
    return collection


async def post_ask(client, session, question=REFUND_QUESTION, **body):
    return await client.post(ASK, json={"question": question, **body}, headers=csrf(session))


async def post_verify(client, session, answer, **body):
    return await client.post(VERIFY, json={"answer": answer, **body}, headers=csrf(session))


async def accepted_id(response) -> str:
    assert response.status_code == 202, response.text
    return response.json()["analysis_id"]


async def wait_done(client, analysis_id: str, timeout_s: float = 30.0) -> dict:
    """Poll `GET /analyses/{id}` until the analysis is final; returns the last body."""
    deadline = time.monotonic() + timeout_s
    while True:
        response = await client.get(f"{ANALYSES}/{analysis_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("succeeded", "failed") or time.monotonic() >= deadline:
            return body
        await asyncio.sleep(0.05)


async def passage_collections(db, tenant_id: UUID) -> dict[str, str]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select id, collection_id from passages")
        return {str(row["id"]): str(row["collection_id"]) for row in await cur.fetchall()}


def evidence_of(body: dict) -> list[dict]:
    return [item for claim in body["claims"] for item in claim["evidence"]]


# --- ask and verify ---------------------------------------------------------------------------


async def test_ask_then_get_full_analysis(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(
        client, alice, db, services, [SHIPPING_REFUNDS, EXCHANGES]
    )
    answer = "Refunds take 30 days [P1]."
    services.llm.responses = [answer, claims_json((0, REFUNDS))]

    accepted = await post_ask(client, alice, collection_ids=[collection["id"]])
    analysis_id = await accepted_id(accepted)
    queued = await client.get(f"{ANALYSES}/{analysis_id}")
    async with background_worker(db, services):
        body = await wait_done(client, analysis_id)

    assert set(accepted.json()) == {"analysis_id", "status"}
    assert accepted.json()["status"] == "queued"
    assert queued.status_code == 200
    assert (queued.json()["status"], queued.json()["claims"], queued.json()["citations"]) == (
        "queued",
        [],
        [],
    )
    assert set(body) == ANALYSIS_KEYS
    assert (body["id"], body["kind"], body["status"], body["error_code"]) == (
        analysis_id,
        "ask",
        "succeeded",
        None,
    )
    assert (body["question"], body["answer"]) == (REFUND_QUESTION, answer)
    assert (body["live"], body["as_of"], body["collection_ids"]) == (
        True,
        None,
        [collection["id"]],
    )
    assert body["progress"]["stage"] == "done"
    assert body["completed_at"] is not None
    assert body["known_at"] is not None
    assert "total" in body["timings"]
    assert body["summary"]["n_claims"] == 1
    assert body["config"]["nli"] == "fake-rule-nli"

    (claim,) = body["claims"]
    assert set(claim) == CLAIM_KEYS
    assert (claim["ordinal"], claim["text"], claim["extraction_method"], claim["verdict"]) == (
        0,
        REFUNDS,
        "llm_atomic",
        "supported",
    )
    assert set(claim["probs"]) == {"supported", "contradicted", "insufficient"}
    assert claim["conflict"] is False
    assert claim["latest_check"] is None
    best = claim["evidence"][0]
    assert set(best) == EVIDENCE_KEYS
    assert set(best["nli"]) == {"entail", "contra", "neutral"}
    assert (best["relation"], best["rank"], best["redacted"]) == ("supports", 1, False)
    assert best["text"] == SHIPPING_REFUNDS
    assert best["span_text"] == REFUNDS
    assert SHIPPING_REFUNDS[best["span_start"] : best["span_end"]] == best["span_text"]
    assert (best["document_title"], best["version_number"], best["passage_is_current"]) == (
        "Document 1",
        1,
        True,
    )

    (citation,) = body["citations"]
    assert set(citation) == CITATION_KEYS
    assert (citation["number"], citation["status"], citation["redacted"]) == (1, "valid", False)
    assert (citation["passage_id"], citation["version_id"]) == (
        best["passage_id"],
        best["version_id"],
    )
    assert (citation["document_title"], citation["version_number"]) == ("Document 1", 1)
    assert answer[citation["marker_start"] : citation["marker_end"]] == "[P1]"


async def test_verify_route(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS, EXCHANGES])

    accepted = await post_verify(
        client,
        alice,
        TWO_FACTS,
        question="How long do refunds and exchanges take?",
        options={"extraction": "sentence"},
    )
    analysis_id = await accepted_id(accepted)
    async with background_worker(db, services):
        body = await wait_done(client, analysis_id)

    assert body["status"] == "succeeded", body
    assert (body["kind"], body["answer"], body["live"]) == ("verify", TWO_FACTS, True)
    assert body["question"] == "How long do refunds and exchanges take?"
    # No collection_ids in the request means every collection Alice can read.
    assert body["collection_ids"] == [collection["id"]]
    claims = body["claims"]
    assert [(c["ordinal"], c["text"], c["extraction_method"]) for c in claims] == [
        (0, REFUNDS, "sentence"),
        (1, EXCHANGES, "sentence"),
    ]
    assert [TWO_FACTS[c["answer_start"] : c["answer_end"]] for c in claims] == [
        REFUNDS,
        EXCHANGES,
    ]
    assert [c["verdict"] for c in claims] == ["supported", "supported"]
    assert body["citations"] == []
    assert body["summary"]["extraction"] == {"sentence": 2}
    assert services.llm.calls == []


async def test_conflict_flag(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS, REFUNDS_45])

    accepted = await post_verify(
        client,
        alice,
        REFUNDS,
        collection_ids=[collection["id"]],
        options={"extraction": "sentence"},
    )
    async with background_worker(db, services):
        body = await wait_done(client, await accepted_id(accepted))

    assert body["status"] == "succeeded", body
    (claim,) = body["claims"]
    assert sorted(item["relation"] for item in claim["evidence"]) == ["contradicts", "supports"]
    assert claim["conflict"] is True


# --- visibility and redaction (R8) -------------------------------------------------------------


async def test_redaction_after_revoked_grant(client, client2, client3, db, services):
    alice, bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    secret = await collection_with_docs(client, alice, db, services, [REFUNDS], "Secret")
    open_ = await collection_with_docs(client, alice, db, services, [EXCHANGES], "Open")
    await grant_read(client, alice, secret["id"], bob)
    await grant_read(client, alice, open_["id"], bob)
    services.llm.responses = [
        "Refunds take 30 days [P1]. Exchanges take 14 days [P2].",
        claims_json((0, REFUNDS), (1, EXCHANGES)),
    ]

    accepted = await post_ask(client2, bob, collection_ids=[secret["id"], open_["id"]])
    analysis_id = await accepted_id(accepted)
    async with background_worker(db, services):
        before = await wait_done(client2, analysis_id)
    await grant_read(client, alice, secret["id"])  # Bob loses read on the secret collection
    after = (await client2.get(f"{ANALYSES}/{analysis_id}")).json()
    alice_view = (await client.get(f"{ANALYSES}/{analysis_id}")).json()

    assert before["status"] == "succeeded", before
    # While the grant stands nothing is withheld.
    assert [item["redacted"] for item in evidence_of(before)] == [False] * 4
    assert all(item["text"] is not None for item in evidence_of(before))
    assert [c["redacted"] for c in before["citations"]] == [False, False]

    owner = await passage_collections(db, tenant_id)
    hidden = [e for e in evidence_of(after) if owner[e["passage_id"]] == secret["id"]]
    visible = [e for e in evidence_of(after) if owner[e["passage_id"]] == open_["id"]]
    assert (len(hidden), len(visible)) == (2, 2)
    kept = {(e["passage_id"], e["version_id"]) for e in evidence_of(before)}
    for item in hidden:
        assert item["redacted"] is True
        assert all(item[field] is None for field in EVIDENCE_NULLED), item
        assert (item["passage_id"], item["version_id"]) in kept
        assert item["relation"] in ("supports", "contradicts", "related")
        assert isinstance(item["rank"], int)
        assert all(item["nli"][name] is not None for name in ("entail", "contra", "neutral"))
        assert isinstance(item["span_start"], int)
    for item in visible:
        assert item["redacted"] is False
        assert item["text"] is not None
        assert item["span_text"] is not None
        assert item["document_title"] == "Document 1"

    by_number = {c["number"]: c for c in after["citations"]}
    assert owner[by_number[1]["passage_id"]] == secret["id"]
    assert by_number[1]["redacted"] is True
    assert all(by_number[1][field] is None for field in CITATION_NULLED)
    assert by_number[1]["version_id"] is not None
    assert by_number[1]["status"] == "valid"
    assert by_number[2]["redacted"] is False
    assert (by_number[2]["document_title"], by_number[2]["version_number"]) == ("Document 1", 1)

    # The claims and the answer stay; only the source material is withheld.
    assert [c["text"] for c in after["claims"]] == [REFUNDS, EXCHANGES]
    assert after["answer"] == before["answer"]
    # The owner reads everything.
    assert [e["redacted"] for e in evidence_of(alice_view)] == [False] * 4
    assert all(e["text"] is not None for e in evidence_of(alice_view))
    assert [c["redacted"] for c in alice_view["citations"]] == [False, False]
    assert all(c["document_title"] == "Document 1" for c in alice_view["citations"])


async def test_other_member_cannot_see(client, client2, client3, client4, db, services):
    alice, bob, _carol = await _team(client, client2, client3, db)
    dana = await dana_joins(client4, alice, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])
    await grant_read(client, alice, collection["id"], bob)
    analysis_id = await accepted_id(await post_ask(client2, bob, collection_ids=[collection["id"]]))

    hidden = await client4.get(f"{ANALYSES}/{analysis_id}")
    hidden_list = await client4.get(ANALYSES)
    own = await client2.get(f"{ANALYSES}/{analysis_id}")
    admin = await client.get(f"{ANALYSES}/{analysis_id}")

    assert dana["tenant"]["role"] == "member"
    assert hidden.status_code == 404
    assert hidden.json()["code"] == "analysis_not_found"
    assert hidden_list.status_code == 200
    assert hidden_list.json()["items"] == []
    assert own.status_code == 200
    assert admin.status_code == 200


async def test_admin_without_grant_reads_everything(
    client, client2, client3, client4, db, services
):
    alice, bob, _carol = await _team(client, client2, client3, db)
    dana = await dana_joins(client4, alice, db, role="admin")
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])
    await grant_read(client, alice, collection["id"], bob)
    services.llm.responses = ["Refunds take 30 days [P1].", claims_json((0, REFUNDS))]
    analysis_id = await accepted_id(await post_ask(client2, bob, collection_ids=[collection["id"]]))
    async with background_worker(db, services):
        await wait_done(client2, analysis_id)

    seen = (await client4.get(f"{ANALYSES}/{analysis_id}")).json()
    listed = (await client4.get(ANALYSES, params={"mine": "false"})).json()

    assert dana["tenant"]["role"] == "admin"
    assert seen["status"] == "succeeded", seen
    assert [e["redacted"] for e in evidence_of(seen)] == [False]
    assert all(e["text"] is not None for e in evidence_of(seen))
    assert [c["redacted"] for c in seen["citations"]] == [False]
    assert [item["id"] for item in listed["items"]] == [analysis_id]


async def test_cross_tenant_404(client, client2, client3, db, services):
    alice, bob, carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])
    analysis_id = await accepted_id(
        await post_ask(client, alice, collection_ids=[collection["id"]])
    )

    seen = await client3.get(f"{ANALYSES}/{analysis_id}")
    listed = await client3.get(ANALYSES, params={"mine": "false"})
    foreign_scope = await post_ask(client3, carol, collection_ids=[collection["id"]])

    assert bob["tenant"]["id"] == alice["tenant"]["id"] != carol["tenant"]["id"]
    assert seen.status_code == 404
    assert seen.json()["code"] == "analysis_not_found"
    assert listed.json()["items"] == []
    assert foreign_scope.status_code == 404
    assert foreign_scope.json()["code"] == "collection_not_found"


async def test_mine_false_requires_admin(client, client2, client3, db, services):
    alice, bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])
    await grant_read(client, alice, collection["id"], bob)
    scope = {"collection_ids": [collection["id"]]}
    bobs = await accepted_id(await post_ask(client2, bob, **scope))
    alices = await accepted_id(await post_ask(client, alice, **scope))

    denied = await client2.get(ANALYSES, params={"mine": "false"})
    bob_mine = await client2.get(ANALYSES)
    alice_all = await client.get(ANALYSES, params={"mine": "false"})
    alice_mine = await client.get(ANALYSES)

    assert denied.status_code == 403
    assert denied.json()["code"] == "forbidden"
    assert [item["id"] for item in bob_mine.json()["items"]] == [bobs]
    assert [item["id"] for item in alice_all.json()["items"]] == [alices, bobs]
    assert [item["id"] for item in alice_mine.json()["items"]] == [alices]


# --- listing ------------------------------------------------------------------------------------


async def test_list_pages_newest_first(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])
    ids = [
        await accepted_id(await post_ask(client, alice, collection_ids=[collection["id"]]))
        for _ in range(3)
    ]

    first = (await client.get(ANALYSES, params={"limit": 2})).json()
    second = (
        await client.get(ANALYSES, params={"limit": 2, "cursor": first["next_cursor"]})
    ).json()
    bad = await client.get(
        ANALYSES, params={"cursor": encode_cursor(datetime.now(UTC), "not-a-uuid")}
    )

    assert [item["id"] for item in first["items"]] == [ids[2], ids[1]]
    assert first["next_cursor"] is not None
    assert [item["id"] for item in second["items"]] == [ids[0]]
    assert second["next_cursor"] is None
    assert bad.status_code == 400
    assert bad.json()["code"] == "bad_cursor"


async def test_list_item_fields(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])
    question = "x" * 300
    as_of = datetime(2020, 1, 1, tzinfo=UTC)

    analysis_id = await accepted_id(
        await post_ask(
            client,
            alice,
            question,
            collection_ids=[collection["id"]],
            as_of=as_of.isoformat(),
        )
    )
    listed = (await client.get(ANALYSES)).json()

    (item,) = listed["items"]
    assert set(item) == SUMMARY_KEYS
    assert item["id"] == analysis_id
    assert item["kind"] == "ask"
    assert item["question_preview"] == question[:200]
    assert (item["status"], item["error_code"], item["completed_at"]) == ("queued", None, None)
    assert item["live"] is False
    assert datetime.fromisoformat(item["as_of"]) == as_of
    assert (item["summary"], item["changed_since"]) == ({}, False)


async def test_changed_since_and_latest_check(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    await collection_with_docs(client, alice, db, services, [REFUNDS])
    analysis_id = await accepted_id(
        await post_verify(client, alice, REFUNDS, options={"extraction": "sentence"})
    )
    async with background_worker(db, services):
        body = await wait_done(client, analysis_id)
    assert body["status"] == "succeeded", body
    claim_id = UUID(body["claims"][0]["id"])
    version_id = UUID(body["claims"][0]["evidence"][0]["version_id"])

    async def add_check(reason: str, verdict: str, changed: bool) -> None:
        async with db.tenant_tx(tenant_id) as conn:
            await analyses.insert_claim_check(
                conn,
                tenant_id=tenant_id,
                claim_id=claim_id,
                reason=reason,
                verdict=verdict,
                p_supported=0.1,
                p_contradicted=0.8,
                p_insufficient=0.1,
                evidence=[],
                checked_version_ids=[version_id],
                changed=changed,
            )

    async def first_listed() -> dict:
        return (await client.get(ANALYSES)).json()["items"][0]

    unchecked = await first_listed()
    await add_check("manual", "supported", False)
    confirmed = await first_listed()
    await add_check("source_changed", "contradicted", True)
    after = (await client.get(f"{ANALYSES}/{analysis_id}")).json()
    changed = await first_listed()

    assert (unchecked["changed_since"], confirmed["changed_since"]) == (False, False)
    assert changed["changed_since"] is True
    check = after["claims"][0]["latest_check"]
    assert set(check) == CHECK_KEYS
    assert (check["reason"], check["verdict"], check["changed"]) == (
        "source_changed",
        "contradicted",
        True,
    )
    assert check["probs"] == {"supported": 0.1, "contradicted": 0.8, "insufficient": 0.1}
    assert check["checked_version_ids"] == [str(version_id)]


# --- validation and errors ----------------------------------------------------------------------


async def test_validation_limits(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    await collection_with_docs(client, alice, db, services, [REFUNDS])

    rejected = [
        await post_ask(client, alice, "q" * 2001),
        await post_ask(client, alice, "   "),
        await post_ask(client, alice, "a\u0000b"),
        await post_ask(client, alice, options={"consistency_samples": 6}),
        await post_ask(client, alice, options={"consistency_samples": -1}),
        await post_ask(client, alice, options={"extraction": "magic"}),
        await post_verify(client, alice, "a" * 20001),
        await post_verify(client, alice, ""),
        await post_verify(client, alice, "ok", question="q" * 2001),
        await post_verify(client, alice, "ok", options={"consistency_samples": 1}),
    ]
    # The limits themselves are accepted.
    accepted = [
        await post_ask(client, alice, "q" * 2000),
        await post_ask(client, alice, options={"consistency_samples": 5}),
        await post_verify(client, alice, "a" * 20000, question="q" * 2000),
    ]

    assert [(r.status_code, r.json()["code"]) for r in rejected] == [(422, "validation_error")] * 10
    assert [r.status_code for r in accepted] == [202, 202, 202]


async def test_options_are_stored(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    options = {"extraction": "sentence", "consistency_samples": 3}

    analysis_id = await accepted_id(await post_ask(client, alice, options=options))

    async with db.tenant_tx(tenant_id) as conn:
        row = await analyses.get_analysis(conn, UUID(analysis_id))
    assert row["options"] == options


async def test_blank_verify_question_is_not_stored(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])

    analysis_id = await accepted_id(await post_verify(client, alice, REFUNDS, question="   "))

    async with db.tenant_tx(tenant_id) as conn:
        row = await analyses.get_analysis(conn, UUID(analysis_id))
    assert (row["kind"], row["question"], row["answer"]) == ("verify", None, REFUNDS)


async def test_unreadable_collection_404(client, client2, client3, db, services):
    alice, bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])

    asked = await post_ask(client2, bob, collection_ids=[collection["id"]])
    verified = await post_verify(client2, bob, REFUNDS, collection_ids=[collection["id"]])
    missing = await post_ask(client2, bob, collection_ids=["00000000-0000-4000-8000-000000000000"])

    for response in (asked, verified, missing):
        assert response.status_code == 404
        assert response.json()["code"] == "collection_not_found"
    assert bob["tenant"]["role"] == "member"
    assert (await client2.get(ANALYSES)).json()["items"] == []


async def test_backpressure_route_429(client, client2, client3, db, services, api_app, monkeypatch):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])
    monkeypatch.setattr(api_app.state.settings, "max_active_analyses", 2)
    scope = {"collection_ids": [collection["id"]]}

    first = await post_ask(client, alice, **scope)
    second = await post_verify(client, alice, REFUNDS, **scope)
    third = await post_ask(client, alice, **scope)

    assert (first.status_code, second.status_code) == (202, 202)
    assert third.status_code == 429
    assert third.json()["code"] == "too_many_analyses"
    assert len((await client.get(ANALYSES)).json()["items"]) == 2
