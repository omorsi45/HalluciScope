"""GET /analyses/{id}/graph, POST /claims/{id}/feedback, and GET /claims/{id}/checks."""

import json
from uuid import UUID, uuid4

import pytest

from backend.db.repos import analyses
from tests.helpers.auth import csrf
from tests.test_analysis_routes import (
    ANALYSES,
    CHECK_KEYS,
    EXCHANGES,
    REFUND_QUESTION,
    REFUNDS,
    REFUNDS_45,
    accepted_id,
    background_worker,
    claims_json,
    collection_with_docs,
    grant_read,
    passage_collections,
    post_ask,
    post_verify,
    wait_done,
)
from tests.test_collections import _team
from tests.test_provenance import add_version

pytestmark = pytest.mark.pg

CLAIMS = "/api/v1/claims"
NODE_KEYS = {"id", "type", "label", "data"}
EDGE_KEYS = {"source", "target", "type"}
# The first sentence cites two passages; the second cites a marker with no passage behind it,
# so it adds neither a sentence node nor a `cites` edge.
ANSWER = "Refunds take 30 days [P1, P2]. Exchanges take 14 days [P9]."
# One sentence well over the 120 character label limit.
LONG_SENTENCE = "Refunds take 30 days " + "and apply to every single order " * 8 + "."


async def asked(client, session, db, services, collection_ids, answer, claims) -> dict:
    """Run an ask analysis to the end; the fake model writes `answer` and extracts `claims`."""
    services.llm.responses = [answer, claims_json(*claims)]
    accepted = await post_ask(client, session, REFUND_QUESTION, collection_ids=collection_ids)
    analysis_id = await accepted_id(accepted)
    async with background_worker(db, services):
        body = await wait_done(client, analysis_id)
    assert body["status"] == "succeeded", body
    return body


@pytest.fixture
def verify_done(db, services):
    """Run a verify analysis (sentence extraction, no model call) to the end."""

    async def run(client, session, text: str = REFUNDS) -> dict:
        analysis_id = await accepted_id(
            await post_verify(client, session, text, options={"extraction": "sentence"})
        )
        async with background_worker(db, services):
            body = await wait_done(client, analysis_id)
        assert body["status"] == "succeeded", body
        return body

    return run


async def get_graph(client, analysis_id: str) -> dict:
    response = await client.get(f"{ANALYSES}/{analysis_id}/graph")
    assert response.status_code == 200, response.text
    return response.json()


def edge_keys(graph: dict) -> list[tuple[str, str, str]]:
    return [(e["source"], e["target"], e["type"]) for e in graph["edges"]]


def nodes_of(graph: dict, kind: str) -> dict[str, dict]:
    return {n["id"]: n for n in graph["nodes"] if n["type"] == kind}


async def lineage(db, tenant_id: UUID) -> dict[str, tuple[str, str]]:
    """Passage id -> (version id, document id) for every passage of the tenant."""
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select p.id, p.version_id, v.document_id "
            "from passages p join document_versions v on v.id = p.version_id"
        )
        return {
            str(row["id"]): (str(row["version_id"]), str(row["document_id"]))
            for row in await cur.fetchall()
        }


# --- graph --------------------------------------------------------------------------------------


async def test_graph_shape(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS, EXCHANGES])
    body = await asked(
        client,
        alice,
        db,
        services,
        [collection["id"]],
        ANSWER,
        [(0, REFUNDS), (1, EXCHANGES)],
    )

    graph = await get_graph(client, body["id"])
    again = await get_graph(client, body["id"])

    claims = body["claims"]
    valid = [c for c in body["citations"] if c["status"] == "valid"]
    assert len(claims) == 2
    assert {c["status"] for c in body["citations"]} == {"valid", "unknown_marker"}
    assert set(graph) == {"nodes", "edges"}
    assert all(set(n) == NODE_KEYS for n in graph["nodes"])
    assert all(set(e) == EDGE_KEYS for e in graph["edges"])
    assert {n["type"] for n in graph["nodes"]} == {
        "claim",
        "sentence",
        "passage",
        "version",
        "document",
    }
    assert {e["type"] for e in graph["edges"]} >= {"supports", "cites", "part_of", "version_of"}
    assert {e["type"] for e in graph["edges"]} <= {
        "supports",
        "contradicts",
        "related",
        "cites",
        "part_of",
        "version_of",
        "superseded_by",
    }

    # Node ids: one per claim, per cited sentence, per passage, per version, per document.
    lines = await lineage(db, tenant_id)
    passage_ids = {e["passage_id"] for c in claims for e in c["evidence"]}
    passage_ids |= {c["passage_id"] for c in valid}
    sentences = {f"sentence:{c['sentence_start']}:{c['sentence_end']}" for c in valid}
    assert set(nodes_of(graph, "claim")) == {f"claim:{c['id']}" for c in claims}
    assert set(nodes_of(graph, "sentence")) == sentences
    assert len(sentences) == 1
    assert len([k for k in edge_keys(graph) if k[2] == "cites"]) == 2
    assert set(nodes_of(graph, "passage")) == {f"passage:{p}" for p in passage_ids}
    assert set(nodes_of(graph, "version")) == {f"version:{lines[p][0]}" for p in passage_ids}
    assert set(nodes_of(graph, "document")) == {f"document:{lines[p][1]}" for p in passage_ids}

    # Edges: claim to passage by relation, sentence to passage for valid citations only.
    expected = set()
    for claim in claims:
        for item in claim["evidence"]:
            expected.add(
                (f"claim:{claim['id']}", f"passage:{item['passage_id']}", item["relation"])
            )
    for citation in valid:
        sentence = f"sentence:{citation['sentence_start']}:{citation['sentence_end']}"
        expected.add((sentence, f"passage:{citation['passage_id']}", "cites"))
    for passage_id in passage_ids:
        version_id, document_id = lines[passage_id]
        expected.add((f"passage:{passage_id}", f"version:{version_id}", "part_of"))
        expected.add((f"version:{version_id}", f"document:{document_id}", "version_of"))
    assert set(edge_keys(graph)) == expected

    # Labels.
    answer = body["answer"]
    for claim in claims:
        assert nodes_of(graph, "claim")[f"claim:{claim['id']}"]["label"] == claim["text"]
    for citation in valid:
        sentence = f"sentence:{citation['sentence_start']}:{citation['sentence_end']}"
        text = answer[citation["sentence_start"] : citation["sentence_end"]]
        assert nodes_of(graph, "sentence")[sentence]["label"] == text
    assert {n["label"] for n in nodes_of(graph, "passage").values()} == {REFUNDS, EXCHANGES}
    assert {n["label"] for n in nodes_of(graph, "version").values()} == {"v1"}
    assert {n["label"] for n in nodes_of(graph, "document").values()} == {
        "Document 1",
        "Document 2",
    }

    # Every edge joins two nodes; the lists are sorted and free of duplicates; a second call
    # returns the same graph.
    node_ids = [n["id"] for n in graph["nodes"]]
    keys = edge_keys(graph)
    assert {k[0] for k in keys} | {k[1] for k in keys} <= set(node_ids)
    assert node_ids == sorted(set(node_ids))
    assert keys == sorted(set(keys))
    assert again == graph


async def test_graph_labels_are_cut_at_120_characters(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await collection_with_docs(client, alice, db, services, [LONG_SENTENCE])
    answer = f"{LONG_SENTENCE[:-1]} [P1]."
    body = await asked(client, alice, db, services, [collection["id"]], answer, [(0, answer[:300])])

    graph = await get_graph(client, body["id"])

    assert len(LONG_SENTENCE) > 120
    (claim,) = nodes_of(graph, "claim").values()
    (sentence,) = nodes_of(graph, "sentence").values()
    (passage,) = nodes_of(graph, "passage").values()
    assert claim["label"] == body["claims"][0]["text"][:120]
    assert sentence["label"] == answer[:120]
    assert passage["label"] == LONG_SENTENCE[:120]
    assert [len(n["label"]) for n in (claim, sentence, passage)] == [120, 120, 120]


async def test_graph_superseded_edge(client, client2, client3, db, services, verify_done):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    await collection_with_docs(client, alice, db, services, [REFUNDS])
    body = await verify_done(client, alice)
    (item,) = body["claims"][0]["evidence"]
    old_version, document_id = item["version_id"], item["document_id"]
    before = await get_graph(client, body["id"])

    await add_version(db, services, tenant_id, UUID(document_id), REFUNDS_45)
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select id from document_versions where document_id = %s and version_number = 2",
            (UUID(document_id),),
        )
        new_version = str((await cur.fetchone())["id"])
    after = await get_graph(client, body["id"])

    assert [k for k in edge_keys(before) if k[2] == "superseded_by"] == []
    assert [k for k in edge_keys(after) if k[2] == "superseded_by"] == [
        (f"version:{old_version}", f"version:{new_version}", "superseded_by")
    ]
    versions = nodes_of(after, "version")
    assert {n["id"]: n["label"] for n in versions.values()} == {
        f"version:{old_version}": "v1",
        f"version:{new_version}": "v2",
    }
    assert set(nodes_of(after, "document")) == {f"document:{document_id}"}
    version_of = {k for k in edge_keys(after) if k[2] == "version_of"}
    assert version_of == {
        (f"version:{old_version}", f"document:{document_id}", "version_of"),
        (f"version:{new_version}", f"document:{document_id}", "version_of"),
    }
    # The analysis still points at the version it was reached on.
    assert [k for k in edge_keys(after) if k[2] == "part_of"] == [
        (f"passage:{item['passage_id']}", f"version:{old_version}", "part_of")
    ]
    node_ids = {n["id"] for n in after["nodes"]}
    assert all(k[0] in node_ids and k[1] in node_ids for k in edge_keys(after))


async def test_graph_redacts_unreadable(client, client2, client3, db, services):
    alice, bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    secret = await collection_with_docs(client, alice, db, services, [REFUNDS], "Secret")
    open_ = await collection_with_docs(client, alice, db, services, [EXCHANGES], "Open")
    await grant_read(client, alice, secret["id"], bob)
    await grant_read(client, alice, open_["id"], bob)
    body = await asked(
        client2,
        bob,
        db,
        services,
        [secret["id"], open_["id"]],
        "Refunds take 30 days [P1]. Exchanges take 14 days [P2].",
        [(0, REFUNDS), (1, EXCHANGES)],
    )
    before = await get_graph(client2, body["id"])
    await grant_read(client, alice, secret["id"])  # Bob loses read on the secret collection
    after = await get_graph(client2, body["id"])
    owner_view = await get_graph(client, body["id"])

    owners = await passage_collections(db, tenant_id)
    lines = await lineage(db, tenant_id)
    secret_passages = {p for p, c in owners.items() if c == secret["id"]}
    open_passages = {p for p, c in owners.items() if c == open_["id"]}
    secret_nodes = {f"passage:{p}" for p in secret_passages}
    secret_versions = {f"version:{lines[p][0]}" for p in secret_passages}
    secret_documents = {f"document:{lines[p][1]}" for p in secret_passages}
    assert len(secret_passages) == len(open_passages) == 1

    # While the grant stands nothing is withheld.
    assert secret_nodes <= set(nodes_of(before, "passage"))
    assert secret_versions <= set(nodes_of(before, "version"))
    assert secret_documents <= set(nodes_of(before, "document"))
    assert all(n["label"] is not None for n in nodes_of(before, "passage").values())

    passages = nodes_of(after, "passage")
    (hidden_id,) = secret_nodes
    assert passages[hidden_id]["label"] is None
    assert passages[hidden_id]["data"] == {"redacted": True}
    (shown,) = [n for n in passages.values() if n["id"] not in secret_nodes]
    assert shown["label"] == EXCHANGES
    assert shown["data"]["redacted"] is False
    # Neither the version nor the document of the withheld passage is named.
    assert not secret_versions & set(nodes_of(after, "version"))
    assert not secret_documents & set(nodes_of(after, "document"))
    keys = edge_keys(after)
    assert not [k for k in keys if k[0] in secret_nodes and k[2] in ("part_of", "version_of")]
    assert not [k for k in keys if k[0] in secret_versions or k[1] in secret_versions]
    assert not [k for k in keys if k[0] in secret_documents or k[1] in secret_documents]
    # The claim and sentence edges to the withheld passage stay: only its source is withheld.
    into_secret = {k for k in keys if k[1] in secret_nodes}
    assert into_secret == {k for k in edge_keys(before) if k[1] in secret_nodes}
    assert {k[2] for k in into_secret} >= {"cites"}
    node_ids = {n["id"] for n in after["nodes"]}
    assert all(k[0] in node_ids and k[1] in node_ids for k in keys)
    # The owner reads everything, as Bob did before the revocation.
    assert owner_view == before


async def test_graph_of_invisible_analysis_404(client, client2, client3, db, services, verify_done):
    alice, bob, carol = await _team(client, client2, client3, db)
    await collection_with_docs(client, alice, db, services, [REFUNDS])
    body = await verify_done(client, alice)

    member = await client2.get(f"{ANALYSES}/{body['id']}/graph")
    foreign = await client3.get(f"{ANALYSES}/{body['id']}/graph")
    missing = await client.get(f"{ANALYSES}/{uuid4()}/graph")

    assert (bob["tenant"]["role"], carol["tenant"]["id"] != alice["tenant"]["id"]) == (
        "member",
        True,
    )
    for response in (member, foreign, missing):
        assert response.status_code == 404
        assert response.json()["code"] == "analysis_not_found"


# --- feedback -----------------------------------------------------------------------------------


async def feedback_rows(db, tenant_id: UUID) -> list[dict]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select claim_id, user_id, label, note from feedback order by user_id, label"
        )
        return await cur.fetchall()


async def test_feedback_upsert_route(client, client2, client3, db, services, verify_done):
    alice, bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection = await collection_with_docs(client, alice, db, services, [REFUNDS])
    await grant_read(client, alice, collection["id"], bob)
    body = await verify_done(client2, bob)
    url = f"{CLAIMS}/{body['claims'][0]['id']}/feedback"

    first = await client2.post(url, json={"label": "correct", "note": "checked"}, headers=csrf(bob))
    second = await client2.post(url, json={"label": "incorrect"}, headers=csrf(bob))
    by_owner = await client.post(url, json={"label": "correct"}, headers=csrf(alice))

    assert [r.status_code for r in (first, second, by_owner)] == [201, 201, 201]
    assert set(first.json()) == {"id"}
    UUID(first.json()["id"])
    # The same user gets the same row back, holding the last label and note.
    assert second.json()["id"] == first.json()["id"]
    assert by_owner.json()["id"] != first.json()["id"]
    rows = await feedback_rows(db, tenant_id)
    mine = [r for r in rows if str(r["user_id"]) == bob["user"]["id"]]
    theirs = [r for r in rows if str(r["user_id"]) == alice["user"]["id"]]
    assert [(r["label"], r["note"]) for r in mine] == [("incorrect", None)]
    assert [(r["label"], r["note"]) for r in theirs] == [("correct", None)]
    assert len(rows) == 2


async def test_feedback_on_invisible_claim_404(client, client2, client3, db, services, verify_done):
    alice, bob, carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    await collection_with_docs(client, alice, db, services, [REFUNDS])
    body = await verify_done(client, alice)
    claim_id = body["claims"][0]["id"]
    payload = {"label": "correct"}

    # Bob is a member who did not create the analysis; Carol belongs to another tenant.
    attempts = [
        await client2.post(f"{CLAIMS}/{claim_id}/feedback", json=payload, headers=csrf(bob)),
        await client3.post(f"{CLAIMS}/{claim_id}/feedback", json=payload, headers=csrf(carol)),
        await client.post(f"{CLAIMS}/{uuid4()}/feedback", json=payload, headers=csrf(alice)),
    ]
    reads = [
        await client2.get(f"{CLAIMS}/{claim_id}/checks"),
        await client3.get(f"{CLAIMS}/{claim_id}/checks"),
        await client.get(f"{CLAIMS}/{uuid4()}/checks"),
    ]

    for response in (*attempts, *reads):
        assert response.status_code == 404, response.text
        assert response.json()["code"] == "claim_not_found"
    assert await feedback_rows(db, tenant_id) == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"label": "maybe"},
        {"label": "correct", "note": "n" * 1001},
        {"label": "correct", "note": "a\ud800b"},
        {"label": "correct", "note": "a\u0000b"},
    ],
    ids=["no_label", "bad_label", "note_too_long", "surrogate", "nul"],
)
async def test_feedback_validation(client, client2, client3, db, services, verify_done, payload):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    await collection_with_docs(client, alice, db, services, [REFUNDS])
    body = await verify_done(client, alice)

    response = await client.post(
        f"{CLAIMS}/{body['claims'][0]['id']}/feedback",
        content=json.dumps(payload),
        headers={**csrf(alice), "Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert await feedback_rows(db, tenant_id) == []


async def test_feedback_note_limit_is_accepted(client, client2, client3, db, services, verify_done):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    await collection_with_docs(client, alice, db, services, [REFUNDS])
    body = await verify_done(client, alice)

    response = await client.post(
        f"{CLAIMS}/{body['claims'][0]['id']}/feedback",
        json={"label": "incorrect", "note": "n" * 1000},
        headers=csrf(alice),
    )

    assert response.status_code == 201
    (row,) = await feedback_rows(db, tenant_id)
    assert (row["label"], row["note"]) == ("incorrect", "n" * 1000)


# --- claim checks -------------------------------------------------------------------------------


async def test_checks_route_empty_then_listed(client, client2, client3, db, services, verify_done):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    await collection_with_docs(client, alice, db, services, [REFUNDS])
    body = await verify_done(client, alice)
    claim_id = UUID(body["claims"][0]["id"])
    version_id = UUID(body["claims"][0]["evidence"][0]["version_id"])
    url = f"{CLAIMS}/{claim_id}/checks"

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

    empty = await client.get(url)
    await add_check("manual", "supported", False)
    one = await client.get(url)
    await add_check("source_changed", "contradicted", True)
    two = await client.get(url)

    assert (empty.status_code, empty.json()) == (200, [])
    assert one.status_code == 200
    (check,) = one.json()
    assert set(check) == CHECK_KEYS
    assert (check["reason"], check["verdict"], check["changed"]) == ("manual", "supported", False)
    assert check["probs"] == {"supported": 0.1, "contradicted": 0.8, "insufficient": 0.1}
    assert check["checked_version_ids"] == [str(version_id)]
    # Newest first.
    assert [c["reason"] for c in two.json()] == ["source_changed", "manual"]
    assert [c["changed"] for c in two.json()] == [True, False]
    assert two.json()[1] == check
