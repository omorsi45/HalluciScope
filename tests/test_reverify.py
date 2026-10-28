"""Re-verification (R3): which claims a changed source touches, the checks written for them,
manual re-verification, the two jobs and their failure hooks, principal_for_user."""

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from backend.auth.service import Principal, principal_for_user
from backend.db.repos import members, tenants, users
from backend.jobs.queue import Job
from backend.jobs.registry import build_failure_handlers
from backend.jobs.worker import JobContext
from backend.provenance.reverify import RecheckCandidate, claims_to_recheck, recheck_claims
from tests.fakes import KeywordReranker, make_test_calibrator
from tests.helpers.auth import csrf
from tests.pg_fixtures import drain, seed_docs
from tests.test_analysis_routes import (
    ANALYSES,
    EXCHANGES,
    REFUNDS,
    TWO_FACTS,
    accepted_id,
    background_worker,
    grant_read,
    passage_collections,
    post_verify,
    wait_done,
)
from tests.test_collections import _create, _team
from tests.test_document_routes import post_version
from tests.test_provenance import add_version
from tests.test_run_analysis import json_logs

pytestmark = pytest.mark.pg

REFUNDS_45 = "Refunds take 45 days."
SHIPPING = "Shipping is free."
# Five one-passage documents. The refund claim reranks the first four at 4, 2, 2 and 2 (shared
# query tokens), so they are its four evidence items; the fifth shares none and is not.
FIVE_DOCS = [
    REFUNDS,
    EXCHANGES,
    "Returns take 10 days.",
    "Orders take 5 days.",
    "Warranty lasts 2 years.",
]
WARRANTY = 4  # index of the fifth document


async def fetch(db, tenant_id: UUID, query: str, params: tuple = ()) -> list[dict]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def checks_of(db, tenant_id: UUID) -> list[dict]:
    return await fetch(db, tenant_id, "select * from claim_checks order by created_at, id")


async def job_rows(db, kind: str) -> list[dict]:
    async with db.system_tx() as conn:
        cur = await conn.execute("select * from jobs where kind = %s order by id", (kind,))
        return await cur.fetchall()


async def version_ids(db, tenant_id: UUID, document_id: UUID) -> list[UUID]:
    rows = await fetch(
        db,
        tenant_id,
        "select id from document_versions where document_id = %s order by version_number",
        (document_id,),
    )
    return [row["id"] for row in rows]


async def collection_docs(client, session, db, services, texts, name="Policies", **seed):
    collection = await _create(client, session, name)
    document_ids = await seed_docs(
        db, services, UUID(session["tenant"]["id"]), UUID(collection["id"]), texts, **seed
    )
    return collection, document_ids


async def run_verify(client, db, services, session, answer, **body) -> dict:
    """Verify the answer sentence by sentence and return the finished analysis."""
    accepted = await post_verify(
        client, session, answer, options={"extraction": "sentence"}, **body
    )
    analysis_id = await accepted_id(accepted)
    async with background_worker(db, services):
        analysis = await wait_done(client, analysis_id)
    assert analysis["status"] == "succeeded", analysis
    return analysis


async def upload_version(client, session, document_id, db, services, text: str) -> None:
    """Upload a new version through the API and run the worker: ingestion, then re-verification."""
    response = await post_version(
        client, session, document_id, text.encode(), filename="policy.txt"
    )
    assert response.status_code == 202, response.text
    await drain(db, services)


async def post_reverify(client, session, analysis_id: str):
    return await client.post(f"{ANALYSES}/{analysis_id}/reverify", headers=csrf(session))


@dataclass(frozen=True)
class Corpus:
    alice: dict
    tenant_id: UUID
    collection_id: UUID
    document_ids: list[UUID]
    analysis: dict  # Alice's finished verify analysis over the five documents

    @property
    def analysis_id(self) -> UUID:
        return UUID(self.analysis["id"])

    @property
    def claim_ids(self) -> list[UUID]:
        return [UUID(claim["id"]) for claim in self.analysis["claims"]]


async def build_corpus(client, client2, client3, db, services, answer: str) -> Corpus:
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection, document_ids = await collection_docs(client, alice, db, services, FIVE_DOCS)
    analysis = await run_verify(
        client, db, services, alice, answer, collection_ids=[collection["id"]]
    )
    for claim in analysis["claims"]:
        titles = {item["document_title"] for item in claim["evidence"]}
        # Four evidence items, none from the warranty document: the trigger tests depend on it.
        assert len(claim["evidence"]) == 4, titles
        assert "Document 5" not in titles, titles
        assert min(item["rerank_score"] for item in claim["evidence"]) == 2.0
    return Corpus(
        alice,
        UUID(alice["tenant"]["id"]),
        UUID(collection["id"]),
        document_ids,
        analysis,
    )


@pytest.fixture
async def corpus(client, client2, client3, db, services) -> Corpus:
    return await build_corpus(client, client2, client3, db, services, REFUNDS)


async def candidates(db, services, corpus: Corpus, index: int, *, evidence_k=4, reranker=None):
    """What `claims_to_recheck` finds for the latest version of document `index`."""
    document_id = corpus.document_ids[index]
    old, new = (await version_ids(db, corpus.tenant_id, document_id))[-2:]
    async with db.tenant_tx(corpus.tenant_id) as conn:
        return await claims_to_recheck(
            conn,
            reranker or services.reranker,
            document_id=document_id,
            old_version_id=old,
            new_version_id=new,
            evidence_k=evidence_k,
        )


# --- the end to end behavior --------------------------------------------------------------------


async def test_changed_evidence_flips_verdict(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, (document_id,) = await collection_docs(client, alice, db, services, [REFUNDS])
    analysis = await run_verify(
        client, db, services, alice, REFUNDS, collection_ids=[collection["id"]]
    )
    (claim,) = analysis["claims"]
    assert claim["verdict"] == "supported"
    assert await checks_of(db, tenant_id) == []

    with json_logs() as stream:
        await upload_version(client, alice, document_id, db, services, REFUNDS_45)

    _v1, v2 = await version_ids(db, tenant_id, document_id)
    (passage,) = await fetch(db, tenant_id, "select id from passages where version_id = %s", (v2,))
    (check,) = await checks_of(db, tenant_id)
    assert (check["claim_id"], check["reason"], check["verdict"], check["changed"]) == (
        UUID(claim["id"]),
        "source_changed",
        "contradicted",
        True,
    )
    assert check["p_contradicted"] > check["p_supported"]
    assert check["checked_version_ids"] == [v2]
    (item,) = check["evidence"]
    assert item == {
        "passage_id": str(passage["id"]),
        "version_id": str(v2),
        "relation": "contradicts",
        "rank": 1,
        "span_start": 0,
        "span_end": len(REFUNDS_45),
        "nli": {"entail": 0.05, "contra": 0.9, "neutral": 0.05},
    }

    after = (await client.get(f"{ANALYSES}/{analysis['id']}")).json()
    (after_claim,) = after["claims"]
    assert after_claim["verdict"] == "supported"  # the stored verdict is history, never rewritten
    latest = after_claim["latest_check"]
    assert (latest["id"], latest["reason"], latest["verdict"], latest["changed"]) == (
        str(check["id"]),
        "source_changed",
        "contradicted",
        True,
    )
    assert latest["checked_version_ids"] == [str(v2)]
    listed = (await client.get(ANALYSES)).json()["items"]
    assert [(i["id"], i["changed_since"]) for i in listed] == [(analysis["id"], True)]
    (job,) = await job_rows(db, "reverify_document")
    assert (job["status"], job["priority"]) == ("succeeded", 5)
    logs = stream.getvalue()
    assert "claims rechecked" in logs  # the capture works
    for text in (REFUNDS, "Refunds take", "30 days", "45 days"):
        assert text not in logs, text


async def test_unrelated_change_skips_claim(client, client2, client3, db, services):
    # Evidence depth 1: the claim's one evidence item is the refunds passage (score 4), so the
    # other document is not evidence and the claim does not have too little evidence.
    services.calibrator = make_test_calibrator(evidence_k=1)
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, (_refunds, other) = await collection_docs(
        client, alice, db, services, [REFUNDS, EXCHANGES]
    )
    await run_verify(client, db, services, alice, REFUNDS, collection_ids=[collection["id"]])

    # Scores 2 against the claim, below the 4 stored for its evidence.
    await upload_version(client, alice, other, db, services, "Exchanges take 21 days.")

    (job,) = await job_rows(db, "reverify_document")
    assert job["status"] == "succeeded"
    assert await checks_of(db, tenant_id) == []


async def test_related_change_in_another_document_rechecks_claim(
    client, client2, client3, db, services
):
    services.calibrator = make_test_calibrator(evidence_k=1)
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, (_refunds, other) = await collection_docs(
        client, alice, db, services, [REFUNDS, EXCHANGES]
    )
    await run_verify(client, db, services, alice, REFUNDS, collection_ids=[collection["id"]])

    # Scores 4 against the claim: equal to its lowest stored evidence score, which counts.
    await upload_version(client, alice, other, db, services, REFUNDS)

    (check,) = await checks_of(db, tenant_id)
    assert (check["reason"], check["verdict"], check["changed"]) == (
        "source_changed",
        "supported",
        False,
    )


async def test_new_passage_triggers_recheck(client, client2, client3, db, services):
    services.calibrator = make_test_calibrator(evidence_k=1)
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, (_refunds, exchanges) = await collection_docs(
        client, alice, db, services, [REFUNDS, EXCHANGES]
    )
    analysis = await run_verify(
        client, db, services, alice, SHIPPING, collection_ids=[collection["id"]]
    )
    (claim,) = analysis["claims"]
    assert claim["verdict"] == "insufficient"

    await upload_version(client, alice, exchanges, db, services, f"{EXCHANGES} {SHIPPING}")

    (check,) = await checks_of(db, tenant_id)
    assert (check["claim_id"], check["reason"], check["verdict"], check["changed"]) == (
        UUID(claim["id"]),
        "source_changed",
        "supported",
        True,
    )


async def test_as_of_analysis_never_rechecked(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, (document_id,) = await collection_docs(
        client,
        alice,
        db,
        services,
        [REFUNDS],
        effective_from=datetime.now(UTC) - timedelta(days=1),
    )
    scope = {"collection_ids": [collection["id"]]}
    live = await run_verify(client, db, services, alice, REFUNDS, **scope)
    as_of = await run_verify(
        client, db, services, alice, REFUNDS, as_of=datetime.now(UTC).isoformat(), **scope
    )
    assert [c["verdict"] for c in as_of["claims"]] == ["supported"]

    await upload_version(client, alice, document_id, db, services, REFUNDS_45)

    # The live analysis is rechecked and the as-of one is not, so the job did run.
    (check,) = await checks_of(db, tenant_id)
    assert check["claim_id"] == UUID(live["claims"][0]["id"])
    as_of_claim = (await client.get(f"{ANALYSES}/{as_of['id']}")).json()["claims"][0]
    assert as_of_claim["latest_check"] is None
    # Not even a direct request touches it: its scope is pinned to what was known then.
    written = await recheck_claims(
        services, tenant_id, UUID(as_of["id"]), [UUID(as_of_claim["id"])], "manual"
    )
    assert written == 0
    assert len(await checks_of(db, tenant_id)) == 1


async def test_creator_lost_access_scope_shrinks(client, client2, client3, db, services):
    alice, bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    secret, _ = await collection_docs(client, alice, db, services, [REFUNDS], "Secret")
    open_, (open_doc,) = await collection_docs(client, alice, db, services, [REFUNDS], "Open")
    await grant_read(client, alice, secret["id"], bob)
    await grant_read(client, alice, open_["id"], bob)
    analysis = await run_verify(
        client2, db, services, bob, REFUNDS, collection_ids=[secret["id"], open_["id"]]
    )
    owners = await passage_collections(db, tenant_id)
    seen = {owners[item["passage_id"]] for item in analysis["claims"][0]["evidence"]}
    assert seen == {secret["id"], open_["id"]}  # both collections were evidence
    await grant_read(client, alice, secret["id"])  # Bob loses read on the secret collection

    await upload_version(client, alice, open_doc, db, services, REFUNDS_45)

    (check,) = await checks_of(db, tenant_id)
    assert (check["reason"], check["verdict"]) == ("source_changed", "contradicted")
    # The secret document still says 30 days. Had it been searched, it would have supported
    # the claim and been among the evidence.
    owners = await passage_collections(db, tenant_id)
    assert [owners[item["passage_id"]] for item in check["evidence"]] == [open_["id"]]
    assert len(check["checked_version_ids"]) == 1


@pytest.mark.parametrize("how", ["removed", "disabled"])
async def test_creator_removed_no_check(client, client2, client3, db, services, how):
    alice, bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, (document_id,) = await collection_docs(client, alice, db, services, [REFUNDS])
    await grant_read(client, alice, collection["id"], bob)
    scope = {"collection_ids": [collection["id"]]}
    bobs = await run_verify(client2, db, services, bob, REFUNDS, **scope)
    alices = await run_verify(client, db, services, alice, REFUNDS, **scope)
    bob_id = UUID(bob["user"]["id"])
    async with db.system_tx() as conn:
        if how == "removed":
            await members.delete_membership(conn, tenant_id, bob_id)
        else:
            await conn.execute("update users set disabled_at = now() where id = %s", (bob_id,))

    await upload_version(client, alice, document_id, db, services, REFUNDS_45)

    # Alice's claim is rechecked, which shows the job ran; Bob's is left alone.
    (check,) = await checks_of(db, tenant_id)
    assert check["claim_id"] == UUID(alices["claims"][0]["id"])
    assert check["claim_id"] != UUID(bobs["claims"][0]["id"])
    (job,) = await job_rows(db, "reverify_document")
    assert job["status"] == "succeeded"


async def test_manual_reverify_route(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, _ = await collection_docs(client, alice, db, services, [REFUNDS, EXCHANGES])
    analysis = await run_verify(
        client, db, services, alice, TWO_FACTS, collection_ids=[collection["id"]]
    )
    assert len(analysis["claims"]) == 2

    response = await post_reverify(client, alice, analysis["id"])

    assert (response.status_code, response.json()) == (202, {"status": "queued"})
    (job,) = await job_rows(db, "reverify_analysis")
    assert job["payload"] == {"analysis_id": analysis["id"]}
    assert (job["tenant_id"], job["priority"], job["status"]) == (tenant_id, 5, "queued")
    await drain(db, services)
    checks = await checks_of(db, tenant_id)
    assert {c["claim_id"]: (c["reason"], c["verdict"], c["changed"]) for c in checks} == {
        UUID(claim["id"]): ("manual", "supported", False) for claim in analysis["claims"]
    }
    assert len(checks) == 2
    after = (await client.get(f"{ANALYSES}/{analysis['id']}")).json()
    assert [c["latest_check"]["reason"] for c in after["claims"]] == ["manual", "manual"]
    assert (await client.get(ANALYSES)).json()["items"][0]["changed_since"] is False


async def test_changed_compares_with_the_latest_check(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, (document_id,) = await collection_docs(client, alice, db, services, [REFUNDS])
    analysis = await run_verify(
        client, db, services, alice, REFUNDS, collection_ids=[collection["id"]]
    )
    await upload_version(client, alice, document_id, db, services, REFUNDS_45)

    await post_reverify(client, alice, analysis["id"])
    await drain(db, services)

    first, second = await checks_of(db, tenant_id)
    # The original verdict was supported. The second check agrees with the first, so no change.
    assert (first["reason"], first["verdict"], first["changed"]) == (
        "source_changed",
        "contradicted",
        True,
    )
    assert (second["reason"], second["verdict"], second["changed"]) == (
        "manual",
        "contradicted",
        False,
    )


async def test_manual_reverify_as_of_409(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection, _ = await collection_docs(
        client, alice, db, services, [REFUNDS], effective_from=datetime.now(UTC) - timedelta(days=1)
    )
    as_of = await run_verify(
        client,
        db,
        services,
        alice,
        REFUNDS,
        as_of=datetime.now(UTC).isoformat(),
        collection_ids=[collection["id"]],
    )

    response = await post_reverify(client, alice, as_of["id"])

    assert response.status_code == 409
    assert response.json()["code"] == "not_live"
    assert await job_rows(db, "reverify_analysis") == []


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
async def test_manual_reverify_unfinished_409(client, client2, client3, db, services, status):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, _ = await collection_docs(client, alice, db, services, [REFUNDS])
    analysis_id = await accepted_id(
        await post_verify(client, alice, REFUNDS, collection_ids=[collection["id"]])
    )
    async with db.tenant_tx(tenant_id) as conn:
        await conn.execute("update analyses set status = %s", (status,))

    response = await post_reverify(client, alice, analysis_id)

    assert response.status_code == 409
    assert response.json()["code"] == "not_finished"
    assert await job_rows(db, "reverify_analysis") == []


async def test_manual_reverify_visibility(client, client2, client3, db, services):
    alice, bob, carol = await _team(client, client2, client3, db)
    collection, _ = await collection_docs(client, alice, db, services, [REFUNDS])
    await grant_read(client, alice, collection["id"], bob)
    scope = {"collection_ids": [collection["id"]]}
    alices = await run_verify(client, db, services, alice, REFUNDS, **scope)
    bobs = await run_verify(client2, db, services, bob, REFUNDS, **scope)

    other_member = await post_reverify(client2, bob, alices["id"])
    other_tenant = await post_reverify(client3, carol, alices["id"])
    unknown = await post_reverify(client, alice, str(uuid4()))
    assert [r.status_code for r in (other_member, other_tenant, unknown)] == [404, 404, 404]
    assert {r.json()["code"] for r in (other_member, other_tenant, unknown)} == {
        "analysis_not_found"
    }
    assert await job_rows(db, "reverify_analysis") == []

    creator = await post_reverify(client2, bob, bobs["id"])
    owner = await post_reverify(client, alice, bobs["id"])  # an owner may re-verify any analysis

    assert (creator.status_code, owner.status_code) == (202, 202)
    assert [j["payload"] for j in await job_rows(db, "reverify_analysis")] == [
        {"analysis_id": bobs["id"]}
    ] * 2


@pytest.mark.parametrize("calibrator", ["missing", "mismatch"])
@pytest.mark.parametrize("via", ["document", "analysis"])
async def test_job_fails_permanently_without_a_usable_calibrator(
    client, client2, client3, db, services, via, calibrator
):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    collection, (document_id,) = await collection_docs(client, alice, db, services, [REFUNDS])
    analysis = await run_verify(
        client, db, services, alice, REFUNDS, collection_ids=[collection["id"]]
    )
    services.calibrator = (
        None if calibrator == "missing" else make_test_calibrator(models={"nli": "other-nli"})
    )

    if via == "document":
        await upload_version(client, alice, document_id, db, services, REFUNDS_45)
    else:
        await post_reverify(client, alice, analysis["id"])
        await drain(db, services)

    (job,) = await job_rows(db, f"reverify_{via}")
    assert (job["status"], job["attempts"], job["last_error"]) == (
        "failed",
        1,
        f"calibrator_{calibrator}",
    )
    assert await checks_of(db, tenant_id) == []


# --- claims_to_recheck: the R3 triggers ---------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "text", "evidence_k", "trigger"),
    [
        # Evidence text that is gone from the new version wins over everything else.
        (0, REFUNDS_45, 4, "evidence_changed"),
        (0, REFUNDS_45, 5, "evidence_changed"),
        # Too few evidence links (4 < 5) wins over a relevant new passage.
        (WARRANTY, "Warranty lasts 3 years.", 5, "thin_evidence"),
        (WARRANTY, REFUNDS, 5, "thin_evidence"),
        # Enough evidence: only a changed passage that reranks at least as well as the weakest
        # stored evidence item (2) counts. 4 and 2 do, 0 does not.
        (WARRANTY, REFUNDS, 4, "new_passage"),
        (WARRANTY, "Orders take 7 days.", 4, "new_passage"),
        (WARRANTY, "Warranty lasts 3 years.", 4, None),
    ],
)
async def test_trigger(db, services, corpus, index, text, evidence_k, trigger):
    await add_version(db, services, corpus.tenant_id, corpus.document_ids[index], text)

    found = await candidates(db, services, corpus, index, evidence_k=evidence_k)

    expected = (
        []
        if trigger is None
        else [RecheckCandidate(corpus.claim_ids[0], corpus.analysis_id, trigger)]
    )
    assert found == expected


def long_sentence(lead: str, tag: str) -> str:
    """One sentence of about 450 characters: two of them never share a passage."""
    return f"{lead} " + " ".join(f"{tag}{i}" for i in range(60)) + "."


async def test_an_evidence_passage_kept_in_the_new_version_is_not_evidence_changed(
    client, client2, client3, db, services
):
    services.calibrator = make_test_calibrator(evidence_k=1)
    alice, _bob, _carol = await _team(client, client2, client3, db)
    tenant_id = UUID(alice["tenant"]["id"])
    kept = long_sentence("Refunds take 30 days", "alpha")
    collection, (document_id,) = await collection_docs(
        client, alice, db, services, [f"{kept} {long_sentence('Warranty lasts 2 years', 'beta')}"]
    )
    analysis = await run_verify(
        client, db, services, alice, REFUNDS, collection_ids=[collection["id"]]
    )
    (claim,) = analysis["claims"]
    assert [item["text"] for item in claim["evidence"]] == [kept]  # the one evidence item
    # Version 2 rewrites only the second passage.
    await add_version(
        db,
        services,
        tenant_id,
        document_id,
        f"{kept} {long_sentence('Warranty lasts 3 years', 'gamma')}",
    )
    old, new = await version_ids(db, tenant_id, document_id)

    async with db.tenant_tx(tenant_id) as conn:
        found = await claims_to_recheck(
            conn,
            services.reranker,
            document_id=document_id,
            old_version_id=old,
            new_version_id=new,
            evidence_k=1,
        )

    # The evidence text survives, the changed passage scores 0 against the claim's 4, and one
    # link is as many as the evidence depth asks for.
    assert found == []


async def test_a_version_with_nothing_new_flags_no_thin_evidence(db, services, corpus):
    # A version that only drops text has no changed passage, so a claim with too few evidence
    # links (4 < 5) has no reason to be rechecked.
    document_id = corpus.document_ids[WARRANTY]
    (version_id,) = await version_ids(db, corpus.tenant_id, document_id)

    async with db.tenant_tx(corpus.tenant_id) as conn:
        found = await claims_to_recheck(
            conn,
            services.reranker,
            document_id=document_id,
            old_version_id=version_id,
            new_version_id=version_id,
            evidence_k=5,
        )

    assert found == []


async def test_links_without_rerank_scores_count_as_a_match(db, services, corpus):
    async with db.tenant_tx(corpus.tenant_id) as conn:
        await conn.execute("update evidence_links set rerank_score = null")
    await add_version(db, services, corpus.tenant_id, corpus.document_ids[WARRANTY], "Gone.")

    found = await candidates(db, services, corpus, WARRANTY)

    assert found == [RecheckCandidate(corpus.claim_ids[0], corpus.analysis_id, "new_passage")]


async def test_only_analyses_over_the_documents_collection_are_considered(
    client, db, services, corpus
):
    other, _ = await collection_docs(client, corpus.alice, db, services, [REFUNDS], "Other")
    await run_verify(client, db, services, corpus.alice, REFUNDS, collection_ids=[other["id"]])
    await add_version(db, services, corpus.tenant_id, corpus.document_ids[0], REFUNDS_45)

    found = await candidates(db, services, corpus, 0)

    # The other analysis has one evidence link (too few) but never looked at this collection.
    assert found == [RecheckCandidate(corpus.claim_ids[0], corpus.analysis_id, "evidence_changed")]


class RecordingReranker(KeywordReranker):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def score(self, query, passages):
        self.calls.append((query, list(passages), threading.get_ident()))
        return super().score(query, passages)


async def test_reranking_runs_off_the_event_loop_once_per_claim(
    client, client2, client3, db, services
):
    corpus = await build_corpus(client, client2, client3, db, services, TWO_FACTS)
    assert len(corpus.claim_ids) == 2
    await add_version(
        db, services, corpus.tenant_id, corpus.document_ids[WARRANTY], "Warranty lasts 3 years."
    )
    reranker = RecordingReranker()

    found = await candidates(db, services, corpus, WARRANTY, reranker=reranker)

    assert found == []  # 0 is below the weakest stored evidence score
    assert [(query, passages) for query, passages, _ in reranker.calls] == [
        (REFUNDS, ["Warranty lasts 3 years."]),
        (EXCHANGES, ["Warranty lasts 3 years."]),
    ]
    assert threading.get_ident() not in {thread for _, _, thread in reranker.calls}


# --- principal_for_user -------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["member", "removed", "disabled", "unknown_user", "other_tenant"])
async def test_principal_for_user(db, case):
    async with db.system_tx() as conn:
        tenant_id = (await tenants.insert_tenant(conn, "acme"))["id"]
        other_id = (await tenants.insert_tenant(conn, "other"))["id"]
        user = await users.insert_user(conn, "dana@example.com", "not-a-real-hash", "Dana")
        member_of = other_id if case == "other_tenant" else tenant_id
        await tenants.add_membership(conn, member_of, user["id"], "admin")
        if case == "removed":
            await members.delete_membership(conn, tenant_id, user["id"])
        if case == "disabled":
            await conn.execute("update users set disabled_at = now() where id = %s", (user["id"],))

    asked_about = uuid4() if case == "unknown_user" else user["id"]

    principal = await principal_for_user(db, tenant_id, asked_about)

    if case == "member":
        assert principal == Principal(
            user["id"], "dana@example.com", "Dana", tenant_id, "admin", "system"
        )
    else:
        assert principal is None


# --- failure hooks and registry -----------------------------------------------------------------


def hook_context(kind: str, *, tenant_id, payload, db=None) -> JobContext:
    job = Job(
        id=41,
        tenant_id=tenant_id,
        kind=kind,
        payload=payload,
        priority=5,
        attempts=3,
        max_attempts=3,
        request_id=None,
    )
    return JobContext(job=job, db=db, worker_id="test-worker", services=None)


@pytest.mark.parametrize("kind", ["reverify_document", "reverify_analysis"])
async def test_failure_hook_only_logs(kind):
    tenant_id = uuid4()
    # No database: a hook that tried to settle a row would fail instead of logging.
    ctx = hook_context(kind, tenant_id=tenant_id, payload={"analysis_id": "SECRET-PAYLOAD"})
    hook = build_failure_handlers(None)[kind]

    with json_logs() as stream:
        assert await hook(ctx, "job_timeout") is None

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [r["msg"] for r in records] == ["reverify job failed"]
    (record,) = records
    assert (record["kind"], record["error_code"]) == (kind, "job_timeout")
    assert (record["tenant_id"], record["job_id"]) == (str(tenant_id), "41")
    assert "SECRET-PAYLOAD" not in stream.getvalue()


@pytest.mark.parametrize("kind", ["reverify_document", "reverify_analysis"])
async def test_failure_hook_never_raises(kind):
    ctx = hook_context(kind, tenant_id=None, payload={})
    hook = build_failure_handlers(None)[kind]

    with json_logs() as stream:
        assert await hook(ctx, "lease_expired") is None

    assert "reverify job failed" in stream.getvalue()
