"""Quick checks: create_quick_analysis, process_upload, and the deprecated /api adapters.

The fakes decide every verdict: the document below has one sentence the answer repeats, one
whose number it changes, and one it never mentions, so the three tiers are predictable.
"""

import asyncio
import re
from uuid import UUID, uuid4

import pytest

from backend.analysis.quick import create_quick_analysis
from backend.analysis.service import run_analysis
from backend.api.errors import ApiError
from backend.api.routes.legacy import analyze_response
from backend.auth.service import Principal
from backend.config import Settings
from backend.ingest.handlers import process_upload
from backend.jobs.worker import PermanentJobError
from backend.llm.client import LLMError, LLMTimeout, LLMUnavailable
from tests.fakes import ScriptedLLM
from tests.helpers.auth import csrf, signup
from tests.test_analysis_routes import (
    accepted_id,
    background_worker,
    claims_json,
    collection_with_docs,
    evidence_of,
    post_ask,
    wait_done,
)
from tests.test_collections import _team

pytestmark = pytest.mark.pg

ANALYZE = "/api/analyze"
ANALYSES = "/api/analyses"
FEEDBACK = "/api/feedback"
V1_ANALYSES = "/api/v1/analyses"

DOCUMENT = "Refunds take 30 days. Exchanges take 14 days."
QUESTION = "How long do refunds take?"
ANSWER = "Refunds take 30 days [P1]. Refunds take 45 days [P1]. Warranty lasts two years [P1]."
CLAIM_TEXTS = ["Refunds take 30 days.", "Refunds take 45 days.", "Warranty lasts two years."]
CLAIMS = claims_json(*enumerate(CLAIM_TEXTS))

DEPRECATION_HEADERS = {"Deprecation": "true", "Link": '</api/v1/ask>; rel="successor-version"'}
ANALYZE_KEYS = {"id", "question", "answer", "scored_claims", "retrieved_chunks", "overall_score"}
SCORED_CLAIM_KEYS = {"claim", "hallucination_score", "tier", "verifier_details"}
NLI_KEYS = {"entailment", "contradiction", "neutral", "matched_chunk"}
LIST_ITEM_KEYS = {"id", "question", "answer", "overall_score", "created_at"}


class SlowLLM(ScriptedLLM):
    """Takes longer than the analysis limit the test sets; bounded so a RED run still ends."""

    async def chat(self, messages, **kwargs):
        await asyncio.sleep(1.0)
        return await super().chat(messages, **kwargs)


def script_llm(services, answer: str = ANSWER, claims: str = CLAIMS) -> None:
    services.llm.responses = [answer, claims]


async def analyze(client, session, document_text: str = DOCUMENT, question: str = QUESTION):
    return await client.post(
        ANALYZE,
        json={"document_text": document_text, "question": question},
        headers=csrf(session),
    )


def tenant_of(session: dict) -> UUID:
    return UUID(session["tenant"]["id"])


def principal_of(session: dict) -> Principal:
    user = session["user"]
    return Principal(
        UUID(user["id"]),
        user["email"],
        user["display_name"],
        tenant_of(session),
        session["tenant"]["role"],
        "session",
    )


async def fetch(db, tenant_id: UUID, query: str, params: tuple = ()) -> list[dict]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def job_kinds(db) -> list[str]:
    async with db.system_tx() as conn:
        cur = await conn.execute("select kind from jobs order by id")
        return [row["kind"] for row in await cur.fetchall()]


def assert_deprecated(response) -> None:
    for name, value in DEPRECATION_HEADERS.items():
        assert response.headers[name] == value


# --- the adapters over the engine ---------------------------------------------------------------


async def test_legacy_analyze_shape(client, db, services):
    alice = await signup(client)
    script_llm(services)

    async with background_worker(db, services):
        response = await analyze(client, alice)

    assert response.status_code == 200, response.text
    assert_deprecated(response)
    body = response.json()
    assert set(body) == ANALYZE_KEYS
    analysis_id = body["id"]
    assert str(UUID(analysis_id)) == analysis_id
    assert (body["question"], body["answer"]) == (QUESTION, ANSWER)
    assert body["retrieved_chunks"] == [DOCUMENT]

    claims = body["scored_claims"]
    assert [claim["claim"] for claim in claims] == CLAIM_TEXTS
    assert [claim["tier"] for claim in claims] == ["supported", "hallucinated", "uncertain"]
    assert all(set(claim) == SCORED_CLAIM_KEYS for claim in claims)
    assert all(set(claim["verifier_details"]) == {"nli", "calibrated"} for claim in claims)
    assert all(set(claim["verifier_details"]["nli"]) == NLI_KEYS for claim in claims)
    # The supported claim's best evidence is the document, and the fake NLI scores it by hand.
    assert claims[0]["verifier_details"]["nli"] == {
        "entailment": 0.9,
        "contradiction": 0.05,
        "neutral": 0.05,
        "matched_chunk": DOCUMENT,
    }

    # The probabilities the adapter reports are the stored ones, read through the v1 API.
    stored = (await client.get(f"{V1_ANALYSES}/{analysis_id}")).json()
    assert [claim["verdict"] for claim in stored["claims"]] == [
        "supported",
        "contradicted",
        "insufficient",
    ]
    for legacy, new in zip(claims, stored["claims"], strict=True):
        probs = new["probs"]
        assert legacy["hallucination_score"] == pytest.approx(1 - probs["supported"])
        assert legacy["verifier_details"]["calibrated"] == {
            "p_supported": probs["supported"],
            "p_contradicted": probs["contradicted"],
            "p_insufficient": probs["insufficient"],
            "verdict": new["verdict"],
        }
    scores = [claim["hallucination_score"] for claim in claims]
    assert body["overall_score"] == pytest.approx(sum(scores) / 3)


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("POST", ANALYZE, {"document_text": DOCUMENT, "question": QUESTION}),
        ("GET", ANALYSES, None),
        ("POST", FEEDBACK, {"analysis_id": str(uuid4()), "claim_index": 0, "is_correct": True}),
    ],
)
async def test_legacy_requires_auth(client, method, path, payload):
    response = await client.request(method, path, json=payload)

    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"
    assert_deprecated(response)


async def test_legacy_llm_down_503(client, db, services):
    alice = await signup(client)
    services.llm.responses = [LLMUnavailable("SECRET-LLM-DETAIL")]

    async with background_worker(db, services):
        response = await analyze(client, alice)

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert_deprecated(response)
    assert response.json()["code"] == "llm_unavailable"
    for secret in (DOCUMENT, QUESTION, "SECRET-LLM-DETAIL"):
        assert secret not in response.text
    (row,) = await fetch(db, tenant_of(alice), "select status, error_code from analyses")
    assert (row["status"], row["error_code"]) == ("failed", "llm_unavailable")


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (LLMTimeout("SECRET-LLM-DETAIL"), 504, "llm_timeout"),
        (LLMError("SECRET-LLM-DETAIL"), 500, "llm_error"),
    ],
)
async def test_legacy_failed_analysis_maps_error_code_to_status(
    client, db, services, failure, status, code
):
    alice = await signup(client)
    services.llm.responses = [failure]

    async with background_worker(db, services):
        response = await analyze(client, alice)

    assert (response.status_code, response.json()["code"]) == (status, code)
    assert "SECRET-LLM-DETAIL" not in response.text


async def test_legacy_analysis_timeout_is_504(client, db, services):
    alice = await signup(client)
    services.llm = SlowLLM()
    services.settings = Settings(analysis_timeout_s=0.2)

    async with background_worker(db, services):
        response = await analyze(client, alice)

    assert (response.status_code, response.json()["code"]) == (504, "analysis_timeout")


async def test_legacy_calibrator_missing_is_500(client, db, services):
    alice = await signup(client)
    services.calibrator = None

    async with background_worker(db, services):
        response = await analyze(client, alice)

    assert (response.status_code, response.json()["code"]) == (500, "calibrator_missing")


async def test_legacy_unreadable_document_reports_the_ingest_code(client, db, services):
    alice = await signup(client)

    async with background_worker(db, services):
        response = await analyze(client, alice, document_text="   \n  ")

    assert (response.status_code, response.json()["code"]) == (500, "no_text")
    assert services.llm.calls == []
    (upload,) = await fetch(db, tenant_of(alice), "select status, error_code from uploads")
    assert (upload["status"], upload["error_code"]) == ("failed", "no_text")


async def test_legacy_wait_expired_504(client, db, services, api_app, monkeypatch):
    alice = await signup(client)
    script_llm(services)
    monkeypatch.setattr(api_app.state.settings, "quick_wait_s", 0.2)

    response = await analyze(client, alice)

    (row,) = await fetch(db, tenant_of(alice), "select id, status from analyses")
    assert response.status_code == 504
    assert response.json()["code"] == "analysis_pending"
    assert response.headers["Location"] == f"/api/v1/analyses/{row['id']}"
    assert_deprecated(response)
    assert row["status"] == "queued"
    # The work is not lost: once a worker runs, the location serves the result.
    async with background_worker(db, services):
        body = await wait_done(client, str(row["id"]))
    assert (body["kind"], body["status"]) == ("quick", "succeeded")


@pytest.mark.parametrize(
    "payload",
    [
        {"document_text": "", "question": QUESTION},
        {"document_text": DOCUMENT, "question": ""},
        {"document_text": DOCUMENT, "question": "q" * 2001},
        {"document_text": "x" * 1_000_001, "question": QUESTION},
        {"document_text": "bad \x00 text", "question": QUESTION},
    ],
)
async def test_legacy_analyze_rejects_invalid_input(client, db, payload):
    alice = await signup(client)

    response = await client.post(ANALYZE, json=payload, headers=csrf(alice))

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert await fetch(db, tenant_of(alice), "select id from analyses") == []


async def test_legacy_analyze_rejects_a_lone_surrogate(client, db):
    alice = await signup(client)
    # httpx cannot encode a lone surrogate, so the escape goes over the wire as written.
    raw = '{"document_text": "lone \\ud800 surrogate", "question": "Why?"}'

    response = await client.post(
        ANALYZE,
        content=raw,
        headers={**csrf(alice), "Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert await fetch(db, tenant_of(alice), "select id from analyses") == []


async def test_legacy_analyze_429_when_too_many_analyses_are_active(
    client, db, api_app, monkeypatch
):
    alice = await signup(client)
    monkeypatch.setattr(api_app.state.settings, "max_active_analyses", 0)

    response = await analyze(client, alice)

    assert (response.status_code, response.json()["code"]) == (429, "too_many_analyses")
    assert_deprecated(response)
    assert await fetch(db, tenant_of(alice), "select id from collections") == []


async def test_legacy_feedback_and_list(client, client2, db, services):
    alice = await signup(client)
    bob = await signup(client2, email="bob@example.com", display_name="Bob")
    script_llm(services)
    async with background_worker(db, services):
        first = (await analyze(client, alice)).json()
    script_llm(services)
    async with background_worker(db, services):
        second = (await analyze(client, alice)).json()

    wrong = await client.post(
        FEEDBACK,
        json={"analysis_id": first["id"], "claim_index": 1, "is_correct": False, "note": "no"},
        headers=csrf(alice),
    )
    again = await client.post(
        FEEDBACK,
        json={"analysis_id": first["id"], "claim_index": 1, "is_correct": True},
        headers=csrf(alice),
    )
    listed = await client.get(ANALYSES)
    limited = await client.get(ANALYSES, params={"limit": 1})
    bobs_list = await client2.get(ANALYSES)

    assert wrong.status_code == 200, wrong.text
    assert_deprecated(wrong)
    assert set(wrong.json()) == {"id"}
    assert str(UUID(wrong.json()["id"])) == wrong.json()["id"]
    # A second answer for the same claim replaces the first: one row, last label wins.
    assert again.json() == wrong.json()
    (row,) = await fetch(
        db,
        tenant_of(alice),
        "select f.label, f.note, c.ordinal, c.analysis_id from feedback f "
        "join claims c on c.id = f.claim_id",
    )
    assert (row["label"], row["note"], row["ordinal"], str(row["analysis_id"])) == (
        "correct",
        None,
        1,
        first["id"],
    )

    assert listed.status_code == 200
    assert_deprecated(listed)
    items = listed.json()
    assert [item["id"] for item in items] == [second["id"], first["id"]]
    assert all(set(item) == LIST_ITEM_KEYS for item in items)
    newest = items[0]
    assert (newest["question"], newest["answer"]) == (QUESTION, ANSWER)
    assert newest["overall_score"] == pytest.approx(second["overall_score"])
    assert isinstance(newest["created_at"], str)
    assert [item["id"] for item in limited.json()] == [second["id"]]
    assert bobs_list.json() == []

    # The claim must exist and belong to the caller; the answer is the same 404 for all three.
    unknown_claim = {"analysis_id": first["id"], "claim_index": 99, "is_correct": True}
    other_analysis = {"analysis_id": str(uuid4()), "claim_index": 0, "is_correct": True}
    not_an_id = {"analysis_id": "17", "claim_index": 0, "is_correct": True}
    for payload in (unknown_claim, other_analysis, not_an_id):
        refused = await client.post(FEEDBACK, json=payload, headers=csrf(alice))
        assert (refused.status_code, refused.json()["code"]) == (404, "claim_not_found")
    foreign = {"analysis_id": first["id"], "claim_index": 0, "is_correct": True}
    refused = await client2.post(FEEDBACK, json=foreign, headers=csrf(bob))
    assert (refused.status_code, refused.json()["code"]) == (404, "claim_not_found")


# --- quick collections ----------------------------------------------------------------------------


async def test_quick_collection_hidden(client, db, services):
    alice = await signup(client)
    animals = await collection_with_docs(
        client, alice, db, services, ["Zebras graze on savanna grass."], name="Animals"
    )
    script_llm(services)
    async with background_worker(db, services):
        response = await analyze(client, alice)
    assert response.status_code == 200, response.text

    listed = await client.get("/api/v1/collections")
    (quick,) = await fetch(
        db, tenant_of(alice), "select id, name from collections where kind = 'quick'"
    )
    by_id = await client.get(f"/api/v1/collections/{quick['id']}")
    services.llm.responses = [
        "Zebras graze on savanna grass [P1].",
        claims_json((0, "Zebras graze on savanna grass.")),
    ]
    async with background_worker(db, services):
        accepted = await post_ask(client, alice, question="How long do refunds take?")
        asked = await wait_done(client, await accepted_id(accepted))

    assert [item["name"] for item in listed.json()["items"]] == ["Animals"]
    assert by_id.status_code == 404
    # A default-scope ask reads the normal collection only, so it never sees the quick document.
    assert asked["status"] == "succeeded", asked
    assert asked["collection_ids"] == [animals["id"]]
    texts = [item["text"] for item in evidence_of(asked)]
    assert texts == ["Zebras graze on savanna grass."]


async def test_quick_check_works_for_a_member_without_collection_grants(
    client, client2, client3, db, services
):
    _alice, bob, _carol = await _team(client, client2, client3, db)
    assert bob["tenant"]["role"] == "member"
    script_llm(services)

    async with background_worker(db, services):
        response = await analyze(client2, bob)

    # The member reaches the hidden collection only through the admin grant created for them.
    assert response.status_code == 200, response.text
    (grant,) = await fetch(db, tenant_of(bob), "select user_id, permission from collection_grants")
    assert (str(grant["user_id"]), grant["permission"]) == (bob["user"]["id"], "admin")


async def test_quick_ingest_inline_only(client, db, services):
    alice = await signup(client)
    script_llm(services)

    async with background_worker(db, services):
        response = await analyze(client, alice)

    assert response.status_code == 200, response.text
    assert await job_kinds(db) == ["run_analysis"]
    (upload,) = await fetch(db, tenant_of(alice), "select status, version_id from uploads")
    assert upload["status"] == "created_version"
    assert upload["version_id"] is not None


# --- create_quick_analysis and process_upload -----------------------------------------------------


async def test_create_quick_analysis_stores_the_request_and_queues_nothing(client, db):
    alice = await signup(client)
    principal = principal_of(alice)

    analysis_id = await create_quick_analysis(
        db,
        principal,
        data=DOCUMENT.encode(),
        filename="handbook.txt",
        question=QUESTION,
        settings=Settings(),
        enqueue=False,
    )

    tenant_id = tenant_of(alice)
    (analysis,) = await fetch(db, tenant_id, "select * from analyses")
    (collection,) = await fetch(db, tenant_id, "select * from collections")
    (grant,) = await fetch(db, tenant_id, "select * from collection_grants")
    (upload,) = await fetch(db, tenant_id, "select * from uploads")
    assert analysis["id"] == analysis_id
    assert (analysis["kind"], analysis["question"], analysis["answer"]) == (
        "quick",
        QUESTION,
        None,
    )
    assert analysis["collection_ids"] == [collection["id"]]
    assert analysis["options"] == {"upload_id": str(upload["id"]), "extraction": "llm_atomic"}
    assert collection["kind"] == "quick"
    assert re.fullmatch(r"quick-[0-9a-f]{32}", collection["name"])
    assert (grant["collection_id"], grant["user_id"], grant["permission"]) == (
        collection["id"],
        principal.user_id,
        "admin",
    )
    assert (upload["status"], upload["filename"]) == ("queued", "handbook.txt")
    assert await job_kinds(db) == []


async def test_create_quick_analysis_checks_the_active_limit_before_storing_anything(client, db):
    alice = await signup(client)

    with pytest.raises(ApiError) as caught:
        await create_quick_analysis(
            db,
            principal_of(alice),
            data=DOCUMENT.encode(),
            filename="handbook.txt",
            question=QUESTION,
            settings=Settings(max_active_analyses=0),
        )

    assert (caught.value.status, caught.value.code) == (429, "too_many_analyses")
    counts = await fetch(
        db,
        tenant_of(alice),
        "select (select count(*) from collections) as collections, "
        "(select count(*) from uploads) as uploads, (select count(*) from blobs) as blobs",
    )
    assert counts == [{"collections": 0, "uploads": 0, "blobs": 0}]


async def test_create_quick_analysis_leaves_no_collection_behind_when_the_file_is_refused(
    client, db
):
    alice = await signup(client)

    with pytest.raises(ApiError) as caught:
        await create_quick_analysis(
            db,
            principal_of(alice),
            data=b"MZ not a document",
            filename="tool.exe",
            question=QUESTION,
            settings=Settings(),
        )

    assert (caught.value.status, caught.value.code) == (415, "unsupported_type")
    assert await fetch(db, tenant_of(alice), "select id from collections") == []
    assert await job_kinds(db) == []


async def test_process_upload_is_idempotent(client, db, services):
    alice = await signup(client)
    await create_quick_analysis(
        db,
        principal_of(alice),
        data=DOCUMENT.encode(),
        filename="handbook.txt",
        question=QUESTION,
        settings=Settings(),
        enqueue=False,
    )
    tenant_id = tenant_of(alice)
    (upload,) = await fetch(db, tenant_id, "select id from uploads")

    first = await process_upload(services, tenant_id, upload["id"])
    second = await process_upload(services, tenant_id, upload["id"])

    assert first is not None
    assert (first.status, first.version_number, first.previous_version_id) == (
        "created_version",
        1,
        None,
    )
    assert first.n_passages >= 1
    assert second is None
    assert len(await fetch(db, tenant_id, "select id from document_versions")) == 1


async def test_quick_analysis_with_a_failed_upload_fails_with_the_upload_code(
    client, db, services
):
    alice = await signup(client)
    analysis_id = await create_quick_analysis(
        db,
        principal_of(alice),
        data=DOCUMENT.encode(),
        filename="handbook.txt",
        question=QUESTION,
        settings=Settings(),
        enqueue=False,
    )
    tenant_id = tenant_of(alice)
    # An earlier attempt already failed the upload, then died before it failed the analysis.
    async with db.tenant_tx(tenant_id) as conn:
        await conn.execute("update uploads set status = 'failed', error_code = 'no_text'")

    with pytest.raises(PermanentJobError) as caught:
        await run_analysis(services, tenant_id, analysis_id)

    assert caught.value.code == "no_text"
    (row,) = await fetch(db, tenant_id, "select status, error_code from analyses")
    assert (row["status"], row["error_code"]) == ("failed", "no_text")
    assert services.llm.calls == []


# --- the legacy response mapping, by hand ---------------------------------------------------------


def _claim(claim_id: str, ordinal: int, text: str, verdict: str, probs: tuple[float, ...]) -> dict:
    p_supported, p_contradicted, p_insufficient = probs
    return {
        "id": claim_id,
        "ordinal": ordinal,
        "text": text,
        "verdict": verdict,
        "p_supported": p_supported,
        "p_contradicted": p_contradicted,
        "p_insufficient": p_insufficient,
    }


def _evidence(claim_id: str, text: str, nli: tuple[float, float, float]) -> dict:
    entail, contra, neutral = nli
    return {
        "claim_id": claim_id,
        "text": text,
        "nli_entail": entail,
        "nli_contra": contra,
        "nli_neutral": neutral,
    }


def test_analyze_response_maps_claims_and_evidence():
    analysis = {"id": UUID(int=7), "question": "Q?", "answer": "A."}
    claims = [
        _claim("c1", 0, "One.", "supported", (0.9, 0.05, 0.05)),
        _claim("c2", 1, "Two.", "contradicted", (0.25, 0.7, 0.05)),
        _claim("c3", 2, "Three.", "insufficient", (0.5, 0.0, 0.5)),
        _claim("c4", 3, "Four.", "insufficient", (0.0, 0.0, 1.0)),
    ]
    evidence = [
        # Rank order within a claim; the best entailment is the second item, not the first.
        _evidence("c1", "alpha", (0.3, 0.2, 0.5)),
        _evidence("c1", "beta", (0.8, 0.1, 0.1)),
        _evidence("c2", "alpha", (0.1, 0.8, 0.1)),
        _evidence("c3", "gamma", (0.4, 0.4, 0.2)),
        _evidence("c3", "delta", (0.4, 0.3, 0.3)),
    ]

    response = analyze_response(analysis, claims, evidence)

    assert response.id == str(UUID(int=7))
    assert (response.question, response.answer) == ("Q?", "A.")
    scored = response.scored_claims
    assert [claim.claim for claim in scored] == ["One.", "Two.", "Three.", "Four."]
    assert [claim.tier for claim in scored] == [
        "supported",
        "hallucinated",
        "uncertain",
        "uncertain",
    ]
    assert [claim.hallucination_score for claim in scored] == pytest.approx([0.1, 0.75, 0.5, 1.0])
    assert scored[0].verifier_details == {
        "nli": {"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1, "matched_chunk": "beta"},
        "calibrated": {
            "p_supported": 0.9,
            "p_contradicted": 0.05,
            "p_insufficient": 0.05,
            "verdict": "supported",
        },
    }
    # A tie in entailment goes to the better ranked item.
    assert scored[2].verifier_details["nli"]["matched_chunk"] == "gamma"
    # No evidence: zeros and an empty chunk.
    assert scored[3].verifier_details["nli"] == {
        "entailment": 0.0,
        "contradiction": 0.0,
        "neutral": 0.0,
        "matched_chunk": "",
    }
    # Distinct texts, claim by claim, in the order first seen.
    assert response.retrieved_chunks == ["alpha", "beta", "gamma", "delta"]
    assert response.overall_score == pytest.approx((0.1 + 0.75 + 0.5 + 1.0) / 4)


def test_analyze_response_without_claims_scores_zero():
    response = analyze_response({"id": UUID(int=1), "question": "Q?", "answer": "A."}, [], [])

    assert (response.scored_claims, response.retrieved_chunks, response.overall_score) == (
        [],
        [],
        0.0,
    )
