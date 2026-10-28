"""The optional self-consistency diagnostic of ask analyses, and the legacy pipeline's removal.

The unit tests drive `consistency_diagnostic` with a scripted LLM and the hash embedder. The
Postgres tests run whole ask analyses through the worker. Imports of the module under test sit
inside the tests so each one fails (or passes) on its own.
"""

import importlib.util
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
import pytest

from backend.analysis.service import STAGES
from backend.db.repos import collections, tenants
from backend.generate.answer import build_answer_messages
from backend.llm.client import LLMError, LLMTimeout, LLMUnavailable
from backend.retrieval.types import PassageHit
from backend.verify.claims import sentence_claims
from tests.fakes import HashEmbedder, ScriptedLLM
from tests.pg_fixtures import drain, seed_docs
from tests.test_run_analysis import (
    EXCHANGES,
    REFUNDS,
    World,
    add_user,
    analysis_row,
    ask,
    claim_rows,
    claims_json,
    verify,
)

TEXT = "Refunds take 30 days."
TWO_CLAIMS = "Refunds take 30 days. Exchanges take 14 days."
SAMPLE_QUESTION = "How long do refunds take?"


def make_hit() -> PassageHit:
    return PassageHit(
        passage_id=UUID(int=1),
        document_id=UUID(int=101),
        version_id=UUID(int=201),
        version_number=1,
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        document_title="Refund Policy",
        text=TEXT,
        start_char=0,
        end_char=len(TEXT),
        page_start=None,
        page_end=None,
        section=None,
        scores={"rrf": 0.5},
        rank=1,
    )


# --- the diagnostic ----------------------------------------------------------------------------


async def test_appearances_counted():
    from backend.verify.consistency import consistency_diagnostic

    llm = ScriptedLLM()
    llm.responses = [
        "Refunds take 30 days [P1].",
        "Refunds take 30 days. Warranty lasts two years.",
        "Warranty lasts two years [P1].",
    ]
    hits = [make_hit()]

    result = await consistency_diagnostic(
        llm,
        HashEmbedder(),
        SAMPLE_QUESTION,
        hits,
        sentence_claims(TEXT),
        n_samples=3,
    )

    assert result == [{"appearances": 2, "n_samples": 3, "failed_samples": 0}]
    # Every sample is the answer prompt of the analysis, asked again at temperature 0.7.
    assert len(llm.calls) == 3
    for call in llm.calls:
        assert call["temperature"] == 0.7
        assert call["messages"] == build_answer_messages(SAMPLE_QUESTION, hits)


async def test_one_entry_per_claim_in_claim_order():
    from backend.verify.consistency import consistency_diagnostic

    llm = ScriptedLLM()
    llm.responses = [
        "Exchanges take 14 days.",
        "Exchanges take 14 days. Refunds take 30 days.",
    ]

    result = await consistency_diagnostic(
        llm,
        HashEmbedder(),
        SAMPLE_QUESTION,
        [make_hit()],
        sentence_claims(TWO_CLAIMS),
        n_samples=2,
    )

    # The first claim is the refunds one; it appears in the second sample only.
    assert result == [
        {"appearances": 1, "n_samples": 2, "failed_samples": 0},
        {"appearances": 2, "n_samples": 2, "failed_samples": 0},
    ]


@pytest.mark.parametrize("failure", [LLMTimeout, LLMUnavailable, LLMError])
async def test_failed_sample_counted(failure):
    from backend.verify.consistency import consistency_diagnostic

    llm = ScriptedLLM()
    llm.responses = [TEXT, failure("SECRET-LLM-DETAIL"), TEXT]

    result = await consistency_diagnostic(
        llm,
        HashEmbedder(),
        SAMPLE_QUESTION,
        [make_hit()],
        sentence_claims(TEXT),
        n_samples=3,
    )

    # The failed sample is zero claims: it stays in n_samples and adds no appearance.
    assert result == [{"appearances": 2, "n_samples": 3, "failed_samples": 1}]


async def test_sample_without_claims_is_not_a_failure():
    from backend.verify.consistency import consistency_diagnostic

    llm = ScriptedLLM()
    llm.responses = ["", TEXT]

    result = await consistency_diagnostic(
        llm,
        HashEmbedder(),
        SAMPLE_QUESTION,
        [make_hit()],
        sentence_claims(TEXT),
        n_samples=2,
    )

    assert result == [{"appearances": 1, "n_samples": 2, "failed_samples": 0}]


async def test_threshold_decides_what_counts_as_the_same_claim():
    from backend.verify.consistency import consistency_diagnostic

    sample = "Refunds take 30 days in total."  # similar to the claim, not the same

    async def appearances(threshold: float) -> int:
        llm = ScriptedLLM()
        llm.responses = [sample]
        (entry,) = await consistency_diagnostic(
            llm,
            HashEmbedder(),
            SAMPLE_QUESTION,
            [make_hit()],
            sentence_claims(TEXT),
            n_samples=1,
            threshold=threshold,
        )
        return entry["appearances"]

    assert await appearances(0.95) == 0
    assert await appearances(0.5) == 1
    # "At least": a threshold equal to the cosine itself still counts.
    claim_vector, sample_vector = HashEmbedder().embed_passages([TEXT, sample])
    assert await appearances(float(np.dot(claim_vector, sample_vector))) == 1


async def test_default_threshold_is_0_85():
    from backend.verify.consistency import consistency_diagnostic

    # Under the hash embedder the claim's cosine with the first sample is 4 / sqrt(4 * 5) = 0.894
    # and with the second 4 / sqrt(4 * 6) = 0.816: the default falls between them.
    llm = ScriptedLLM()
    llm.responses = ["Refunds take 30 days total.", "Refunds take 30 days in total."]

    (entry,) = await consistency_diagnostic(
        llm,
        HashEmbedder(),
        SAMPLE_QUESTION,
        [make_hit()],
        sentence_claims(TEXT),
        n_samples=2,
    )

    assert entry["appearances"] == 1


async def test_no_claims_makes_no_llm_calls():
    from backend.verify.consistency import consistency_diagnostic

    llm = ScriptedLLM()

    result = await consistency_diagnostic(
        llm, HashEmbedder(), SAMPLE_QUESTION, [make_hit()], [], n_samples=3
    )

    assert result == []
    assert llm.calls == []


# --- inside an analysis ------------------------------------------------------------------------

ANSWER = "Refunds take 30 days [P1]. Exchanges take 14 days [P2]."
EXTRACTION = claims_json((0, "Refunds take 30 days."), (1, "Exchanges take 14 days."))


@pytest.fixture
async def world(db) -> World:
    async with db.system_tx() as conn:
        tenant_id = (await tenants.insert_tenant(conn, "acme"))["id"]
    owner = await add_user(db, tenant_id, "owner")
    async with db.tenant_tx(tenant_id) as conn:
        collection = await collections.insert_collection(
            conn, tenant_id, "Policies", "", owner.user_id
        )
    return World(tenant_id, owner, collection["id"])


@pytest.fixture
async def seeded(db, services, world) -> list[UUID]:
    return await seed_docs(db, services, world.tenant_id, world.collection_id, [REFUNDS, EXCHANGES])


def verdict_columns(rows: list[dict]) -> list[tuple]:
    return [
        (
            row["text"],
            row["verdict"],
            row["p_supported"],
            row["p_contradicted"],
            row["p_insufficient"],
            row["features"],
        )
        for row in rows
    ]


@pytest.mark.pg
async def test_ask_with_consistency_stores_diagnostic(db, services, world, seeded):
    plain_id = await ask(db, world)
    services.llm.responses = [ANSWER, EXTRACTION]
    assert await drain(db, services) == 1
    plain_calls = len(services.llm.calls)

    # The two samples: the first repeats both claims, the second only the refunds claim.
    services.llm.responses = [ANSWER, EXTRACTION, ANSWER, "Refunds take 30 days [P1]."]
    sampled_id = await ask(db, world, options={"consistency_samples": 2})
    assert await drain(db, services) == 1

    assert plain_calls == 2  # answer and extraction: no samples without the option
    sampled_calls = services.llm.calls[plain_calls:]
    assert len(sampled_calls) == 4
    # The samples are the answer prompt again, after verification, at temperature 0.7.
    assert [call["temperature"] for call in sampled_calls] == [0.0, 0.0, 0.7, 0.7]
    assert sampled_calls[2]["messages"] == sampled_calls[3]["messages"]
    assert sampled_calls[2]["messages"] == sampled_calls[0]["messages"]

    row = await analysis_row(db, world, sampled_id)
    assert (row["status"], row["error_code"]) == ("succeeded", None)
    # The diagnostic runs under the verifying stage: no stage of its own.
    assert row["progress"]["stages_done"] == list(STAGES)
    assert set(row["timings"]) == {*STAGES, "total"}

    plain = await claim_rows(db, world, plain_id)
    sampled = await claim_rows(db, world, sampled_id)
    assert [claim["diagnostics"].get("consistency") for claim in plain] == [None, None]
    assert [claim["diagnostics"]["consistency"] for claim in sampled] == [
        {"appearances": 2, "n_samples": 2, "failed_samples": 0},
        {"appearances": 1, "n_samples": 2, "failed_samples": 0},
    ]
    # The verdict-time diagnostics stay beside it.
    assert all(claim["diagnostics"]["premise_mode"] == "passage" for claim in sampled)
    # Never a feature: verdicts, probabilities and features equal the run without samples.
    assert verdict_columns(sampled) == verdict_columns(plain)


@pytest.mark.pg
async def test_failed_samples_do_not_fail_the_analysis(db, services, world, seeded):
    services.llm.responses = [ANSWER, EXTRACTION, LLMTimeout("SECRET-LLM-DETAIL"), ANSWER]
    analysis_id = await ask(db, world, options={"consistency_samples": 2})

    assert await drain(db, services) == 1

    row = await analysis_row(db, world, analysis_id)
    assert (row["status"], row["error_code"]) == ("succeeded", None)
    claims = await claim_rows(db, world, analysis_id)
    assert [claim["diagnostics"]["consistency"] for claim in claims] == [
        {"appearances": 1, "n_samples": 2, "failed_samples": 1},
        {"appearances": 1, "n_samples": 2, "failed_samples": 1},
    ]


@pytest.mark.pg
async def test_verify_analysis_never_samples(db, services, world, seeded):
    services.llm.responses = [EXTRACTION]
    # The API refuses samples for verify; the service must not honor them if asked directly.
    analysis_id = await verify(db, world, ANSWER, options={"consistency_samples": 2})

    assert await drain(db, services) == 1

    assert len(services.llm.calls) == 1  # the claim extraction only
    claims = await claim_rows(db, world, analysis_id)
    assert len(claims) == 2
    assert all("consistency" not in claim["diagnostics"] for claim in claims)


# --- the legacy pipeline is gone ---------------------------------------------------------------

LEGACY_MODULES = (
    "backend.core.pipeline",
    "backend.core.chunker",
    "backend.core.ensemble",
    "backend.core.generator",
    "backend.core.decomposer",
    "backend.core.verifiers",
    "backend.db.models",
    "backend.db.repository",
    "evaluation.benchmarks",
    "evaluation.ablation",
    "evaluation.calibration",
)


def module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # the parent package is gone too
        return False


def test_legacy_modules_removed():
    assert [name for name in LEGACY_MODULES if module_exists(name)] == []
    # The document reader stays: it delegates to the new parsers.
    assert module_exists("backend.core.document")
