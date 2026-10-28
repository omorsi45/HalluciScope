"""VerificationEngine: calibrated claim verdicts, evidence span localization, one NLI batch."""

import dataclasses
import uuid
from datetime import UTC, datetime

import numpy as np
import pytest

from backend.calibration.artifact import CalibratorMismatch
from backend.retrieval.service import PostgresRetriever
from backend.retrieval.types import PassageHit, SearchScope
from backend.verify.engine import VerificationEngine, collect_signals
from backend.verify.features import FEATURE_NAMES
from backend.verify.types import VERDICT_ORDER, Claim, NLIResult, Verdict
from tests.fakes import HashEmbedder, KeywordReranker, RuleNLIScorer, make_test_calibrator
from tests.pg_fixtures import seed_docs
from tests.test_bm25 import make_scope

pg = pytest.mark.pg

TEXTS = ["Refunds take 30 days. Shipping is free.", "Exchanges take 14 days."]
SUPPORTED_CLAIM = "Refunds take 30 days."
CONTRADICTED_CLAIM = "Refunds take 45 days."
INSUFFICIENT_CLAIM = "The office has a cafeteria."
SENTENCE = "Refunds take 30 days."


class SpyNLI:
    """RuleNLIScorer that records every batch it is asked to score."""

    model_id = RuleNLIScorer.model_id

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self._inner = RuleNLIScorer()

    def score(self, pairs):
        self.calls.append(list(pairs))
        return self._inner.score(pairs)


def make_claim(ordinal: int, text: str) -> Claim:
    return Claim(
        ordinal=ordinal, text=text, answer_start=0, answer_end=len(text), method="sentence"
    )


def make_engine(db, *, nli=None, calibrator=None, depth: int = 8) -> VerificationEngine:
    embedder = HashEmbedder()
    reranker = KeywordReranker()
    return VerificationEngine(
        PostgresRetriever(db, embedder, reranker),
        reranker,
        nli if nli is not None else RuleNLIScorer(),
        embedder,
        calibrator if calibrator is not None else make_test_calibrator(),
        depth=depth,
    )


async def seeded(db, services):
    """A tenant with two seeded documents; returns the tenant id, collection id and scope."""
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, TEXTS)
    return tenant_id, collection_id, SearchScope(tenant_id, (collection_id,))


async def version_text(db, tenant_id, version_id) -> str:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select text from document_versions where id = %s", (version_id,))
        return (await cur.fetchone())["text"]


def slice_of(item) -> str:
    """The span of an evidence item, cut out of its passage text."""
    hit = item.hit
    return hit.text[item.span_start - hit.start_char : item.span_end - hit.start_char]


class FixedRetriever:
    """Retriever stub that returns the same hits for every query."""

    def __init__(self, hits) -> None:
        self.hits = hits

    async def search(self, query, scope, **kwargs):
        return list(self.hits)


class ScriptedNLI:
    """NLI stub that answers by premise text; unlisted premises get `default`."""

    model_id = "scripted"

    def __init__(self, table: dict[str, NLIResult], default: NLIResult) -> None:
        self.table = table
        self.default = default

    def score(self, pairs):
        return [self.table.get(premise, self.default) for premise, _ in pairs]


PASSAGE = "Alpha one. Beta two. Gamma three."
NEUTRAL = NLIResult(entail=0.1, contra=0.1, neutral=0.8)


def make_hit(text: str) -> PassageHit:
    return PassageHit(
        passage_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        version_number=1,
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        document_title="Policy",
        text=text,
        start_char=0,
        end_char=len(text),
        page_start=None,
        page_end=None,
        section=None,
        scores={},
        rank=1,
    )


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        # Best entail 0.7 is at least best contra 0.6: the entail window's result stands.
        (NLIResult(0.7, 0.1, 0.2), NLIResult(0.2, 0.6, 0.2), NLIResult(0.7, 0.1, 0.2)),
        # Best entail 0.5 is below best contra 0.6: the contra window's result stands.
        (NLIResult(0.5, 0.1, 0.4), NLIResult(0.1, 0.6, 0.3), NLIResult(0.1, 0.6, 0.3)),
        # Best entail 0.5 equals best contra 0.5: entail wins the tie.
        (NLIResult(0.5, 0.1, 0.4), NLIResult(0.1, 0.5, 0.4), NLIResult(0.5, 0.1, 0.4)),
    ],
)
async def test_window_result_is_entail_unless_contra_is_higher(first, second, expected):
    nli = ScriptedNLI({"Alpha one.": first, "Beta two.": second}, NEUTRAL)
    scope = SearchScope(uuid.uuid4(), (uuid.uuid4(),))

    ((_, signals),) = await collect_signals(
        FixedRetriever([make_hit(PASSAGE)]),
        KeywordReranker(),
        nli,
        HashEmbedder(),
        [make_claim(0, "Alpha one is beta")],
        scope,
        premise_mode="window",
    )

    (signal,) = signals
    assert signal.nli == expected
    assert signal.text == PASSAGE


@pg
async def test_three_verdicts(db, services):
    _, _, scope = await seeded(db, services)
    claims = [
        make_claim(0, SUPPORTED_CLAIM),
        make_claim(1, CONTRADICTED_CLAIM),
        make_claim(2, INSUFFICIENT_CLAIM),
    ]

    results = await make_engine(db).verify(claims, scope)

    assert [r.claim for r in results] == claims
    assert [r.verdict for r in results] == [
        Verdict.SUPPORTED,
        Verdict.CONTRADICTED,
        Verdict.INSUFFICIENT,
    ]
    for result in results:
        assert list(result.probs) == [verdict.value for verdict in VERDICT_ORDER]
        assert sum(result.probs.values()) == pytest.approx(1.0)
        assert list(result.features) == list(FEATURE_NAMES)
        assert len(result.signals) == result.diagnostics["n_candidates"] == 2
        assert result.diagnostics["premise_mode"] == "passage"
    assert results[0].features["max_entail"] == 0.9
    assert results[1].features["max_contra"] == 0.9
    assert results[1].probs["contradicted"] > 0.5


@pg
async def test_no_evidence_is_insufficient(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    scope = SearchScope(tenant_id, (collection_id,))

    (result,) = await make_engine(db).verify([make_claim(0, SUPPORTED_CLAIM)], scope)

    assert result.verdict is Verdict.INSUFFICIENT
    assert result.evidence == ()
    assert result.signals == ()
    assert result.features["max_entail"] == 0.0
    assert result.features["neutral_at_best"] == 1.0
    assert result.diagnostics == {"n_candidates": 0, "premise_mode": "passage"}


@pg
async def test_no_claims_gives_no_results(db, services):
    _, _, scope = await seeded(db, services)
    engine = make_engine(db)

    assert await engine.collect([], scope) == []
    assert await engine.verify([], scope) == []


@pg
async def test_evidence_spans_inside_passage_and_exact(db, services):
    tenant_id, _, scope = await seeded(db, services)
    claims = [
        make_claim(0, SUPPORTED_CLAIM),
        make_claim(1, CONTRADICTED_CLAIM),
        make_claim(2, INSUFFICIENT_CLAIM),
    ]

    results = await make_engine(db).verify(claims, scope)

    items = [item for result in results for item in result.evidence]
    assert len(items) == 6
    for item in items:
        hit = item.hit
        assert hit.start_char <= item.span_start < item.span_end <= hit.end_char
        text = await version_text(db, tenant_id, hit.version_id)
        assert text[item.span_start : item.span_end] == slice_of(item)


@pg
async def test_supporting_span_localized(db, services):
    _, _, scope = await seeded(db, services)

    (result,) = await make_engine(db).verify([make_claim(0, SUPPORTED_CLAIM)], scope)

    by_text = {item.hit.text: item for item in result.evidence}
    supporting = by_text[TEXTS[0]]
    assert supporting.relation == "supports"
    assert slice_of(supporting) == SENTENCE
    # An item that is neither the best entailment nor the best contradiction keeps its passage.
    other = by_text[TEXTS[1]]
    assert (other.span_start, other.span_end) == (other.hit.start_char, other.hit.end_char)


@pg
async def test_contradicting_span_localized(db, services):
    _, _, scope = await seeded(db, services)

    (result,) = await make_engine(db).verify([make_claim(0, CONTRADICTED_CLAIM)], scope)

    by_text = {item.hit.text: item for item in result.evidence}
    contradicting = by_text[TEXTS[0]]
    assert contradicting.relation == "contradicts"
    assert slice_of(contradicting) == SENTENCE


@pg
async def test_evidence_items_carry_their_signal(db, services):
    _, _, scope = await seeded(db, services)

    (result,) = await make_engine(db).verify([make_claim(0, SUPPORTED_CLAIM)], scope)

    embedder = HashEmbedder()
    claim_vector = embedder.embed_passages([SUPPORTED_CLAIM])[0]
    assert len(result.evidence) == 2
    for item in result.evidence:
        expected_cos = float(np.dot(claim_vector, embedder.embed_passages([item.hit.text])[0]))
        assert item.dense_cos == pytest.approx(expected_cos)
        assert item.rerank == item.hit.scores["rerank"]
    # Evidence is ordered by rerank: four shared claim words beat two.
    assert [item.hit.text for item in result.evidence] == TEXTS
    assert [item.rerank for item in result.evidence] == [4.0, 2.0]


@pg
async def test_pinned_hit_used(db, services):
    tenant_id, (x, y) = await make_scope(db, collections=2)
    await seed_docs(db, services, tenant_id, x, TEXTS)
    (pinned_hit, *_) = await PostgresRetriever(db, HashEmbedder(), KeywordReranker()).search(
        SUPPORTED_CLAIM, SearchScope(tenant_id, (x,)), k=1, rerank=False
    )
    assert pinned_hit.text == TEXTS[0]
    empty_scope = SearchScope(tenant_id, (y,))
    engine = make_engine(db)
    claim = make_claim(0, SUPPORTED_CLAIM)

    (without,) = await engine.verify([claim], empty_scope)
    (with_pin,) = await engine.verify([claim], empty_scope, {0: [pinned_hit]})

    assert without.verdict is Verdict.INSUFFICIENT
    assert without.evidence == ()
    assert with_pin.verdict is Verdict.SUPPORTED
    assert [item.hit.passage_id for item in with_pin.evidence] == [pinned_hit.passage_id]


@pg
async def test_parity_mismatch_refuses(db):
    calibrator = make_test_calibrator(models={"nli": "other"})

    with pytest.raises(CalibratorMismatch) as raised:
        make_engine(db, calibrator=calibrator)

    assert raised.value.field == "nli"
    assert raised.value.actual == RuleNLIScorer.model_id


@pg
async def test_window_premise_mode(db, services):
    _, _, scope = await seeded(db, services)
    calibrator = dataclasses.replace(make_test_calibrator(), premise_mode="window")
    spy = SpyNLI()
    claims = [
        make_claim(0, SUPPORTED_CLAIM),
        make_claim(1, CONTRADICTED_CLAIM),
        make_claim(2, INSUFFICIENT_CLAIM),
    ]

    results = await make_engine(db, nli=spy, calibrator=calibrator).verify(claims, scope)

    assert [r.verdict for r in results] == [
        Verdict.SUPPORTED,
        Verdict.CONTRADICTED,
        Verdict.INSUFFICIENT,
    ]
    assert {r.diagnostics["premise_mode"] for r in results} == {"window"}
    # The first batch scored the sentence windows of the passages, not only the passages.
    premises = {premise for premise, _ in spy.calls[0]}
    assert {SENTENCE, "Shipping is free."} <= premises


@pg
async def test_single_nli_batch_call(db, services):
    _, _, scope = await seeded(db, services)
    spy = SpyNLI()
    claims = [
        make_claim(0, SUPPORTED_CLAIM),
        make_claim(1, CONTRADICTED_CLAIM),
        make_claim(2, INSUFFICIENT_CLAIM),
    ]

    collected = await make_engine(db, nli=spy).collect(claims, scope)

    assert len(spy.calls) == 1
    assert sorted(spy.calls[0]) == sorted((text, claim.text) for claim in claims for text in TEXTS)
    assert [len(signals) for _, signals in collected] == [2, 2, 2]
    assert [len(hits) for hits, _ in collected] == [2, 2, 2]
