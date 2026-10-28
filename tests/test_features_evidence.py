import asyncio
import math
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from backend.retrieval.service import PostgresRetriever
from backend.retrieval.types import PassageHit, SearchScope
from backend.verify.evidence import gather_evidence, relation_for, window_spans
from backend.verify.features import FEATURE_NAMES, compute_features, order_signals
from backend.verify.types import Claim, EvidenceSignal, NLIResult
from tests.fakes import HashEmbedder, KeywordReranker
from tests.pg_fixtures import seed_docs
from tests.test_bm25 import make_scope

pg = pytest.mark.pg

CLAIM = "Refunds take 30 days"


def _signal(passage_id, text, nli, rerank, dense_cos):
    return EvidenceSignal(
        passage_id=passage_id, text=text, nli=nli, rerank=rerank, dense_cos=dense_cos
    )


def _s1():
    return _signal("p1", "Refunds take 30 days.", NLIResult(0.9, 0.05, 0.05), 2.0, 0.8)


def _s2():
    return _signal("p2", "Shipping is free.", NLIResult(0.1, 0.2, 0.7), -1.0, 0.3)


def _feature(name, vector):
    return vector[FEATURE_NAMES.index(name)]


def test_feature_names_length_12():
    assert len(FEATURE_NAMES) == 12
    assert FEATURE_NAMES == (
        "max_entail",
        "max_contra",
        "entail_minus_contra",
        "neutral_at_best",
        "n_entail_ge_half",
        "top_rerank",
        "max_dense_cos",
        "number_coverage",
        "has_numbers",
        "content_coverage",
        "claim_len",
        "contra_and_entail",
    )


def test_feature_vector_literal():
    vector = compute_features(CLAIM, [_s1(), _s2()], evidence_k=4)

    expected = [
        0.9,
        0.2,
        0.7,
        0.05,
        math.log1p(1),
        0.8807970779778823,
        0.8,
        1.0,
        1.0,
        1.0,
        1.6094379124341003,
        0.2,
    ]
    assert vector.dtype == np.float64
    assert vector.shape == (12,)
    np.testing.assert_allclose(vector, expected, rtol=0, atol=1e-12)


def test_evidence_k_truncates_by_rerank():
    # s2 is listed first but ranks second by rerank, so evidence_k=1 keeps only s1.
    vector = compute_features(CLAIM, [_s2(), _s1()], evidence_k=1)

    assert _feature("max_contra", vector) == 0.05


def test_number_mismatch_coverage():
    vector = compute_features("Refunds take 14 days", [_s1()], evidence_k=4)

    assert _feature("number_coverage", vector) == 0.0
    assert _feature("has_numbers", vector) == 1.0


def test_money_and_percent_numbers():
    money = _signal("p1", "The fee is 1200 dollars.", NLIResult(0.8, 0.1, 0.1), 1.0, 0.5)
    percent = _signal("p2", "Growth was 5% last year.", NLIResult(0.8, 0.1, 0.1), 1.0, 0.5)
    bare = _signal("p3", "Growth was 5 last year.", NLIResult(0.8, 0.1, 0.1), 1.0, 0.5)

    fee = compute_features("The fee is $1,200", [money], evidence_k=4)
    growth = compute_features("Growth was 5%", [percent], evidence_k=4)
    no_percent = compute_features("Growth was 5%", [bare], evidence_k=4)

    assert _feature("number_coverage", fee) == 1.0
    assert _feature("number_coverage", growth) == 1.0
    assert _feature("number_coverage", no_percent) == 0.0


def test_no_signals_vector():
    vector = compute_features(CLAIM, [], evidence_k=4)

    expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, math.log1p(4), 0.0]
    np.testing.assert_allclose(vector, expected, rtol=0, atol=1e-12)


def test_no_signals_claim_without_numbers_or_content_tokens():
    vector = compute_features("It is the", [], evidence_k=4)

    assert _feature("number_coverage", vector) == 1.0
    assert _feature("has_numbers", vector) == 0.0
    assert _feature("content_coverage", vector) == 1.0
    assert _feature("claim_len", vector) == math.log1p(3)


def test_none_rerank_ordered_last():
    tied_first = _signal("a", "x", NLIResult(0.3, 0.3, 0.4), 1.0, 0.1)
    unscored = _signal("b", "x", NLIResult(0.3, 0.3, 0.4), None, 0.1)
    tied_second = _signal("c", "x", NLIResult(0.3, 0.3, 0.4), 1.0, 0.1)
    lowest = _signal("d", "x", NLIResult(0.3, 0.3, 0.4), -2.0, 0.1)
    also_unscored = _signal("e", "x", NLIResult(0.3, 0.3, 0.4), None, 0.1)

    ordered = order_signals([unscored, lowest, tied_first, also_unscored, tied_second])

    # Rerank descending; equal scores and the unscored pair keep their input order.
    assert [s.passage_id for s in ordered] == ["a", "c", "d", "b", "e"]


def test_top_rerank_zero_when_no_signal_has_a_rerank_score():
    unscored = _signal("p1", "Refunds take 30 days.", NLIResult(0.9, 0.05, 0.05), None, 0.8)

    vector = compute_features(CLAIM, [unscored], evidence_k=4)

    assert _feature("top_rerank", vector) == 0.0


# --- Evidence gathering and span localization ---------------------------------------------

KEYWORD_RERANKER = KeywordReranker()
REFUND_CLAIM = "Refunds take thirty days for customers"


def _hit(text: str, *, start_char: int = 0) -> PassageHit:
    return PassageHit(
        passage_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        version_number=1,
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        document_title="Policy",
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
        page_start=None,
        page_end=None,
        section=None,
        scores={},
        rank=1,
    )


def _claim(ordinal: int, text: str) -> Claim:
    return Claim(
        ordinal=ordinal, text=text, answer_start=0, answer_end=len(text), method="sentence"
    )


def test_window_spans_absolute():
    text = "Alpha is first. Beta comes second. Gamma closes it."
    hit = _hit(text, start_char=100)

    windows = window_spans(hit)

    assert len(windows) == 5
    # Width 1 for each sentence, then width 2 for each pair; offsets are absolute.
    assert [(start, end) for start, end, _ in windows] == [
        (100, 115),
        (116, 134),
        (135, 151),
        (100, 134),
        (116, 151),
    ]
    for start, end, window_text in windows:
        assert hit.text[start - 100 : end - 100] == window_text
    assert [window_text for _, _, window_text in windows][3] == (
        "Alpha is first. Beta comes second."
    )


def test_window_spans_width_larger_than_sentence_count_is_skipped():
    hit = _hit("Alpha is first. Beta comes second.", start_char=7)

    assert [(s, e) for s, e, _ in window_spans(hit, widths=(1, 2, 3))] == [
        (7, 22),
        (23, 41),
        (7, 41),
    ]


@pytest.mark.parametrize("text", ["Refunds take thirty days", "   "])
def test_window_spans_without_sentence_boundary_is_whole_passage(text):
    hit = _hit(text, start_char=40)

    assert window_spans(hit) == [(40, 40 + len(text), text)]


@pytest.mark.parametrize(
    ("result", "relation"),
    [
        (NLIResult(entail=0.9, contra=0.05, neutral=0.05), "supports"),
        (NLIResult(entail=0.05, contra=0.9, neutral=0.05), "contradicts"),
        (NLIResult(entail=0.05, contra=0.05, neutral=0.9), "related"),
        # Entailment ties neutral for the max and beats contradiction.
        (NLIResult(entail=0.4, contra=0.2, neutral=0.4), "supports"),
        # Contradiction ties entailment for the max: entail is not above contra.
        (NLIResult(entail=0.4, contra=0.4, neutral=0.2), "contradicts"),
        # Contradiction ties neutral for the max.
        (NLIResult(entail=0.2, contra=0.4, neutral=0.4), "contradicts"),
        # Neutral is the largest, so even a high entailment is only related.
        (NLIResult(entail=0.3, contra=0.1, neutral=0.6), "related"),
    ],
)
def test_relation_for_cases(result, relation):
    assert relation_for(result) == relation


class RecordingRetriever:
    """Retriever stub that records each search and how many ran at once."""

    def __init__(self, hits: Sequence[PassageHit] = ()) -> None:
        self.hits = list(hits)
        self.calls: list[tuple[str, SearchScope, dict]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def search(self, query, scope, **kwargs):
        self.calls.append((query, scope, kwargs))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
        finally:
            self.in_flight -= 1
        return list(self.hits)


async def test_gather_searches_hybrid_without_rerank():
    retriever = RecordingRetriever()
    scope = SearchScope(uuid.uuid4(), (uuid.uuid4(),))

    result = await gather_evidence(
        retriever, KEYWORD_RERANKER, [_claim(0, REFUND_CLAIM)], scope, candidates=13
    )

    assert result == [[]]
    assert retriever.calls == [
        (
            REFUND_CLAIM,
            scope,
            {"k": 13, "mode": "hybrid", "rerank": False, "candidates": 13},
        )
    ]


async def test_gather_bounds_concurrency():
    retriever = RecordingRetriever()
    scope = SearchScope(uuid.uuid4(), (uuid.uuid4(),))
    claims = [_claim(n, f"Claim number {n}") for n in range(6)]

    result = await gather_evidence(retriever, KEYWORD_RERANKER, claims, scope, concurrency=2)

    assert result == [[]] * 6
    assert [query for query, _, _ in retriever.calls] == [claim.text for claim in claims]
    assert retriever.max_in_flight == 2


async def test_gather_puts_pinned_first_and_dedupes_by_passage():
    found_a, found_b, pinned_c = _hit("Alpha one."), _hit("Beta two."), _hit("Gamma three.")
    pinned_a = replace(found_a, scores={"pinned": 1.0})
    retriever = RecordingRetriever([found_a, found_b])
    scope = SearchScope(uuid.uuid4(), (uuid.uuid4(),))

    (hits,) = await gather_evidence(
        retriever,
        KEYWORD_RERANKER,
        [_claim(0, "Unrelated claim words")],
        scope,
        pinned={0: [pinned_c, pinned_a]},
    )

    # No claim token occurs in a passage, so every rerank score ties and the union order
    # decides: pinned first, and the pinned copy of a passage wins over the retrieved copy.
    assert [hit.passage_id for hit in hits] == [
        pinned_c.passage_id,
        found_a.passage_id,
        found_b.passage_id,
    ]
    assert hits[1].scores == {"pinned": 1.0, "rerank": 0.0}


def make_retriever(db) -> PostgresRetriever:
    return PostgresRetriever(db, HashEmbedder(), KEYWORD_RERANKER)


@pg
async def test_gather_keeps_pinned_and_depth(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    texts = [
        "Refunds take thirty days for customers.",
        "Refunds take thirty days.",
        "Refunds are processed by finance.",
        "Shipping is free on large orders.",
        "Weather report on tuesday afternoon.",
    ]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))
    retriever = make_retriever(db)
    (pinned_hit,) = await retriever.search("weather tuesday", scope, mode="bm25", rerank=False)
    assert pinned_hit.text == texts[4]
    claim = _claim(0, REFUND_CLAIM)

    unpinned = await gather_evidence(retriever, KEYWORD_RERANKER, [claim], scope, depth=3)
    pinned = await gather_evidence(
        retriever, KEYWORD_RERANKER, [claim], scope, pinned={0: [pinned_hit]}, depth=3
    )

    assert [hit.text for hit in unpinned[0]] == texts[:3]
    (hits,) = pinned
    # The pinned passage matches no claim token, yet it takes the third slot from texts[2].
    assert [hit.text for hit in hits] == [texts[0], texts[1], texts[4]]
    assert [hit.scores["rerank"] for hit in hits] == [6.0, 4.0, 0.0]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert len({hit.passage_id for hit in hits}) == 3


@pg
async def test_gather_keeps_every_pinned_hit_beyond_depth(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    texts = [
        "Refunds take thirty days for customers.",
        "Shipping is free on large orders.",
        "Weather report on tuesday afternoon.",
    ]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))
    retriever = make_retriever(db)
    low = await retriever.search("report shipping", scope, mode="bm25", rerank=False)
    assert {hit.text for hit in low} == {texts[1], texts[2]}

    (hits,) = await gather_evidence(
        retriever,
        KEYWORD_RERANKER,
        [_claim(0, REFUND_CLAIM)],
        scope,
        pinned={0: low},
        depth=1,
    )

    # Two pinned hits exceed depth 1: both stay, and the best unpinned passage is dropped.
    assert {hit.text for hit in hits} == {texts[1], texts[2]}


@pg
async def test_gather_results_follow_claim_order_with_per_claim_pins(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    texts = [
        "Refunds take thirty days for customers.",
        "Refunds are processed by finance.",
        "Shipping is free on large orders.",
        "Weather report on tuesday afternoon.",
    ]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))
    retriever = make_retriever(db)
    (weather,) = await retriever.search("weather tuesday", scope, mode="bm25", rerank=False)
    claims = [_claim(0, REFUND_CLAIM), _claim(1, "Shipping is free on large orders")]

    results = await gather_evidence(
        retriever,
        KEYWORD_RERANKER,
        claims,
        scope,
        pinned={1: [weather]},
        depth=2,
        concurrency=2,
    )

    # Claim 0 has no pin and gets its two best passages. Claim 1 keeps its pin (rerank 1.0)
    # beneath its own best passage, and the pin does not leak into claim 0.
    assert [[hit.text for hit in hits] for hits in results] == [
        [texts[0], texts[1]],
        [texts[2], texts[3]],
    ]


@pg
async def test_gather_respects_scope(db, services):
    tenant_id, (x, y) = await make_scope(db, collections=2)
    in_scope = await seed_docs(
        db,
        services,
        tenant_id,
        x,
        ["Refunds take thirty days for customers.", "Shipping is free on large orders."],
    )
    out_of_scope = await seed_docs(
        db,
        services,
        tenant_id,
        y,
        [f"Refunds take thirty days for customers in region {n}." for n in range(4)],
    )
    scope = SearchScope(tenant_id, (x,))

    (hits,) = await gather_evidence(
        make_retriever(db), KEYWORD_RERANKER, [_claim(0, REFUND_CLAIM)], scope
    )

    assert {hit.document_id for hit in hits} == set(in_scope)
    assert not {hit.document_id for hit in hits} & set(out_of_scope)


@pg
async def test_gather_empty_scope(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    scope = SearchScope(tenant_id, (collection_id,))

    result = await gather_evidence(
        make_retriever(db), KEYWORD_RERANKER, [_claim(0, REFUND_CLAIM)], scope
    )

    assert result == [[]]
