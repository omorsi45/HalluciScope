import json

import pytest

from backend.generate.answer import NO_INFO_ANSWER, parse_citations
from backend.ingest.types import Sentence
from backend.llm.client import LLMError, LLMTimeout, LLMUnavailable
from backend.llm.prompts import EXTRACT_SYSTEM
from backend.verify.claims import (
    CLAIMS_SCHEMA,
    STOPWORDS,
    claim_sentences,
    llm_atomic_claims,
    sentence_claims,
)
from backend.verify.types import VERDICT_ORDER, Verdict
from tests.fakes import ScriptedLLM


def _llm(*replies: str | Exception) -> ScriptedLLM:
    llm = ScriptedLLM()
    llm.responses = list(replies)
    return llm


def _reply(*items: object) -> str:
    return json.dumps({"claims": list(items)})


def _many_sentences(count: int) -> str:
    return " ".join(f"Item {n} ships today." for n in range(count))


def test_verdict_order_matches_probability_columns():
    assert [v.value for v in VERDICT_ORDER] == ["supported", "contradicted", "insufficient"]
    assert Verdict("contradicted") is Verdict.CONTRADICTED


def test_sentence_claims_strip_markers():
    answer = "Refunds take 30 days [P1]. Shipping is free [P2]."

    claims = sentence_claims(answer)

    assert [c.text for c in claims] == ["Refunds take 30 days.", "Shipping is free."]
    assert [c.cited_numbers for c in claims] == [(1,), (2,)]
    assert [c.ordinal for c in claims] == [0, 1]
    assert {c.method for c in claims} == {"sentence"}
    # The span is the source sentence in the original answer; only the claim text drops markers.
    assert [answer[c.answer_start : c.answer_end] for c in claims] == [
        "Refunds take 30 days [P1].",
        "Shipping is free [P2].",
    ]


def test_claim_sentences_returns_sentence_clean_text_and_numbers():
    answer = "Refunds take 30 days [P1]. Shipping is free [P2]."

    assert claim_sentences(answer) == [
        (Sentence(0, 26), "Refunds take 30 days.", (1,)),
        (Sentence(27, 49), "Shipping is free.", (2,)),
    ]


def test_marker_after_sentence_end_cites_the_preceding_sentence():
    answer = "Refunds take 30 days. [P1] Shipping is free. [P2, P3]"

    claims = sentence_claims(answer)

    assert [c.text for c in claims] == ["Refunds take 30 days.", "Shipping is free."]
    assert [c.cited_numbers for c in claims] == [(1,), (2, 3)]
    assert [answer[c.answer_start : c.answer_end] for c in claims] == [
        "Refunds take 30 days.",
        "Shipping is free.",
    ]
    # Claim sentences and citation sentences are the same spans.
    assert {(c.answer_start, c.answer_end) for c in claims} == {
        (ref.sentence_start, ref.sentence_end) for ref in parse_citations(answer, 3)
    }


def test_leading_marker_cites_the_first_sentence():
    claims = sentence_claims("[P2] Refunds take 30 days. Shipping is free.")

    assert [c.text for c in claims] == ["Refunds take 30 days.", "Shipping is free."]
    assert [c.cited_numbers for c in claims] == [(2,), ()]


def test_mid_sentence_markers_removed_and_numbers_deduplicated_in_order():
    answer = "Refunds take 30 days [P2] and shipping is free [P1, P2]."

    claims = sentence_claims(answer)

    assert [c.text for c in claims] == ["Refunds take 30 days and shipping is free."]
    assert claims[0].cited_numbers == (2, 1)


@pytest.mark.parametrize(
    "answer",
    [
        NO_INFO_ANSWER,
        NO_INFO_ANSWER.upper().rstrip("."),
        f"{NO_INFO_ANSWER} [P1]",
        "[P1].",
        "[P1]",
        "",
        "   ",
        "It is what it was.",
    ],
)
def test_skips_no_info_and_empty(answer):
    assert sentence_claims(answer) == []


def test_cap_fifty_claims():
    claims = sentence_claims(_many_sentences(80))

    assert len(claims) == 50
    assert [c.ordinal for c in claims] == list(range(50))
    assert claims[0].text == "Item 0 ships today."
    assert claims[49].text == "Item 49 ships today."


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        pytest.param(
            " ".join(["alpha"] * 100) + ".", " ".join(["alpha"] * 66), id="last-whitespace"
        ),
        # The 400 characters end exactly at a word end, so the whole prefix is kept.
        pytest.param(
            " ".join(["abcd"] * 79 + ["abcde", "tail", "words"]) + ".",
            " ".join(["abcd"] * 79 + ["abcde"]),
            id="word-ends-at-400",
        ),
        pytest.param("x" * 500 + ".", "x" * 400, id="no-whitespace"),
    ],
)
def test_long_sentence_truncated_at_last_whitespace_before_400(sentence, expected):
    claims = sentence_claims(sentence)

    assert [c.text for c in claims] == [expected]
    # The span still covers the whole source sentence.
    assert (claims[0].answer_start, claims[0].answer_end) == (0, len(sentence))


def test_stopwords_are_function_words_only():
    assert isinstance(STOPWORDS, frozenset)
    assert {"the", "a", "an", "it", "they", "is", "was", "have", "will", "of", "in", "for"} <= (
        STOPWORDS
    )
    assert {"and", "but", "or", "if", "because", "to", "with"} <= STOPWORDS
    content = {
        "take",
        "days",
        "refunds",
        "shipping",
        "free",
        "office",
        "cafeteria",
        "claim",
        "supported",
    }
    assert STOPWORDS.isdisjoint(content)


async def test_llm_atomic_valid():
    answer = "Refunds take 30 days and shipping is free [P1]."
    llm = _llm(
        _reply(
            {"sentence": 0, "text": "Refunds take 30 days."},
            {"sentence": 0, "text": "Shipping is free."},
        )
    )

    claims = await llm_atomic_claims(llm, answer)

    assert [c.text for c in claims] == ["Refunds take 30 days.", "Shipping is free."]
    assert {c.method for c in claims} == {"llm_atomic"}
    assert [c.ordinal for c in claims] == [0, 1]
    assert {c.cited_numbers for c in claims} == {(1,)}
    assert {(c.answer_start, c.answer_end) for c in claims} == {(0, len(answer))}
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["json_schema"] == CLAIMS_SCHEMA
    assert call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": "0. Refunds take 30 days and shipping is free."},
    ]


async def test_llm_atomic_numbers_only_the_kept_sentences():
    answer = "Refunds take 30 days [P1]. It is what it was. Shipping is free [P2]."
    llm = _llm(
        _reply(
            {"sentence": 0, "text": "Refunds take 30 days."},
            {"sentence": 1, "text": "Shipping is free."},
        )
    )

    claims = await llm_atomic_claims(llm, answer)

    assert llm.calls[0]["messages"][1]["content"] == (
        "0. Refunds take 30 days.\n1. Shipping is free."
    )
    assert [answer[c.answer_start : c.answer_end] for c in claims] == [
        "Refunds take 30 days [P1].",
        "Shipping is free [P2].",
    ]
    assert [c.cited_numbers for c in claims] == [(1,), (2,)]


async def test_llm_atomic_orders_claims_by_sentence():
    llm = _llm(
        _reply(
            {"sentence": 1, "text": "Shipping is free."},
            {"sentence": 0, "text": "Refunds take 30 days."},
        )
    )

    claims = await llm_atomic_claims(llm, "Refunds take 30 days. Shipping is free.")

    assert [(c.ordinal, c.text) for c in claims] == [
        (0, "Refunds take 30 days."),
        (1, "Shipping is free."),
    ]


async def test_llm_atomic_filters_invalid_items():
    answer = "Refunds take 30 days. Shipping is free."
    llm = _llm(
        _reply(
            {"sentence": 0, "text": "Refunds take 30 days."},
            {"sentence": 7, "text": "Seven is out of range."},
            {"sentence": -1, "text": "Negative indexes must not wrap around."},
            {"sentence": True, "text": "A boolean is not a sentence index."},
            {"sentence": 0, "text": "ok"},
            {"sentence": 0, "text": "x" * 401},
            {"sentence": 0, "text": 5},
            {"sentence": 0},
            "not an object",
        )
    )

    claims = await llm_atomic_claims(llm, answer)

    # Sentence 1 has no surviving item, so it falls back to its own text.
    assert [(c.method, c.text) for c in claims] == [
        ("llm_atomic", "Refunds take 30 days."),
        ("sentence", "Shipping is free."),
    ]


async def test_llm_atomic_keeps_text_bounds_inclusive():
    llm = _llm(
        _reply(
            {"sentence": 0, "text": "Yes"},
            {"sentence": 0, "text": "y" * 400},
        )
    )

    claims = await llm_atomic_claims(llm, "Refunds take 30 days.")

    assert [c.text for c in claims] == ["Yes", "y" * 400]
    assert {c.method for c in claims} == {"llm_atomic"}


async def test_llm_atomic_collapses_whitespace_in_claim_text():
    llm = _llm(_reply({"sentence": 0, "text": "  Refunds take\n30   days.  "}))

    claims = await llm_atomic_claims(llm, "Refunds take 30 days.")

    assert [c.text for c in claims] == ["Refunds take 30 days."]


async def test_llm_atomic_keeps_at_most_eight_per_sentence():
    llm = _llm(_reply(*({"sentence": 0, "text": f"Fact number {n} holds."} for n in range(1, 10))))

    claims = await llm_atomic_claims(llm, "Refunds take 30 days.")

    assert [c.text for c in claims] == [f"Fact number {n} holds." for n in range(1, 9)]
    assert [c.ordinal for c in claims] == list(range(8))


async def test_llm_atomic_keeps_at_most_fifty_overall_in_sentence_order():
    answer = _many_sentences(80)
    items = []
    for n in range(80):
        items.append({"sentence": n, "text": f"Item {n} ships today."})
        items.append({"sentence": n, "text": f"Item {n} is real."})
    llm = _llm(_reply(*items))

    claims = await llm_atomic_claims(llm, answer)

    assert len(claims) == 50
    assert [c.ordinal for c in claims] == list(range(50))
    assert claims[49].text == "Item 24 is real."
    assert answer[claims[49].answer_start : claims[49].answer_end] == "Item 24 ships today."


async def test_uncovered_sentence_gets_sentence_claim():
    answer = "Refunds take 30 days [P1]. Shipping is free [P2]."
    llm = _llm(_reply({"sentence": 0, "text": "Refunds take 30 days."}))

    claims = await llm_atomic_claims(llm, answer)

    assert [(c.ordinal, c.method, c.text, c.cited_numbers) for c in claims] == [
        (0, "llm_atomic", "Refunds take 30 days.", (1,)),
        (1, "sentence", "Shipping is free.", (2,)),
    ]
    assert answer[claims[1].answer_start : claims[1].answer_end] == "Shipping is free [P2]."


async def test_no_kept_sentences_skips_the_llm():
    llm = _llm(_reply({"sentence": 0, "text": "Refunds take 30 days."}))

    assert await llm_atomic_claims(llm, NO_INFO_ANSWER) == []
    assert llm.calls == []


@pytest.mark.parametrize("failure", [LLMUnavailable, LLMTimeout, LLMError])
async def test_llm_failure_falls_back(failure):
    answer = "Refunds take 30 days [P1]. Shipping is free [P2]."
    llm = _llm(failure("down"))

    claims = await llm_atomic_claims(llm, answer)

    assert [(c.ordinal, c.method, c.text, c.cited_numbers) for c in claims] == [
        (0, "sentence_fallback", "Refunds take 30 days.", (1,)),
        (1, "sentence_fallback", "Shipping is free.", (2,)),
    ]
    assert [answer[c.answer_start : c.answer_end] for c in claims] == [
        "Refunds take 30 days [P1].",
        "Shipping is free [P2].",
    ]


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "",
        "[]",
        '["claims"]',
        '"claims"',
        "null",
        "{}",
        '{"claims": "Refunds take 30 days."}',
        pytest.param("[" * 100_000, id="deeply-nested"),
    ],
)
async def test_invalid_json_falls_back(reply):
    llm = _llm(reply)

    claims = await llm_atomic_claims(llm, "Refunds take 30 days. Shipping is free.")

    assert [(c.method, c.text) for c in claims] == [
        ("sentence_fallback", "Refunds take 30 days."),
        ("sentence_fallback", "Shipping is free."),
    ]
