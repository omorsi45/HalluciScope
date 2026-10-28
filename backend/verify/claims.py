"""Split an answer into claims, each tied to the span of the sentence it came from.

Sentences come from the same segmentation that locates the answer's `[P#]` citation markers
(`backend.generate.answer`), so a claim's span and its `cited_numbers` always agree with the
stored citations. The claim text drops the markers; the span keeps them.

`llm_atomic_claims` asks the LLM to split sentences into atomic claims and validates every
item, because the reply is untrusted. When the LLM fails or returns something unusable, the
sentence claims are used and marked `sentence_fallback`.
"""

import json
import re
from typing import Any, Literal

from backend.generate.answer import _MARKER, _sentence_spans, parse_citations
from backend.ingest.types import Sentence
from backend.llm.client import LLMClient, LLMFailure
from backend.llm.prompts import EXTRACT_SYSTEM, NO_INFO_ANSWER
from backend.verify.types import Claim

MAX_CLAIMS = 50
MAX_PER_SENTENCE = 8
MIN_CHARS = 3
MAX_CHARS = 400

CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["sentence", "text"],
            },
        }
    },
    "required": ["claims"],
}

# English function words only: articles, pronouns, auxiliaries, prepositions, conjunctions.
# Content verbs, nouns, adjectives and negations ("not", "no", "never") are deliberately left
# out because they carry meaning; "may" is left out because it is also a month.
# fmt: off
STOPWORDS: frozenset[str] = frozenset({
    # articles and demonstratives
    "a", "an", "the", "this", "that", "these", "those",
    # pronouns and question words
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "there", "here",
    # auxiliaries and modals
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "might", "must",
    # prepositions
    "of", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "over", "under", "out", "off", "within", "without", "across", "along", "among", "around",
    "behind", "beside", "beyond", "near", "toward", "towards", "upon", "onto", "per", "via",
    "inside", "outside", "throughout", "despite",
    # conjunctions
    "and", "but", "or", "nor", "so", "yet", "if", "as", "than", "because", "while",
    "although", "though", "unless", "whether", "until", "since", "both", "either", "neither",
    # fragments left by splitting contractions on the apostrophe ("it's" -> "it", "s")
    "s", "t", "d", "ll", "m", "re", "ve",
})
# fmt: on
_WORD = re.compile(r"[A-Za-z0-9]+")
# A marker and the whitespace before it, so "days [P1]." becomes "days." and not "days .".
_MARKER_AND_GAP = re.compile(r"\s*" + _MARKER.pattern)
_NO_INFO = NO_INFO_ANSWER.lower().removesuffix(".")


def claim_sentences(answer: str) -> list[tuple[Sentence, str, tuple[int, ...]]]:
    """The sentences of `answer` worth turning into claims.

    Each item is the sentence span in `answer`, its text with citation markers removed and
    whitespace collapsed, and the passage numbers cited in it (first appearance order, no
    repeats). Sentences with no usable text, no content word, or the no-information reply
    are skipped.
    """
    cited: dict[tuple[int, int], list[int]] = {}
    for ref in parse_citations(answer, 0):
        numbers = cited.setdefault((ref.sentence_start, ref.sentence_end), [])
        if ref.number not in numbers:
            numbers.append(ref.number)

    kept = []
    for start, end in _sentence_spans(answer):
        text = " ".join(_MARKER_AND_GAP.sub("", answer[start:end]).split())
        if len(text) < MIN_CHARS or not _has_content_word(text) or _is_no_info(text):
            continue
        kept.append((Sentence(start, end), text, tuple(cited.get((start, end), ()))))
    return kept


def sentence_claims(
    answer: str, *, method: Literal["sentence", "sentence_fallback"] = "sentence"
) -> list[Claim]:
    """One claim per kept sentence, at most `MAX_CLAIMS`."""
    return [
        Claim(
            ordinal=ordinal,
            text=_truncate(text),
            answer_start=sentence.start,
            answer_end=sentence.end,
            method=method,
            cited_numbers=cited,
        )
        for ordinal, (sentence, text, cited) in enumerate(claim_sentences(answer)[:MAX_CLAIMS])
    ]


async def llm_atomic_claims(llm: LLMClient, answer: str) -> list[Claim]:
    """Atomic claims from the LLM; sentence claims marked `sentence_fallback` if it fails."""
    sentences = claim_sentences(answer)
    if not sentences:
        return []

    listing = "\n".join(f"{index}. {text}" for index, (_, text, _) in enumerate(sentences))
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": listing},
    ]
    try:
        reply = await llm.chat(messages, temperature=0.0, json_schema=CLAIMS_SCHEMA)
    except LLMFailure:
        return sentence_claims(answer, method="sentence_fallback")
    items = _reply_items(reply)
    if items is None:
        return sentence_claims(answer, method="sentence_fallback")

    texts_by_sentence: dict[int, list[str]] = {}
    for item in items:
        valid = _valid_item(item, len(sentences))
        if valid is None:
            continue
        index, text = valid
        kept_texts = texts_by_sentence.setdefault(index, [])
        if len(kept_texts) < MAX_PER_SENTENCE:
            kept_texts.append(text)

    claims: list[Claim] = []
    for index, (sentence, sentence_text, cited) in enumerate(sentences):
        method: Literal["llm_atomic", "sentence"] = "llm_atomic"
        texts = texts_by_sentence.get(index)
        if not texts:
            method, texts = "sentence", [_truncate(sentence_text)]
        for text in texts:
            claims.append(
                Claim(
                    ordinal=len(claims),
                    text=text,
                    answer_start=sentence.start,
                    answer_end=sentence.end,
                    method=method,
                    cited_numbers=cited,
                )
            )
    return claims[:MAX_CLAIMS]


def _has_content_word(text: str) -> bool:
    return any(word.lower() not in STOPWORDS for word in _WORD.findall(text))


def _is_no_info(text: str) -> bool:
    return text.lower().removesuffix(".") == _NO_INFO


def _truncate(text: str) -> str:
    """Cut `text` (single-spaced) at the last space within `MAX_CHARS` characters."""
    if len(text) <= MAX_CHARS:
        return text
    cut = text.rfind(" ", 0, MAX_CHARS + 1)
    return text[:cut] if cut > 0 else text[:MAX_CHARS]


def _reply_items(reply: str) -> list[Any] | None:
    """The `claims` list of the LLM reply, or None when the reply is not usable JSON."""
    try:
        data = json.loads(reply)
    except (ValueError, RecursionError):
        # RecursionError: the reply nests arrays or objects deeper than the parser allows.
        return None
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return None
    return data["claims"]


def _valid_item(item: Any, n_sentences: int) -> tuple[int, str] | None:
    """`(sentence index, claim text)` for a well-formed item, else None."""
    if not isinstance(item, dict):
        return None
    index, text = item.get("sentence"), item.get("text")
    # `type() is int` rejects True and False, which would otherwise index sentences 1 and 0.
    if type(index) is not int or not 0 <= index < n_sentences:
        return None
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return None
    return index, text
