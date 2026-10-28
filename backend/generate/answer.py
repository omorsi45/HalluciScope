"""Generate an answer from retrieved passages and locate its `[P#]` citation markers.

Passage numbers are 1-based positions in the context passed to the model. The model's output is
untrusted: markers it invents are kept as `unknown_marker`, never dropped or repaired.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from backend.ingest.segment import segment_sentences
from backend.llm.client import LLMClient
from backend.llm.prompts import ANSWER_SYSTEM, NO_INFO_ANSWER
from backend.retrieval.types import PassageHit

# Case-sensitive "P"; no spaces right inside the brackets. After the first number, the "P" of
# later numbers is optional: [P1, P3] and [P1, 3] both list two passages.
_MARKER = re.compile(r"\[(P\d+(?:\s*,\s*P?\d+)*)\]")
_NUMBER = re.compile(r"\d+")
# int() raises ValueError on very long digit strings. A number this long is far beyond any
# context size, so it is recorded as this out-of-range value instead.
_MAX_DIGITS = 9
_OUT_OF_RANGE = 10**9


@dataclass(frozen=True)
class CitationRef:
    number: int  # 1-based P number as written
    marker_start: int  # offsets of the whole bracket in the answer
    marker_end: int
    sentence_start: int  # the containing sentence
    sentence_end: int
    status: Literal["valid", "unknown_marker"]


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    citations: tuple[CitationRef, ...]
    context: tuple[PassageHit, ...]


def render_context(hits: Sequence[PassageHit]) -> str:
    """Number the hits `[P1]`, `[P2]`, ... and wrap each passage text in `<passage>` tags."""
    blocks = []
    for number, hit in enumerate(hits, start=1):
        header = f"[P{number}] {hit.document_title}, v{hit.version_number}"
        if hit.page_start is not None:
            header += f", p. {hit.page_start}"
        # Passage text is untrusted: it must not be able to close its own block.
        text = hit.text.replace("</passage>", "</ passage>")
        blocks.append(f"{header}\n<passage>{text}</passage>")
    return "\n\n".join(blocks)


def build_answer_messages(question: str, hits: Sequence[PassageHit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"{render_context(hits)}\n\nQuestion: {question}"},
    ]


def parse_citations(answer: str, n_context: int) -> list[CitationRef]:
    """Find every `[P#]` marker in `answer` and tie it to its sentence.

    A bracket listing several passages yields one `CitationRef` per number, all sharing the
    bracket offsets. A number outside `1..n_context` is `unknown_marker`.
    """
    matches = list(_MARKER.finditer(answer))
    if not matches:
        return []
    sentences = _sentence_spans(answer)
    citations = []
    for match in matches:
        sentence_start, sentence_end = _containing_sentence(sentences, match.start())
        for digits in _NUMBER.findall(match.group(1)):
            number = int(digits) if len(digits) <= _MAX_DIGITS else _OUT_OF_RANGE
            citations.append(
                CitationRef(
                    number=number,
                    marker_start=match.start(),
                    marker_end=match.end(),
                    sentence_start=sentence_start,
                    sentence_end=sentence_end,
                    status="valid" if 1 <= number <= n_context else "unknown_marker",
                )
            )
    return citations


def _sentence_spans(answer: str) -> list[tuple[int, int]]:
    """Sentence spans of `answer`, with the citation markers treated as gaps between sentences.

    Models usually write "Refunds take 30 days. [P1] Shipping is free. [P2]". Segmenting the
    raw text would start the second sentence at "[P1]" and attach that marker to the wrong
    sentence. Blanking each marker with spaces of the same length keeps every offset valid for
    `answer` and leaves the marker in the gap after the sentence it follows.
    """
    blanked = _MARKER.sub(lambda match: " " * len(match.group(0)), answer)
    sentences = segment_sentences(blanked)
    if not sentences:
        # The answer has no text besides markers, so the raw text is its only sentence.
        sentences = segment_sentences(answer)
    return [(sentence.start, sentence.end) for sentence in sentences]


def _containing_sentence(sentences: list[tuple[int, int]], offset: int) -> tuple[int, int]:
    """The sentence holding `offset`; in a gap, the preceding sentence; before all, the first."""
    chosen = sentences[0]
    for sentence in sentences:
        if sentence[0] > offset:
            break
        chosen = sentence
    return chosen


async def generate_answer(
    llm: LLMClient, question: str, hits: Sequence[PassageHit]
) -> GeneratedAnswer:
    if not hits:
        return GeneratedAnswer(text=NO_INFO_ANSWER, citations=(), context=())
    text = await llm.chat(build_answer_messages(question, hits), temperature=0.0)
    return GeneratedAnswer(
        text=text,
        citations=tuple(parse_citations(text, len(hits))),
        context=tuple(hits),
    )
