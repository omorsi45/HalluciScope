"""Sentence-aligned passage builder.

A passage is a run of whole sentences taken from one section of the canonical text, so
`text[p.start:p.end]` is exactly the passage text. Offsets are Python `str` indices (Unicode
code points), which Postgres `substr` on UTF-8 text also uses.
"""

import hashlib
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from backend.ingest.segment import segment_sentences
from backend.ingest.types import PageSpan, PassageDraft, SectionSpan, Sentence


def build_passages(
    text: str,
    pages: Sequence[PageSpan] = (),
    sections: Sequence[SectionSpan] = (),
    *,
    target_chars: int = 800,
    max_chars: int = 1200,
    overlap_sentences: int = 1,
) -> list[PassageDraft]:
    """Group the sentences of `text` into passages without crossing section boundaries.

    Pages must not overlap each other. Sections are clipped so regions never overlap.
    """
    ordered_pages = sorted(pages, key=lambda page: page.start)
    page_starts = [page.start for page in ordered_pages]
    page_ends = [page.end for page in ordered_pages]

    drafts: list[PassageDraft] = []
    for region_start, region_end, section in _regions(len(text), sections):
        sentences = _region_sentences(text, region_start, region_end, max_chars)
        for start, end in _group(sentences, target_chars, max_chars, overlap_sentences):
            first = bisect_right(page_ends, start)  # first page ending after `start`
            last = bisect_left(page_starts, end) - 1  # last page starting before `end`
            has_pages = first <= last
            passage_text = text[start:end]
            drafts.append(
                PassageDraft(
                    ordinal=len(drafts),
                    start=start,
                    end=end,
                    text=passage_text,
                    page_start=ordered_pages[first].page if has_pages else None,
                    page_end=ordered_pages[last].page if has_pages else None,
                    section=section,
                    text_sha256=hashlib.sha256(passage_text.encode("utf-8")).digest(),
                )
            )
    return drafts


def _regions(length: int, sections: Sequence[SectionSpan]) -> list[tuple[int, int, str | None]]:
    """Section spans plus the gaps between them, in document order."""
    regions: list[tuple[int, int, str | None]] = []
    cursor = 0
    for span in sorted(sections, key=lambda s: s.start):
        start = max(span.start, cursor)
        end = min(span.end, length)
        if start >= end:
            continue
        if start > cursor:
            regions.append((cursor, start, None))
        regions.append((start, end, span.path))
        cursor = end
    if cursor < length:
        regions.append((cursor, length, None))
    return regions


def _region_sentences(text: str, start: int, end: int, max_chars: int) -> list[Sentence]:
    sentences: list[Sentence] = []
    for sentence in segment_sentences(text, start, end):
        if sentence.end - sentence.start > max_chars:
            sentences.extend(_split_long(text, sentence, max_chars))
        else:
            sentences.append(sentence)
    return sentences


def _split_long(text: str, sentence: Sentence, max_chars: int) -> list[Sentence]:
    """Cut at the last whitespace at or before `max_chars`, or hard at `max_chars`."""
    pieces: list[Sentence] = []
    start = sentence.start
    while sentence.end - start > max_chars:
        cut = start + max_chars
        while cut > start and not text[cut].isspace():
            cut -= 1
        if cut == start:  # no whitespace in the window
            pieces.append(Sentence(start, start + max_chars))
            start += max_chars
            continue
        end = cut
        while text[end - 1].isspace():
            end -= 1
        pieces.append(Sentence(start, end))
        start = cut
        while text[start].isspace():
            start += 1
    pieces.append(Sentence(start, sentence.end))
    return pieces


def _group(
    sentences: Sequence[Sentence], target_chars: int, max_chars: int, overlap_sentences: int
) -> list[tuple[int, int]]:
    """Greedy runs of sentences as `(start, end)` offsets, with sentence overlap."""
    spans: list[tuple[int, int]] = []
    count = len(sentences)
    first = 0
    last = 0
    while first < count:
        while last + 1 < count and sentences[last + 1].end - sentences[first].start <= target_chars:
            last += 1
        spans.append((sentences[first].start, sentences[last].end))
        following = last + 1
        if following >= count:
            break
        overlap_first = following - overlap_sentences
        if (
            last - first + 1 > overlap_sentences
            and sentences[following].end - sentences[overlap_first].start <= max_chars
        ):
            first = overlap_first
        else:
            first = following
        # The next sentence always joins the passage, so every passage makes progress.
        last = following
    return spans
