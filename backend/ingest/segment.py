"""Offset-preserving sentence segmentation.

`segment_sentences` returns spans whose offsets are Python `str` indices into the original
text, so `text[s.start:s.end]` is the sentence. One left-to-right scan, linear in the input.
"""

import re

from backend.ingest.types import Sentence

ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "inc",
        "ltd",
        "jr",
        "sr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "no",
        "fig",
        "eq",
        "co",
        "corp",
        "dept",
        "approx",
    }
)

# Anchored at the position given to `match`, so no leading `^` is needed.
_LIST_MARKER = re.compile(r"(?:[-*+\u2022]|\d{1,3}[.)])\s")
_TERMINATORS = ".!?"
_CLOSERS = "\"'\u201d\u2019)]"
_OPENERS = "\"'\u201c\u2018([{"
_TERMINATOR_RUN = _TERMINATORS + _CLOSERS


def segment_sentences(text: str, start: int = 0, end: int | None = None) -> list[Sentence]:
    """Split `text[start:end]` into sentences with offsets into `text`.

    Sentences exclude surrounding whitespace. A single newline that is not a list or heading
    break stays inside the sentence (a PDF line wrap).
    """
    lo = max(start, 0)
    hi = len(text) if end is None else min(end, len(text))
    sentences: list[Sentence] = []
    sent_start = -1
    i = lo
    while i < hi:
        ch = text[i]
        if sent_start < 0:
            if ch.isspace():
                i += 1
                continue
            sent_start = i
        if ch == "\n":
            if _newline_breaks(text, sent_start, i, hi):
                sentences.append(_trimmed(text, sent_start, i))
                sent_start = -1
        elif ch in _TERMINATORS:
            j = i + 1
            while j < hi and text[j] in _TERMINATOR_RUN:
                j += 1
            if _ends_sentence(text, lo, sent_start, i, j, hi):
                sentences.append(Sentence(sent_start, j))
                sent_start = -1
            i = j
            continue
        i += 1
    if sent_start >= 0:
        sentences.append(_trimmed(text, sent_start, hi))
    return sentences


def _trimmed(text: str, start: int, end: int) -> Sentence:
    while end > start and text[end - 1].isspace():
        end -= 1
    return Sentence(start, end)


def _newline_breaks(text: str, sent_start: int, i: int, hi: int) -> bool:
    """Whether the newline at `i` ends the sentence (rules 1 and 2)."""
    if i + 1 < hi and text[i + 1] == "\n":
        return True
    k = i
    while k > sent_start and text[k - 1] in " \t":
        k -= 1
    if k > sent_start and text[k - 1] == ":":
        return True
    return _LIST_MARKER.match(text, i + 1, hi) is not None


def _ends_sentence(text: str, lo: int, sent_start: int, i: int, j: int, hi: int) -> bool:
    """Whether the terminator at `i` (its run of closers ends before `j`) ends the sentence."""
    if j >= hi or not text[j].isspace():
        return False
    k = j
    while k < hi and text[k].isspace():
        k += 1
    if k >= hi:
        return False
    nxt = text[k]
    if not (
        nxt.isupper()
        or nxt.isdecimal()
        or nxt in _OPENERS
        or _LIST_MARKER.match(text, k, hi) is not None
    ):
        return False
    if text[i] != ".":
        return True
    token_start = i
    while token_start > lo and (text[token_start - 1].isalnum() or text[token_start - 1] == "."):
        token_start -= 1
    token = text[token_start:i]
    if token.lower() in ABBREVIATIONS:
        return False
    if len(token) == 1 and token.isupper():
        return False
    # A numbered list marker such as "1." at the start of a sentence is not a sentence.
    return not (token_start == sent_start and len(token) <= 3 and token.isdecimal())
