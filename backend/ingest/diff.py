"""Sentence-level diff between two canonical texts.

Both texts are split with `segment_sentences` and the sentence strings are compared with
`difflib.SequenceMatcher`. Every hunk carries offsets (Python `str` indices) into both texts.

- `equal`: `from_text` and `to_text` are empty, the offsets span the equal sentences.
- `insert`: `from_start == from_end` is the boundary in the old text, `to_*` span the new text.
- `delete`: `to_start == to_end` is the boundary in the new text, `from_*` span the old text.
- `replace`: both sides span their sentences.

An empty range sits at the end of the preceding sentence, at the start of the first sentence
when nothing precedes it, and at `len(text)` when it is past the last sentence.
"""

import difflib
from dataclasses import dataclass
from typing import Literal, cast

from backend.ingest.segment import segment_sentences
from backend.ingest.types import Sentence

Op = Literal["equal", "insert", "delete", "replace"]


@dataclass(frozen=True)
class DiffHunk:
    op: Op
    from_start: int
    from_end: int
    to_start: int
    to_end: int
    from_text: str  # "" for equal and insert hunks
    to_text: str  # "" for equal and delete hunks


def diff_texts(old: str, new: str) -> list[DiffHunk]:
    """The sentence-level hunks that turn `old` into `new`, in text order."""
    if old == new:
        return [DiffHunk("equal", 0, len(old), 0, len(new), "", "")]
    old_spans = segment_sentences(old)
    new_spans = segment_sentences(new)
    matcher = difflib.SequenceMatcher(
        None,
        [old[span.start : span.end] for span in old_spans],
        [new[span.start : span.end] for span in new_spans],
        autojunk=False,
    )
    hunks = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        from_start, from_end = _range(old, old_spans, i1, i2)
        to_start, to_end = _range(new, new_spans, j1, j2)
        hunks.append(_hunk(op, from_start, from_end, to_start, to_end, old, new))
    return _merge_adjacent(hunks, old, new)


def _range(text: str, spans: list[Sentence], lo: int, hi: int) -> tuple[int, int]:
    """Offsets covering `spans[lo:hi]`; an empty selection sits at the neighboring boundary."""
    if hi > lo:
        return spans[lo].start, spans[hi - 1].end
    if lo >= len(spans):
        return len(text), len(text)
    boundary = spans[lo - 1].end if lo > 0 else spans[0].start
    return boundary, boundary


def _hunk(
    op: str, from_start: int, from_end: int, to_start: int, to_end: int, old: str, new: str
) -> DiffHunk:
    return DiffHunk(
        cast(Op, op),  # difflib's opcode tags are exactly these four names
        from_start,
        from_end,
        to_start,
        to_end,
        old[from_start:from_end] if op in ("delete", "replace") else "",
        new[to_start:to_end] if op in ("insert", "replace") else "",
    )


def _merge_adjacent(hunks: list[DiffHunk], old: str, new: str) -> list[DiffHunk]:
    """Join neighboring hunks that have the same op into one."""
    merged: list[DiffHunk] = []
    for hunk in hunks:
        if merged and merged[-1].op == hunk.op:
            first = merged.pop()
            hunk = _hunk(
                hunk.op, first.from_start, hunk.from_end, first.to_start, hunk.to_end, old, new
            )
        merged.append(hunk)
    return merged
