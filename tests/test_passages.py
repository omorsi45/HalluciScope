import hashlib
import random

import pytest

from backend.ingest.passages import build_passages
from backend.ingest.segment import segment_sentences
from backend.ingest.types import PageSpan, SectionSpan

MAX_CHARS = 1200

# Mixed scripts and combining marks so offsets are checked in code points, not bytes.
WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "caf\u00e9",
    "na\u00efve",
    "e\u0301clair",
    "\U0001f600",
    "\u65e5\u672c\u8a9e",
    "tokens",
    "refund",
    "policy",
]
OPENERS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Refunds", "Policy"]


def sentence(index: int, length: int = 100) -> str:
    """A sentence of exactly `length` characters that the segmenter keeps whole."""
    body = f"S{index}"
    return body + "b" * (length - len(body) - 1) + "."


def generated_text(seed: int) -> str:
    rng = random.Random(seed)
    paragraphs = []
    for _ in range(30):
        sentences = []
        for _ in range(rng.randint(1, 12)):
            words = [rng.choice(WORDS) for _ in range(rng.randint(5, 40))]
            words[0] = rng.choice(OPENERS)
            sentences.append(" ".join(words) + rng.choice(".!?"))
        paragraphs.append(" ".join(sentences))
    unpunctuated = "Unpunctuated " + " ".join(rng.choice(WORDS) for _ in range(600))
    paragraphs.insert(15, unpunctuated[:3000])
    return "\n\n".join(paragraphs)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_invariants_on_generated_text(seed):
    text = generated_text(seed)
    sentences = segment_sentences(text)
    assert any(s.end - s.start > MAX_CHARS for s in sentences)

    passages = build_passages(text)

    assert len(passages) > 30
    assert [p.ordinal for p in passages] == list(range(len(passages)))
    starts = [p.start for p in passages]
    assert starts == sorted(starts)
    for p in passages:
        assert p.text == text[p.start : p.end]
        assert 0 < len(p.text) <= MAX_CHARS
        assert p.text_sha256 == hashlib.sha256(p.text.encode()).digest()
        assert (p.page_start, p.page_end, p.section) == (None, None, None)

    covered = bytearray(len(text))
    for p in passages:
        covered[p.start : p.end] = b"\x01" * (p.end - p.start)
    for s in sentences:
        if s.end - s.start <= MAX_CHARS:
            assert any(p.start <= s.start and s.end <= p.end for p in passages)
        else:
            # Pieces of an over-long sentence are trimmed at the cuts, so only whitespace
            # may be left uncovered.
            assert all(covered[i] or text[i].isspace() for i in range(s.start, s.end))


def test_overlap_one_sentence():
    text = " ".join(sentence(i) for i in range(10))

    passages = build_passages(text, target_chars=350)

    # Sentence i spans [101 * i, 101 * i + 100). Each passage starts at the previous
    # passage's last sentence.
    assert [(p.start, p.end) for p in passages] == [
        (0, 302),
        (202, 504),
        (404, 706),
        (606, 908),
        (808, 1009),
    ]


def test_no_overlap_when_single_sentence_passages():
    text = " ".join(sentence(i, 700) for i in range(5))

    passages = build_passages(text)

    assert [(p.start, p.end) for p in passages] == [
        (0, 700),
        (701, 1401),
        (1402, 2102),
        (2103, 2803),
        (2804, 3504),
    ]


def test_overlap_skipped_when_it_would_exceed_max_chars():
    text = " ".join([sentence(0, 300), sentence(1, 600), sentence(2, 700)])

    passages = build_passages(text, target_chars=1000)

    # Repeating the 600-character sentence before the 700-character one would need 1301.
    assert [(p.start, p.end) for p in passages] == [(0, 901), (902, 1602)]


def test_overlap_may_exceed_target_chars_within_max_chars():
    text = " ".join([sentence(0, 300), sentence(1, 300), sentence(2, 600)])

    passages = build_passages(text, target_chars=700)

    # Overlap plus the next sentence is 901 characters: over target, under max, so it is kept.
    assert [(p.start, p.end) for p in passages] == [(0, 601), (301, 1202)]


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        ("aaaa bbbb cccc dddd", 11, [(0, 9), (10, 19)]),
        ("aaaa bbbb cccc", 9, [(0, 9), (10, 14)]),
        ("abcdefghijklmnop", 10, [(0, 10), (10, 16)]),
        ("aaaa   bbbb", 6, [(0, 4), (7, 11)]),
    ],
    ids=["last_whitespace", "whitespace_at_limit", "hard_cut", "trimmed_pieces"],
)
def test_overlong_sentence_is_split(text, max_chars, expected):
    passages = build_passages(text, target_chars=max_chars, max_chars=max_chars)

    assert [(p.start, p.end) for p in passages] == expected
    assert all(p.text == text[p.start : p.end] for p in passages)


def test_sections_never_crossed():
    intro = "Intro text here."
    block_a = " ".join(sentence(i) for i in range(4))
    block_b = " ".join(sentence(i) for i in range(4))
    text = intro + "\n\n" + block_a + "\n\n" + block_b
    a_start = len(intro) + 2
    b_start = a_start + len(block_a) + 2
    sections = [
        SectionSpan(a_start, a_start + len(block_a), "A"),
        SectionSpan(b_start, len(text), "B"),
    ]

    passages = build_passages(text, sections=sections, target_chars=250)

    assert [p.section for p in passages] == [None, "A", "A", "A", "B", "B", "B"]
    regions = [(0, len(intro)), (a_start, a_start + len(block_a)), (b_start, len(text))]
    for p in passages:
        assert any(lo <= p.start and p.end <= hi for lo, hi in regions)
    # Sentence i of a block spans [101 * i, 101 * i + 100); overlap restarts inside a section.
    assert [(p.start - a_start, p.end - a_start) for p in passages[1:4]] == [
        (0, 201),
        (101, 302),
        (202, 403),
    ]
    assert passages[4].start == b_start


def test_pages_mapping():
    text = sentence(0) + " " + sentence(1) + "\n\n" + sentence(2) + " " + sentence(3)
    pages = [PageSpan(1, 0, 201), PageSpan(2, 203, 404)]

    passages = build_passages(text, pages, target_chars=250)

    assert [(p.start, p.end, p.page_start, p.page_end) for p in passages] == [
        (0, 201, 1, 1),
        (101, 303, 1, 2),
        (203, 404, 2, 2),
    ]


def test_empty_text():
    assert build_passages("") == []
