import random

import pytest

from backend.ingest.assemble import TextAssembler
from backend.ingest.errors import IngestError
from backend.ingest.normalize import normalize
from backend.ingest.types import PageSpan, SectionSpan

ALPHABET = ["a", " ", "\t", "\n", "\r", "\x00", "\u00a0", "e", "\u0301", "\U0001f600", "."]

INGEST_ERROR_CODES = [
    "unsupported_type",
    "bad_encoding",
    "bad_pdf",
    "encrypted_pdf",
    "no_text",
    "too_many_rows",
    "bad_csv",
    "too_large",
    "too_long",
    "too_many_passages",
    "parse_timeout",
    "effective_from_out_of_order",
    "document_deleted",
]


def _random_strings(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    return [
        "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 40))) for _ in range(count)
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb\rc", "a\nb\nc"),
        ("a\t\u00a0 b", "a b"),
        ("a \x01 b", "a b"),
        ("x  \ny", "x\ny"),
        ("a\n\n\n\nb", "a\n\nb"),
        ("  \n a \n ", "a"),
        ("e\u0301", "\u00e9"),
    ],
)
def test_normalize_rules(raw, expected):
    assert normalize(raw) == expected


def test_normalize_idempotent():
    for s in _random_strings(seed=7, count=200):
        once = normalize(s)
        assert normalize(once) == once, repr(s)


def test_normalize_composes_marks_exposed_by_control_removal():
    # A control character between a base letter and a combining mark is removed, so the
    # pair becomes adjacent and must compose to the single code point.
    assert normalize("e\x00\u0301") == "\u00e9"


def test_assembler_offsets_and_pages():
    asm = TextAssembler()
    first = asm.add("Page one.", page=1)
    skipped = asm.add("  ", page=2)
    third = asm.add("Page\tthree.", page=3)
    doc = asm.build({}, "txt", "1")

    assert skipped is None
    assert doc.text == "Page one.\n\nPage three."
    assert doc.pages == (PageSpan(1, 0, 9), PageSpan(3, 11, 22))
    assert first == (0, 9)
    assert third == (11, 22)
    assert doc.text[first[0] : first[1]] == "Page one."
    assert doc.text[third[0] : third[1]] == "Page three."


def test_assembler_extends_open_page_span_for_the_same_page():
    asm = TextAssembler()
    asm.add("One.", page=1)
    asm.add("Two.", page=1)
    asm.add("Three.", page=2)
    doc = asm.build({}, "txt", "1")

    assert doc.text == "One.\n\nTwo.\n\nThree."
    assert doc.pages == (PageSpan(1, 0, 10), PageSpan(2, 12, 18))


def test_assembler_page_less_piece_does_not_extend_a_page_span():
    asm = TextAssembler()
    asm.add("One.", page=1)
    asm.add("Loose.")
    asm.add("Two.", page=1)
    doc = asm.build({}, "txt", "1")

    assert doc.pages == (PageSpan(1, 0, 4), PageSpan(1, 14, 18))


def test_assembler_sections():
    asm = TextAssembler()
    asm.start_section("A")
    asm.add("Alpha one.")
    asm.add("Alpha two.")
    asm.start_section("A > B")
    asm.add("Beta.")
    asm.start_section(None)
    asm.add("Tail.")
    doc = asm.build({"title": "T"}, "md", "2")

    assert doc.text == "Alpha one.\n\nAlpha two.\n\nBeta.\n\nTail."
    assert doc.sections == (SectionSpan(0, 22, "A"), SectionSpan(24, 29, "A > B"))
    assert doc.metadata == {"title": "T"}
    assert (doc.parser, doc.parser_version) == ("md", "2")


def test_assembler_drops_sections_without_pieces():
    asm = TextAssembler()
    asm.start_section("Empty")
    asm.start_section("Blank")
    asm.add("   ")
    asm.start_section("Real")
    asm.add("Body.")
    asm.start_section("Trailing")
    doc = asm.build({}, "md", "2")

    assert doc.sections == (SectionSpan(0, 5, "Real"),)


def test_assembled_text_is_normalized():
    rng = random.Random(11)
    for _ in range(100):
        asm = TextAssembler()
        for piece in _random_strings(seed=rng.randint(0, 10_000), count=rng.randint(0, 6)):
            asm.add(piece, page=rng.choice([None, 1, 2]))
        doc = asm.build({}, "txt", "1")
        assert normalize(doc.text) == doc.text, repr(doc.text)


@pytest.mark.parametrize("code", INGEST_ERROR_CODES)
def test_ingest_error_accepts_known_codes(code):
    err = IngestError(code)
    assert err.code == code


@pytest.mark.parametrize("code", ["unknown_code", "", "NO_TEXT", "no_text "])
def test_unknown_ingest_error_code_rejected(code):
    with pytest.raises(ValueError):
        IngestError(code)
