import time

import pytest

from backend.ingest.errors import IngestError
from backend.ingest.normalize import normalize
from backend.ingest.parsers.markdown import parse_markdown
from backend.ingest.parsers.text import decode_utf8, parse_text


def md(source: str):
    return parse_markdown(source.encode("utf-8"))


def test_text_paragraphs_and_bom():
    doc = parse_text(b"\xef\xbb\xbfFirst para.\r\n\r\nSecond.")
    assert doc.text == "First para.\n\nSecond."
    assert (doc.parser, doc.parser_version) == ("text", "text/1")
    assert doc.pages == ()
    assert doc.sections == ()
    assert doc.metadata == {}


def test_text_blank_runs_split_paragraphs_and_single_breaks_stay():
    doc = parse_text(b"a\n\n\n   \nb\nc\n")
    assert doc.text == "a\n\nb\nc"


def test_text_empty_input_gives_empty_text():
    assert parse_text(b"").text == ""


def test_text_bare_carriage_returns_are_line_breaks():
    assert parse_text(b"a\r\rb\rc").text == "a\n\nb\nc"


def test_text_bad_utf8():
    with pytest.raises(IngestError) as excinfo:
        parse_text(b"\xff\xfe\x00")
    assert excinfo.value.code == "bad_encoding"


def test_decode_utf8_rejects_truncated_sequence():
    with pytest.raises(IngestError) as excinfo:
        decode_utf8("caf\u00e9".encode("utf-8")[:-1])
    assert excinfo.value.code == "bad_encoding"


def test_markdown_bad_utf8():
    with pytest.raises(IngestError) as excinfo:
        parse_markdown(b"\xff\xfe\x00")
    assert excinfo.value.code == "bad_encoding"


def test_markdown_strips_syntax():
    doc = md(
        "# Policy\n"
        "\n"
        "**Refunds** are [allowed](http://x) within *30 days*.\n"
        "\n"
        "Use `code` here.\n"
        "\n"
        "![logo](x.png)\n"
        "\n"
        "[ref]: http://y\n"
    )
    assert (doc.parser, doc.parser_version) == ("markdown", "markdown/1")
    assert doc.text == "Policy\n\nRefunds are allowed within 30 days.\n\nUse code here.\n\nlogo"
    for banned in ("**", "http://", "[ref]"):
        assert banned not in doc.text


def test_markdown_sections():
    doc = md("# A\n\nalpha para.\n\n## B\n\nbeta para.\n\n# C\n\ngamma para.\n")
    assert doc.text == "A\n\nalpha para.\n\nB\n\nbeta para.\n\nC\n\ngamma para."
    assert [s.path for s in doc.sections] == ["A", "A > B", "C"]
    assert [doc.text[s.start : s.end] for s in doc.sections] == [
        "A\n\nalpha para.",
        "B\n\nbeta para.",
        "C\n\ngamma para.",
    ]
    beta_start = doc.text.index("beta para.")
    section_b = doc.sections[1]
    assert section_b.start <= beta_start
    assert beta_start + len("beta para.") <= section_b.end


def test_markdown_heading_levels_trailing_hashes_and_skips():
    doc = md("# A\n\n### C ###\n\n## B\n\n#### D\n")
    assert [s.path for s in doc.sections] == ["A", "A > C", "A > B", "A > B > D"]
    assert doc.text == "A\n\nC\n\nB\n\nD"


def test_markdown_text_before_first_heading_has_no_section():
    doc = md("intro line\n\n# A\n\nbody")
    assert [(s.path, doc.text[s.start : s.end]) for s in doc.sections] == [("A", "A\n\nbody")]


def test_markdown_section_offsets_are_code_points():
    doc = md("# Caf\u00e9 \U0001f600\n\nna\u00efve text\n\n## \u00dcn\u00ef\n\nlast")
    assert [(s.path, doc.text[s.start : s.end]) for s in doc.sections] == [
        ("Caf\u00e9 \U0001f600", "Caf\u00e9 \U0001f600\n\nna\u00efve text"),
        ("Caf\u00e9 \U0001f600 > \u00dcn\u00ef", "\u00dcn\u00ef\n\nlast"),
    ]


def test_markdown_title_metadata():
    doc = md("# The **Refund** Policy\n\nbody\n\n# Second\n")
    assert doc.metadata == {"title": "The Refund Policy"}


def test_markdown_heading_path_and_title_are_canonical():
    doc = md("# Two  Spaces\n\nbody\n\n## Cafe\u0301\n\nmore")
    assert doc.metadata == {"title": "Two Spaces"}
    assert [s.path for s in doc.sections] == ["Two Spaces", "Two Spaces > Caf\u00e9"]


def test_markdown_no_level_one_heading_means_no_title():
    assert md("## Sub\n\ntext").metadata == {}


def test_markdown_code_fence_verbatim():
    code = "print('**not bold**')\nurl = [a](http://x)\n* star\n# not a heading"
    doc = md(f"Intro text.\n\n```python\n{code}\n```\n\nOutro.")
    assert doc.text == f"Intro text.\n\n{code}\n\nOutro."
    assert doc.sections == ()
    assert doc.metadata == {}


def test_markdown_tilde_fence_and_unclosed_fence():
    assert md("~~~\n**a**\n~~~\n\nafter").text == "**a**\n\nafter"
    assert md("before\n\n```\n**a**\nrest of file").text == "before\n\n**a**\nrest of file"


def test_markdown_shorter_or_different_fence_does_not_close_a_fence():
    longer = md("````\ninner\n```\nstill code\n````\n\nafter")
    assert longer.text == "inner\n```\nstill code\n\nafter"
    mixed = md("```\na\n~~~\nb\n```\n\nafter")
    assert mixed.text == "a\n~~~\nb\n\nafter"


def test_markdown_fence_inside_comment_is_not_a_fence():
    doc = md("<!--\n```\n-->\n\nafter")
    assert doc.text == "after"


def test_markdown_triple_backticks_closed_on_one_line_are_inline_code():
    doc = md("```x``` is code\n\nnext")
    assert doc.text == "x is code\n\nnext"


def test_markdown_nul_bytes_cannot_forge_code_span_placeholders():
    doc = md("a \x000\x00 b")
    assert doc.text == "a 0 b"


def test_markdown_table_rows():
    doc = md("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert doc.text == "a; b\n1; 2"


def test_markdown_table_alignment_separator_dropped_and_cells_cleaned():
    doc = md("| h1 | h2 |\n| :--- | ---: |\n| x | **y** |\n")
    assert doc.text == "h1; h2\nx; y"


def test_snake_case_kept():
    doc = md("Set max_retry_count and _private_ flag to my_var_name.")
    assert doc.text == "Set max_retry_count and private flag to my_var_name."


def test_markdown_markers_around_spaces_are_kept():
    doc = md("Compute 2 * 3 * 4 and a * b.")
    assert doc.text == "Compute 2 * 3 * 4 and a * b."


def test_markdown_code_spans_shield_markers():
    doc = md("Use `*args` and `**kwargs` here, and `[a](b)` too.")
    assert doc.text == "Use *args and **kwargs here, and [a](b) too."


def test_markdown_link_text_may_contain_code():
    doc = md("See [`foo()`](http://x/y) now.")
    assert doc.text == "See foo() now."


def test_markdown_strong_and_emphasis_variants():
    doc = md("__bold__ and **bold** and _it_ and *it* and ***both***.")
    assert doc.text == "bold and bold and it and it and both."


def test_markdown_lists_and_blockquotes():
    doc = md("* one\n+ two\n- three\n1. four\n\n> quoted line\n> > nested\n")
    assert doc.text == "- one\n- two\n- three\n1. four\n\nquoted line\nnested"


def test_markdown_drops_rules_and_comments():
    doc = md(
        "Before.\n\n---\n\n***\n\n___\n\n"
        "Mid <!-- inline --> text.\n\n"
        "<!-- multi\nline\ncomment -->\n\n"
        "After."
    )
    assert doc.text == "Before.\n\nMid text.\n\nAfter."


def test_markdown_consecutive_lines_form_one_piece():
    doc = md("line one\nline two\n\nline three")
    assert doc.text == "line one\nline two\n\nline three"


def test_markdown_output_normalized():
    doc = md(
        "\ufeff# T\u00e9st  Doc\r\n\r\n"
        "Para\twith   odd\u00a0spaces.   \r\n"
        "Second line\r\n\r\n\r\n\r\n"
        "| a  | b |\r\n|---|---|\r\n| 1 | 2 |\r\n\r\n"
        "```\r\ncode  here\r\n```\r\n"
    )
    assert doc.text
    assert normalize(doc.text) == doc.text


# Each input has many openers and no closer, which makes a naive unbounded pattern rescan the
# rest of the line per opener (quadratic). A regex cannot be interrupted by the parse timeout,
# so the parser itself has to stay fast. 200k characters take a minute or more when quadratic.
_HOSTILE = {
    "star_openers": "*a " * 66_000,
    "double_star_openers": "**a " * 50_000,
    "underscore_openers": "_a " * 66_000,
    "dunder_openers": "__a " * 50_000,
    "open_brackets": "[" * 200_000,
    "links_without_paren": "[a](" * 50_000,
    "growing_backtick_runs": "".join("`" * k + "a " for k in range(1, 600)),
    "heading_with_long_space_run": "# a" + " " * 200_000 + "x",
}


@pytest.mark.parametrize("source", _HOSTILE.values(), ids=_HOSTILE.keys())
def test_markdown_hostile_lines_stay_fast(source):
    started = time.perf_counter()
    md(source)
    assert time.perf_counter() - started < 5
