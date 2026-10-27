import time

import pytest

from backend.ingest.errors import IngestError
from backend.ingest.normalize import normalize
from backend.ingest.parsers.csv_table import parse_csv
from backend.ingest.parsers.html import parse_html


def html(source: str):
    return parse_html(source.encode("utf-8"))


def csv_doc(source: str):
    return parse_csv(source.encode("utf-8"))


def error_code(call, data: bytes) -> str:
    with pytest.raises(IngestError) as excinfo:
        call(data)
    return excinfo.value.code


# ---------------------------------------------------------------- HTML


def test_html_drops_scripts_and_styles():
    doc = html(
        "<html><head><title>Refund Policy</title>"
        "<style>p { color: red }</style><script>alert('head')</script></head>"
        "<body><p>Body text stays.</p>"
        "<script>alert(1)</script><style>.x { color: blue }</style>"
        "<noscript>Enable scripts</noscript><template><p>Hidden</p></template>"
        "<svg><text>Chart label</text></svg><p>After.</p></body></html>"
    )
    assert (doc.parser, doc.parser_version) == ("html", "html/1")
    assert doc.text == "Body text stays.\n\nAfter."
    assert "alert(" not in doc.text
    assert "color:" not in doc.text
    assert doc.metadata == {"title": "Refund Policy"}
    assert doc.pages == ()


def test_html_script_content_is_never_parsed_as_markup():
    doc = html("<p>before</p><script>var s = '</p><p>fake</p>';</script><p>after</p>")
    assert doc.text == "before\n\nafter"


def test_html_blocks_lists_tables():
    doc = html(
        "<p>One</p><p>Two</p><ul><li>a</li><li>b</li></ul>"
        "<table><tr><th>k</th><th>v</th></tr><tr><td>x</td><td>1</td></tr></table>"
    )
    assert doc.text == "One\n\nTwo\n\n- a\n\n- b\n\nk; v\n\nx; 1"


def test_html_inline_tags_do_not_split_words_or_paragraphs():
    doc = html("<p>Re<b>fund</b> <i>within</i>\n   30   days</p>")
    assert doc.text == "Refund within 30 days"


def test_html_entities_and_br():
    doc = html("<p>Fish &amp; chips &lt;3 &#169; caf&eacute;<br>second line<br/>third</p>")
    assert doc.text == "Fish & chips <3 \u00a9 caf\u00e9\nsecond line\nthird"


def test_html_br_does_not_leave_leading_or_trailing_spaces_on_lines():
    doc = html("<p>one <br> two</p>")
    assert doc.text == "one\ntwo"


def test_html_sections_from_headings():
    doc = html(
        "<h1>A</h1><p>alpha para.</p><h2>B</h2><p>beta para.</p><h1>C</h1><p>gamma para.</p>"
    )
    assert doc.text == "A\n\nalpha para.\n\nB\n\nbeta para.\n\nC\n\ngamma para."
    assert [s.path for s in doc.sections] == ["A", "A > B", "C"]
    assert [doc.text[s.start : s.end] for s in doc.sections] == [
        "A\n\nalpha para.",
        "B\n\nbeta para.",
        "C\n\ngamma para.",
    ]


def test_html_heading_levels_skip_and_pop():
    doc = html("<h1>A</h1><h3>C</h3><h2>B</h2><h4>D</h4>")
    assert [s.path for s in doc.sections] == ["A", "A > C", "A > B", "A > B > D"]
    assert doc.text == "A\n\nC\n\nB\n\nD"


def test_html_text_before_first_heading_has_no_section():
    doc = html("<p>intro</p><h1>A</h1><p>body</p>")
    assert [(s.path, doc.text[s.start : s.end]) for s in doc.sections] == [("A", "A\n\nbody")]


def test_html_heading_text_is_one_canonical_line():
    doc = html("<h1>The <b>Refund</b>  Policy<br>Cafe\u0301</h1><p>x</p>")
    assert doc.text == "The Refund Policy Caf\u00e9\n\nx"
    assert [s.path for s in doc.sections] == ["The Refund Policy Caf\u00e9"]


def test_html_section_offsets_are_code_points():
    doc = html(
        "<h1>Caf\u00e9 \U0001f600</h1><p>na\u00efve text</p><h2>\u00dcn\u00ef</h2><p>last</p>"
    )
    assert [(s.path, doc.text[s.start : s.end]) for s in doc.sections] == [
        ("Caf\u00e9 \U0001f600", "Caf\u00e9 \U0001f600\n\nna\u00efve text"),
        ("Caf\u00e9 \U0001f600 > \u00dcn\u00ef", "\u00dcn\u00ef\n\nlast"),
    ]


def test_html_empty_heading_is_ignored():
    doc = html("<h1>A</h1><h2>  </h2><p>body</p>")
    assert [s.path for s in doc.sections] == ["A"]
    assert doc.text == "A\n\nbody"


def test_html_title_only_comes_from_title_element():
    assert html("<h1>Heading</h1><p>x</p>").metadata == {}
    doc = html("<title>  Spaced \n  Title </title><p>x</p><title>Second</title>")
    assert doc.metadata == {"title": "Spaced Title"}
    assert doc.text == "x"


def test_html_svg_title_is_not_the_document_title():
    doc = html("<svg><title>Icon</title></svg><p>x</p>")
    assert doc.metadata == {}
    assert doc.text == "x"


def test_html_unclosed_title_gives_no_title():
    assert html("<title>Never closed").metadata == {}


def test_html_unclosed_head_ends_at_body_content():
    doc = html("<html><head><title>T</title><meta charset=utf-8><p>content</p>")
    assert doc.text == "content"
    assert doc.metadata == {"title": "T"}


def test_html_pre_keeps_newlines():
    doc = html("<p>Before</p><pre>line one\nline two\n\nline four</pre><p>After</p>")
    assert doc.text == "Before\n\nline one\nline two\n\nline four\n\nAfter"


def test_html_blockquote_and_definition_lists_and_figures_are_pieces():
    doc = html(
        "<blockquote>quote</blockquote><dl><dt>term</dt><dd>meaning</dd></dl>"
        "<figure><figcaption>caption</figcaption></figure>"
    )
    assert doc.text == "quote\n\nterm\n\nmeaning\n\ncaption"


def test_html_nested_list_items_are_separate_prefixed_pieces():
    doc = html("<ul><li>a<ul><li>b</li></ul></li><li>c</li></ul>")
    assert doc.text == "- a\n\n- b\n\n- c"


def test_html_empty_list_item_adds_no_piece_and_no_stray_prefix():
    doc = html("<ul><li></li></ul><p>after</p>")
    assert doc.text == "after"


def test_html_list_item_with_block_child_keeps_its_prefix():
    doc = html("<ul><li><p>wrapped</p></li></ul>")
    assert doc.text == "- wrapped"


def test_html_table_omits_empty_cells_and_blank_rows():
    doc = html(
        "<table><tr><td>a</td><td></td><td>c</td></tr>"
        "<tr><td> </td><td></td></tr><tr><td>d</td></tr></table>"
    )
    assert doc.text == "a; c\n\nd"


def test_html_table_cell_with_paragraphs_stays_in_the_row():
    doc = html("<table><tr><td><p>x</p><p>y</p></td><td>z</td></tr></table><p>after</p>")
    assert doc.text == "x; y; z\n\nafter"


def test_html_unclosed_table_cells_and_rows():
    doc = html("<table><tr><td>a<td>b<tr><td>c<td>d</table>")
    assert doc.text == "a; b\n\nc; d"


def test_html_comments_and_doctype_are_dropped():
    doc = html("<!DOCTYPE html><!-- hidden --><p>shown<!-- also hidden --></p>")
    assert doc.text == "shown"


def test_html_malformed_does_not_raise():
    doc = html("<p>open <b>bold <div>x")
    assert doc.text == "open bold\n\nx"


@pytest.mark.parametrize(
    "source",
    [
        "",
        "<",
        "<<<>>>",
        "</p></div></li>",
        "<p",
        "<p a='unterminated>text",
        "<!--",
        "<![CDATA[ x",
        "<script>never closed",
        "<style>never closed",
        "<svg><g><path>",
        "<table><tr><td><table><tr><td>deep",
        "<h1>unclosed <h2>nested <h3>headings",
        "<ul><li><li><li>",
        "<pre>unclosed pre\n\n\n  text",
        "&#xFFFFFFFF; &#0; &bogus; &amp",
    ],
)
def test_html_garbage_does_not_raise(source):
    html(source)


def test_html_text_without_any_tags():
    assert html("just text").text == "just text"


def test_html_empty_input_gives_empty_text():
    assert html("").text == ""


def test_html_bad_utf8():
    assert error_code(parse_html, b"<p>\xff\xfe</p>") == "bad_encoding"


def test_html_bom_is_dropped():
    assert parse_html(b"\xef\xbb\xbf<p>x</p>").text == "x"


def test_html_output_normalized():
    doc = html(
        "<h1>T\u00e9st  Doc</h1><p>Para\twith   odd\u00a0spaces.   <br>  Second</p>"
        "<pre>code  here\r\n\r\n\r\n\r\nmore</pre><table><tr><td>a  </td><td>b</td></tr></table>"
    )
    assert doc.text
    assert normalize(doc.text) == doc.text


# Each input is shaped to make a naive implementation rescan or copy a growing amount per
# tag (quadratic). A regex or a pure-Python loop cannot be interrupted by the parse timeout,
# so the parser itself has to stay fast. A quadratic implementation takes a minute or more.
_HOSTILE_HTML = {
    "open_angle_brackets": "<" * 200_000,
    "unterminated_tags": "<a " * 66_000,
    "unterminated_attribute": "<a b='" * 40_000,
    "unterminated_comments": "<!--" * 50_000,
    "bare_ampersands": "&" * 200_000,
    "entities_without_semicolon": "&amp" * 40_000,
    "unclosed_paragraphs": "<p>x" * 66_000,
    "unclosed_nested_divs": "<div>x" * 40_000,
    "unclosed_svg": "<svg>" * 40_000 + "text",
    "mismatched_skip_end_tags": "<svg><head>" * 20_000
    + "</template>" * 20_000
    + "</head></svg>" * 20_000,
    "unclosed_scripts": "<script>" * 30_000,
    "script_with_many_end_like_runs": "<script>" + "</scr" * 40_000,
    "many_titles": "<title>t</title>" * 30_000,
    "unclosed_titles": "<title>t" * 30_000,
    "unclosed_list_items": "<li>x" * 66_000,
    "unclosed_table_cells": "<table><tr>" + "<td>x" * 50_000,
    "many_line_breaks": "<p>" + "<br>" * 100_000 + "x",
    "many_empty_headings": "<h1></h1>" * 25_000,
    "deep_heading_stack": "<h1>a</h1><h2>b</h2><h3>c</h3><h4>d</h4><h5>e</h5><h6>f</h6>" * 4_000,
}


@pytest.mark.parametrize("source", _HOSTILE_HTML.values(), ids=_HOSTILE_HTML.keys())
def test_html_hostile_input_stays_fast(source):
    started = time.perf_counter()
    html(source)
    assert time.perf_counter() - started < 5


# ---------------------------------------------------------------- CSV


def test_csv_rows_render():
    doc = parse_csv(b"name,days\nRefund,30\nExchange,\n")
    assert doc.text == "name: Refund; days: 30\n\nname: Exchange"
    assert (doc.parser, doc.parser_version) == ("csv", "csv/1")
    assert doc.pages == ()
    assert doc.sections == ()
    assert doc.metadata == {}


def test_csv_blank_header():
    doc = csv_doc("name,,  ,days\nA,B,C,D\n")
    assert doc.text == "name: A; column_2: B; column_3: C; days: D"


def test_csv_values_beyond_the_header_use_column_numbers():
    doc = csv_doc("a,b\n1,2,3,4\n")
    assert doc.text == "a: 1; b: 2; column_3: 3; column_4: 4"


def test_csv_quoted_fields_commas_and_newlines():
    doc = csv_doc('name,note\n"Smith, J.","said ""hi""\nand left"\n')
    assert doc.text == 'name: Smith, J.; note: said "hi"\nand left'


def test_csv_whitespace_only_values_are_empty_and_rows_without_values_are_skipped():
    doc = csv_doc("a,b\n , \n1,  \n,,\n\n2,3\n")
    assert doc.text == "a: 1\n\na: 2; b: 3"


def test_csv_crlf_bom_and_non_ascii():
    doc = parse_csv("\ufeffn\u00e4me,city\r\nZo\u00eb,\U0001f600 Paris\r\n".encode("utf-8"))
    assert doc.text == "n\u00e4me: Zo\u00eb; city: \U0001f600 Paris"


def test_csv_blank_lines_before_the_header_are_ignored():
    assert csv_doc("\n\na,b\n1,2\n").text == "a: 1; b: 2"


def test_csv_output_normalized():
    doc = csv_doc("a,b\n  x\u00a0 y ,Cafe\u0301\t z\n")
    assert doc.text == "a: x y; b: Caf\u00e9 z"
    assert normalize(doc.text) == doc.text


def test_csv_row_limit():
    rows = "x\n" * 50_001
    assert error_code(parse_csv, f"a\n{rows}".encode()) == "too_many_rows"


def test_csv_row_limit_boundary_is_inclusive():
    rows = "x\n" * 50_000
    doc = csv_doc(f"a\n{rows}")
    assert doc.text.count("a: x") == 50_000


def test_csv_header_only_no_text():
    assert error_code(parse_csv, b"name,days\n") == "no_text"
    assert error_code(parse_csv, b"name,days") == "no_text"


def test_csv_without_any_text_is_no_text():
    assert error_code(parse_csv, b"") == "no_text"
    assert error_code(parse_csv, b"\n\n") == "no_text"
    assert error_code(parse_csv, b"a,b\n,\n , \n") == "no_text"


def test_csv_bad_encoding():
    assert error_code(parse_csv, b"a,b\n\xff,\xfe\n") == "bad_encoding"


def test_csv_oversized_field_is_bad_csv():
    assert error_code(parse_csv, b"a\n" + b"x" * 200_000 + b"\n") == "bad_csv"


def test_csv_error_does_not_carry_document_text():
    with pytest.raises(IngestError) as excinfo:
        parse_csv(b"secret\n" + b"x" * 200_000)
    assert "secret" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


# (source, expected ingest error code or None when the input parses).
_HOSTILE_CSV = {
    "one_giant_empty_row": ("," * 5_000_000, "no_text"),
    "giant_header_then_row": ("a," * 1_000_000 + "\n" + "b," * 1_000_000, None),
    "many_blank_lines": ("a\n" + "\n" * 5_000_000, "no_text"),
    "quote_runs": ('a\n"' + '""' * 60_000 + '"\n', None),
    "unterminated_quote": ('a\n"' + "x," * 50_000, None),
    "many_empty_rows": ("a,b\n" + ",\n" * 50_000, "no_text"),
    "wide_rows": ("a\n" + ("," * 400 + "x\n") * 2_000, None),
}


@pytest.mark.parametrize(
    ("source", "expected_code"), _HOSTILE_CSV.values(), ids=_HOSTILE_CSV.keys()
)
def test_csv_hostile_input_stays_fast(source, expected_code):
    started = time.perf_counter()
    if expected_code is None:
        assert csv_doc(source).text
    else:
        assert error_code(parse_csv, source.encode("utf-8")) == expected_code
    assert time.perf_counter() - started < 5
