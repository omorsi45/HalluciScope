# Plan B Task 5 report: HTML and CSV parsers

Status: DONE_WITH_CONCERNS (one concern, shared with Task 4; see Concerns)

Commit: `ea463f9 feat: add html and csv parsers` on `feature/evidencegraph` (not pushed). Pre-commit gitleaks hook ran and passed ("no leaks found"). Staged by name: 3 files.

## What was built

- `backend/ingest/parsers/html.py`: `parse_html(data) -> ParsedDocument`, parser `html`, version `html/1`. An `html.parser.HTMLParser` subclass (`convert_charrefs=True`), streaming and non-recursive, fed the whole `decode_utf8(data)` string.
- `backend/ingest/parsers/csv_table.py`: `parse_csv(data) -> ParsedDocument`, parser `csv`, version `csv/1`. Stdlib `csv.reader` (dialect `excel`) over `io.StringIO(decode_utf8(data), newline="")`.
- `tests/test_parsers_html_csv.py`: 88 tests (parametrized hostile-input cases included).

### HTML behavior (all per brief)

- `script, style, noscript, template, svg, head` content dropped. `<title>` text becomes `metadata["title"]` (first closed title wins; a title inside svg/template/noscript is ignored).
- Block tags `p, div, section, article, header, footer, main, nav, aside, ul, ol, li, table, tr, pre, blockquote, dl, dt, dd, figure, figcaption, hr, form` flush the paragraph on start and end tags. `td`/`th` are cell boundaries; a row is the non-empty cells joined with `"; "`. `<br>` inserts `"\n"`. `<li>` content gets `"- "`. `<pre>` keeps newlines. `h1..h6` use the same level stack and `start_section(" > ".join(stack))` pattern as Markdown and are added as their own piece.
- Whitespace outside `<pre>` collapses with `[ \t\r\n\f]+` to one space (HTML rules; a plain character class, linear).
- No exception is raised for unclosed or garbage markup (17 garbage inputs in a parametrized test).

### CSV behavior (all per brief)

- First non-blank record is the header; blank (after `strip`) header cells become `column_<1-based index>`; each data row becomes `"h1: v1; h2: v2"`, empty (after `strip`) values omitted, rows with no values skipped.
- More than 50,000 data rows raises `too_many_rows`; `csv.Error` raises `bad_csv` (`from None`, so no document text in the traceback); no text at all raises `no_text`; bad UTF-8 raises `bad_encoding` (via `decode_utf8`).

## RED evidence

Tests written first, run before any implementation existed:

```
$ wsl.sh timeout 300 python -m pytest tests/test_parsers_html_csv.py -q -x
ERROR collecting tests/test_parsers_html_csv.py
tests/test_parsers_html_csv.py:7: in <module>
    from backend.ingest.parsers.csv_table import parse_csv
E   ModuleNotFoundError: No module named 'backend.ingest.parsers.csv_table'
1 error in 0.63s     (exit 2)
```

This is the expected "feature missing" failure (neither module existed). No expensive behavior can run in RED because the import fails first.

## GREEN evidence

```
$ wsl.sh timeout 300 python -m pytest tests/test_parsers_html_csv.py -q
88 passed in 3.85s
```

Slowest cases (`--durations`): CSV `giant_header_then_row` (2M cells) 1.12 s; every HTML hostile case 0.26 s or less. All hostile tests assert under 5 s.

Linear-time check (ad hoc script, deleted afterwards): for 13 hostile HTML shapes (unterminated attribute, `<!--`, `<`, `&amp`, unclosed `<p>`, `<script></scr`, unclosed `<td>`, `<svg><head></template>` mismatches, CDATA, PI, doctype, bad end tags, quoted attributes) timing at 50k and 200k repeats scaled about 4x for 4x the input (largest: 0.68 s for 4.4 MB).

## Full suite and lint

```
$ wsl.sh timeout 900 python -m pytest -q
631 passed, 2 warnings in 63.29s
```

The 2 warnings are pre-existing (`tests/test_generator.py`, `backend/core/generator.py:50` un-awaited AsyncMock coroutine); not touched by this task. No failures.

```
$ wsl.sh timeout 120 ruff check .
All checks passed!
$ wsl.sh timeout 120 ruff format --check backend/ingest tests/test_parsers_html_csv.py
13 files already formatted
```

(`ruff format --check backend tests` over the whole tree reports 24 files that would be reformatted. None are mine; they are pre-existing and out of scope.) New files are LF (`git ls-files --eol`: `w/lf`). No em dashes in the new files (a pre-existing one exists in `backend/core/generator.py`, not touched).

## Decisions

1. Implied `</head>`: HTML5 lets `</head>` be omitted. If `head` is the only open skipped element and a start tag outside the head-content set (`title, meta, link, base, script, style, noscript, template, head`) arrives, the head is treated as closed. Without this, a document with an omitted `</head>` would lose its entire body. Not in the brief; small and tested.
2. Unclosed `<title>` yields no title (otherwise Python 3.14's title handling would swallow the whole document into the title). The title is set only on `</title>`.
3. Tables: empty cells are omitted from a row (matches the Markdown table behavior); a block element inside a cell (for example `<p>x</p><p>y</p>`) stays in the same row as separate cells joined by `"; "`. A row ends at the next `tr`/`table` tag, a heading, or end of input.
4. `<li>` prefix is applied only when the item produces text, and survives a block child (`<li><p>wrapped</p></li>` gives `- wrapped`); empty `<li>` adds nothing.
5. Skip stack is a stack plus a count dict, each entry pushed and popped once, so mismatched end tags stay amortized constant time.
6. Headings are collapsed to one line (`br` and whitespace runs become single spaces); empty headings are ignored. No `<h1>` fallback for the title: the brief says only `<title>` sets `metadata["title"]`.
7. CSV: a data row of only empty values is skipped but counted toward the 50,000 limit; truly blank lines are not records and are not counted (also ignored before the header). Cells beyond the header length use `column_<index>`. If the whole result has no text (empty file, header only, every row empty) the parser raises `no_text`, same code path as the header-only case in the brief.
8. CSV keeps the stdlib `csv.field_size_limit()` default (131,072 characters per field): an oversized field is a `csv.Error`, therefore `bad_csv`. The global is not modified.

## Concerns

1. Memory amplification from section paths (HTML here, and the same pattern already exists in `parse_markdown` from Task 4). Each `SectionSpan.path` is a fresh `" > ".join(...)` string, so a long heading followed by many sub-headings holds `sections x len(ancestors)` characters. Measured on `parse_html`: a 220 KB input (`<h1>` of 200,000 chars followed by 2,000 `<h2>b</h2>`) holds 400,208,000 path characters (about 400 MB RSS). A 20 MB upload could need tens of GB before any downstream limit (`too_long`, `too_many_passages`) runs. Time stays linear (the join is a memcpy); the problem is memory. I did not change `assemble.py` or Markdown (earlier tasks, out of scope). Suggested fix for the controller: cap the length of each path component (or the total path) inside `TextAssembler.start_section`, plus an early `too_long` check in the assembler, so every parser (HTML, Markdown, later PDF) is covered at once.
2. Python's stdlib `HTMLParser` has had quadratic edge cases in older releases. The venv runs Python 3.14.9 and all 13 shapes above scaled linearly, but the guarantee depends on the interpreter version in production.
3. Minor, deferred: `<br>` inside `<pre>` is a literal newline like text newlines; `<textarea>`, `<select>`, `<option>`, `<label>` content is treated as ordinary inline text; `<caption>` text becomes its own paragraph before the table rows; no `<h1>` fallback title; non-UTF-8 HTML (for example windows-1252 or a `<meta charset>`) raises `bad_encoding` because `decode_utf8` is strict, as for the other text parsers.
