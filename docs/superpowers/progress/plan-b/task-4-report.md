# Plan B Task 4 report: plain text and Markdown parsers

Status: DONE_WITH_CONCERNS (concerns are scope notes, nothing blocking)

Commit: `38b31bd feat: add text and markdown parsers` on `feature/evidencegraph` (not pushed).
Working tree clean after the commit.

## What was built

Files (all new, LF endings, no em dashes):

- `backend/ingest/parsers/__init__.py` (empty, as the brief says)
- `backend/ingest/parsers/text.py`
  - `decode_utf8(data) -> str`: strict `utf-8-sig`; failure raises `IngestError("bad_encoding")` with `from None` so the offending bytes stay out of tracebacks.
  - `parse_text(data) -> ParsedDocument`: splits on blank lines (`\n\s*\n`), adds each paragraph to a `TextAssembler`; parser `text`, version `text/1`, no pages, no sections, empty metadata.
- `backend/ingest/parsers/markdown.py`
  - `parse_markdown(data) -> ParsedDocument`: line based, no Markdown library, built on `TextAssembler`; parser `markdown`, version `markdown/1`.
  - All brief rules implemented: ATX headings with a level stack, `start_section(" > ".join(stack))`, heading text as its own piece, first level-1 heading in `metadata["title"]`; fenced code (``` and ~~~) verbatim as one piece; inline code, images, links, emphasis (`**`, `__`, `*`, `_`, intra-word underscores kept), blockquote prefixes, `*`/`+` list markers to `-`, numbered markers kept, horizontal rules, reference definitions, HTML comments (multi-line too), pipe tables (separator rows dropped, cells joined with `"; "`); consecutive non-blank lines join with `"\n"` into one piece.
- `tests/test_parsers_text_md.py`: 40 tests (the 8 named in the brief plus coverage of every other rule in the brief and the hostile-input checks below).

## RED evidence

Command (all runs through the WSL wrapper, under `timeout 300`):
`wsl -d Ubuntu -- bash ".../.local/wsl.sh" timeout 300 python -m pytest tests/test_parsers_text_md.py -q -p no:cacheprovider`

1. Tests only, no production code:
   ```
   ERROR collecting tests/test_parsers_text_md.py
   E   ModuleNotFoundError: No module named 'backend.ingest.parsers'
   1 error in 0.63s
   ```
2. Stubs that raise `NotImplementedError` (so each test fails at the call, not at import):
   ```
   26 failed in 0.41s
   (every failure: NotImplementedError from parsers/text.py or parsers/markdown.py)
   ```

## GREEN evidence

First implementation (26 tests): `26 passed in 0.31s`.

Review of my own test set with a throwaway mutation script (outside the repo, in the Windows temp dir; it restored each source file after every mutation). First pass left 4 survivors (fence close length, fence close character, fence inside a comment, bare `\r` handling). I added tests for the first three and found the fourth was an equivalent mutant in `text.py` (the assembler joins paragraphs with the same `"\n\n"` that `normalize` collapses blank runs to, so splitting is unobservable in the text output; I removed the redundant `\r` replaces instead of keeping dead code). Final pass: 22 mutations, 0 survivors:

```
killed(fail)    drop normalize() on heading text/title/path
killed(fail)    drop NUL removal in _clean_inline
killed(fail)    drop backtick-fence lookahead
killed(fail)    heading stack pop >= to >
killed(fail)    fence closes on any length
killed(fail)    fence closes on any character
killed(fail)    fence opens inside an HTML comment
killed(fail)    skip list marker normalization
killed(fail)    skip blockquote stripping
killed(fail)    keep reference definitions
killed(fail)    keep horizontal rules
killed(fail)    keep table separator rows
killed(fail)    join cells with space instead of "; "
killed(fail)    title from any heading level
killed(fail)    skip code span shielding
killed(fail)    intra-word underscore guard removed
killed(fail)    drop image handling
killed(fail)    drop final paragraph flush
killed(fail)    utf-8 instead of utf-8-sig
killed(fail)    _MAX_SPAN 300 -> 100000
killed(timeout) lazy heading regex (quadratic)
killed(timeout) non-possessive, unanchored code span (quadratic)
baseline: pass
```

Focused run, final: `40 passed in 1.67s`.

## Full suite and lint

- Full suite: `timeout 900 python -m pytest -q` -> `543 passed, 2 warnings in 58.58s`. The 2 warnings are the existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` in `tests/test_generator.py` (not touched by this task).
- Lint: `timeout 120 ruff check .` -> `All checks passed!`
- `ruff format --check backend/ingest tests/test_parsers_text_md.py` -> `11 files already formatted`. Running `ruff format --check backend tests` repo-wide reports 24 other files that would be reformatted; all pre-date this task and I did not touch them.
- Hooks (gitleaks) ran on commit: no leaks found.

## Decisions (brief ambiguous or silent)

1. Fence delimiter lines (the ``` or ~~~ line and its info string) are dropped; the code lines between them are kept verbatim as one piece. The brief says fenced code lines are kept verbatim; the delimiters are syntax, and `python` info strings would be noise in retrieval. An unclosed fence runs to the end of the file. A closing fence must use the same character and be at least as long as the opener. A backtick fence line may not contain further backticks (otherwise it is an inline code span).
2. Heading regex. The brief's `^(#{1,6})\s+(.*?)\s*#*\s*$` backtracks quadratically on a long whitespace run (200k spaces would run for minutes and a regex cannot be interrupted by the 60 s parse timeout). I match `(#{1,6})\s+(.*)` and trim with `rstrip().rstrip("#").rstrip()`, which gives the same group for every input I compared by reasoning (trailing whitespace, then closing hashes, then whitespace).
3. Heading text, section path, and `metadata["title"]` all go through `normalize()`, so they match the canonical text the section span covers (for example `Cafe` + combining accent becomes the NFC form in the path). Empty headings are ignored.
4. Heading levels use a stack of `(level, text)`: a new heading pops everything at the same or deeper level, so skipped levels simply do not appear in the path (`# A`, `### C` gives `A > C`).
5. Blank handling. A line that is blank, only a comment, a horizontal rule, a reference definition, or empty after inline cleaning ends the current paragraph. Table separator rows and all-empty table rows are skipped without ending it, so a whole table is one piece.
6. Tables: a row is a line whose stripped form starts with `|`. Cells are split on `|`, cleaned individually, and empty cells are dropped.
7. Inline processing order: stash code spans behind NUL placeholders (content never treated as syntax, so `` `*args` `` and `` `**kwargs` `` survive), then links and images, then emphasis, then restore. NUL is removed from the input first so a hostile file cannot forge a placeholder (that would be an `IndexError` instead of an `IngestError`).
8. Blockquote prefixes are stripped before heading and list detection, so `> # Title` is a heading.
9. The text parser does not raise `no_text` for an empty file; the brief puts that in Task 10.

## Hostile input (untrusted uploads)

My first implementation used unbounded lazy patterns and was quadratic. Measured on the first version at 20,000 characters: `*a ` repeated 0.79 s, `**a ` repeated 0.60 s, `_a ` repeated 0.80 s, `__a ` repeated 1.29 s (so about 80 to 130 s at 200k). Fixes:

- Emphasis, code span, and link text repeats are bounded by `_MAX_SPAN = 300` (longer spans keep their markers); closing markers are matched before their lookbehind so the lazy scan stays cheap.
- Code spans anchor at the start of a backtick run and take the whole run possessively (a long backtick run was otherwise tried once per length and per start).
- Links and images are found with a bounded `[text](` regex plus `str.find(")")`, which handles a URL of any length (inline base64 data URIs) in linear time and stops as soon as no `)` is left.
- Heading regex as in decision 2.

After: same 20k-character inputs take 0.02 to 0.04 s. At 2,000,000 characters (22 hostile shapes: unmatched `*`, `**`, `_`, `__`, `[`, `[a](`, `![a](`, backtick runs of growing length, long space runs after a heading, table pipes, and others) the worst case is 3.63 s (`__a ` repeated), next 3.47 s (`[` repeated); plain prose of 2M characters takes 0.43 s. The suite keeps `test_markdown_hostile_lines_stay_fast` (8 shapes at 200k characters, each must finish in under 5 s; they take 0.2 to 0.35 s). That test was written after the fix, so it has no recorded RED run (running it against the naive version would take over a minute per case, which the rules forbid for RED); the `_MAX_SPAN` and regex mutations above show it fails when the bounds are lost.

## Concerns

1. The brief's heading pattern strips trailing hashes without requiring a preceding space, so a heading like `# C#` becomes `C`. I kept that behavior (decision 2 preserves it); CommonMark would keep `C#`.
2. Emphasis spans longer than 300 characters keep their `*` or `_` markers. Emphasis and links that wrap across a line break are not stripped (the parser is line based, as specified).
3. Not in the brief, so not handled: setext headings (`Title` over `---`; the `---` is dropped as a rule), YAML front matter (the `---` fences are dropped but the `key: value` lines stay as a paragraph), escaped characters such as `\*` and `\|`, reference-style links `[text][ref]`, HTML tags other than comments, indented code blocks.
4. `[Note]: some sentence` matches the brief's reference-definition pattern (no end anchor) and is dropped.
5. Worst-case hostile line at the 20 MiB byte limit scales to roughly 35 s in this parser (3.6 s per 2M characters, linear), under the 60 s timeout but not by a wide margin; the 2,000,000 character limit in a later task should be applied before parsing where possible, or the parse should run in a killable process.
6. Pre-existing repo formatting drift (24 files fail `ruff format --check`) and the `test_generator` RuntimeWarning are unrelated and untouched.
