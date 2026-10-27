# Plan B Task 2 report: offset-preserving sentence segmentation

Status: DONE
Commit: `ac4ef14 feat: add offset-preserving sentence segmentation` (branch `feature/evidencegraph`, not pushed)

## What I built

- `backend/ingest/segment.py`: `ABBREVIATIONS` (frozenset, exact values from the brief) and
  `segment_sentences(text, start=0, end=None) -> list[Sentence]`.
  Single left-to-right scanner, precompiled list-marker regex, linear time (backward token
  scans stop at whitespace, so each token is scanned at most once).
- `tests/test_segment.py`: the 9 brief tests verbatim in intent, plus 2 extra (see Decisions).

Files changed (staged by name): `backend/ingest/segment.py`, `tests/test_segment.py`. Nothing else.

## How the rules map to code

1. `"\n\n"` ends a sentence: `_newline_breaks` returns True when the next char is `\n`.
2. Single `"\n"`: ends the sentence when the current line (trailing spaces/tabs ignored) ends with
   `:` or the next line matches the list-marker regex; otherwise the scan continues, so the
   newline stays inside the sentence text (PDF wrap).
3. Terminator run: after the first of `.!?`, the run of `.!?"'\u201d\u2019)]` is consumed
   atomically, then the sentence ends if whitespace follows and the next non-space char is an
   uppercase letter, a decimal digit, an opening quote or bracket (`"'\u201c\u2018([{`), or a
   list marker.
4. No split after an abbreviation (token before the `.`, lowercased, may contain dots such as
   `e.g`), after a single uppercase letter (initials), or inside a number (`3.14`, `1.5x` have no
   whitespace after the `.`, so rule 3 never fires).
5. Trailing text without terminal punctuation is emitted at the end of the scan.

Sentences are trimmed of surrounding whitespace; offsets are indices into the original `text`,
and `start`/`end` are clamped to `[0, len(text)]`.

## RED

Command:
`wsl -d Ubuntu -- bash ".../.local/wsl.sh" timeout 300 python -m pytest tests/test_segment.py -q`

```
ERROR collecting tests/test_segment.py
tests/test_segment.py:1: in <module>
    from backend.ingest.segment import segment_sentences
E   ModuleNotFoundError: No module named 'backend.ingest.segment'
1 error in 0.50s
```

Fails for the expected reason (feature missing), exit 2. The run was written before any
production code, and the module did not exist.

Per-test RED for the extra numbered-list test: after implementing, I temporarily removed only the
numbered-list guard (last line of `_ends_sentence`, replaced by `return True`), ran the file, saw
the test fail, then restored the guard:

```
.........F.
FAILED tests/test_segment.py::test_numbered_list_marker_is_not_a_sentence
E   At index 1 diff: '1.' != '1. First step'
E   Left contains 2 more items, first extra item: '2.'
1 failed, 10 passed in 0.51s
```

The restore was verified by the GREEN runs below (11 passed).

## GREEN

```
tests/test_segment.py -q
...........                                                              [100%]
11 passed in 0.23s
```

Full suite (`timeout 900 python -m pytest -q`, run once before committing):

```
489 passed, 2 warnings in 59.43s   (exit 0)
```

The 2 warnings are pre-existing (`tests/test_generator.py`: "coroutine ... was never awaited" at
`backend/core/generator.py:50`), unrelated to this task. No failures.

Lint:

```
ruff check .                                   -> All checks passed!   (exit 0)
ruff format --check segment.py test_segment.py -> 2 files already formatted (exit 0)
```

Em dash (U+2014) count in the two new files: 0.

Commit hooks: the pre-commit hook ran gitleaks ("no leaks found"); no hook bypassed. Working tree is
clean after the commit.

## Decisions

1. No conflict between the brief's rules and its expected outputs. All 9 acceptance cases pass as
   specified, including `test_paragraph_and_wrap` (`"Line one wraps\nhere."` keeps its inner
   newline) and `test_offsets_with_range_and_unicode` (`segment_sentences(s, 6)`, offsets exact
   with an accented letter and an emoji, which is one code point and one `str` index).
2. List-marker regex: the brief writes `^(?:[-*+\u2022]|\d{1,3}[.)])\s`. I compiled it without
   `^` and call `pattern.match(text, pos, endpos)`, which anchors at `pos`. With `^` and no
   MULTILINE flag, `match(text, pos)` would fail at any `pos` that is not index 0, and rule 3 needs
   to match mid-line. Same pattern otherwise.
3. Numbered list guard (small addition to rule 4, "inside a number"): a `.` after 1 to 3 digits
   that start the current sentence is not a boundary. Without it, rule 3 splits `"1. First step"`
   into `"1."` and `"First step"` because `F` is uppercase, which contradicts rule 2 treating
   `1.` as a list marker. Covered by `test_numbered_list_marker_is_not_a_sentence`
   (RED shown above).
4. Extra test `test_end_bound_limits_scan` covers the `end` argument, which the brief's tests do
   not exercise: `segment_sentences("One. Two. Three.", 0, 9)` gives `[(0, 4), (5, 9)]`.

## Concerns (non-blocking, behavior follows the brief's rules)

- Abbreviation list is applied regardless of what follows: a real sentence ending in `No.`, `St.`,
  `Co.`, `Inc.`, or a single capital such as `... plan B.` followed by an uppercase word will not
  split. This is the trade-off the brief's rule 4 implies.
- Rule 1 is exactly `"\n\n"`. A blank line containing spaces (`"\n \n"`) is not a paragraph break;
  `normalize` strips trailing spaces, so canonical text does not contain it.
- Only `. ! ?` terminate sentences; full-width CJK punctuation is not handled (not in the brief).
