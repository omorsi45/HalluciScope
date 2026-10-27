# Plan B Task 1 report: ingestion types, normalization, text assembler

Status: DONE_WITH_CONCERNS (one deviation from the brief's step order, see Decisions)

Commit: `dee9a0b feat: add ingestion types, normalization, and text assembler` on `feature/evidencegraph` (not pushed).

## What was built

New package `backend/ingest/`:

- `types.py`: frozen dataclasses `PageSpan`, `SectionSpan`, `ParsedDocument` (with `sections`), `Sentence`, `PassageDraft`, exactly the fields in the brief.
- `errors.py`: `INGEST_ERROR_CODES` (the 13 codes from Global Constraints, including `document_deleted`) and `IngestError(code)`; an unknown code raises `ValueError` with a constant message (the code is not echoed).
- `normalize.py`: `normalize(text) -> str`.
- `assemble.py`: `TextAssembler` with `add`, `start_section`, `build`.
- `__init__.py`: docstring only.

Tests: `tests/test_normalize.py` (32 test cases including parametrized ones).

## Files changed (6, all new, 328 insertions)

backend/ingest/__init__.py, backend/ingest/types.py, backend/ingest/errors.py, backend/ingest/normalize.py, backend/ingest/assemble.py, tests/test_normalize.py

## RED evidence

Tests written first. Command:

```
wsl -d Ubuntu -- bash ".../.worktrees/evidencegraph/.local/wsl.sh" timeout 300 python -m pytest tests/test_normalize.py -q
```

Output (modules missing, as the brief expects):

```
tests/test_normalize.py:5: in <module>
    from backend.ingest.assemble import TextAssembler
E   ModuleNotFoundError: No module named 'backend.ingest'
ERROR tests/test_normalize.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.49s
```

Second RED (found while going GREEN): after implementing the brief's literal step order, 3 tests failed for a real reason:

```
FAILED tests/test_normalize.py::test_normalize_idempotent
FAILED tests/test_normalize.py::test_normalize_composes_marks_exposed_by_control_removal
FAILED tests/test_normalize.py::test_assembled_text_is_normalized
3 failed, 29 passed in 0.66s
assert normalize("e\x00\u0301") == "\u00e9"   ->   got 'e\u0301'
```

Root cause: the brief orders NFC (step 1) before control-character removal (step 3). A control char between a base letter and a combining mark gets removed, the two become adjacent, and only a second `normalize` pass composes them. That breaks the two explicit requirements `normalize(normalize(s)) == normalize(s)` and `normalize(result.text) == result.text`.

## GREEN evidence

```
tests/test_normalize.py -q  ->  32 passed in 0.33s
```

Full suite (`timeout 900 python -m pytest -q`): `478 passed, 2 warnings in 59.62s`. The 2 warnings are the existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` from `tests/test_generator.py` (`backend/core/generator.py:50`), untouched by this task.

Lint (`timeout 120 ruff check .`): `All checks passed!`

Hygiene: no em dashes and no CR characters in the new files (grep clean). gitleaks pre-commit: `no leaks found`. Only the 6 files were staged, by name.

## Decisions

1. NFC placement (deviation from the brief's literal order). `normalize` applies: CRLF/CR to LF, remove `Cc` except `\n` and `\t`, then NFC, then tab/`Zs` to space, collapse spaces, strip line-trailing spaces, collapse 3+ newlines, strip. The only change is NFC moved from step 1 to just after step 3 (brief steps 2, 3, 1, 4, 5, 6, 7, 8). Every exact case in the brief's `test_normalize_rules` still holds; output differs from the literal order only on inputs where control-char removal exposes a base+mark pair, which is exactly where the literal order is not idempotent.
   Verification (scratch fuzz in WSL, 300,000 seeded random strings over the brief alphabet plus extra marks, `U+2000`, `U+2028`, `U+0085`, `U+200B`, `U+001F`; scratch file deleted afterward): my `normalize` was not idempotent 0 times; it equalled "brief order plus a second NFC after control removal" on all 300,000; the brief's literal order was not idempotent 26,057 times.
   Spec check: the design spec has no normalization rules beyond "canonical text normalization", so there is nothing to override the idempotency requirement.
2. Page span extension. `add(..., page=p)` extends the previous span only if the immediately preceding piece had the same page. A page-less piece (`page=None`) in between closes the open span, so a later piece on the same page starts a new `PageSpan`. Reason: otherwise the span would claim the page-less piece as part of that page. Covered by `test_assembler_page_less_piece_does_not_extend_a_page_span`.
3. Section boundaries. A section ends at the end of its last piece (not at the following separator) and starts at its first piece. With the brief's example the sections are `SectionSpan(0, 22, "A")` and `SectionSpan(24, 29, "A > B")`. The piece that follows `start_section(None)` belongs to no section. A section that never gets a piece (including one replaced by another `start_section` call, or a trailing one) is dropped.
4. Extra tests beyond the brief's list, all for behavior the brief specifies or that I decided: `test_normalize_composes_marks_exposed_by_control_removal` (pins decision 1 with a literal), `test_assembler_extends_open_page_span_for_the_same_page`, `test_assembler_page_less_piece_does_not_extend_a_page_span`, `test_assembler_drops_sections_without_pieces`, and `test_ingest_error_accepts_known_codes` (all 13 codes). `test_unknown_ingest_error_code_rejected` is parametrized over `unknown_code`, empty string, `NO_TEXT`, and `no_text ` (trailing space).
5. `IngestError` stores the code in `args` via `super().__init__(code)`; the `ValueError` for an unknown code does not include the code (logging hygiene: never put caller-supplied text in exception messages).
6. `INGEST_ERROR_CODES` is a public frozenset in `errors.py`, so later tasks can reuse it without duplicating the list.

## Concerns

- Decision 1 deviates from the literal step order in the brief. Behavior for all stated examples is identical, and it is required for the explicit idempotency invariants, but anything downstream (or a future plan text) that assumes "NFC first" should know. Plan B Task 2 onward should call `normalize` only through `backend.ingest.normalize`.
- `PassageDraft.text_sha256` is `bytes` per the brief; Task 9 must keep that in mind when mapping to the bytea column.
