# Plan C Task 3 report: claim extraction

Status: DONE_WITH_CONCERNS (concerns are design notes, nothing blocking)
Commit: `d028bae feat: add atomic claim extraction with span alignment` on `feature/evidencegraph` (not pushed)
Working tree after commit: clean.

## What was built

- `backend/verify/__init__.py`: package docstring.
- `backend/verify/types.py`: `Verdict` (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`), `VERDICT_ORDER`, frozen `Claim(ordinal, text, answer_start, answer_end, method, cited_numbers=())`.
- `backend/verify/claims.py`:
  - constants `MAX_CLAIMS = 50`, `MAX_PER_SENTENCE = 8`, `MIN_CHARS = 3`, `MAX_CHARS = 400`; `CLAIMS_SCHEMA` exactly as in the brief.
  - `STOPWORDS` module-level frozenset of function words only.
  - `claim_sentences(answer)`: sentences from `backend.generate.answer._sentence_spans` (markers blanked to same-length spaces before segmenting, so a marker after a sentence end attaches to the preceding sentence). Cited numbers come from the public `parse_citations(answer, 0)` grouped by `(sentence_start, sentence_end)`. Text has markers removed (with the whitespace before each marker) and whitespace collapsed. Skipped: empty, no content word (`[A-Za-z0-9]+` token not in `STOPWORDS`), `NO_INFO_ANSWER` (case-insensitive, final period ignored).
  - `sentence_claims(answer, *, method="sentence")`: one claim per kept sentence, ordinals from 0, at most 50, text cut at the last space within 400 characters (hard cut at 400 if there is no space).
  - `llm_atomic_claims(llm, answer)`: system prompt `EXTRACT_SYSTEM`, user message is the kept sentences as `"<i>. <text>"` joined by newlines, `temperature=0.0`, `json_schema=CLAIMS_SCHEMA`. Item validation drops non-dict items, non-`int` sentence (bool rejected) or out of range index, non-string text, text outside 3 to 400 characters after whitespace collapse. Keeps at most 8 per sentence and 50 overall in sentence order. Kept sentences with no surviving item get one `sentence` claim. `LLMFailure`, invalid JSON, a non-object, or a reply without a `claims` list returns `sentence_claims(answer, method="sentence_fallback")`.
- `backend/llm/prompts.py`: added `EXTRACT_SYSTEM` (atomic, decontextualized claims, pronouns resolved, one fact each, no hedges or meta-statements, JSON only, sentences treated as untrusted).
- `tests/test_claims.py`: 41 tests.

Every claim's span is its source sentence span in the original answer (markers inside the sentence stay in the span; only claim text drops them). `cited_numbers` is the sentence's citations.

## RED evidence

Tests written first, no production code. Command:

`wsl -d Ubuntu -- bash ".../.local/wsl.sh" timeout 300 python -m pytest tests/test_claims.py -q`

```
ERROR collecting tests/test_claims.py
tests/test_claims.py:8: in <module>
    from backend.llm.prompts import EXTRACT_SYSTEM
E   ImportError: cannot import name 'EXTRACT_SYSTEM' from 'backend.llm.prompts'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.88s
```

Fails because the feature (prompt, `backend.verify` package) does not exist. No LLM, network, or model was reachable from the RED run (all LLM tests use `ScriptedLLM`).

## GREEN evidence

- First GREEN run after implementing: `40 passed in 0.44s`.
- After adding the 400-character boundary case: `tests/test_claims.py` `41 passed in 0.42s`.
- Full suite (`timeout 900`, 143 s wall): `1097 passed, 2 deselected, 2 warnings in 139.90s`. Exit 0. The 2 deselected are the `models` marker (default `addopts`). The 2 warnings are the existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` in `tests/test_generator.py` (legacy `backend/core/generator.py`), unrelated to this task.
- `ruff check .`: `All checks passed!` (exit 0). `ruff format --check` on the 5 changed files: already formatted.
- Pre-commit hook (gitleaks): no leaks found. Files are LF, no BOM, no U+2014.

## Mutation checks (tests catch real breaks)

Each mutation was applied temporarily to `claims.py`, the file restored afterwards (hash compared to a backup: identical).

| Mutation | Result |
|---|---|
| `except (ValueError, RecursionError)` to `except ValueError` | `test_invalid_json_falls_back[deeply-nested]` fails |
| `type(index) is not int` to `not isinstance(index, int)` | `test_llm_atomic_filters_invalid_items` fails |
| Use raw `segment_sentences(answer)` instead of `_sentence_spans` | `test_marker_after_sentence_end_cites_the_preceding_sentence` fails |
| Remove per-sentence cap of 8 | `test_llm_atomic_keeps_at_most_eight_per_sentence` fails |
| `return claims[:MAX_CLAIMS]` to `return claims` | `test_llm_atomic_keeps_at_most_fifty_overall_in_sentence_order` fails |
| Do not deduplicate cited numbers | `test_mid_sentence_markers_removed_and_numbers_deduplicated_in_order` fails |
| Truncation `rfind(" ", 0, MAX_CHARS + 1)` to `MAX_CHARS` | first run passed (gap), added `word-ends-at-400` case, then it fails |
| Drop `0 <=` from the index range check | still passes: equivalent mutant. Negative indexes only create an unused dict key, so they cannot wrap around. The check is kept as explicit validation and the `-1` item stays in the test as a guard for a list-indexed refactor |

## Decisions and deviations

1. `Verdict` is a `StrEnum`, not `(str, Enum)` as written in the brief. The repo already uses `StrEnum` (`backend/core/ensemble.py`) and ruff `UP042` flags `(str, Enum)` on py311. Values and `VERDICT_ORDER` are as specified.
2. `claims.py` imports two private names from `backend/generate/answer.py` (`_MARKER`, `_sentence_spans`) so claim sentences and citation sentences cannot drift. `answer.py` was not modified (outside this task's file list). Follow-up option: expose them under public names.
3. `cited_numbers` are deduplicated in first-appearance order (brief does not say).
4. A kept sentence must also be at least `MIN_CHARS` (3) characters after marker removal, so every `Claim.text` honors the 3 to 400 limit of the global constraints. (A sentence such as `5.` is skipped.)
5. A reply that is a JSON object without a list under `claims` (for example `{}`) is treated as unusable and falls back to `sentence_fallback`. The brief names only invalid JSON, a non-object, and `LLMFailure`. Rationale: `claims` is required by `CLAIMS_SCHEMA`, and the `sentence_fallback` label tells later stages that the LLM did not produce usable output. `{"claims": []}` (valid, empty) gives `sentence` claims, not a fallback.
6. `RecursionError` is caught alongside `ValueError` around `json.loads`, since the reply is untrusted and deeply nested arrays raise it. Covered by a test.
7. No kept sentences returns `[]` without calling the LLM.
8. LLM claim text has whitespace collapsed before the 3 to 400 length check. Citation markers are not stripped from LLM text (the brief does not ask for it).
9. The user message lists all kept sentences even when there are more than 50. The 50 cap is applied to claims afterward (the first 50 sentences always produce at least one claim each, so nothing is lost).
10. `STOPWORDS` contains articles, demonstratives, pronouns and wh-words, auxiliaries and modals, prepositions, conjunctions, and contraction fragments (`s t d ll m re ve`). Deliberately excluded: content verbs, nouns, adjectives, negations (`not`, `no`, `never`), and `may` (also a month). Test asserts `take, days, refunds, shipping, free, office, cafeteria, claim, supported` are not members.
11. Review Focus numbering in the brief was ignored, as instructed; the task's own test list was followed (all 8 named tests are present with those names, plus extras).

## Concerns

- Decision 2 (private imports) is a small coupling to `answer.py` internals. If `_sentence_spans` or `_MARKER` are renamed, `claims.py` breaks at import, which the tests would catch immediately.
- Decision 5 differs from a literal reading of the brief for `{}` and `{"claims": "x"}` replies (method label only: `sentence_fallback` instead of `sentence`).
- `STOPWORDS` membership is a judgment call (for example `up`, `down`, `over`, `out`, `off` are included as prepositions). Later tasks that use it for features may want to adjust it.
