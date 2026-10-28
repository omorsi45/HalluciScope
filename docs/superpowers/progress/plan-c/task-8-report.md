# Plan C Task 8 report: evidence signals and the feature vector

Status: DONE

Commit: `7306d32 feat: add claim evidence feature extraction` (branch `feature/evidencegraph`, not pushed)

## What was built

- `backend/verify/types.py`: added frozen `EvidenceSignal(passage_id, text, nli, rerank, dense_cos)`.
- `backend/verify/features.py`:
  - `FEATURE_NAMES` (12 names, brief order), `NUMBER_RE`, `order_signals`, `compute_features`.
  - `order_signals`: stable sort by rerank descending, `None` last, ties keep input order.
  - `compute_features`: float64 vector of length 12 from `order_signals(signals)[:evidence_k]`, with the
    no-signal vector exactly as the brief defines it.
  - Number tokens: `NUMBER_RE` matches with `$` and `,` removed, compared as strings (`$1,200` matches
    `1200`; `5%` matches `5%` but not `5`).
  - Word tokens: lowercase `\w+`; content tokens exclude `backend.verify.claims.STOPWORDS`.
- `tests/test_features_evidence.py`: 9 tests (the 7 from the brief plus 2 small extras, listed below).

Files in the commit: `backend/verify/features.py`, `backend/verify/types.py`, `tests/test_features_evidence.py`
(3 files, 258 insertions). Staged by name only.

## RED evidence

Tests written first, before `features.py` or `EvidenceSignal` existed:

```
wsl.sh timeout 300 python -m pytest tests/test_features_evidence.py -q
tests/test_features_evidence.py:5: in <module>
    from backend.verify.features import FEATURE_NAMES, compute_features, order_signals
E   ModuleNotFoundError: No module named 'backend.verify.features'
ERROR tests/test_features_evidence.py
1 error in 0.79s
```

The failure is the missing feature module, which is the expected reason. It was a collection error, so the
individual tests did not run in the RED step; they are covered by the GREEN run below.

## GREEN evidence

```
wsl.sh timeout 300 python -m pytest tests/test_features_evidence.py -v
test_feature_names_length_12 PASSED
test_feature_vector_literal PASSED
test_evidence_k_truncates_by_rerank PASSED
test_number_mismatch_coverage PASSED
test_money_and_percent_numbers PASSED
test_no_signals_vector PASSED
test_no_signals_claim_without_numbers_or_content_tokens PASSED
test_none_rerank_ordered_last PASSED
test_top_rerank_zero_when_no_signal_has_a_rerank_score PASSED
9 passed in 0.34s
```

`test_feature_vector_literal` asserts the brief's literal vector within 1e-12 (and dtype float64, shape (12,)).

## Full suite and lint

- `wsl.sh timeout 900 python -m pytest -q`: `1175 passed, 3 deselected, 2 warnings in 148.84s`, exit 0.
  The 2 warnings are `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` in
  `tests/test_generator.py::test_generate_answer` and `::test_generate_answer_with_temperature`
  (`backend/core/generator.py:50`). They are in code this task did not touch.
- `wsl.sh timeout 120 ruff check .`: `All checks passed!`
- No em dashes in the three changed files (grep count 0). `types.py` was LF in index and working copy
  (`i/lf w/lf`), so the edit preserved its line endings.
- The pre-commit gitleaks hook ran: `no leaks found`. Hooks were not bypassed.

## Decisions (the brief left these open)

1. Coverage fractions use sets, so a repeated claim number or content token counts once. The brief says
   "fraction of claim numbers/content tokens" and does not say whether duplicates count; sets are the
   conventional reading and the literal test cannot distinguish them.
2. `claim_len` counts all `\w+` tokens of the claim (stopwords and repeats included), as the brief says
   "number of \w+ tokens"; only `content_coverage` filters stopwords.
3. `top_rerank` uses a two-branch sigmoid. For `x >= 0` it is the same expression as `1 / (1 + exp(-x))`,
   so the literal `0.8807970779778823` is unchanged; for very negative scores (below about -709) the
   one-branch form would raise `OverflowError`, the two-branch form returns a value near 0.
4. Claim tokens are lowercased once and compared against lowercased signal text, so coverage is
   case-insensitive for words. Numbers are compared as raw digit strings (no lowercasing needed).
5. Two tests beyond the brief's list: `test_no_signals_claim_without_numbers_or_content_tokens` (the
   `1.0 if no numbers` and `0.0 if the claim has content tokens else 1.0` branches of the empty vector) and
   `test_top_rerank_zero_when_no_signal_has_a_rerank_score`. `test_money_and_percent_numbers` also asserts that
   `5%` does not match a bare `5`, which pins the normalization rule.

## Concerns

- None blocking. Worth knowing for the calibrator tasks: because `\w+` matches digits, numbers also appear in
  the content tokens (the claim `Refunds take 14 days` has content coverage 3/4 against s1 while number
  coverage is 0.0). That follows the brief's definitions literally.
- `evidence_k <= 0` is not guarded (`[:evidence_k]` with a negative value would slice from the end). The brief
  does not define it and callers use the config default of 4; `0` correctly yields the no-signal vector.
