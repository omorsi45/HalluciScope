# Plan C Task 11 report: calibrated claim verification engine

Status: DONE
Commit: `0461788 feat: add calibrated claim verification engine` (branch `feature/evidencegraph`, not pushed)

## What was built

- `backend/verify/types.py`: added frozen `EvidenceItem` and `ClaimVerdict` exactly as in the brief (imports `PassageHit` from `backend.retrieval.types`; no import cycle).
- `backend/verify/engine.py`:
  - `collect_signals(...)`: `gather_evidence(depth, candidates)`, then every NLI pair of every claim in one `await asyncio.to_thread(nli.score, pairs)` (skipped when there are no pairs). `passage` mode scores `hit.text`; `window` mode scores every `window_spans(hit)` window and keeps the entail window when best entail >= best contra, else the contra window. Unique texts (claims that have hits plus all hit texts) are embedded in one `embed_passages` call in a thread; `dense_cos` is the dot product of the normalized rows. Signals keep hit order, `passage_id=str(hit.passage_id)`, `text=hit.text`, `rerank=hit.scores.get("rerank")`. Zero claims returns `[]`.
  - `VerificationEngine.__init__` runs `calibrator.check_parity(...)` (raises `CalibratorMismatch`). `collect` calls `collect_signals` with the calibrator's `candidates` and `premise_mode`. `verify` builds the features per claim, calls `predict_proba` once on the stacked matrix, `decide`, builds evidence items from `order_signals(signals)[:evidence_k]` with `relation_for`, then localizes spans. Zero claims returns `[]`.
  - Localization: the highest-entail item always, the highest-contra item when its relation is `contradicts` (contra is its largest probability, ties count, same rule as `relation_for`). All windows of all targets across all claims are scored in one extra NLI batch. Best window = highest target probability, ties keep the earliest in `window_spans` order (width 1 first). Other items keep the passage span.
  - `STAGE_SECONDS{stage="nli"}` observed around each scorer call (collection batch and localization batch), `{stage="calibrate"}` around features, `predict_proba` and `decide`. No LLM import or call anywhere in the module.
  - `diagnostics = {"n_candidates": len(hits), "premise_mode": calibrator.premise_mode}`; `probs` keyed by `VERDICT_ORDER` values; `features` keyed by `FEATURE_NAMES`.
- `tests/test_engine.py`: 14 tests (11 `pg`, 3 parametrized pure-Python).

## Files changed

- Modified: `backend/verify/types.py`
- Created: `backend/verify/engine.py`, `tests/test_engine.py`
- Staged by name; `docs/superpowers/`, `.superpowers/`, `.local/` untouched. New files are LF (checked: 0 CR bytes, no U+2014).

## RED evidence

Tests written first. Command:
`wsl ... wsl.sh timeout 300 python -m pytest tests/test_engine.py -q`

```
tests/test_engine.py:11: in <module>
    from backend.verify.engine import VerificationEngine
E   ModuleNotFoundError: No module named 'backend.verify.engine'
ERROR tests/test_engine.py
1 error in 0.90s     (exit=2)
```

Fails because the feature is missing (the module is new), nothing slow or dangerous ran.

Gap found after the first GREEN: my first test set passed even with the window-result rule inverted (the fake scorer's windows tie on the other class, so best entail and best contra were the same window). I added `test_window_result_is_entail_unless_contra_is_higher` (3 literal cases: entail 0.7 vs contra 0.6, entail 0.5 vs contra 0.6, tie 0.5 vs 0.5) and ran it against the inverted rule first:

```
FAILED tests/test_engine.py::test_window_result_is_entail_unless_contra_is_higher[first0-second0-expected0]
FAILED ...[first1-second1-expected1]
FAILED ...[first2-second2-expected2]
3 failed, 11 deselected in 0.98s
```

Then restored the correct rule.

## Mutation checks (temporary edits, reverted)

- Localization tie-break reversed (latest window wins): `test_supporting_span_localized` and `test_contradicting_span_localized` fail (`'Refunds take 30 days. Shipping is free.' != 'Refunds take 30 days.'`), 2 failed, 9 passed.
- Window-result rule inverted: caught only after adding the dedicated test above (see RED).
- Not mutation-tested: other branches covered by named tests (per-claim NLI batching by `test_single_nli_batch_call`, pinned ignored by `test_pinned_hit_used`, parity skipped by `test_parity_mismatch_refuses`).

## GREEN evidence

```
tests/test_engine.py -q   ->  14 passed in 1.26s
```

Brief tests present and passing: `test_three_verdicts`, `test_no_evidence_is_insufficient`, `test_evidence_spans_inside_passage_and_exact`, `test_supporting_span_localized`, `test_pinned_hit_used`, `test_parity_mismatch_refuses`, `test_window_premise_mode`, `test_single_nli_batch_call`. Extra: `test_contradicting_span_localized`, `test_evidence_items_carry_their_signal` (rerank and dense_cos vs HashEmbedder), `test_no_claims_gives_no_results`, `test_window_result_is_entail_unless_contra_is_higher` (x3).

## Full suite and lint

- `timeout 900 python -m pytest -q` -> `1280 passed, 3 deselected, 2 warnings in 194.35s (0:03:14)`, exit 0. The 2 warnings are the existing `tests/test_generator.py` unawaited-coroutine RuntimeWarning in `backend/core/generator.py`, unrelated to this task. The 3 deselected are the default-deselected `models` tests.
- `ruff check .` -> `All checks passed!`
- `ruff format --check .` -> 31 files repo-wide would be reformatted (pre-existing, for example `tests/test_worker.py`); my three files pass `ruff format --check`. I did not reformat other files.
- Commit hook ran gitleaks: no leaks found.

## Decisions

1. Signal `text` is the passage text in both premise modes (the signal is "a retrieved passage scored against the claim"); only `nli` differs in window mode. Feature `number_coverage` and `content_coverage` therefore read the passage text.
2. Embeddings: one batched `embed_passages` call over unique texts (claims that have at least one hit, plus all hit texts), not one call per text. Equivalent for row-independent embedders; the brief's wording is "unique texts embedded once per call".
3. Same item is both the best-entail and best-contra target: the contra window wins (that item's relation is `contradicts` by construction). The brief does not define this case.
4. Localization re-scores windows with its own NLI batch even in window mode, because `collect` returns hits and signals only, not per-window results.
5. `collect_signals` skips the thread hop when no pairs or no texts exist (empty collection).

## Concerns

- `depth` is an engine constructor argument and is not validated against `evidence_k`; a depth below `evidence_k` just yields fewer evidence items.
- The localization extra NLI batch means a `verify` call makes two scorer calls in passage mode (one for signals, one for windows); `collect` alone makes exactly one, as the brief requires. Real DeBERTa cost of the second batch is small (at most 2 items per claim, a few windows each) but not measured.
- Item/ClaimVerdict `relation` is typed `str` per the brief, not a `Literal`.
