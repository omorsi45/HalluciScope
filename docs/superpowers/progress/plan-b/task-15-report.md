# Plan B Task 15 report: reciprocal rank fusion and reranking

Status: DONE

Commit: `239e120 feat: add reciprocal rank fusion and reranking` on `feature/evidencegraph` (not pushed).

## What I built

- `backend/retrieval/fusion.py`: `rrf_fuse(rankings, k=60) -> list[tuple[K, float]]`. Score is the sum of `1 / (k + rank)` over rankings (1-based). Ordering is score desc, then best individual rank asc, then key asc. A key repeated inside one ranking raises `ValueError` (constant message, no key in it). Empty input returns `[]`.
- `backend/retrieval/rerank.py`: `async rerank_hits(reranker, query, hits, k) -> list[PassageHit]`. Empty input returns `[]` without touching the model. Otherwise scores via `await asyncio.to_thread(reranker.score, query, [h.text for h in hits])`, sorts by rerank logit desc then previous `rank` asc, keeps the top `k`, and builds each result with `dataclasses.replace(hit, scores={**hit.scores, "rerank": logit}, rank=position)` (position from 1). The scores dict is copied, so the input hit is never mutated. Fewer than `k` hits in gives fewer than `k` out.
- `tests/test_fusion_rerank.py`: 9 tests (the 6 named in the brief plus 3 extra behavior tests, see below).

## Files changed

- Created `backend/retrieval/fusion.py`
- Created `backend/retrieval/rerank.py`
- Created `tests/test_fusion_rerank.py`

No other files touched. New files are LF with no CR bytes and no U+2014 (checked by byte scan). Git reported the expected "LF will be replaced by CRLF" warning only.

## Tests

Brief tests:

- `test_rrf_literals`: `[["a","b","c"], ["b","c","d"]]` gives keys `b, c, a, d` with scores `0.03252247488101534`, `0.03200204813108039`, `0.01639344262295082`, `0.015873015873015872` within 1e-15.
- `test_rrf_tie_break`: `[["b"],["a"]]` gives `a, b`. With `k=1`, `rrf_fuse([["m","a","c","d","e"],["x","a","y","w","m"]])`: `m` and `a` both score exactly `0.6666666666666666` (asserted with `==`), `m` first because of best rank 1 vs 2. I also assert the whole order `m, a, x, c, y, d, w, e`; I derived it by hand: x = 1/2; c and y tie at 1/4 with best rank 3; d and w tie at 1/5 with best rank 4 (both pairs fall back to key order); e = 1/6.
- `test_rrf_duplicate_raises` (also checks that the same key in two different rankings is fine).
- `test_rrf_empty` (`[]` and `[[], []]`).
- `test_rerank_orders_and_renumbers`: `KeywordReranker`, query `red fox jumps`, three hits (previous ranks 1, 2, 3, texts scoring 1, 0, 3). Expected order is hit3, hit1, hit2, ranks `[1,2,3]`, `scores["rerank"] == [3.0, 1.0, 0.0]`, earlier `scores["rrf"]` preserved, and the input hits' `scores` dicts and `rank` values are unchanged afterwards.
- `test_rerank_empty_no_call`: a reranker whose `score` raises `AssertionError` is never called.

Extra tests (each guards a behavior in the dispatch or the brief interface text, no new features):

- `test_rerank_truncates_to_k_and_returns_fewer_for_short_input`: `k=2` returns 2; `k=10` with 3 hits returns 3.
- `test_rerank_equal_scores_keep_previous_rank_order`: two hits tie on the logit; the input list is deliberately not in rank order so only the `rank` field can order them.
- `test_rerank_scores_off_the_event_loop_thread`: the fake records `threading.get_ident()`; it must differ from the loop thread's id (guards against dropping `asyncio.to_thread`).

Mutation check (reasoned, not run): removing the best-rank tie-break fails `test_rrf_tie_break` (`a` would precede `m`); mutating the original `scores` dict in place fails the unchanged-inputs assertions; calling `score` directly on the loop thread fails the thread test; sorting by logit only fails the equal-score test; skipping the empty guard fails `test_rerank_empty_no_call`.

## RED evidence

Command (from the worktree, via the WSL wrapper):

```
wsl -d Ubuntu -- bash ".../.worktrees/evidencegraph/.local/wsl.sh" timeout 300 python -m pytest tests/test_fusion_rerank.py -q
```

Output (before the two modules existed):

```
ERROR collecting tests/test_fusion_rerank.py
tests/test_fusion_rerank.py:10: in <module>
    from backend.retrieval.fusion import rrf_fuse
E   ModuleNotFoundError: No module named 'backend.retrieval.fusion'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.63s
exit=2
```

This is the expected failure: the feature modules are missing. It fails fast; nothing expensive runs. (A missing module is a collection error rather than per-test failures; I did not add stub production code just to turn it into assertion failures.)

## GREEN evidence

Same command after implementing both modules:

```
.........                                                                [100%]
9 passed in 0.31s
exit=0
```

## Full suite

`wsl.sh timeout 900 python -m pytest -q`:

```
887 passed, 2 deselected, 2 warnings in 133.00s (0:02:12)
```

No failures. The 2 deselected are the default-deselected marker tests (config deselects them; not from this task). The 2 warnings are pre-existing `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` from `tests/test_generator.py` (`test_generate_answer`, `test_generate_answer_with_temperature`, via `backend/core/generator.py:50`); untouched by this task.

Note: the exit code I echoed after the full-suite command was the PowerShell pipeline's (I piped through `Select-Object -Last`), so the evidence for the full suite is the pytest summary line above, not that exit code.

## Lint

- `ruff check .` -> `All checks passed!` (exit 0)
- `ruff format --check` on the three new files -> `3 files already formatted` (exit 0)

## Commit

- `239e120 feat: add reciprocal rank fusion and reranking` (3 files, 222 insertions). Staged by name. The pre-commit gitleaks hook ran and reported no leaks. No push, no amend, hooks not bypassed.

## Decisions

- `zip(hits, logits, strict=True)` in `rerank_hits`: if a model returns a different number of scores than passages, it raises `ValueError` instead of silently dropping hits. This adds no code beyond the keyword, and the Reranker protocol says one logit per passage.
- The `ValueError` in `rrf_fuse` uses a constant message that does not include the key (logging hygiene rule: no content from exception messages).
- `k` is not validated in either function (the brief specifies no validation). `k <= 0` in `rerank_hits` follows normal slice behavior; callers are expected to pass a positive `k`.
- Default `k=60` is a literal in the signature, as in the brief.

## Concerns

None blocking. One note for later tasks: `rerank_hits` returns raw logits in `scores["rerank"]`, so any caller that wants a comparable 0..1 number must normalize itself.
