# Task 2 report: read NLI label order from model config

Status: DONE. Commit: 80f4a98 `fix: read NLI label order from model config`.

## What was built
- `NLIVerifier.__init__` reads `model.config.id2label` (int or str keys, names lowercased) into `self.label_index` with keys contradiction, entailment, neutral. Raises `ValueError("NLI model labels must include contradiction, entailment, neutral")` otherwise.
- `_compute_hallucination_score(self, contradiction, neutral, entailment)` returns `contradiction + 0.5 * neutral`.
- `verify` indexes probabilities via `label_index`; tokenizer uses `truncation="only_first"`. Details keys and min-over-chunks selection unchanged.
- AGENTS.md NLI bullet now says labels are read from `id2label`.

## Files changed
`backend/core/verifiers/nli.py`, `backend/core/verifiers/AGENTS.md`, `tests/test_nli.py`.

## RED
`.venv/bin/python -m pytest tests/test_nli.py -q`: 6 failed. Reasons: score mapping TypeError (old signature), entailment test saw `entailment: 0.995` on neutral logits at index 2 (old order assumption), `DID NOT RAISE ValueError`, `truncation` was `True` not `'only_first'`, count test failed (strong logit at index 1 read as neutral).

## GREEN
`tests/test_nli.py`: 6 passed. Full suite: 52 passed, 2 warnings (pre-existing unawaited coroutine warning in test_generator). 

## Lint
`ruff check .` reports 46 errors repo-wide (pre-existing). For the two files I touched, 2 errors remain (I001 import sorting); the same 2 exist at HEAD, so unchanged. Left alone as out of scope.

## Decisions / notes
- Grep confirmed no callers of `_compute_hallucination_score` outside `nli.py` and `tests/test_nli.py`. Other NLIVerifier constructors (cli/main.py, api/startup.py, evaluation/ablation.py) pass real models; no other test builds one with a MagicMock model (full suite green).
- `numpy` import retained in nli.py (still used for argmin and array typing).
- No concerns.
