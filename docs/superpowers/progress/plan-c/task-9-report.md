# Plan C Task 9 report: calibrator artifact

Status: DONE
Commit: `cb56d07 feat: add versioned calibrator artifact with parity checks` (branch `feature/evidencegraph`, not pushed)

## What I built

- `backend/calibration/artifact.py` (new): `ARTIFACT_SCHEMA_VERSION = 1`, `CalibratorError(ValueError)`, `CalibratorMismatch(RuntimeError)` (attributes `field`, `expected`, `actual`), and the frozen `Calibrator` dataclass with `predict_proba`, `decide`, `to_dict`, `from_dict`, `load`, `save`, `artifact_id`, `check_parity`, exactly as in the brief.
- `tests/fakes.py`: `make_test_calibrator(*, tau_s=0.5, tau_c=0.5, evidence_k=4, models=None)` with the brief's values (12 features, mean 0, std 1, `W[0, max_entail] = 8`, `W[1, max_contra] = 8`, `b = [0, 0, 3]`, T = 1, candidates 20, `premise_mode = "passage"`, alpha_s 0.10, alpha_c 0.05, delta 0.05). Default model ids come from `RuleNLIScorer.model_id`, `HashEmbedder.model_id`, `KeywordReranker.model_id` (`fake-rule-nli`, `fake-hash-384`, `fake-keyword`), so they cannot drift from the fakes.
- `backend/config.py`: `calibrator_path: str = "backend/calibration/artifacts/default.json"`.
- `tests/test_calibrator_artifact.py` (new, 72 test cases). `default.json` was not created.

Files in the commit: `backend/calibration/artifact.py`, `backend/config.py`, `tests/fakes.py`, `tests/test_calibrator_artifact.py` (staged by name).

## Tests

The six brief tests plus extras: canonical-JSON file format, `artifact_id` formula and sensitivity, `to_dict` independence, `predict_proba` shape guard, `tau_c = None`, inclusive thresholds, supported-over-contradicted priority, fake defaults and overrides, fake/parity agreement with the real fake classes, a parametrized invalid-artifact table (34 bad-value rows plus 18 missing-key rows, one per field), non-object artifact, invalid artifact via `load`, and parity order.

Expected values are hand-derived: the subset test uses non-trivial mean/std/temperature and a feature order opposite to `FEATURE_NAMES` plus junk in the unused columns, with logits worked out by hand (row 0 logits `[2.8, 0, 0.5]`, row 1 `[0, 0.5, 0.5]`, softmax computed with `math.exp` in the test).

### RED

Command: `wsl.sh timeout 300 python -m pytest tests/test_calibrator_artifact.py -q -p no:cacheprovider`

```
ERROR collecting tests/test_calibrator_artifact.py
tests/test_calibrator_artifact.py:11: in <module>
    from backend.calibration.artifact import (
E   ModuleNotFoundError: No module named 'backend.calibration.artifact'
1 error in 0.81s
```

Fails for the right reason (module missing). It ran in under a second, nothing expensive is involved.

### GREEN

```
72 passed in 0.41s
```

### Mutation check (done after GREEN, file restored afterwards; re-run showed 72 passed)

Each realistic mutation of `artifact.py` made at least one test fail: positional instead of by-name column selection, `>` instead of `>=` for tau_s, contradicted checked before supported, temperature ignored, embedder dropped from parity, parity order reversed, `sort_keys` removed from `artifact_id`, `std > 0` check removed, `tau_s` left unvalidated.

## Full suite and lint

- Full suite (`timeout 900 python -m pytest -q`): `1247 passed, 3 deselected, 2 warnings in 145.32s`. The 3 deselected are the `models` marker (default `addopts`). The 2 warnings are a pre-existing `RuntimeWarning` in `tests/test_generator.py` (`coroutine ... was never awaited` from `backend/core/generator.py`), unrelated to this task.
- `ruff check .`: `All checks passed!`
- `ruff format --check` on the 4 changed files: `4 files already formatted` (I ran `ruff format` on `artifact.py` and `fakes.py` once, which touched only my additions).
- Commit hook: gitleaks ran, no leaks found.
- No em dashes in any changed file. `git ls-files --eol`: `tests/fakes.py` stays `i/lf w/lf`, `backend/config.py` stays CRLF in the working copy (not converted); new files are LF.

## Decisions (brief silent or ambiguous)

1. Message format: every `CalibratorError` message starts with `<field>:` (for example `W: expected shape (3, 12), got (2, 12)`), and tests assert that prefix. A non-object artifact uses the name `artifact`.
2. Beyond the brief's list, `from_dict` also validates: `std` strictly positive (a zero would make predictions inf/nan), `mean`/`std`/`b` shapes, `alpha_s`/`alpha_c`/`delta` finite and in `[0, 1]`, `candidates` an integer >= 1, `models` an object whose `nli`/`embedder`/`reranker` are non-empty strings (extra keys are kept so the round trip is stable), `metadata` an object. Every field is required: a missing key, including `tau_s`/`tau_c`, is an error rather than a silent null (a truncated file must not turn into "never supports"). Booleans are rejected where a number or integer is expected.
3. `predict_proba` requires a 2-D array of width 12 and raises `ValueError` otherwise (a single 1-D vector is rejected). A `(0, 12)` input returns `(0, 3)`.
4. `make_test_calibrator(models=...)` merges over the default ids rather than replacing them, so a partial override like `{"nli": "other"}` stays a valid artifact.
5. `save` writes LF line ends explicitly (`newline="\n"`) so a committed `default.json` is identical on Windows and Linux.
6. `load` lets `FileNotFoundError` propagate for a missing file (the worker maps that to `calibrator_missing` in a later task). It does not wrap invalid JSON; `json.JSONDecodeError` is a `ValueError` but not a `CalibratorError`.
7. No test for the `calibrator_path` default: it is a bare constant with no behavior until the worker task reads it.

## Concerns

- `Calibrator` is `@dataclass(frozen=True)` as specified, so `==` on two instances compares ndarray fields and raises an ambiguity error; compare `to_dict()` outputs (as the tests do). Later tasks should do the same.
- Decision 6: if the worker wants one exception type for "artifact unusable", it should catch `CalibratorError` and `json.JSONDecodeError` (or Task 11 can ask for load to wrap it).
