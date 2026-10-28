# Plan C Task 4 report: DeBERTa NLI scorer

Status: DONE
Commit: `ea282c9 feat: add DeBERTa NLI scorer with config label mapping` (branch `feature/evidencegraph`, not pushed)

## What was built

- `backend/verify/types.py`: frozen `NLIResult(entail, contra, neutral)` and `NLIScorer` Protocol (`model_id`, `score(pairs) -> list[NLIResult]`).
- `backend/verify/nli.py` (new):
  - `label_indices(id2label)`: int or numeric-string keys, labels matched lowercased, returns `{"entailment", "contradiction", "neutral"}` columns, `ValueError` naming the missing classes.
  - `DebertaNLIScorer(tokenizer, model, *, model_id, device, batch_size=16)`: reads columns from `model.config.id2label` in the constructor (so a non-NLI head fails at load), `model.to(device)`, `model.eval()`, batches of `batch_size`, tokenizer called with `(premises, hypotheses, truncation="only_first", max_length=512, padding=True, return_tensors="pt")`, `torch.inference_mode()`, softmax on `logits.float()`, empty input returns `[]` without touching the tokenizer or model. `torch` is imported inside `score`, so importing the module is free of ML libraries.
- `backend/models/loader.py`: `load_nli(settings) -> DebertaNLIScorer` (lazy `transformers` import inside the function, `resolve_device(settings.model_device)`, `settings.nli_batch_size`). The loader imports `DebertaNLIScorer` at module level, which is safe because `nli.py` has no heavy imports; the existing `test_importing_loader_does_not_import_ml_libraries` still passes.
- `backend/config.py`: `nli_model: str = "cross-encoder/nli-deberta-v3-large"`, `nli_batch_size: int = 16`. The legacy `nli_model_name` stays untouched (used by the legacy pipeline and `get_nli_model`).
- `tests/fakes.py`: `RuleNLIScorer` per ruling R-C2 (contradiction only when the hypothesis has numbers, none occur in the premise, the premise has numbers, and EVERY non-number content token of the hypothesis occurs in the premise; else entailment when every hypothesis content token, numbers included, occurs in the premise; else neutral; chosen 0.9, others 0.05; `model_id = "fake-rule-nli"`).
- `tests/test_nli_scorer.py` (new): 26 default tests plus 1 `models` test.

## Files changed

`backend/verify/nli.py` (new), `backend/verify/types.py`, `backend/models/loader.py`, `backend/config.py`, `tests/fakes.py`, `tests/test_nli_scorer.py` (new). Staged by name; nothing else committed. New files are LF; `loader.py` and `config.py` stay CRLF in the working copy, `types.py` and `fakes.py` stay LF (checked with `git ls-files --eol`).

## RED evidence

Default run before any production code:

```
wsl.sh timeout 300 python -m pytest tests/test_nli_scorer.py -q
E   ModuleNotFoundError: No module named 'backend.verify.nli'
ERROR tests/test_nli_scorer.py
1 error in 1.67s   (exit 2)
```

Models run before any production code (fails at import, so nothing is downloaded):

```
wsl.sh timeout 300 python -m pytest tests/test_nli_scorer.py -q -m models
E   ModuleNotFoundError: No module named 'backend.verify.nli'
1 error in 1.68s   (exit 2)
```

Second RED for the `batch_size` guard (added after the first implementation pass; I removed the guard line, added the test, ran, then restored the guard):

```
FAILED tests/test_nli_scorer.py::test_scorer_rejects_batch_size_below_one[0]   Failed: DID NOT RAISE ValueError
FAILED tests/test_nli_scorer.py::test_scorer_rejects_batch_size_below_one[-1]  Failed: DID NOT RAISE ValueError
2 failed, 24 passed, 1 deselected in 7.51s
```

Mutation checks (temporary edits, reverted, final code re-run green):

- RuleNLIScorer with the old "at least half" rule instead of "every": 3 R-C2 cases fail (`Exchanges take 14 days.`/`Refunds take 30 days.` neutral, the symmetric case, and `Refunds take 14 business days.` neutral).
- Scorer reading `row[0]`, `row[1]`, `row[2]` instead of the config columns: `test_scorer_maps_labels_and_batches[layout0]`, `[layout1]` and `test_load_nli_uses_settings_and_the_loaded_models_labels` fail.

## GREEN evidence

Default, focused:

```
wsl.sh timeout 300 python -m pytest tests/test_nli_scorer.py -q
26 passed, 1 deselected in 5.57s   (exit 0)
```

Models run (real `cross-encoder/nli-deberta-v3-large` on CPU):

```
wsl.sh timeout 1800 python -m pytest tests/test_nli_scorer.py -q -m models
1 passed, 26 deselected, 1 warning in 7.15s   (exit 0)
```

The warning is torch's own `torch.jit.script` FutureWarning on Python 3.14. The checkpoint's config, read with `AutoConfig`: `id2label = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}`, so `label_indices` returns `{'entailment': 1, 'contradiction': 0, 'neutral': 2}`. The real-model test asserts entail > 0.9 for the SNLI pair ("A soccer game with multiple males playing." / "Some men are playing a sport."), contra > 0.9 for ("The store opens at 9 am." / "The store never opens."), probabilities summing to 1, input order preserved when the list is reversed, and `score([]) == []`.

Full suite:

```
wsl.sh timeout 900 python -m pytest -q
1123 passed, 3 deselected, 2 warnings in 144.95s   (exit 0)
```

The 2 warnings are the existing `tests/test_generator.py` "coroutine ... was never awaited" RuntimeWarnings from the legacy generator tests (not from this task). The 3 deselected are the `models` tests (2 existing adapter tests, 1 new).

Lint:

```
wsl.sh timeout 120 ruff check .          -> All checks passed!
wsl.sh timeout 120 ruff format --check <the 6 touched files> -> 6 files already formatted
```

## Tests in `tests/test_nli_scorer.py`

- `label_indices`: from config; string keys and any label case; each of the three labels missing raises `ValueError` naming it.
- Scorer (fake tokenizer and model, no download): labels mapped and 5 pairs at batch size 2 give 3 forward passes (2, 2, 1) in input order, run under both a DeBERTa-like head layout and a layout that differs in every column; premise passed first and hypothesis second; `truncation="only_first"`, `max_length=512`, `padding=True`; model moved to the device, `eval()` called, forward runs under `torch.inference_mode()`; empty input; non-NLI head rejected; `batch_size < 1` rejected; importing `backend.verify.nli` imports no torch/transformers (subprocess).
- `load_nli` wiring with `AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained` monkeypatched: settings model name, batch size, device, and the loaded model's own labels are used.
- `RuleNLIScorer`: the brief's three cases unchanged, plus R-C2 cases: `("Exchanges take 14 days.", "Refunds take 30 days.")` neutral and the reverse, partial number overlap neutral, premise without numbers neutral, extra hypothesis word (`business`) neutral, premise superset entails; input order and empty input.

## Decisions

1. Ruling R-C2 applied instead of the brief's "at least half" text, as instructed.
2. Config field `nli_model` is a new field next to the legacy `nli_model_name` (the brief names `nli_model`; the spec section on configuration lists `NLI_MODEL`). Both default to the same checkpoint; the legacy one is left alone.
3. Added a `batch_size >= 1` guard in `DebertaNLIScorer` (not in the brief): `range(0, n, 0)` raises an unhelpful error and a negative step silently scores nothing. Two tests cover it. Small, easy to drop if unwanted.
4. Added `test_load_nli_uses_settings_and_the_loaded_models_labels` and the import-cleanliness test for `backend.verify.nli`, so the config fields and the lazy-import property are covered by default runs and not only by the 1.7 GB models test.
5. My first draft of the models test also asserted `neutral > 0.5` for ("The store opens at 9 am." / "Dolphins are mammals."). The model returned contradiction 0.999 for it (known NLI behavior for unrelated sentences), so that was a wrong assumption in my test, not a defect in the scorer. The brief does not require a neutral check; I removed that third pair rather than weaken any assertion the brief asks for.
6. Two files (`types.py`, `fakes.py`) were briefly converted to CRLF by the edit tool; I restored them to LF so their working-copy endings match what they were before.

## Concerns

- RuleNLIScorer edge cases follow the ruling literally: a hypothesis with no content tokens (all stopwords) is vacuously entailed, and a hypothesis of only numbers (for example "30") against a premise with other numbers is vacuously "every non-number token present" and so contradicted. Neither case is expected in the planned fixtures; flagging in case later fixtures use such hypotheses.
- The first models run took 25 s end to end (including weight loading) and its captured output shows only the "Loading weights" bar, so I could not tell from the output whether the 1.7 GB checkpoint was downloaded in that run or already cached; the second run took 7 s.
- `load_nli` does not pass a `dtype` to `from_pretrained`; the checkpoint is fp32 and the scorer softmaxes in float32 regardless. On a GPU or MPS device someone may later want half precision, which is out of scope here.
