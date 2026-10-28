# Task 26 report: end-to-end latency benchmark

Status: DONE_WITH_CONCERNS (concerns are notes for Task 27, none block the commit).

Commit: `eab71d9 feat: add end-to-end latency benchmark` on `feature/evidencegraph` (not pushed). Staged by name: `evaluation/run_latency.py`, `evaluation/report.py`, `tests/test_latency.py`. The pre-commit gitleaks scan found no leaks.

## What I built

- `evaluation/run_latency.py` (new): `python -m evaluation.run_latency [--requests 50] [--database-url URL] [--fake] [--out evaluation/results/latency.json]`.
  - `build_questions(families, n)`: `"What is the <fact phrase> for <entity>?"`, one per fact stated in version 1, going round the families (first fact of each family, then the second of each, facts in key order), starting over when `n` exceeds the pool. The fact phrase is the fact key with underscores replaced by spaces (`return_window` -> `return window`); `FactSpec` has no other name field.
  - `ingest_corpus(services, families, name)`: a new bench tenant (`_make_tenant` from `run_temporal`, so same user and system principal setup) and version 1 of the 10 families uploaded through `_upload` (`intake_upload` + `process_upload`, explicit `effective_from`). No distractors or foreign copies: the brief says version 1 of 10 families.
  - `measure(services, tenant, questions)`: per question `create_analysis_request(kind="ask", options={}, enqueue=False, collection_kinds=("benchmark",))`, then `run_analysis` in-process, sequentially. A `PermanentJobError` is caught only to learn the code; the analysis row is then read back and must carry that same code (else RuntimeError). Returns the finished `analyses` rows.
  - `summarize(rows)`: per stage (`retrieving, generating, extracting, verifying, persisting`) and `total`, `p50_s`, `p95_s`, `mean_s` over the rows with status `succeeded`; everything else counted under `error_code` (or `status` for a non-final row, which would be a bug). No success means `stages == {}`. Percentiles come from `run_retrieval.percentile` (nearest rank, `ceil(p*n/100)`, 1-based, already tested there), reused rather than duplicated.
  - `fake_reply(messages)`: the `ScriptedLLM.default` of `--fake`. Claim extraction (system message equals `EXTRACT_SYSTEM`) gets `{"claims": []}` (valid JSON, each sentence stands as its own claim); an answer is the first sentence of the first context passage plus ` [P1]`.
  - `run(args)`: `--fake` uses `load_models("fake","fake")` (HashEmbedder, KeywordReranker, RuleNLIScorer, per R-C2), a `ScriptedLLM` with `fake_reply`, and `make_test_calibrator()`. Real uses `load_models("real","real")`, `OllamaClient(settings.llm_model = llama3.2:3b)`, and `Calibrator.load(settings.calibrator_path)`. Output is exactly `{"n", "stages", "models", "env", "failures"}`: `n` = requests issued, `models` = `{llm, nli, embedder, reranker}`, `env` = `collect_env(db)`.
- `evaluation/report.py`: `render_latency(results)` (intro, environment with commit/CPU/RAM/Python/torch/PostgreSQL/pgvector/models, `Requests: N, failed: K (code count, ...)`, table `| Stage | p50 s | p95 s | mean s |` in the order of the results file, 3 decimals; a "no request succeeded" line instead of the table when `stages` is empty; an "excluded" note only when there are failures) and the `latency` subcommand (`python -m evaluation.report latency <results.json> --out docs/results/latency.md`, LF output). Module docstring updated.
- `tests/test_latency.py` (new, 17 tests).

## RED

Tests written first. Command (worktree root, through the Windows-level killer):

```
.venv\Scripts\python.exe .local\run.py 340 .local\t26-red.log -- wsl -d Ubuntu -- bash ".../.local/wsl.sh" timeout 280 python -m pytest tests/test_latency.py -q
```

```
tests/test_latency.py:15: in <module>
    from evaluation import report, run_latency
E   ImportError: cannot import name 'run_latency' from 'evaluation'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.16s
exit=2
```

The module did not exist, so the whole file failed at collection (feature missing, not a typo). No Ollama or model download was possible: the RED run imported nothing but the missing module.

## GREEN

```
tests/test_latency.py -q  ->  17 passed in 10.88s   exit=0
```

Includes both Postgres tests (not skipped): `test_latency_smoke_fake` (`--fake --requests 3`: `n == 3`, `failures == {}`, every stage and `total` present with `p50_s <= p95_s`, model ids `fake-rule-nli / fake-hash-384 / fake-keyword / fake-llm`, exactly the five brief keys, file equals the returned result) and `test_a_failed_request_is_counted_and_left_out_of_the_percentiles` (first answer raises `LLMUnavailable`: rows are `failed, succeeded, succeeded`, `failures == {"llm_unavailable": 1}`, the successful answers have `n_citations >= 1`, `n_invalid_citations == 0`, `n_claims >= 1`, and `total.p95_s` equals the max of the two successful totals).

Mutation check of the exclusion rule: temporarily changing `summarize` to select rows by `row["timings"]` instead of `status == "succeeded"` made `test_summarize_counts_failures_by_error_code_and_leaves_them_out_of_the_stages` fail (`.F...` for the `summarize or failed_request` selection); the file was restored byte for byte (the grep of the line afterwards shows the original) before the gates below.

## Full suite and lint (after the final edit, before the commit)

- `python -m pytest -q` (inner 900, outer 960): `1654 passed, 3 deselected in 301.72s`, `exit=0`. The 3 deselected are the `models`-marked tests that the project's `addopts` excludes.
- `ruff check .`: `All checks passed!`
- `ruff format --check evaluation/run_latency.py evaluation/report.py tests/test_latency.py`: `3 files already formatted` (a first `--diff` showed two line-joins in my own code, applied by hand).
- `python -m evaluation.run_latency --help` works; the real benchmark was not run.
- Line endings: the three files are LF in the working copy (`git ls-files --eol` shows `w/lf` for the neighbours). No em dash in any `.py` file of the repo.

## Decisions

1. `n` is the number of requests issued; failures are separate and not subtracted, and the successful count is `n` minus the failure total. The brief's output shape has no other count, so I added none (no timestamps, calibrator id or success count either).
2. No `--artifact` flag (the brief's interface has none): a real run loads `Settings().calibrator_path`, and `Calibrator.load` raises early if the file is missing instead of producing 50 `calibrator_missing` failures. `--fake` always uses `make_test_calibrator()`.
3. Default analysis options (`options={}`), so claim extraction goes through the LLM (`llm_atomic`), which is what the `extracting` stage measures in the worker. Default `analysis_timeout_s` (300 s) applies, unlike `run_temporal` which raises it, so a request slower than that is a counted `analysis_timeout` failure, as it would be for a user.
4. Exit code is always 0 after the file is written (the brief defines no failure exit). A run where every request failed writes `stages: {}` and the failure counts, and the report says "No request succeeded".
5. Reused private helpers of `run_temporal` (`_make_tenant`, `_upload`, `_Tenant`, `_log`, `_positive_int`) by import instead of duplicating or renaming them; ruff has no rule against it here, and renaming would have touched an earlier task's file.
6. `fake_reply` lives in `run_latency.py` (the `--fake` flag needs it at run time, like `load_models` imports `tests.fakes` lazily); `tests/fakes.py` is unchanged.

## Concerns / notes for Task 27

- No warm-up request: the first real request includes Ollama's model load and the first NLI/embedder forward passes. It cannot move p50 or p95 at 50 requests (rank 25 and 48), but it does lift `mean_s` and the maximum. If a steadier mean is wanted, a discarded warm-up request would be a deliberate change to this runner.
- Questions ask for one fact each over only 10 documents with no distractors, so retrieval is easy and the context is small; it measures the pipeline's cost, not retrieval at scale.
- The real calibrator `backend/calibration/artifacts/default.json` must exist and match the real NLI/embedder/reranker ids, otherwise every request fails (`calibrator_missing` or `calibrator_mismatch`); that would show in `failures`, not as a crash.
- Bench tenants are created per run (`latency-<8 hex>`) and not removed, like the temporal bench; they accumulate in `evidencegraph_bench`.
