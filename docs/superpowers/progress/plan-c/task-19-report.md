# Plan C Task 19 report: self-consistency diagnostic, legacy pipeline removal

Status: DONE_WITH_CONCERNS (see Concerns)

Commit: `e424e8f refactor: replace the legacy pipeline with the evidence engine` on `feature/evidencegraph` (parent `2ae414f`). Not pushed. 44 files changed, 587 insertions, 2224 deletions. Working tree clean afterwards. The commit hook (gitleaks) ran and passed.

## What was built

`backend/verify/consistency.py`: `consistency_diagnostic(llm, embedder, question, context, claims, *, n_samples, threshold=0.85) -> list[dict]`.
- Draws `n_samples` answers concurrently (`asyncio.gather`) from `build_answer_messages(question, context)` at temperature 0.7, splits each with `sentence_claims`, embeds the claim texts and all sample-claim texts once with `embed_passages` (off the event loop, `asyncio.to_thread`, like the engine), and counts per claim the samples whose best cosine is `>= threshold`.
- Returns one `{"appearances", "n_samples", "failed_samples"}` per claim, in claim order. A sample whose LLM call raises an `LLMFailure` is zero claims and adds one to `failed_samples`. No claims means no LLM call and `[]`.

`backend/analysis/service.py`: after `engine.verify` and inside the same `stages.enter("verifying")` block, for `kind == "ask"` with `options.consistency_samples > 0`, each verdict gets `diagnostics["consistency"]` (via `dataclasses.replace`, the other diagnostics keys stay). No new stage; `stages_done` and `timings` keys are unchanged. Stored by the existing `analyses.save_results` into `claims.diagnostics`. Verdicts, probabilities and features are untouched.

## Removed

Deleted with `git rm` (all by name):
- `backend/core/`: `pipeline.py`, `chunker.py`, `ensemble.py`, `generator.py`, `decomposer.py`, `AGENTS.md`, and the whole `verifiers/` package (`__init__.py`, `base.py`, `consistency.py`, `nli.py`, `similarity.py`, `AGENTS.md`). Kept: `backend/core/__init__.py`, `backend/core/document.py`.
- `backend/db/models.py`, `backend/db/repository.py`.
- `evaluation/benchmarks.py`, `evaluation/ablation.py`, `evaluation/calibration.py`, `evaluation/datasets/.gitkeep`, `evaluation/datasets/demo.jsonl`.
- Tests: `tests/test_chunker.py`, `test_decomposer.py`, `test_ensemble.py`, `test_pipeline.py`, `test_consistency.py`, `test_nli.py`, `test_similarity.py`, `test_generator.py`, `test_db.py`.
- Also removed from code (not whole files):
  - `backend/cli/main.py`: `_build_pipeline` and the `evaluate` command (controller ruling). The `Path` and `Settings` imports are still used by `check`.
  - `backend/models/loader.py`: `get_embedding_model`, `get_nli_model`, `get_settings` (used only by those two), and the `lru_cache`/`TYPE_CHECKING` imports they needed. Plan B/C adapters untouched; the cheap-import test is green.
  - `backend/config.py`: `ollama_model`, `embedding_model_name`, `nli_model_name`, `chunk_size`, `chunk_overlap`, `top_k_chunks`, `consistency_temperature`, `consistency_similarity_threshold`, `nli_weight`, `consistency_weight`, `similarity_weight`, `db_path`, `index_cache_maxsize`. `consistency_samples` kept, default changed from 5 to 0 as the brief says.
  - `pyproject.toml`: `langchain-text-splitters==1.1.3`, `aiosqlite==0.22.1`. Uninstalled from the WSL venv: `uv pip uninstall --python /home/omar_morsi/venvs/evidencegraph/bin/python langchain-text-splitters aiosqlite` printed `- aiosqlite==0.22.1` and `- langchain-text-splitters==1.1.3`. The Windows `.venv` was not touched.
  - `tests/conftest.py`: the legacy fixtures `sample_document_text`, `sample_claims`, `tmp_db_path` (used only by deleted tests).
  - `tests/test_api_startup.py`: `test_startup_opens_no_sqlite_database` (see decisions).
  - Leftover git-ignored bytecode dir `backend/core/verifiers/__pycache__` removed from disk; it would have made `backend.core.verifiers` importable as a namespace package and hidden the removal from `test_legacy_modules_removed`.

AGENTS.md: rewritten `AGENTS.md`, `backend/AGENTS.md`, `evaluation/AGENTS.md`, `tests/AGENTS.md` (new layout, the Docker and `scripts/dev-postgres.sh start` Postgres commands, `python -m backend.cli.main migrate` and `worker`, `python -m pytest -q` needs Postgres, the invariants, the committed-artifacts rule). `backend/core/AGENTS.md` and `backend/core/verifiers/AGENTS.md` removed with `git rm`. `frontend/AGENTS.md` untouched. No em dash in any changed file (scanned).

## Tests written (TDD)

New `tests/test_consistency_diag.py` (13 cases + parametrization): `test_appearances_counted`, `test_one_entry_per_claim_in_claim_order`, `test_failed_sample_counted` (x3: LLMTimeout, LLMUnavailable, LLMError), `test_sample_without_claims_is_not_a_failure`, `test_threshold_decides_what_counts_as_the_same_claim` (includes threshold equal to the cosine still counts), `test_default_threshold_is_0_85`, `test_no_claims_makes_no_llm_calls`, `@pg test_ask_with_consistency_stores_diagnostic`, `@pg test_failed_samples_do_not_fail_the_analysis`, `@pg test_verify_analysis_never_samples`, `test_legacy_modules_removed` (11 deleted module names, plus `backend.core.document` still importable).
Other changed tests: `tests/test_config.py` (`test_default_settings` now pins `consistency_samples == 0`; new `test_legacy_settings_removed`; the weights-sum test is gone with the weights), `tests/test_cli_module.py` (`test_cli_offers_only_the_evidence_engine_commands`: help lists `check`, `migrate`, `worker`, not `evaluate`, and `evaluate --help` exits 2), `tests/test_model_adapters.py` (`test_legacy_loaders_removed`; the `embedding_model_name` assertion removed).

### RED (before any production change)

Command: `python -m pytest tests/test_consistency_diag.py tests/test_config.py tests/test_cli_module.py tests/test_model_adapters.py -q -rf` (log `.local/t19-red.log`). Result: `16 failed, 15 passed, 2 deselected`. Failures, each for the expected reason:
- 9 diagnostic unit tests: `ModuleNotFoundError: No module named 'backend.verify.consistency'`.
- `test_ask_with_consistency_stores_diagnostic`: `assert 2 == 4` (no sample LLM calls were made).
- `test_failed_samples_do_not_fail_the_analysis`: `KeyError: 'consistency'`.
- `test_legacy_modules_removed`: all 11 listed modules still importable.
- `test_default_settings`: `assert 5 == 0`; `test_legacy_settings_removed`: 13 legacy fields present.
- `test_cli_offers_only_the_evidence_engine_commands`: `evaluate  Run evaluation benchmark on a dataset.` in the help.
- `test_legacy_loaders_removed`: `get_embedding_model`, `get_nli_model` present.
`test_verify_analysis_never_samples` passed in RED because it guards absence of behavior; it is covered by mutation M2 below.

### GREEN

`python -m pytest tests/test_consistency_diag.py tests/test_config.py tests/test_cli_module.py tests/test_model_adapters.py tests/test_api_startup.py tests/test_document.py -q`: `40 passed, 2 deselected` (before the later threshold-boundary assertion was added; the file passes in the full run below).

### Mutation checks (each mutation applied, tests run, file restored from a backup copy)

| Mutation | Result |
|---|---|
| M1 `best >= threshold` to `best > threshold` | `test_threshold_decides...` fails |
| M2 verify analyses also sample | `test_verify_analysis_never_samples` fails |
| M3 sample temperature 0.7 to 0.0 | `test_appearances_counted`, `test_ask_with_consistency_stores_diagnostic` fail |
| M4 `failed_samples` always 0 | 3x `test_failed_sample_counted` and `test_failed_samples_do_not_fail_the_analysis` fail |
| M5 sample when `samples >= 0` (i.e. also with 0) | `test_ask_with_consistency_stores_diagnostic` fails |

## Full suite and lint (final tree, before the commit)

- `python -m pytest -q` (WSL, Postgres `eg-pg`, inner timeout 900): `1441 passed, 3 deselected in 264.02s`, exit 0. The 3 deselected are the `models`-marked tests that the default `addopts` skip. Task 17's last full run was 1434 passed; the delta is the removed legacy tests (-) and the new diagnostic, config, CLI and loader tests (+).
- `ruff check .`: `All checks passed!`.
- `ruff format --check` on every Python file I created or edited (except `tests/conftest.py`): `10 files already formatted`. `tests/conftest.py` starts with a blank line that was there before this task; I left it.
- Git: commit made with Windows git from PowerShell, deletions via `git rm`, every other path staged by name; no push, no amend, no hook skipped.

## Decisions

1. `failed_samples` is part of every per-claim dict (`{"appearances", "n_samples", "failed_samples"}`), because the brief says the result list carries it and the list is one dict per claim. `n_samples` stays the requested number, so a failed sample counts in the denominator and adds no appearance.
2. Only `LLMFailure` is absorbed per sample (that is "a failed sample"). Any other exception (embedder bug, for example) propagates and the analysis ends `internal_error` like any other stage failure; I did not swallow it.
3. The diagnostic runs for `kind == "ask"` only, as the brief says, not for `quick`, even though quick analyses also generate an answer. A direct `create_analysis_request(kind="verify", options={"consistency_samples": 2})` (which the API refuses) is ignored by the service; tested.
4. `consistency_samples` stays in `Settings` with default 0 per the brief. Nothing reads it: a request's own `options.consistency_samples` (validated 0..5 in `AnalysisOptionsIn`) decides. See concerns.
5. The `evaluate` CLI test: `tests/test_cli.py` contained no test of `evaluate` or `_build_pipeline` (it is the Postgres-backed `check` suite), so nothing there needed changing. The "evaluate is gone" guard went into `tests/test_cli_module.py` next to the existing `--help` test, which runs without Postgres.
6. Removed `test_startup_opens_no_sqlite_database` from `tests/test_api_startup.py`: it set `HALLUCISCOPE_DB_PATH` to prove startup does not create a SQLite file. `db_path` no longer exists and `aiosqlite` is uninstalled, so the test no longer tests anything (it would pass for any value).
7. `evaluation/AGENTS.md` mentions the RAGTruth test split rule (`--split test --final`) from the Global Constraints as a rule only; the factuality runner is a later task and the file names no module that does not exist yet. The layout and command lists name only existing modules (`run_retrieval`, `data`, `beir`, `metrics`, `report`).
8. The root `AGENTS.md` Docker note says the container is bare Postgres: the app commands also need the `eg_owner`/`eg_app` roles and the `evidencegraph` database that `scripts/dev-postgres.sh` creates (run the same statements); the test suite creates its own roles and databases against `HALLUCISCOPE_TEST_ADMIN_URL`. I verified both against `scripts/dev-postgres.sh` and `tests/pg_fixtures.py`.

## Concerns

- `README.md` still documents the removed `halluciscope evaluate` commands, the ensemble weights and the old architecture. It is not in the brief's file list and Plan D rewrites it (its file list includes `README.md`), so I left it.
- `Settings.consistency_samples` is now an unused setting (kept because the brief says to keep it). Plan D's operations doc test ("every `Settings.model_fields` name appears as `HALLUCISCOPE_<NAME>`") will require it to be documented.
- `.gitignore` still has `evaluation/datasets/*.json`, `*.db`, `*.sqlite` lines that belonged to the removed pipeline. Harmless; not in scope.
- `langchain-core` (and other transitive packages of the removed splitter) may still sit in the WSL venv as orphans; I uninstalled exactly the two named packages.
- `halluciscope.egg-info/` (untracked, ignored) still lists the old `requires.txt`; it is regenerated by the next editable install.
