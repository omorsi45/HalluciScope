# Plan C Task 13 report: run_analysis job

Status: DONE. The implementer session was interrupted after GREEN on focused tests; the controller ran the full suite (1349 passed, 3 deselected, ruff clean) and committed 05433e5 `feat: add analysis job for ask and verify workflows` with the files listed below.

## Review of the cancelled attempt's uncommitted tests

- tests/test_run_analysis.py (31 test cases from 24 functions): matches the brief's 11 named tests plus extras (calibrator mismatch, retrieval_failed, internal_error log hygiene, sentence extraction, request storage, collection kinds, failure hook x4). No assertion contradicts the brief. Changes: 3 lines over 100 characters wrapped, `ruff format` applied. No weakening.
- tests/pg_fixtures.py: `services` fixture with `llm=ScriptedLLM()`, `nli=RuleNLIScorer()`, `calibrator=make_test_calibrator()`. Correct as is.
- tests/test_worker.py harness: fakes for `load_nli`, `OllamaClient`, `timeouts_s`, calibrator path (set to a missing file by default). Two defects fixed:
  1. **Hang in RED.** `FakeWorker.run` waited on `stop.wait()` forever. The CLI tests (`CliRunner().invoke(app, ["worker"])`) never send SIGINT, so with no implementation the worker started and the run hung until the outer kill (this is what stalled the previous attempt). Now `asyncio.wait_for(stop.wait(), LIMIT_S)`: an unstopped worker fails after 10 s.
  2. **Wrong expectation.** The rejected-artifact test asserted that `backend.cli.main` logged exactly one message, but `_run_worker` also logs "worker metrics disabled" before the models load (the existing test pins that). The test now filters to messages starting with "calibrator artifact". It also asserts `isinstance(result.exception, SystemExit)` so an unexpected crash (exit code 1 as well) cannot satisfy `exit_code == 1`.

## RED

Commands run through `.local/run.py` (Windows-level killer) and `wsl.sh` with an inner `timeout`.

`tests/test_run_analysis.py`: collection error, feature missing:

```
tests/test_run_analysis.py:14: in <module>
    from backend.analysis.service import STAGES, create_analysis_request
E   ImportError: cannot import name 'STAGES' from 'backend.analysis.service'
1 error in 1.00s
```

`tests/test_worker.py -k "registries or import_paths or run_worker or calibrator"`, before the harness fix: exit=124 (killed after 300 s). Hang located with `-v` and a 90 s killer: `test_worker_exits_non_zero_when_the_calibrator_artifact_is_rejected[malformed-temperature]` (worker started, nothing stops it). After the harness fix, 33 s:

```
8 failed, 8 passed, 44 deselected in 33.37s
E  assert {'ingest_upload', 'search'} == {'ingest_uplo...is', 'search'}          (registries)
E  CalledProcessError ... import backend.analysis.service ...                    (import paths)
E  ValidationError: analysis_timeout_s  Extra inputs are not permitted           (timeout test)
E  AttributeError: 'Services' object has no attribute 'calibrator'               (services wiring)
E  assert isinstance(TimeoutError(), SystemExit)                                 (3 rejected-artifact cases)
```

## Implementation

Files (all staged by name for the commit):

- `backend/analysis/service.py` (extended; `CitationRecord` kept): `STAGES`, `create_analysis_request`, `run_analysis`, `resolve_citations`, helpers.
- `backend/analysis/handlers.py` (new): `handle_run_analysis`, `fail_run_analysis`.
- `backend/jobs/registry.py`: `run_analysis` registered in both registries.
- `backend/services.py`: `llm`, `nli`, `calibrator` fields (None defaults), `retriever()`.
- `backend/config.py`: `analysis_timeout_s = 300.0`, `max_active_analyses = 3`.
- `backend/cli/main.py`: `_load_calibrator`, `_reject_calibrator`, `_run_worker` wiring.
- Tests: `tests/test_run_analysis.py` (new), `tests/test_worker.py`, `tests/pg_fixtures.py`.

## Decisions

1. `known_at` is the database clock (`select now()`), as in `POST /search`, because it is compared to `created_at` which the database stamps.
2. Retrieval failures: a small `_GuardedRetriever` wraps the retriever for both the ask retrieval and the engine's claim evidence gathering, so either path reports `retrieval_failed` and no exception message travels. NLI errors stay `internal_error`.
3. `VerificationEngine` is built before the first stage, so a calibrator trained on other models fails as `calibrator_mismatch` before any LLM call.
4. Progress: after `save_results`, `set_progress(conn, id, "done")` runs in the same transaction, which moves `persisting` into `stages_done` (the brief requires `stages_done == list(STAGES)`). `stage` is then `"done"`. A verify analysis shows `["extracting", "verifying", "persisting"]`.
5. `timings["persisting"]` covers the stage commit and building the records. The insert itself is stamped with these timings, so it cannot be included.
6. Worker: the calibrator file is read and validated before the models load (a bad artifact should not cost a DeBERTa load); parity (`check_parity`) runs after the three models are loaded. Rejection logs `calibrator artifact rejected: <field>` (field from `CalibratorError`'s `<field>:` prefix, `exc.field` for a mismatch, `json` for `JSONDecodeError`) and raises `typer.Exit(1)`. A missing file logs nothing: the existing no-token test pins the `backend.cli.main` log to exactly one message. The httpx client is closed in the final `finally`, before the pool.
7. The failure hook uses direct SQL with `status in ('queued', 'running')` (like the ingest and search hooks) because `set_status` has no terminal guard.

## GREEN

```
tests/test_worker.py -k "registries or import_paths or run_worker or calibrator":  16 passed, 44 deselected in 3.56s
tests/test_run_analysis.py:                                                        31 passed in 4.98s
ruff check .:                                                                      All checks passed!
```

Mutation checks (temporary, reverted): making the live analysis use `known_at` failed `test_as_of_analysis_sees_only_what_was_known_when_it_was_requested`; dropping the status guard of the hook failed 3 hook tests.
