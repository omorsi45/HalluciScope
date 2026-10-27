# Task 9 report: structured logging, metrics registry, metrics server

Status: DONE. Commit: f608866 `feat: add structured logging and metrics registry`.

## Built
- `backend/obs/logging.py`: context vars, `bind`, `ALLOWED_EXTRA`, `JsonFormatter`, `configure_logging`.
- `backend/obs/metrics.py`: `Counter`, `Gauge`, `Histogram`, `Registry`, `DEFAULT_BUCKETS`, `REGISTRY` and the seven named metrics (created at import).
- `backend/obs/metrics_server.py`: `start_metrics_server` (ThreadingHTTPServer, daemon thread, silenced `log_message`).
- Tests: `tests/test_obs_logging.py` (6), `tests/test_obs_metrics.py` (15). Package `backend/obs/__init__.py` is empty.

## RED
`pytest tests/test_obs_logging.py tests/test_obs_metrics.py -q` before implementation: 2 collection errors
(`ModuleNotFoundError: No module named 'backend.obs'`, then `ImportError: cannot import name 'metrics'`
after creating the empty package). Feature missing, not typos.

## GREEN
Same command: `21 passed`. Also `-W error` on the two files: 21 passed (an earlier run caught my own test
helper leaking `HTTPError` objects as ResourceWarning; fixed by closing them).

## Full suite / lint
`pytest -q`: 156 passed, 2 warnings (pre-existing `tests/test_generator.py` unawaited-coroutine warnings,
not from this task). `ruff check .`: all checks passed. `ruff format` applied to the new files.
Pre-commit gitleaks hook ran clean.

## Decisions
- Auth followed the brief (the dispatch note said only "other paths 404"): token None -> 404 for everything;
  non-/metrics -> 404; missing or wrong bearer -> 401 with `WWW-Authenticate: Bearer`; compare via
  `hmac.compare_digest` on UTF-8 bytes of the full `Bearer <token>` header.
- Unlabeled metrics render a `0.0` sample before any observation (HELP and TYPE always present); labeled
  metrics with no children render only HELP and TYPE.
- Output format: values as Python float repr (`2.0`), `le` as `repr(float(bound))` (`0.005`, `1.0`, `300.0`), `+Inf`.
  Samples within a metric are sorted by label values for deterministic output.
- Exception frames use `basename(file):line in func` via `traceback.extract_tb`; no message or `exc_text`.
- One `threading.Lock` per metric, shared with its children. Counter rejects negative increments with
  `ValueError`. `Registry.register` rejects duplicate names and returns the metric. Histogram buckets must be
  finite and strictly increasing. These were small safety checks beyond the brief text.
- `configure_logging` replaces root and `uvicorn` handlers with one JSON stream handler (stderr), sets
  `uvicorn.propagate = False`, disables `uvicorn.access`. Not wired into any app yet (out of scope).
- Module docstring documents that log messages must be constant strings.

## Concerns
- `configure_logging` replaces existing root handlers, which would drop pytest's caplog handler if called
  inside a test; the test restores state. Callers should invoke it once at startup.


## Fix round 1

Findings: task-9-findings-r1.md (blank token, exception text via non-string msg, plus ruled NaN and arbitrary-object items).

### Changes
- `backend/obs/metrics_server.py`: `expected` is built only for a truthy token (`if token`), so `""` is unconfigured and every request gets 404.
- `backend/obs/logging.py`: `msg` is `getMessage()` only for `str` messages, else `type(record.msg).__name__`. New `_safe_value`: whitelisted values that are str/int/bool/None pass, non-finite floats become `None`, anything else becomes its type name. `json.dumps` now uses `allow_nan=False` and no `default=str`. Module docstring updated.
- Tests added: `test_non_string_message_logs_type_name_only`, `test_non_finite_float_emitted_as_null` (nan, inf, -inf; parsed with `parse_constant` that raises), `test_arbitrary_object_in_whitelisted_key_emitted_as_type_name` (also checks int, None, bool pass through) in `tests/test_obs_logging.py`; `test_metrics_server_blank_token_is_unconfigured` (no header, `Bearer ` header, other token: all 404) in `tests/test_obs_metrics.py`.

### RED
`perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_obs_logging.py tests/test_obs_metrics.py -q`
-> `6 failed, 21 passed`. Failures: the exception-msg test, nan/inf/-inf null tests, the arbitrary-object test (`'SECRET-OBJ' is contained here: ,"route":"SECRET-OBJ"`), and the blank-token test (`assert 401 == 404`).

### GREEN
Same command -> `27 passed`.
Full suite `perl -e 'alarm 600; exec @ARGV' .venv/bin/python -m pytest -q` -> `162 passed, 2 warnings` (pre-existing test_generator warnings).
`.venv/bin/ruff check .` -> All checks passed; `ruff format --check` on touched files -> already formatted.
