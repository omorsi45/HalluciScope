# Task 10 report: API skeleton

Status: DONE. Commit f204849 `feat: add problem errors, request context, security headers, body limits`.

## Built
- `backend/api/errors.py`: `ApiError`, `problem_response` (application/problem+json), `install_error_handlers` (ApiError, RequestValidationError with loc+msg only, Starlette HTTPException with code map, generic Exception -> 500 `internal_error`), plus `unhandled_error_response()` (logs `"unhandled error"` via `logger.exception`, returns the generic 500).
- `backend/api/middleware.py`: pure ASGI `RequestContextMiddleware`, `SecurityHeadersMiddleware`, `BodySizeLimitMiddleware`. No BaseHTTPMiddleware.
- `backend/api/app.py`: `create_app(settings=None, lifespan=None)`; order outermost first RequestContext, SecurityHeaders, CORS allowlist, BodySizeLimit; empty `v1` APIRouter mounted at `/api/v1`; legacy `/health`, `/api/analyze`, `/api/analyses`, `/api/feedback` unchanged.
- `backend/api/startup.py`: lifespan calls `configure_logging(settings.log_level)`, opens `Database` into `app.state.db`, closes on shutdown; passes `settings` to `create_app`; docstring documents the uvicorn `--log-config` / `log_config=None` requirement.
- `tests/test_api_skeleton.py`: 16 tests (the 11 in the brief plus 404 problem, metrics label with route template, CSP only on /api/, multipart uses upload limit, chunked overflow wins over FastAPI's own 400 body-parse error, overflow after response start closes by raising).

## RED
`pytest tests/test_api_skeleton.py -q` -> collection error `ModuleNotFoundError: No module named 'backend.api.errors'`.

## GREEN
- `pytest tests/test_api_skeleton.py tests/test_api.py -q` -> 19 passed.
- Full suite: `178 passed, 2 warnings` (162 before + 16 new; warnings are the pre-existing unawaited AsyncMock in a generator test).
- `ruff check .` clean; `ruff format --check` clean on the files I touched (22 other files in the repo were already unformatted before this task, left alone).

## Decisions
1. Unhandled exceptions are answered inside `RequestContextMiddleware` (it catches, builds the same generic 500 via `unhandled_error_response()`, sends it through its own send wrapper). Reason: Starlette's `Exception` handler runs in `ServerErrorMiddleware`, outside all user middleware, so that 500 would lack `X-Request-ID`, security headers and CORS headers, and would skip the access log/metrics. The `Exception` handler from the brief is still installed as the fallback for anything raised outside the middleware stack. If the response already started, the exception is re-raised.
2. Body limit overflow: FastAPI turns any exception during body read into its own 400. So on overflow the wrapped `receive` raises a private `_BodyTooLarge`, the wrapped `send` drops whatever the app then emits (when nothing has started), and 413 is sent after the app returns. If the app had already started its response, `_BodyTooLarge` is re-raised to close the connection.
3. Metrics status label is `str(status)`; route is `scope["route"].path` read after the inner app returns, else `"unmatched"`.
4. 413 body detail is the constant "Request body exceeds the size limit."; HTTPException details are passed through as strings (framework/developer controlled), input is never echoed.

## Concerns
- `create_configured_app` lifespan now needs Postgres at startup (per the brief); legacy sqlite repo still opened too.
- Content-Length that is not a digit string is ignored by the early check (the streaming count still applies).


## Fix round 1

Base: f204849. Findings: task-10-findings-r1.md (2 review findings + 4 controller rulings).

### Changes
- `backend/api/middleware.py`
  - New innermost `UnhandledErrorMiddleware`: catches an unhandled exception before the response starts and answers the generic 500 there, so the 500 passes back out through CORS and SecurityHeaders and carries nosniff, CSP, Referrer-Policy, X-Frame-Options, `Access-Control-Allow-*`, `Access-Control-Expose-Headers` and `X-Request-ID` (finding 1). It re-raises `_BodyTooLarge` so the body limit still wins. `RequestContextMiddleware` keeps its catch as a fallback; the misleading comment was corrected.
  - Metric `method` label bounded to GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS, else `OTHER` (finding 2). The access log uses the same bounded value.
  - Request id validated with `re.fullmatch` (trailing newline rejected).
  - Content-Length used for the early reject only if ASCII digits and at most 19 characters; otherwise treated as absent and the streamed count applies (this fixes a `ValueError` crash on `"\u00b2"`).
- `backend/api/app.py`: `UnhandledErrorMiddleware` added innermost (order from outermost unchanged: RequestContext, SecurityHeaders, CORS, BodySizeLimit, then the new innermost one); `create_app` raises `ValueError` when `"*"` is in `cors_origins`.
- `backend/api/startup.py`: after `db.open()` the rest of startup runs in try/finally; shutdown closes http client, repo, db each in its own try/finally (db is closed even if startup fails or another close raises). Docstring corrected: `configure_logging` replaces root handlers with the JSON handler, points `uvicorn` at it with propagate off (so `uvicorn.error` goes through it), and empties, un-propagates and disables `uvicorn.access`; it runs after uvicorn's own config so run uvicorn normally; `--log-config /dev/null` is invalid, programmatic `log_config=None` is fine.
- `backend/obs/` untouched.

### Tests added
`tests/test_api_skeleton.py` (+9 cases): `test_unhandled_error_carries_security_and_cors_headers`, `test_unknown_methods_share_one_metric_label`, `test_request_id_with_trailing_newline_is_replaced` (raw ASGI scope), `test_odd_content_length_is_ignored_not_fatal` (5 params: `\xb2`, 20 digits, `-5`, `1 2`, empty; raw ASGI), `test_create_app_rejects_wildcard_cors_origin`.
New `tests/test_api_startup.py` (2): `test_startup_failure_after_db_open_closes_pool`, `test_shutdown_closes_every_resource_even_if_one_fails` (fake Database/Repository, models and verifiers patched; no Postgres, no model load).

### RED
Command: `perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_api_skeleton.py tests/test_api_startup.py -q`
Result: `8 failed, 19 passed in 6.19s`
```
FAILED test_unhandled_error_carries_security_and_cors_headers   KeyError: 'x-content-type-options'
FAILED test_unknown_methods_share_one_metric_label              assert not {'BREW0','BREW1','BREW2','BREW3','BREW4'}
FAILED test_request_id_with_trailing_newline_is_replaced        re.fullmatch(...) is None for 'abc-1\n' (id echoed)
FAILED test_odd_content_length_is_ignored_not_fatal[\xb2]       ValueError: invalid literal for int() with base 10: '²'
FAILED test_odd_content_length_is_ignored_not_fatal[1111...]    assert [] == ['/x'] (20 digit value rejected instead of ignored)
FAILED test_create_app_rejects_wildcard_cors_origin             Failed: DID NOT RAISE ValueError
FAILED test_startup_failure_after_db_open_closes_pool           assert False is True (pool left open)
FAILED test_shutdown_closes_every_resource_even_if_one_fails    assert False is True (db not closed)
```
All fail for the intended reasons; no expensive behavior is triggered.

### GREEN
- `perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_api_skeleton.py tests/test_api.py -q` -> `28 passed in 0.89s` (before ruff-only test edit); with `tests/test_api_startup.py` -> `30 passed in 10.27s`.
- Full suite once: `perl -e 'alarm 600; exec @ARGV' .venv/bin/python -m pytest -q` -> `189 passed, 2 warnings in 13.69s` (178 + 11 new; warnings are the pre-existing unawaited AsyncMock).
- `perl -e 'alarm 120; exec @ARGV' .venv/bin/ruff check .` -> `All checks passed!` (first run flagged RUF012 in the new startup test; fixed with `ClassVar`). `ruff format --check` on touched files: clean.

### Decisions
- A 20-digit Content-Length is treated as absent per the ruling; the streamed count still enforces the limit.
- The 413 path is unchanged: when the limit is exceeded, anything the inner stack emits (including a 500 from the new middleware) is dropped by BodySizeLimit and 413 is sent.

### Concerns
None new.
