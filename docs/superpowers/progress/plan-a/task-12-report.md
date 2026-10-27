# Task 12 report: auth dependencies, CSRF, auth routes

Status: DONE

## Built
- `backend/auth/deps.py`: get_db, get_auth, current_principal (async; Bearer API key skips CSRF and never falls back to the cookie; else `eg_session` cookie; CSRF via `hmac.compare_digest` on unsafe methods; sets `tenant_id_var`/`user_id_var` with `str(uuid)`), require_role.
- `backend/api/schemas_v1.py`, `backend/api/routes/auth.py` (register 201, login, logout 204, me, switch-tenant), mounted in `app.py` on `v1` at `/auth`; `startup.py` sets `app.state.auth`.
- `tests/helpers/{__init__,auth}.py`; `tests/pg_fixtures.py`: api_settings, api_app, client, client2 (added to `__all__`).
- `tests/test_auth_routes.py`: 16 tests (the 10 in the brief plus: me unauthenticated, login returns session, logout revokes server side, invalid bearer no cookie fallback, require_role, context vars per request).

## Evidence
- RED: `pytest tests/test_auth_routes.py -q` -> `ModuleNotFoundError: No module named 'backend.auth.deps'` (collection error, 1 error).
- GREEN: `16 passed`.
- Mutation checks (reverted): removing `_check_csrf` fails `test_unsafe_request_without_csrf_rejected`; removing `tenant_id_var.set` fails `test_principal_context_vars_are_per_request`.
- Full suite: `245 passed, 2 warnings` (the 2 warnings are pre-existing in tests/test_generator.py). `ruff check .` clean; my files are `ruff format` clean.

## Context var scoping (verified by probe plus reasoning)
`RequestContextMiddleware` does NOT give each request its own context: it only `bind`s request_id and resets it. Isolation comes from the server running each request in its own asyncio task (uvicorn does), so `.set()` in the async dependency dies with the task. Probe: with ASGITransport and a direct `await client.get(...)`, `tenant_id_var` DID leak into the test's context (value stayed set after the call), because the transport runs the app in the caller's task. So the test `test_principal_context_vars_are_per_request` models the server by running requests via `asyncio.create_task`, asserting the authed request sees its ids, a later anonymous request sees None, and the test context stays None. Side effect: the access log line (emitted in the middleware's `finally`) now carries tenant_id/user_id for authenticated requests. The dependency must stay `async def`: a sync dependency runs in a threadpool with a copied context and the `.set()` would be lost.

## Decisions
- Logout requires a valid session plus CSRF (goes through `current_principal`), so an expired session gets 401 rather than 204. API-key callers get 204 without a revoke.
- Cookie Max-Age = `session_ttl_days * 86400` (604800 by default). Delete cookies repeat the Secure/SameSite attributes.
- Non-Bearer `Authorization` schemes are ignored (cookie path). Any `Bearer` value is API-key only.
- Register runs `register` then `login`, so login throttling applies to it.
- `registration_disabled` is raised by `AuthService.register` (already mapped to 403).
- `MeOut.tenant` comes from the principal's tenant in the membership list; if missing, 401.

## Concerns
- None blocking. `ruff format --check` reports 21 pre-existing unformatted files elsewhere; not touched.
