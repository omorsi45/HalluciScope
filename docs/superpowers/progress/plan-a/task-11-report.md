# Task 11 report: auth service, global repositories, audit log

Status: DONE_WITH_CONCERNS (minor deviations listed below)

## Built
- `backend/db/repos/{users,tenants,sessions,api_keys,audit}.py` per the brief, plus `throttle.py` (see decisions).
- `backend/auth/service.py`: `Principal`, `LoginResult`, `AuthError`, `AuthService` (register, login with throttling, logout, resolve_session, resolve_api_key, switch_tenant, create/list/revoke API keys).
- `backend/api/errors.py`: `AuthError` handler (problem with the error's code and status; titles from `_AUTH_TITLES`).
- `tests/test_auth_service.py`: 30 tests (all brief tests plus extras).

## RED
`pytest tests/test_auth_service.py -q` before implementation: collection error `ModuleNotFoundError: No module named 'backend.auth.service'` (fast, no expensive calls).

## GREEN
- First run after implementation: 29 passed, 1 failed (`test_audit_rows_have_no_secrets`). Root cause was a test arithmetic error (a successful login clears the email counter, so 4 later failures do not reach the limit of 5); fixed the test to 5 failures. The throttle code was correct.
- `tests/test_auth_service.py`: 30 passed (4.5 s).
- Full suite: `219 passed, 2 warnings` (the 2 warnings are pre-existing, in tests/test_generator.py). Was 189 before.
- `ruff check .`: All checks passed. Task files are `ruff format` clean (the repo has 21 other pre-existing unformatted files; untouched).

## Decisions
- Added `backend/db/repos/throttle.py` (get_failures, record_failure, clear): the brief lists no repo for `auth_throttle`; SQL stays out of the service. Upsert is `on conflict (key, window_start) do update set failures = auth_throttle.failures + 1`.
- Added error code `invalid_display_name` (422): the brief's code list has none for an invalid display name.
- Login does throttle check, credential check, throttle upsert and audit in one `system_tx`, and raises `AuthError` after the transaction closes so failure and throttle rows commit. Password hashing/verification runs in `asyncio.to_thread` so scrypt does not block the event loop.
- A disabled user, or a user with no memberships, whose password is right is treated as `invalid_credentials` and counted as a failure.
- `sessions.revoke_session`, `api_keys.revoke` and `api_keys.touch` keep the brief's signatures and set `revoked_at` / `last_used_at` with SQL `now()`. These are informational columns; no expiry or throttle decision uses them. All expiry and window logic uses the injected clock passed as parameters.
- `api_keys.get(conn, key_id, tenant_id)` added so revoke can check ownership (member revokes own keys only; owner/admin any key in the tenant; otherwise `not_found`). `list_api_keys` returns the caller's own keys in the active tenant including revoked ones (`revoked_at` set), never `key_hash`.
- Audit rows: `register`, `login`, `login_failed`, `login_throttled`; `details` holds only `{"ip": ip}` for login events. No audit for api key create/revoke or tenant switch (not requested).
- Extra tests: invalid email/weak password/registration disabled, successful login clears email failures, malformed/unknown tokens, removed membership invalidates API keys, switch_tenant (success, foreign tenant, API key), AuthError problem response.

## Concerns
- Principal for `switch_tenant` rejects any principal without `session_id` as `api_key_tenant_fixed`.
- The IP is stored in audit details; revisit if IPs count as personal data under a retention policy.


## Fix round 1

Findings file: task-11-findings-r1.md. Base head fed9518.

### Changes
- `backend/db/repos/throttle.py`: replaced `get_failures` / `record_failure` with `reserve(conn, key, window, limit) -> bool` (single conditional upsert: `insert ... on conflict (key, window_start) do update set failures = auth_throttle.failures + 1 where auth_throttle.failures < %s returning failures`; no row means throttled), `release` (decrement, floor 0) and the existing `clear`.
- `backend/auth/service.py` `login` split in three: (a) short tx: `get_login_record`, reserve email slot then ip slot (email refunded if the ip key is full), audit `login_throttled` when refused; (b) `verify_password` / `verify_dummy` in `asyncio.to_thread` with no transaction or connection held; (c) short tx: on success list memberships, clear the email key, release the ip slot (ip key counts failures only), insert session, audit `login`; otherwise audit `login_failed`. Errors are raised after the transactions close. A disabled user (checked before `list_memberships`) costs the same queries as a wrong password.
- `backend/db/repos/users.py`: `get_user_by_email` and `get_user` select explicit columns (no `password_hash`); new `get_login_record(conn, email)` is the only lookup returning the hash.
- `backend/db/repos/api_keys.py`: `list_for_user` returns only keys with `revoked_at is null`.
- Existing tests unchanged (none used the removed throttle functions or the changed signatures).

### New tests (tests/test_auth_service.py, 30 -> 40)
- `test_concurrent_wrong_passwords_respect_email_limit`: fixture `wide_db` = `Database(pg_urls["app"], max_size=10)`, 20 concurrent wrong-password logins via `asyncio.gather`; asserts exactly 5 `invalid_credentials`, 15 `too_many_attempts`, email counter row 5.
- `test_no_connection_held_during_password_verification[True|False]`: `Database(max_size=3)`; stubs for `auth_service.verify_password` and `verify_dummy` run in the worker thread and, via `run_coroutine_threadsafe`, try to hold all 3 connections at once (2 s timeout each); records the result and returns False; asserts `[True]`. Known and unknown email paths.
- `test_ip_throttled_attempt_does_not_count_against_email`, `test_successful_login_does_not_consume_ip_budget` (guard the refund and release rules).
- Probed-behavior tests: `test_disabled_user_rejected_everywhere`, `test_forged_api_key_with_matching_prefix_rejected`, `test_owner_of_other_tenant_cannot_revoke_key`.
- Repo tests: `test_list_api_keys_returns_only_active_keys`, `test_only_login_lookup_returns_password_hash`.

### RED
Command: `perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_auth_service.py -q --no-header` (before implementation)
```
E  AssertionError: assert 14 == 5            (concurrent: 14 invalid_credentials against limit 5)
E  assert [False] == [True]                  (x2, pool-hold, known and unknown email)
E  Left contains one more item: UUID(...)     (list_api_keys returned the revoked key)
E  AttributeError: module 'backend.db.repos.users' has no attribute 'get_login_record'
FAILED test_concurrent_wrong_passwords_respect_email_limit
FAILED test_no_connection_held_during_password_verification[True]
FAILED test_no_connection_held_during_password_verification[False]
FAILED test_list_api_keys_returns_only_active_keys
FAILED test_only_login_lookup_returns_password_hash
5 failed, 35 passed in 12.33s
```
The other new tests passed on the old code (they pin ruled behavior that was already correct or is guarded for the new design).

### GREEN
- `perl -e 'alarm 300; exec @ARGV' .venv/bin/python -m pytest tests/test_auth_service.py -q --no-header`: `40 passed in 8.28s` (rerun after `ruff format` of the test file: 40 passed).
- Concurrency and pool-hold tests rerun 3 times: `3 passed` each time.
- Full suite once: `perl -e 'alarm 600; exec @ARGV' .venv/bin/python -m pytest -q --no-header`: `229 passed, 2 warnings in 20.02s` (warnings pre-existing in tests/test_generator.py).
- `.venv/bin/ruff check .`: All checks passed. `ruff format --check` on touched files: clean.

### Decisions
- Throttled attempts and unknown-email attempts keep their reservation (they count as failures, as before). A user with a correct password but no memberships also keeps the reservation (failure).
- Reservation order is email then ip; an ip refusal refunds the email slot, so an attempt refused by either key leaves both counters unchanged.
- Window and "now" for the whole login are taken once at the start, so a clear/release in phase (c) targets the same window as the reservation.
