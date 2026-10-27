# Plan A final fix wave report

Base 15bc7d6 (410 tests). Head 536b023. Final: 446 passed in 87 s (`perl -e 'alarm 600'` pytest), `ruff check .` clean, `.venv/bin/python -m backend.cli.main migrate` prints `up to date`. The 2 pytest warnings are the pre-existing `tests/test_generator.py` un-awaited AsyncMock warning, not from this wave.

All focused runs used `perl -e 'alarm 300; exec @ARGV'`. Worker-loop tests are bounded with `asyncio.wait_for` (LIMIT_S 10). No venv, file flag, docs/superpowers, or .superpowers changes were committed.

## Commits (oldest first)

| SHA | Subject | Items |
|---|---|---|
| a75c568 | fix: bound auth input fields and reject lone surrogates | I1 |
| d54eba9 | feat: call per-kind failure hooks when jobs fail or die | I2 |
| 3a73dba | fix: scope collection lookups by kind | I3 |
| 6b34a73 | test: cover the kinds keyword of require_collection | I3 (see deviations) |
| be89b1a | test: isolate tests from HALLUCISCOPE environment | M1 |
| 23a4860 | fix: align error codes for grants and API keys | M2 |
| d8f3952 | refactor: make trivial FastAPI dependencies async | M3 |
| 8a32e35 | chore: wrap the page_params signature | M3 (ruff E501 fix-up) |
| 4d6bae3 | fix: reject non-UUID tenant ids in tenant_tx | M4 |
| ad47bb9 | test: assert the tenant isolation policy on every tenant table | M5 |
| f4433a0 | fix: fail when no migrations are found | M6 |
| 5629515 | fix: disable worker metrics when no usable token is configured | M7 |
| 9f7e2c8 | fix: replace dead pooled connections on checkout | M8 |
| c77a66e | fix: reserve the ip throttle key before the email key | M9 |
| 536b023 | fix: write api key last_used_at at most once a minute | M10 |

## I1. Auth input limits and lone surrogates

Changed: `backend/api/schemas_v1.py` gains `Email` (max 320), `Password` (max 256), `DisplayName` (max 100), plus `Name` and `Description` aliases. Each ends in an `AfterValidator` that rejects any char in U+D800..U+DFFF. Applied to every request str field: RegisterIn, LoginIn, MemberAddIn, ApiKeyCreateIn, CollectionIn, CollectionPatch. `backend/auth/service.py` `login` returns `invalid_credentials` (401) for a normalized email over 320 chars before any throttle access.

Tests: `tests/test_auth_routes.py` (`test_login_rejects_oversized_email_with_422`, `test_login_email_over_320_characters_is_422`, `test_register_rejects_lone_surrogate_password_with_422`, `test_register_rejects_oversized_password_and_display_name`), `tests/test_auth_service.py::test_login_with_oversized_email_skips_the_throttle_table`, `tests/test_collections.py::test_lone_surrogate_in_collection_name_is_422`.

RED:
```
assert 401 == 422   (4 KB email)
assert 401 == 422   (321-char email)
assert 500 == 422   (\ud800 password)
assert [{'?column?': 1}] == []   (email throttle row created for oversized email)
4 failed, 1 passed
```
(The oversize password/display_name test passed before the fix because the service already rejected them with 422; it is a schema-level regression guard.)
GREEN: 114 passed across auth_routes, auth_service, collections, members, api_keys.

Note: stacking `StringConstraints` on an already `AfterValidator`-annotated type skipped `strip_whitespace` (an api key name of `"   "` returned 201). The aliases therefore list the constraints first and the surrogate validator last.

## I2. Failure hooks for dead jobs

Changed:
- `backend/jobs/queue.py`: `requeue_stale_detailed(conn, stale_after_s=120.0) -> tuple[int, list[Job]]` (UPDATE ... RETURNING *, dead-lettered rows returned as `Job`); `requeue_stale` keeps its signature and delegates.
- `backend/jobs/worker.py`: `FailureHandler` type; `Worker(..., failure_handlers=None)`; `_finish` calls the kind's hook after `queue.fail` returns `failed` or `dead` (not for `queued` retries, not on lease loss, not on success); the requeue loop uses the detailed form and calls the hook with `lease_expired` per dead job. Hooks run in `_run_failure_hook` with try/except and `logger.exception("job failure hook error", extra=<worker_id, kind, attempt>)`, never raised.
- `backend/jobs/registry.py`: `build_failure_handlers(services) -> {}`.
- `backend/cli/main.py`: passes `failure_handlers=build_failure_handlers(services)` to every Worker.

Tests (`tests/test_worker.py`, `tests/test_jobs_queue.py`): `test_failure_hook_runs_once_when_retries_are_exhausted` (RetryableJobError x3, hook once with `upstream_down`), `test_failure_hook_runs_once_for_a_permanent_failure`, `test_failure_hook_is_not_called_for_success_or_a_pending_retry`, `test_failure_hook_only_runs_for_its_own_kind`, `test_failure_hook_runs_for_a_dead_lettered_stale_job` (once, `lease_expired`), `test_failure_hook_error_is_logged_and_does_not_stop_the_loop` (second job still succeeds, no `worker loop error`, exception text not in logs), `test_requeue_stale_detailed_returns_the_dead_lettered_jobs`, `test_run_worker_passes_the_failure_handlers_to_every_worker`, and `build_failure_handlers(None) == {}`.

RED: `TypeError: Worker.__init__() got an unexpected keyword argument 'failure_handlers'` (x6), `AttributeError: module 'backend.jobs.queue' has no attribute 'requeue_stale_detailed'`; 7 failed. For the CLI pass-through test, RED was shown by temporarily removing the kwarg line: `assert [None, None] == [{'ingest': ...}]`, then restored.
GREEN: tests/test_worker.py + tests/test_jobs_queue.py 101 passed.

## I3. Collection kind scoping

Changed: `backend/authz/policy.py` `require_collection(..., *, kinds=("normal",))` adds `and c.kind = any(%s)`; another kind is 404 `collection_not_found`. `effective_permission` had no caller (grep over backend and tests found only its definition), so it was deleted instead of getting the keyword.

Tests (`tests/test_collections.py`): `test_by_id_routes_do_not_expose_non_normal_collections[quick|benchmark]` (GET, PATCH, grants GET, grants PUT, DELETE all 404, row untouched) and `test_require_collection_kinds_keyword_opts_into_other_kinds`.
RED: `assert [200, 200, 204, 404, 404] == [404]*5` for both kinds (the first version of the test deleted before the grants calls, which hid the grants routes; reordered so DELETE is last).
GREEN: tests/test_collections.py 13 passed.

## M1. Environment isolation

Changed: `tests/conftest.py` autouse `_isolate_from_halluciscope_environment` deletes every `HALLUCISCOPE_*` var except `HALLUCISCOPE_TEST_ADMIN_URL` via monkeypatch. Test: `tests/test_env_isolation.py` runs a pytest subprocess for `test_auth_routes.py -k register_sets_session` with `HALLUCISCOPE_ALLOW_REGISTRATION=false` and `HALLUCISCOPE_METRICS_TOKEN=x`.
RED: subprocess exit non-zero, `test_register_sets_session_and_me_works` failing in `signup`. GREEN: passes (51 passed with worker and config tests).

## M2. Error codes

Changed: PUT grants with a non-member now `grantee_not_member` (422); `not_a_member` stays for switch-tenant 404. `revoke_api_key` raises `api_key_not_found` (404); titles dict loses the now unused `not_found` entry and gains `"api_key_not_found": "API key not found"`. Tests updated in test_collections.py, test_api_keys.py (also asserts the title), test_auth_service.py.
RED: 7 failed (`'not_a_member' == 'grantee_not_member'`, `'not_found' == 'api_key_not_found'` x6). GREEN: 94 passed.

## M3. Async dependencies

Changed: `get_db`, `get_auth` (`backend/auth/deps.py`) and `page_params` (`backend/api/pagination.py`) are `async def`; `current_principal` now `await`s `get_auth`. Test: `tests/test_auth_routes.py::test_trivial_dependencies_are_coroutines[get_db|get_auth|page_params]` (inspect.iscoroutinefunction).
RED: 3 failed. GREEN: 78 passed on the route test files. The first commit left a ruff E501 on `page_params`; fixed in 8a32e35.

## M4. tenant_tx type check

Changed: `Database.tenant_tx` raises `TypeError("tenant_id must be a UUID")` for non-UUID input, at call time. Tests: `tests/test_pool.py::test_tenant_tx_rejects_non_uuid_tenant[None|str|int]`.
RED: `DID NOT RAISE TypeError` x3. The full suite then showed `tests/test_members.py::test_removed_member_loses_session_and_grants` passing a str tenant id (the exact misuse); fixed that test to pass `UUID(...)`. GREEN: 27 passed (members + pool).

## M5. RLS coverage

Changed (test only): `tests/test_rls.py` includes relkind `'p'` and adds `test_every_tenant_table_has_the_tenant_isolation_policy`, which reads `pg_policies` and requires a `tenant_isolation` policy on each tenant table whose `qual` and `with_check` both equal `(tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)`.
No RED is possible against current code (the migrations already comply). Mutation check instead: changing the expected string (`true` to `false`) made the new test fail (1 failed, 7 passed); restored, 8 passed.

## M6. Empty migrations

Changed: `migrate()` raises `MigrationError("no migrations found")` right after `discover()` returns an empty list, before connecting. Test: `tests/test_migrate.py::test_migrate_refuses_an_empty_migrations_directory` (unreachable DB URL, so only the early error passes).
RED: `psycopg.OperationalError: connection failed ... Connection refused`. GREEN: 8 passed.

## M7. Worker metrics token

Changed: `start_metrics_server` treats an empty or whitespace-only token as unconfigured (404). The CLI `_run_worker` starts the server only for a non-blank token, otherwise logs `worker metrics disabled` at INFO (logger `backend.cli.main`). Tests: `tests/test_obs_metrics.py::test_metrics_server_treats_a_blank_token_as_unconfigured` and `tests/test_worker.py::test_run_worker_skips_the_metrics_server_without_a_token[None|""|"   "]`. Two existing harness tests now pass `metrics_token="t"` since they exercise the server.
RED: `assert 401 == 404` (blank tokens), `'metrics.start' not in [... 'metrics.start' ...]` x3. GREEN: 71 passed (obs_metrics, worker, cli_module).

## M8. Pool health check

Changed: `AsyncConnectionPool(..., check=AsyncConnectionPool.check_connection)`. Test: `tests/test_pool.py::test_connection_killed_between_transactions_is_replaced` (single-connection pool, backend killed with `pg_terminate_backend` from the admin URL, second `system_tx` works on a new pid).
RED: `psycopg.errors.AdminShutdown: terminating connection due to administrator command`. GREEN: 4 passed.

## M9. Throttle reservation order

Changed: `_reserve_attempt` reserves the IP key first, the email key second, and refunds the IP reservation if the email key is full. Tests: `test_full_ip_key_creates_no_email_throttle_row` (RED: row for `email:fresh@example.com` existed with failures 0) and `test_full_email_key_gives_the_ip_reservation_back`. Existing throttle tests unchanged and green: 44 passed.

## M10. api_keys.touch

Changed: SQL condition `and (last_used_at is null or last_used_at < now() - interval '60 seconds')`. Test: `test_api_key_last_used_is_written_at_most_once_a_minute` (second resolve leaves the value unchanged; after aging it to 61 s it advances).
RED: the timestamps differed (`...508273 == ...505419` failed). GREEN: 64 passed (auth_service + api_keys).

## Deviations and decisions

- I1: the shared annotated type is not a single `Text` alias. Constraints must precede the surrogate validator (see note under I1), so each field alias (`Email`, `Password`, `DisplayName`, `Name`, `Description`) carries its constraints plus the same `AfterValidator`. No unused `Text` alias was left behind.
- I1: the service guard returns `invalid_credentials` without an audit row (no throttle or audit access), matching "without touching the throttle table".
- I3: the first commit omitted the `kinds` keyword test because a chained shell command failed silently on a BSD sed error; the test was added in follow-up 6b34a73.
- M3: a ruff E501 slipped into d8f3952 (the hooks run gitleaks only); fixed in 8a32e35.
- I2: hooks receive the same `JobContext` shape as handlers (job, db, worker_id, services). No timeout is applied to hooks (not in the ruling).
- Out of scope per findings (not touched): I4 proxy headers, extra indexes, row purging, registration throttle, API key cap.

## Concerns

- `tests/test_worker.py` now sends SIGINT to the pytest process in two tests; both wait until a Worker has been constructed (which happens after `_run_worker` installs its signal handlers), so the signal is always handled.
- `ruff format --check` reports 22 files that would be reformatted; this is pre-existing and not part of the lint gate (`ruff check .` is clean).
