# Plan A final review: fix wave findings and controller rulings (binding)

Source: final whole-plan review of 697f8d5..15bc7d6. One fix wave covers everything below. Each item needs a test written first (watch it fail), then the fix.

## Important

I1. Auth input fields have no length limits, so clients can trigger 500s. A 4 KB email on POST /auth/login hits ProgramLimitExceeded on the auth_throttle_pkey index; a JSON "\ud800" surrogate in the register password raises UnicodeEncodeError in hash_password. Where: backend/api/schemas_v1.py RegisterIn, LoginIn, MemberAddIn (plain str) and backend/auth/service.py.
Ruling:
- Use `Annotated[str, StringConstraints(max_length=...)]`: email 320, password 256, display_name 100.
- Reject lone surrogates in every str input with 422 validation_error. Do it once, in a shared annotated type in schemas_v1.py (for example a `Text` type with an AfterValidator that rejects any character in U+D800..U+DFFF), applied to every request str field in schemas_v1.py.
- The service layer also guards: login with an email over 320 characters returns invalid_credentials without touching the throttle table.
- Tests: the two probes above return 422; a 321-character email returns 422.

I2. Dead jobs leave their domain rows stuck (for example an upload stuck in processing forever).
Ruling (option a):
- Add a failure hook. `FailureHandler = Callable[[JobContext, str], Awaitable[None]]`, called with the final error_code.
- `Worker(..., failure_handlers: Mapping[str, FailureHandler] | None = None)` calls the kind's hook after `queue.fail` returns `failed` or `dead`.
- `backend/jobs/queue.py` gains `async def requeue_stale_detailed(conn, stale_after_s: float = 120.0) -> tuple[int, list[Job]]`, which returns the requeued count and the jobs it dead-lettered. `requeue_stale` keeps its signature and delegates. The worker's requeue loop uses the detailed form and calls the hook for each dead-lettered job, with error_code "lease_expired".
- Hooks run in their own try/except: logged with `logger.exception("job failure hook error")` plus kind, never raised.
- `backend/jobs/registry.py` gains `build_failure_handlers(services) -> dict[str, FailureHandler]` (empty in Plan A). The CLI passes it to every Worker.
- Tests:
  - A job whose handler raises RetryableJobError three times calls the hook exactly once with its error code.
  - A permanent failure calls the hook once.
  - A dead-lettered stale job calls the hook once with lease_expired.
  - A hook that raises does not kill the loop.

I3. require_collection ignores the collection kind, so tenant admins can address quick and benchmark collections through by-id routes.
Ruling:
- `require_collection(conn, principal, collection_id, needed, *, kinds: Sequence[str] = ("normal",))` filters `c.kind = any(%s)`. A collection of another kind is 404 collection_not_found.
- `effective_permission` gets the same keyword. If it has no caller, delete it, and say so in the report.
- Tests: GET, PATCH, DELETE, and grants on a quick-kind collection id return 404 for its tenant owner.

## Minor items fixed in this wave

M1. Tests read HALLUCISCOPE_* from the developer environment. Add an autouse fixture in tests/conftest.py that removes (monkeypatch.delenv) every HALLUCISCOPE_* variable except HALLUCISCOPE_TEST_ADMIN_URL before each test. Test: run a subprocess `pytest tests/test_auth_routes.py -q -k register_sets_session` with HALLUCISCOPE_ALLOW_REGISTRATION=false and HALLUCISCOPE_METRICS_TOKEN=x in its environment and assert it passes (it fails without the fixture).

M2. Error code consistency:
- PUT grants with a non-member uses code `grantee_not_member` (422). `not_a_member` stays for switch-tenant 404.
- Revoking an unknown or forbidden API key uses `api_key_not_found` (404).
- Update the titles dict and tests.

M3. get_db, get_auth (backend/auth/deps.py) and page_params (backend/api/pagination.py) become `async def`.

M4. Database.tenant_tx raises TypeError when tenant_id is not a UUID instance. Test with None and with a str.

M5. RLS coverage test (tests/test_rls.py):
- Also include relkind 'p'.
- Assert that each tenant table has a policy named tenant_isolation whose qual and with_check, as read from pg_policies, both equal `(tenant_id = (NULLIF(current_setting('app.tenant_id'::text, true), ''::text))::uuid)`.

M6. migrate() raises MigrationError("no migrations found") when discover() returns an empty list. Test it.

M7. The worker metrics server (backend/obs/metrics_server.py) treats a whitespace-only token as unconfigured. The CLI does not start the metrics server at all when no token is configured (log one info line, "worker metrics disabled"). Tests for both.

M8. The pool is built with `check=AsyncConnectionPool.check_connection`, so dead connections are replaced after a Postgres restart. Test: a connection killed with pg_terminate_backend between two system_tx calls does not fail the second call.

M9. _reserve_attempt reserves the IP key first and the email key second. If the email key is full, it refunds the IP reservation. Keep the existing throttle tests green and add one for the new order: a full IP key creates no email row.

M10. api_keys.touch only writes when last_used_at is null or older than 60 seconds (condition in SQL). Test: two requests in quick succession write once.

## Not in this wave (controller handles in plan text or later)
- I4 proxy headers for the throttle IP: Plan D Task 12 brief.
- Indexes memberships(user_id), collection_grants(tenant_id, user_id), sessions and api_keys (tenant_id, user_id): Plan B Task 7 migration.
- Purging old rows, a registration throttle, and an API key cap: before release (ledger).
