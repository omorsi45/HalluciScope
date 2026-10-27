# Task 15 report: tenant members and audit routes

Status: DONE. Commit: see `git log -1` (feat: add tenant member management and audit routes).

## Built
- `GET/POST /api/v1/tenants/current/members`, `DELETE .../members/{user_id}`, `GET .../audit`.
- Schemas MemberOut, MemberAddIn, AuditOut (schemas_v1.py); router mounted in app.py.
- `backend/db/repos/members.py`: list_members, member_out, lock_owner_ids (select ... for update), delete_membership, list_audit.
- `backend/api/pagination.py`: added `decode_int_cursor` (bigint key; non-decimal, negative or > 2^63-1 gives 400 bad_cursor), next to decode_uuid_cursor.

## Files
Created: backend/db/repos/members.py, backend/api/routes/members.py, tests/test_members.py. Modified: backend/api/schemas_v1.py, backend/api/app.py, backend/api/pagination.py.

## RED
`pytest tests/test_members.py -q` before implementation: `14 failed` (routes returned 404 not_found, e.g. `assert 404 == 403`, `assert 404 == 201`).

## GREEN
`pytest tests/test_members.py -q`: `16 passed` (8 brief tests + extras listed below, bad-cursor parametrized x3).
Full suite: `290 passed, 2 warnings` (warnings are pre-existing, tests/test_generator.py unawaited AsyncMock). `ruff check .`: All checks passed.

## Decisions
- Removal uses one system_tx: lock owner rows, load target membership, role checks, delete, audit. Grants are removed by the FK cascade (no explicit delete); test asserts they are gone. Session 401 comes from resolve_session; test asserts it.
- Email normalized with strip().lower() before lookup. Audit details are `{"role": ...}` only, target_type "user", target_id user id.
- Admin adding owner: 403 `insufficient_permission` (brief gave only 403). Member on add/audit: 403 `forbidden` from require_role.
- Removing a user who is not a member: 404 `member_not_found` (brief silent).
- Already-member detected via UniqueViolation on the memberships pkey (race-safe), mapped to 409 `already_member`.
- Audit cursor key is str(id), compared `(at, id) < (%s, %s::bigint)`.
- Extra tests beyond the brief: other-owner removal succeeds, admin cannot remove owner, non-member removal 404, audit pagination/order, bad cursor keys, member cannot read audit.

## Concerns
- `ruff format --check` reports 20 pre-existing files unformatted (repo does not enforce it); I did not reformat.
- Disabled users (users.disabled_at) can still be added as members; brief does not address it.


## Fix round 1

Status: DONE. Base a1b6c99. Commit: `fix: require a session for membership changes and revoke removed members' credentials`.

### Changes
- `backend/auth/deps.py`: new reusable dependency `require_session` (api_key principal -> `ApiError(403, "session_required", "This action requires signing in with a session")`). Used on `POST /members` and `DELETE /members/{user_id}`; GET routes unchanged.
- `backend/db/repos/members.py`: `revoke_user_credentials` (sets `revoked_at = now()` on the user's sessions and api_keys where `tenant_id` matches and `revoked_at is null`); `lock_owner_ids` now `order by user_id for update`.
- `backend/api/routes/members.py`: removal calls `revoke_user_credentials` in the same transaction as the delete and audit; an owner principal whose user_id is not in the locked owner set gets 403 `insufficient_permission`.
- `backend/api/pagination.py`: `decode_int_cursor` rejects keys longer than 19 characters before `int()` (400 `bad_cursor`).
- `tests/test_members.py`: 8 new tests (below).

### Tests added
- `test_bearer_key_cannot_add_member`, `test_bearer_key_cannot_remove_member` (403 `session_required`, code and title; nothing changed).
- `test_bearer_key_can_list_members_and_audit` (guard: GETs stay allowed).
- `test_member_cannot_remove` (403 `forbidden`; coverage gap, passed before the fix).
- `test_removed_member_keys_stay_revoked_after_readd` (key and session from the removed membership stay 401 after re-add; the same user's key and session in another tenant stay 200).
- `test_audit_cursor_key_too_long_400` (6720-character key).
- `test_owner_locks_are_taken_in_user_id_order` (8 owners, `lock_owner_ids == sorted(...)`).
- `test_owner_who_lost_ownership_mid_request_cannot_remove` (resolve_session wrapped to delete the caller's membership after authentication; expects 403 `insufficient_permission`).

### Commands
All under `perl -e 'alarm N; exec @ARGV'` (300 s focused, 600 s full, 120 s ruff):
- `.venv/bin/python -m pytest tests/test_members.py -q`
- `.venv/bin/python -m pytest tests/test_api_keys.py -q`
- `.venv/bin/python -m pytest tests/test_auth_routes.py -q`
- `.venv/bin/python -m pytest -q`
- `.venv/bin/ruff check .`

### RED (before implementation)
`pytest tests/test_members.py -q`: `6 failed, 18 passed`. Failures:
- test_bearer_key_cannot_add_member: `assert 201 == 403`
- test_bearer_key_cannot_remove_member: `assert 204 == 403`
- test_removed_member_keys_stay_revoked_after_readd: `assert 200 == 401`
- test_audit_cursor_key_too_long_400: `assert 500 == 400`
- test_owner_locks_are_taken_in_user_id_order: list not equal to its sorted copy
- test_owner_who_lost_ownership_mid_request_cannot_remove: `assert 409 == 403`
The two guard tests (list GETs with a key, member DELETE) passed already, as expected.

### GREEN
- tests/test_members.py: `24 passed`; tests/test_api_keys.py: `19 passed`; tests/test_auth_routes.py: `16 passed` (combined run: `59 passed`).
- Full suite: `298 passed, 2 warnings` (same pre-existing AsyncMock warnings).
- `ruff check .`: All checks passed.

### Decisions
- `require_session` is listed before `_ADMINS` in the route signatures, so a Bearer caller always gets `session_required` (even a member-role key); unauthenticated callers still get 401.
- Re-add after removal creates a fresh membership; old sessions and keys of that tenant stay revoked, credentials in other tenants are untouched (tested).
- The ownership-lost check applies only to owner principals, per the ruling.

### Concerns
- `ruff format --check` still flags pre-existing formatting in tests/test_members.py and one other file; not reformatted (repo does not enforce it).
