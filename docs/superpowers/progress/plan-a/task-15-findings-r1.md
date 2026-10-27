# Task 15 open findings (round 1), from the review, with controller rulings

1. (Important) backend/api/routes/members.py:44,73: Bearer API key principals can add and remove members (require_role checks role only; bearer skips CSRF). A leaked owner key can add an attacker account as owner, giving durable session access that survives key revocation.

Controller ruling (binding): POST /tenants/current/members and DELETE /tenants/current/members/{user_id} with an api_key principal return 403 session_required (same code and title as Task 13). GET routes stay allowed. Implement once as a reusable dependency `require_session` in backend/auth/deps.py (raises ApiError(403, "session_required", "This action requires signing in with a session")) and use it on these two routes. Tests: Bearer POST and DELETE get 403 session_required.
Same round (cheap, same files):
- Removing a member also revokes that user's API keys in this tenant (set revoked_at = now where tenant_id and user_id match and revoked_at is null) in the same transaction, so re-adding a user never revives old keys. Sessions pinned to the tenant are revoked the same way. Test: a removed member's API key stays 401 after the user is re-added.
- decode_int_cursor rejects keys longer than 19 characters before int() (400 bad_cursor). Test with a 6720-character key.
- The owner-lock query orders by user_id (`order by user_id for update`).
- When the principal is an owner, require principal.user_id to be in the locked owner set, else 403 insufficient_permission.
- Test: a member's DELETE gets 403 forbidden.
