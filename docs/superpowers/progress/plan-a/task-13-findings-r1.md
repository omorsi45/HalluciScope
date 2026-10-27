# Task 13 open findings (round 1), from the review, with controller ruling

1. (Important) backend/api/routes/api_keys.py:22-39 and backend/auth/service.py:299, 319: Bearer API keys can manage API keys. A leaked key can mint a replacement that survives revoking the leaked one, and key count is uncapped.

Controller ruling (binding): creating and revoking API keys requires a session principal. POST /api-keys and DELETE /api-keys/{id} with an api_key principal return 403 with code `session_required` (title "This action requires signing in with a session"). Enforce it in AuthService.create_api_key and revoke_api_key (raise AuthError("session_required", 403)) so every caller is covered, map the code in the errors title dict. GET stays allowed for API keys.
Same round (cheap, same files):
- Revoke permission check uses an allow-list: owners and admins may revoke any key in the tenant; anyone else only their own.
- Tests: Bearer POST and DELETE get 403 session_required; cookie POST and DELETE without X-CSRF-Token get 403 csrf_failed; revoking another tenant's key gets 404 not_found and the key still resolves; an owner revokes a member's key (204); the "key still works" check uses a cookie-less client; assert audit details is None; a 100-character name is accepted.
