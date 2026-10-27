"""FastAPI dependencies: principal resolution, CSRF enforcement and role checks."""

import hmac
from collections.abc import Awaitable, Callable

from fastapi import Depends, Request

from backend.api.errors import ApiError
from backend.auth.service import AuthService, Principal
from backend.auth.tokens import token_hash
from backend.db.pool import Database
from backend.obs.logging import tenant_id_var, user_id_var

SESSION_COOKIE = "eg_session"
CSRF_COOKIE = "eg_csrf"
CSRF_HEADER = "X-CSRF-Token"

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_auth(request: Request) -> AuthService:
    return request.app.state.auth


def _unauthenticated() -> ApiError:
    return ApiError(401, "unauthenticated", "Authentication required")


def _bearer_token(request: Request) -> str | None:
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def _check_csrf(request: Request, principal: Principal) -> None:
    if request.method in _SAFE_METHODS:
        return
    supplied = request.headers.get(CSRF_HEADER)
    if (
        not supplied
        or principal.csrf_hash is None
        or not hmac.compare_digest(token_hash(supplied), principal.csrf_hash)
    ):
        raise ApiError(403, "csrf_failed", "CSRF check failed")


async def current_principal(request: Request) -> Principal:
    """Resolve the caller from a Bearer API key or the session cookie.

    This must stay ``async def``: a sync dependency runs in a worker thread with a copied
    context, so the ``.set()`` calls below would not reach the request's log context.
    """
    auth = get_auth(request)
    bearer = _bearer_token(request)
    if bearer is not None:
        # A presented API key never falls back to a cookie, and carries no CSRF risk.
        principal = await auth.resolve_api_key(bearer)
        if principal is None:
            raise _unauthenticated()
    else:
        session_token = request.cookies.get(SESSION_COOKIE)
        if not session_token:
            raise _unauthenticated()
        principal = await auth.resolve_session(session_token)
        if principal is None:
            raise _unauthenticated()
        _check_csrf(request, principal)

    tenant_id_var.set(str(principal.tenant_id))
    user_id_var.set(str(principal.user_id))
    return principal


def require_role(*roles: str) -> Callable[[Principal], Awaitable[Principal]]:
    async def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if principal.role not in roles:
            raise ApiError(403, "forbidden", "Insufficient role")
        return principal

    return dependency


async def require_session(principal: Principal = Depends(current_principal)) -> Principal:
    """Reject API key callers, so a leaked key cannot grant itself durable access."""
    if principal.auth == "api_key":
        raise ApiError(403, "session_required", "This action requires signing in with a session")
    return principal
