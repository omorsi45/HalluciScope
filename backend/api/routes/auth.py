"""Session authentication routes under /api/v1/auth."""

from fastapi import APIRouter, Depends, Request, Response

from backend.api.errors import ApiError
from backend.api.schemas_v1 import (
    LoginIn,
    MeOut,
    RegisterIn,
    SessionOut,
    SwitchTenantIn,
    TenantOut,
    UserOut,
)
from backend.auth.deps import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    current_principal,
    get_auth,
    get_db,
)
from backend.auth.service import AuthService, Principal
from backend.config import Settings
from backend.db.pool import Database
from backend.db.repos import tenants

router = APIRouter()

_SECONDS_PER_DAY = 86400


def _user_out(principal: Principal) -> UserOut:
    return UserOut(id=principal.user_id, email=principal.email, display_name=principal.display_name)


async def _me(db: Database, principal: Principal) -> MeOut:
    async with db.system_tx() as conn:
        memberships = await tenants.list_memberships(conn, principal.user_id)
    tenant_list = [
        TenantOut(id=m["tenant_id"], name=m["tenant_name"], role=m["role"]) for m in memberships
    ]
    current = next((t for t in tenant_list if t.id == principal.tenant_id), None)
    if current is None:
        raise ApiError(401, "unauthenticated", "Authentication required")
    return MeOut(user=_user_out(principal), tenant=current, tenants=tenant_list)


def _set_session_cookies(
    response: Response, settings: Settings, session_token: str, csrf_token: str
) -> None:
    secure = not settings.dev_insecure_cookies
    max_age = settings.session_ttl_days * _SECONDS_PER_DAY
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        max_age=max_age,
        path="/",
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    # The SPA reads this one to echo it in the X-CSRF-Token header, so it is not HttpOnly.
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        max_age=max_age,
        path="/",
        secure=secure,
        httponly=False,
        samesite="lax",
    )


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def _open_session(
    request: Request,
    response: Response,
    email: str,
    password: str,
    db: Database,
    auth: AuthService,
) -> SessionOut:
    result = await auth.login(email, password, ip=_client_ip(request))
    _set_session_cookies(
        response, request.app.state.settings, result.session_token, result.csrf_token
    )
    me = await _me(db, result.principal)
    return SessionOut(user=me.user, tenant=me.tenant, csrf_token=result.csrf_token)


@router.post("/register", response_model=SessionOut, status_code=201)
async def register(
    body: RegisterIn,
    request: Request,
    response: Response,
    db: Database = Depends(get_db),
    auth: AuthService = Depends(get_auth),
) -> SessionOut:
    await auth.register(body.email, body.password, body.display_name)
    return await _open_session(request, response, body.email, body.password, db, auth)


@router.post("/login", response_model=SessionOut)
async def login(
    body: LoginIn,
    request: Request,
    response: Response,
    db: Database = Depends(get_db),
    auth: AuthService = Depends(get_auth),
) -> SessionOut:
    return await _open_session(request, response, body.email, body.password, db, auth)


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    principal: Principal = Depends(current_principal),
    auth: AuthService = Depends(get_auth),
) -> Response:
    session_token = request.cookies.get(SESSION_COOKIE)
    if principal.auth == "session" and session_token:
        await auth.logout(session_token)
    settings: Settings = request.app.state.settings
    response = Response(status_code=204)
    for name, httponly in ((SESSION_COOKIE, True), (CSRF_COOKIE, False)):
        response.delete_cookie(
            name,
            path="/",
            secure=not settings.dev_insecure_cookies,
            httponly=httponly,
            samesite="lax",
        )
    return response


@router.get("/me", response_model=MeOut)
async def me(
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
) -> MeOut:
    return await _me(db, principal)


@router.post("/switch-tenant", response_model=MeOut)
async def switch_tenant(
    body: SwitchTenantIn,
    principal: Principal = Depends(current_principal),
    db: Database = Depends(get_db),
    auth: AuthService = Depends(get_auth),
) -> MeOut:
    switched = await auth.switch_tenant(principal, body.tenant_id)
    return await _me(db, switched)
