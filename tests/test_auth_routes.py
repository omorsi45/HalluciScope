import asyncio
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import Depends

from backend.api.app import create_app
from backend.api.errors import ApiError
from backend.auth.deps import current_principal, require_role
from backend.auth.service import AuthService, Principal
from backend.config import Settings
from backend.obs.logging import tenant_id_var, user_id_var
from tests.helpers.auth import PASSWORD, csrf, signup

pytestmark = pytest.mark.pg

AUTH = "/api/v1/auth"


def _set_cookies(response: httpx.Response) -> dict[str, str]:
    return {header.split("=", 1)[0]: header for header in response.headers.get_list("set-cookie")}


async def test_register_sets_session_and_me_works(client):
    session = await signup(client)

    assert session["user"]["email"] == "alice@example.com"
    assert session["tenant"]["role"] == "owner"
    assert session["csrf_token"]
    me = await client.get(f"{AUTH}/me")
    assert me.status_code == 200
    body = me.json()
    assert body["user"] == session["user"]
    assert body["tenant"] == session["tenant"]
    assert [t["id"] for t in body["tenants"]] == [session["tenant"]["id"]]
    assert body["tenants"][0]["name"] == "Alice's workspace"


async def test_me_without_credentials_is_unauthenticated(client):
    response = await client.get(f"{AUTH}/me")

    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"


async def test_login_wrong_password_problem(client):
    await signup(client)

    response = await client.post(
        f"{AUTH}/login", json={"email": "alice@example.com", "password": "not the password"}
    )

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "invalid_credentials"


async def test_login_returns_session(client, client2):
    await signup(client)

    response = await client2.post(
        f"{AUTH}/login", json={"email": "alice@example.com", "password": PASSWORD}
    )

    assert response.status_code == 200
    assert response.json()["csrf_token"]
    assert (await client2.get(f"{AUTH}/me")).status_code == 200


async def test_unsafe_request_without_csrf_rejected(client):
    session = await signup(client)
    payload = {"tenant_id": session["tenant"]["id"]}

    rejected = await client.post(f"{AUTH}/switch-tenant", json=payload)
    wrong = await client.post(
        f"{AUTH}/switch-tenant", json=payload, headers={"X-CSRF-Token": "wrong"}
    )
    accepted = await client.post(f"{AUTH}/switch-tenant", json=payload, headers=csrf(session))

    assert rejected.status_code == 403
    assert rejected.json()["code"] == "csrf_failed"
    assert wrong.status_code == 403
    assert wrong.json()["code"] == "csrf_failed"
    assert accepted.status_code == 200
    assert accepted.json()["tenant"]["id"] == session["tenant"]["id"]


async def test_logout_then_me_unauthenticated(client):
    session = await signup(client)

    response = await client.post(f"{AUTH}/logout", headers=csrf(session))

    assert response.status_code == 204
    cookies = _set_cookies(response)
    assert "max-age=0" in cookies["eg_session"].lower()
    assert "max-age=0" in cookies["eg_csrf"].lower()
    assert (await client.get(f"{AUTH}/me")).status_code == 401


async def test_logout_revokes_session_server_side(client, client2):
    session = await signup(client)
    stolen = client.cookies["eg_session"]

    await client.post(f"{AUTH}/logout", headers=csrf(session))
    replay = await client2.get(f"{AUTH}/me", headers={"Cookie": f"eg_session={stolen}"})

    assert replay.status_code == 401


async def test_cookie_flags_dev_insecure(client):
    response = await client.post(
        f"{AUTH}/register",
        json={"email": "a@example.com", "password": PASSWORD, "display_name": "A"},
    )

    cookies = _set_cookies(response)
    session_cookie = cookies["eg_session"].lower()
    assert "httponly" in session_cookie
    assert "samesite=lax" in session_cookie
    assert "path=/" in session_cookie
    assert "max-age=604800" in session_cookie
    assert "secure" not in session_cookie
    csrf_cookie = cookies["eg_csrf"].lower()
    assert "httponly" not in csrf_cookie
    assert "samesite=lax" in csrf_cookie
    assert "secure" not in csrf_cookie


async def test_cookie_flags_secure_by_default(db):
    settings = Settings(dev_insecure_cookies=False)
    app = create_app(settings)
    app.state.db = db
    app.state.auth = AuthService(db, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as secure_client:
        response = await secure_client.post(
            f"{AUTH}/register",
            json={"email": "a@example.com", "password": PASSWORD, "display_name": "A"},
        )

    assert response.status_code == 201
    cookies = _set_cookies(response)
    assert "secure" in cookies["eg_session"].lower().replace("; ", ";").split(";")
    assert "secure" in cookies["eg_csrf"].lower().replace("; ", ";").split(";")


async def test_registration_disabled(db):
    settings = Settings(dev_insecure_cookies=True, allow_registration=False)
    app = create_app(settings)
    app.state.db = db
    app.state.auth = AuthService(db, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as closed_client:
        response = await closed_client.post(
            f"{AUTH}/register",
            json={"email": "a@example.com", "password": PASSWORD, "display_name": "A"},
        )

    assert response.status_code == 403
    assert response.json()["code"] == "registration_disabled"


async def test_switch_to_non_member_tenant_404(client, client2):
    session = await signup(client)
    other = await signup(client2, email="bob@example.com", display_name="Bob")

    response = await client.post(
        f"{AUTH}/switch-tenant",
        json={"tenant_id": other["tenant"]["id"]},
        headers=csrf(session),
    )

    assert response.status_code == 404
    assert response.json()["code"] == "not_a_member"


async def test_bearer_api_key_needs_no_csrf(client, client2, api_app):
    session = await signup(client)
    principal = Principal(
        user_id=UUID(session["user"]["id"]),
        email=session["user"]["email"],
        display_name=session["user"]["display_name"],
        tenant_id=UUID(session["tenant"]["id"]),
        role="owner",
        auth="session",
    )
    _, token = await api_app.state.auth.create_api_key(principal, "ci")

    me = await client2.get(f"{AUTH}/me", headers={"Authorization": f"Bearer {token}"})
    switched = await client2.post(
        f"{AUTH}/switch-tenant",
        json={"tenant_id": session["tenant"]["id"]},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert me.status_code == 200
    assert me.json()["tenant"]["id"] == session["tenant"]["id"]
    assert switched.status_code == 403
    assert switched.json()["code"] == "api_key_tenant_fixed"


async def test_invalid_bearer_does_not_fall_back_to_cookie(client):
    await signup(client)

    response = await client.get(f"{AUTH}/me", headers={"Authorization": "Bearer egk_bad"})

    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"


async def test_login_throttled_route(client):
    await signup(client)
    body = {"email": "alice@example.com", "password": "wrong password here"}

    statuses = [(await client.post(f"{AUTH}/login", json=body)).status_code for _ in range(5)]
    sixth = await client.post(f"{AUTH}/login", json=body)

    assert statuses == [401] * 5
    assert sixth.status_code == 429
    assert sixth.json()["code"] == "too_many_attempts"


async def test_require_role_allows_listed_and_rejects_others():
    dependency = require_role("owner", "admin")

    def principal(role: str) -> Principal:
        return Principal(uuid4(), "a@example.com", "A", uuid4(), role, "session")

    assert (await dependency(principal("admin"))).role == "admin"
    with pytest.raises(ApiError) as info:
        await dependency(principal("member"))
    assert (info.value.status, info.value.code) == (403, "forbidden")


async def test_principal_context_vars_are_per_request(client, api_app):
    @api_app.get("/api/v1/_whoami")
    async def whoami(_: Principal = Depends(current_principal)):
        return {"tenant": tenant_id_var.get(), "user": user_id_var.get()}

    @api_app.get("/api/v1/_ctx")
    async def ctx():
        return {"tenant": tenant_id_var.get(), "user": user_id_var.get()}

    session = await signup(client)

    # A server runs each request in its own task (own context); model that with a task.
    authed = await asyncio.create_task(client.get("/api/v1/_whoami"))
    anonymous = await asyncio.create_task(client.get("/api/v1/_ctx"))

    assert authed.json() == {"tenant": session["tenant"]["id"], "user": session["user"]["id"]}
    assert anonymous.json() == {"tenant": None, "user": None}
    assert tenant_id_var.get() is None
    assert user_id_var.get() is None


async def test_login_rejects_oversized_email_with_422(client):
    response = await client.post(
        f"{AUTH}/login", json={"email": "a" * 4096 + "@example.com", "password": PASSWORD}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_login_email_over_320_characters_is_422(client):
    email = "a" * 309 + "@example.com"  # 321 characters
    assert len(email) == 321

    response = await client.post(f"{AUTH}/login", json={"email": email, "password": PASSWORD})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_register_rejects_lone_surrogate_password_with_422(client):
    # Raw bytes: a Python client cannot serialize a lone surrogate, a JSON escape can.
    raw = b'{"email": "a@example.com", "password": "\\ud800 correct horse", "display_name": "A"}'

    response = await client.post(
        f"{AUTH}/register", content=raw, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_register_rejects_oversized_password_and_display_name(client):
    base = {"email": "a@example.com", "password": PASSWORD, "display_name": "A"}

    long_password = await client.post(f"{AUTH}/register", json={**base, "password": "p" * 257})
    long_name = await client.post(f"{AUTH}/register", json={**base, "display_name": "n" * 101})

    assert long_password.status_code == 422
    assert long_name.status_code == 422
