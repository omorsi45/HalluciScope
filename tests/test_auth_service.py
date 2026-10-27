import asyncio
import uuid
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from backend.api.errors import install_error_handlers
from backend.auth import service as auth_service
from backend.auth.service import AuthError, AuthService, Principal
from backend.auth.tokens import token_hash
from backend.config import Settings
from backend.db.pool import Database
from backend.db.repos import users

pytestmark = pytest.mark.pg

PASSWORD = "correct horse battery"


class Clock:
    """Mutable fake clock; tests advance `now` to move time."""

    def __init__(self) -> None:
        self.now = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def auth(db, clock):
    return AuthService(db, Settings(), clock=clock)


async def _fetch(db, query, params=()):
    async with db.system_tx() as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def _register(auth, email="alice@example.com", name="Alice"):
    return await auth.register(email, PASSWORD, name)


async def _expect_code(awaitable, code, status):
    with pytest.raises(AuthError) as info:
        await awaitable
    assert (info.value.code, info.value.status) == (code, status)


async def test_register_creates_owner_tenant(auth, db):
    user_id, tenant_id = await _register(auth)

    rows = await _fetch(
        db,
        "select t.name, m.role, u.password_hash from memberships m "
        "join tenants t on t.id = m.tenant_id join users u on u.id = m.user_id "
        "where m.user_id = %s and m.tenant_id = %s",
        (user_id, tenant_id),
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice's workspace"
    assert rows[0]["role"] == "owner"
    assert rows[0]["password_hash"].startswith("scrypt$")
    assert PASSWORD not in rows[0]["password_hash"]


async def test_email_normalized_and_unique(auth):
    await _register(auth, " Alice@Example.com ")

    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")
    assert result.principal.email == "alice@example.com"

    await _expect_code(_register(auth, "ALICE@example.com", "Other"), "email_taken", 409)


@pytest.mark.parametrize("email", ["", "no-at-sign", "a@b@c", "has space@example.com", "x@"])
async def test_invalid_email_rejected(auth, email):
    await _expect_code(auth.register(email, PASSWORD, "Alice"), "invalid_email", 422)


async def test_weak_password_rejected(auth):
    await _expect_code(auth.register("a@example.com", "short", "Alice"), "weak_password", 422)


async def test_registration_disabled(db, clock):
    service = AuthService(db, Settings(allow_registration=False), clock=clock)
    await _expect_code(
        service.register("a@example.com", PASSWORD, "Alice"), "registration_disabled", 403
    )


async def test_only_token_hashes_stored(auth, db):
    await _register(auth)
    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")

    rows = await _fetch(db, "select token_hash, csrf_hash from sessions")
    assert len(rows) == 1
    assert bytes(rows[0]["token_hash"]) == token_hash(result.session_token)
    assert bytes(rows[0]["csrf_hash"]) == token_hash(result.csrf_token)


async def test_wrong_password_and_unknown_email_same_error(auth):
    await _register(auth)

    await _expect_code(
        auth.login("alice@example.com", "wrong password!!", ip="1.1.1.1"),
        "invalid_credentials",
        401,
    )
    await _expect_code(
        auth.login("nobody@example.com", PASSWORD, ip="1.1.1.1"), "invalid_credentials", 401
    )


async def test_email_throttle_and_window_reset(auth, clock):
    await _register(auth)
    for _ in range(5):
        await _expect_code(
            auth.login("alice@example.com", "wrong password!!", ip="1.1.1.1"),
            "invalid_credentials",
            401,
        )

    await _expect_code(
        auth.login("alice@example.com", PASSWORD, ip="1.1.1.1"), "too_many_attempts", 429
    )

    clock.now += timedelta(minutes=16)
    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")
    assert result.principal.email == "alice@example.com"


async def test_ip_throttle_across_emails(auth):
    await _register(auth)
    for i in range(20):
        await _expect_code(
            auth.login(f"user{i}@example.com", PASSWORD, ip="9.9.9.9"),
            "invalid_credentials",
            401,
        )

    await _expect_code(
        auth.login("alice@example.com", PASSWORD, ip="9.9.9.9"), "too_many_attempts", 429
    )
    result = await auth.login("alice@example.com", PASSWORD, ip="8.8.8.8")
    assert result.principal.email == "alice@example.com"


async def test_successful_login_clears_email_failures(auth):
    await _register(auth)
    for _ in range(4):
        await _expect_code(
            auth.login("alice@example.com", "wrong password!!", ip="1.1.1.1"),
            "invalid_credentials",
            401,
        )
    await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")

    for _ in range(4):
        await _expect_code(
            auth.login("alice@example.com", "wrong password!!", ip="1.1.1.1"),
            "invalid_credentials",
            401,
        )
    await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")


async def test_login_resolves_principal_with_oldest_tenant(auth):
    _, tenant_id = await _register(auth)

    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")
    principal = await auth.resolve_session(result.session_token)

    assert principal is not None
    assert principal.tenant_id == tenant_id
    assert principal.role == "owner"
    assert principal.auth == "session"
    assert principal.csrf_hash == token_hash(result.csrf_token)
    assert principal.display_name == "Alice"


async def test_logout_revokes(auth):
    await _register(auth)
    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")
    assert await auth.resolve_session(result.session_token) is not None

    await auth.logout(result.session_token)

    assert await auth.resolve_session(result.session_token) is None


async def test_expired_session_rejected(auth, clock):
    await _register(auth)
    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")

    clock.now += timedelta(days=6)
    assert await auth.resolve_session(result.session_token) is not None
    clock.now += timedelta(days=2)
    assert await auth.resolve_session(result.session_token) is None


async def test_unknown_session_token_rejected(auth):
    assert await auth.resolve_session("not-a-real-token") is None


async def test_removed_membership_invalidates_session(auth, db):
    user_id, tenant_id = await _register(auth)
    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")

    async with db.system_tx() as conn:
        await conn.execute(
            "delete from memberships where user_id = %s and tenant_id = %s", (user_id, tenant_id)
        )

    assert await auth.resolve_session(result.session_token) is None


async def test_removed_membership_invalidates_api_key(auth, db):
    user_id, tenant_id = await _register(auth)
    principal = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal
    _, token = await auth.create_api_key(principal, "ci")

    async with db.system_tx() as conn:
        await conn.execute(
            "delete from memberships where user_id = %s and tenant_id = %s", (user_id, tenant_id)
        )

    assert await auth.resolve_api_key(token) is None


async def test_api_key_resolve_touch_and_revoke(auth, db):
    await _register(auth)
    principal = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal

    info, token = await auth.create_api_key(principal, "ci")
    assert "key_hash" not in info
    assert info["name"] == "ci"
    assert token.startswith(f"egk_{info['prefix']}_")
    stored = await _fetch(db, "select key_hash, last_used_at from api_keys")
    assert bytes(stored[0]["key_hash"]) == token_hash(token)
    assert stored[0]["last_used_at"] is None

    resolved = await auth.resolve_api_key(token)
    assert resolved is not None
    assert resolved.auth == "api_key"
    assert resolved.api_key_id == info["id"]
    assert resolved.tenant_id == principal.tenant_id
    assert resolved.role == "owner"
    assert resolved.session_id is None
    assert (await _fetch(db, "select last_used_at from api_keys"))[0]["last_used_at"] is not None

    assert [k["id"] for k in await auth.list_api_keys(principal)] == [info["id"]]

    await auth.revoke_api_key(principal, info["id"])
    assert await auth.resolve_api_key(token) is None


async def test_api_key_principal_cannot_create_or_revoke_keys(auth):
    await _register(auth)
    session_principal = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal
    info, token = await auth.create_api_key(session_principal, "ci")
    key_principal = await auth.resolve_api_key(token)
    assert key_principal is not None

    await _expect_code(auth.create_api_key(key_principal, "minted"), "session_required", 403)
    await _expect_code(auth.revoke_api_key(key_principal, info["id"]), "session_required", 403)

    assert [k["id"] for k in await auth.list_api_keys(session_principal)] == [info["id"]]
    assert await auth.resolve_api_key(token) is not None


async def test_malformed_api_key_rejected(auth):
    assert await auth.resolve_api_key("not-an-api-key") is None


async def _add_member(db, tenant_id, email, role):
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "insert into users (email, password_hash, display_name) "
            "values (%s, 'x', %s) returning id",
            (email, email),
        )
        user_id = (await cur.fetchone())["id"]
        await conn.execute(
            "insert into memberships (tenant_id, user_id, role) values (%s, %s, %s)",
            (tenant_id, user_id, role),
        )
    return user_id


async def test_member_cannot_revoke_others_key(auth, db):
    _, tenant_id = await _register(auth)
    owner = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal
    info, token = await auth.create_api_key(owner, "owners")

    await _add_member(db, tenant_id, "member@example.com", "member")
    # The member has no real password, so build the principal directly.
    member = Principal(
        user_id=await _user_id(db, "member@example.com"),
        email="member@example.com",
        display_name="member@example.com",
        tenant_id=tenant_id,
        role="member",
        auth="session",
    )

    await _expect_code(auth.revoke_api_key(member, info["id"]), "not_found", 404)
    assert await auth.resolve_api_key(token) is not None

    own, own_token = await auth.create_api_key(member, "mine")
    await auth.revoke_api_key(member, own["id"])
    assert await auth.resolve_api_key(own_token) is None

    # Owners and admins revoke any key in the tenant.
    other, other_token = await auth.create_api_key(member, "other")
    await auth.revoke_api_key(owner, other["id"])
    assert await auth.resolve_api_key(other_token) is None


async def _user_id(db, email):
    return (await _fetch(db, "select id from users where email = %s", (email,)))[0]["id"]


async def test_revoke_unknown_key_not_found(auth):
    await _register(auth)
    principal = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal
    await _expect_code(auth.revoke_api_key(principal, uuid.uuid4()), "not_found", 404)


async def test_switch_tenant(auth, db):
    user_id, first_tenant = await _register(auth)
    other = await _register(auth, "bob@example.com", "Bob")
    other_tenant = other[1]
    async with db.system_tx() as conn:
        await conn.execute(
            "insert into memberships (tenant_id, user_id, role) values (%s, %s, 'member')",
            (other_tenant, user_id),
        )
    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")
    assert result.principal.tenant_id == first_tenant

    switched = await auth.switch_tenant(result.principal, other_tenant)

    assert switched.tenant_id == other_tenant
    assert switched.role == "member"
    resolved = await auth.resolve_session(result.session_token)
    assert resolved is not None
    assert resolved.tenant_id == other_tenant


async def test_switch_to_foreign_tenant_rejected(auth):
    await _register(auth)
    _, bob_tenant = await _register(auth, "bob@example.com", "Bob")
    principal = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal

    await _expect_code(auth.switch_tenant(principal, bob_tenant), "not_a_member", 404)


async def test_api_key_cannot_switch_tenant(auth):
    await _register(auth)
    principal = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal
    _, token = await auth.create_api_key(principal, "ci")
    key_principal = await auth.resolve_api_key(token)
    assert key_principal is not None

    await _expect_code(
        auth.switch_tenant(key_principal, principal.tenant_id), "api_key_tenant_fixed", 403
    )


async def test_audit_rows_have_no_secrets(auth, db):
    await _register(auth)
    await _expect_code(
        auth.login("alice@example.com", "wrong password!!", ip="1.1.1.1"),
        "invalid_credentials",
        401,
    )
    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")
    for _ in range(5):
        await _expect_code(
            auth.login("alice@example.com", "wrong password!!", ip="1.1.1.1"),
            "invalid_credentials",
            401,
        )
    await _expect_code(
        auth.login("alice@example.com", PASSWORD, ip="1.1.1.1"), "too_many_attempts", 429
    )

    rows = await _fetch(db, "select action, details::text as details from audit_log")
    actions = {row["action"] for row in rows}
    assert {"register", "login_failed", "login", "login_throttled"} <= actions
    for row in rows:
        text = row["details"] or ""
        assert PASSWORD not in text
        assert "wrong password" not in text
        assert result.session_token not in text
        assert result.csrf_token not in text


async def test_failed_login_audit_survives_the_error(auth, db):
    await _register(auth)
    await _expect_code(
        auth.login("alice@example.com", "wrong password!!", ip="1.1.1.1"),
        "invalid_credentials",
        401,
    )

    rows = await _fetch(db, "select action from audit_log where action = 'login_failed'")
    assert len(rows) == 1


async def test_auth_error_maps_to_problem_response():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom():
        raise AuthError("too_many_attempts", 429)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["code"] == "too_many_attempts"
    assert body["status"] == 429
    assert body["title"] == "Too many attempts"


@pytest_asyncio.fixture
async def wide_db(pg_urls, db):
    """Ten-connection Database on the same database; `db` empties the tables afterwards."""
    database = Database(pg_urls["app"], max_size=10)
    await database.open()
    try:
        yield database
    finally:
        await database.close()


async def test_concurrent_wrong_passwords_respect_email_limit(auth, wide_db, clock):
    await _register(auth)
    concurrent = AuthService(wide_db, Settings(), clock=clock)

    results = await asyncio.gather(
        *(
            concurrent.login("alice@example.com", "wrong password!!", ip="1.1.1.1")
            for _ in range(20)
        ),
        return_exceptions=True,
    )

    codes = [r.code for r in results if isinstance(r, AuthError)]
    assert len(codes) == 20
    assert codes.count("invalid_credentials") == 5
    assert codes.count("too_many_attempts") == 15
    rows = await _fetch(
        wide_db, "select failures from auth_throttle where key = 'email:alice@example.com'"
    )
    assert [r["failures"] for r in rows] == [5]


async def _all_connections_free(database, size):
    """True when `size` connections can be held at once, i.e. none is held elsewhere."""
    try:
        async with AsyncExitStack() as stack:
            for _ in range(size):
                await asyncio.wait_for(stack.enter_async_context(database.system_tx()), 2)
    except TimeoutError:
        return False
    return True


@pytest.mark.parametrize("known_email", [True, False])
async def test_no_connection_held_during_password_verification(
    monkeypatch, pg_urls, db, clock, known_email
):
    small = Database(pg_urls["app"], max_size=3)
    await small.open()
    try:
        service = AuthService(small, Settings(), clock=clock)
        await _register(service)
        loop = asyncio.get_running_loop()
        observed: list[bool] = []

        def stub(*_args):
            future = asyncio.run_coroutine_threadsafe(_all_connections_free(small, 3), loop)
            observed.append(future.result(timeout=15))
            return False

        monkeypatch.setattr(auth_service, "verify_password", stub)
        monkeypatch.setattr(auth_service, "verify_dummy", stub)

        email = "alice@example.com" if known_email else "nobody@example.com"
        await _expect_code(
            service.login(email, "wrong password!!", ip="1.1.1.1"), "invalid_credentials", 401
        )

        assert observed == [True]
    finally:
        await small.close()


async def test_ip_throttled_attempt_does_not_count_against_email(auth):
    await _register(auth)
    for i in range(20):
        await _expect_code(
            auth.login(f"user{i}@example.com", PASSWORD, ip="9.9.9.9"),
            "invalid_credentials",
            401,
        )

    await _expect_code(
        auth.login("alice@example.com", "wrong password!!", ip="9.9.9.9"),
        "too_many_attempts",
        429,
    )

    for _ in range(5):
        await _expect_code(
            auth.login("alice@example.com", "wrong password!!", ip="8.8.8.8"),
            "invalid_credentials",
            401,
        )


async def test_successful_login_does_not_consume_ip_budget(auth):
    await _register(auth)
    for _ in range(25):
        await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")

    for i in range(20):
        await _expect_code(
            auth.login(f"user{i}@example.com", PASSWORD, ip="1.1.1.1"),
            "invalid_credentials",
            401,
        )


async def test_disabled_user_rejected_everywhere(auth, db):
    user_id, _ = await _register(auth)
    result = await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")
    _, token = await auth.create_api_key(result.principal, "ci")

    async with db.system_tx() as conn:
        await conn.execute("update users set disabled_at = now() where id = %s", (user_id,))

    assert await auth.resolve_session(result.session_token) is None
    assert await auth.resolve_api_key(token) is None
    await _expect_code(
        auth.login("alice@example.com", PASSWORD, ip="1.1.1.1"), "invalid_credentials", 401
    )


async def test_forged_api_key_with_matching_prefix_rejected(auth):
    await _register(auth)
    principal = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal
    info, token = await auth.create_api_key(principal, "ci")
    forged = f"egk_{info['prefix']}_" + "A" * 43

    assert forged != token
    assert await auth.resolve_api_key(forged) is None
    assert await auth.resolve_api_key(token) is not None


async def test_owner_of_other_tenant_cannot_revoke_key(auth):
    await _register(auth)
    alice = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal
    info, token = await auth.create_api_key(alice, "ci")
    await _register(auth, "bob@example.com", "Bob")
    bob = (await auth.login("bob@example.com", PASSWORD, ip="1.1.1.1")).principal
    assert bob.role == "owner"
    assert bob.tenant_id != alice.tenant_id

    await _expect_code(auth.revoke_api_key(bob, info["id"]), "not_found", 404)
    assert await auth.resolve_api_key(token) is not None


async def test_list_api_keys_returns_only_active_keys(auth):
    await _register(auth)
    principal = (await auth.login("alice@example.com", PASSWORD, ip="1.1.1.1")).principal
    kept, _ = await auth.create_api_key(principal, "kept")
    revoked, _ = await auth.create_api_key(principal, "revoked")

    await auth.revoke_api_key(principal, revoked["id"])

    assert [k["id"] for k in await auth.list_api_keys(principal)] == [kept["id"]]


async def test_only_login_lookup_returns_password_hash(auth, db):
    user_id, _ = await _register(auth)

    async with db.system_tx() as conn:
        by_email = await users.get_user_by_email(conn, "alice@example.com")
        by_id = await users.get_user(conn, user_id)
        record = await users.get_login_record(conn, "alice@example.com")

    assert by_email is not None and by_id is not None and record is not None
    assert "password_hash" not in by_email
    assert "password_hash" not in by_id
    assert record["password_hash"].startswith("scrypt$")
    assert record["id"] == user_id


async def test_login_with_oversized_email_skips_the_throttle_table(auth, db):
    await _expect_code(
        auth.login("a" * 4096 + "@example.com", PASSWORD, ip="1.1.1.1"), "invalid_credentials", 401
    )

    assert await _fetch(db, "select 1 from auth_throttle where left(key, 6) = 'email:'") == []
