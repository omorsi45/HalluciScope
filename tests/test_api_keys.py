from uuid import uuid4

import httpx
import pytest

from backend.db.repos import tenants
from tests.helpers.auth import csrf, signup

pytestmark = pytest.mark.pg

KEYS = "/api/v1/api-keys"
ME = "/api/v1/auth/me"


async def _create(client, session, name="ci"):
    response = await client.post(KEYS, json={"name": name}, headers=csrf(session))
    assert response.status_code == 201, response.text
    return response.json()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _audit(db, action):
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select tenant_id, user_id, target_type, target_id, details::text as details "
            "from audit_log where action = %s",
            (action,),
        )
        return await cur.fetchall()


async def test_create_returns_token_once(client):
    session = await signup(client)

    created = await _create(client, session, name="ci")
    listed = await client.get(KEYS)

    assert created["token"].startswith(f"egk_{created['prefix']}_")
    assert created["name"] == "ci"
    assert created["last_used_at"] is None
    assert listed.status_code == 200
    items = listed.json()
    assert [item["id"] for item in items] == [created["id"]]
    assert all("token" not in item for item in items)


async def test_bearer_key_authenticates_me(client, api_app):
    session = await signup(client)
    created = await _create(client, session)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://test"
    ) as bearer_client:
        response = await bearer_client.get(ME, headers=_bearer(created["token"]))

    assert response.status_code == 200
    assert response.json()["user"]["email"] == "alice@example.com"


async def test_revoked_key_rejected(client, client2):
    session = await signup(client)
    created = await _create(client, session)
    assert (await client2.get(ME, headers=_bearer(created["token"]))).status_code == 200

    deleted = await client.delete(f"{KEYS}/{created['id']}", headers=csrf(session))

    assert deleted.status_code == 204
    after = await client2.get(ME, headers=_bearer(created["token"]))
    assert after.status_code == 401
    assert (await client.get(KEYS)).json() == []


async def test_member_cannot_revoke_other_users_key(client, client2, db):
    alice = await signup(client)
    created = await _create(client, alice)
    bob = await signup(client2, email="bob@example.com", display_name="Bob")
    async with db.system_tx() as conn:
        await tenants.add_membership(conn, alice["tenant"]["id"], bob["user"]["id"], "member")
    switched = await client2.post(
        "/api/v1/auth/switch-tenant",
        json={"tenant_id": alice["tenant"]["id"]},
        headers=csrf(bob),
    )
    assert switched.status_code == 200
    assert switched.json()["tenant"]["role"] == "member"

    response = await client2.delete(f"{KEYS}/{created['id']}", headers=csrf(bob))

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    still = await client.get(ME, headers=_bearer(created["token"]))
    assert still.status_code == 200
    assert [item["id"] for item in (await client.get(KEYS)).json()] == [created["id"]]


async def test_revoke_unknown_key_is_not_found(client):
    session = await signup(client)

    response = await client.delete(f"{KEYS}/{uuid4()}", headers=csrf(session))

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


@pytest.mark.parametrize("name", ["", "   ", "x" * 101])
async def test_key_name_validation(client, name):
    session = await signup(client)

    response = await client.post(KEYS, json={"name": name}, headers=csrf(session))

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert (await client.get(KEYS)).json() == []


async def test_key_name_is_stripped(client):
    session = await signup(client)

    created = await _create(client, session, name="  deploy  ")

    assert created["name"] == "deploy"


async def test_create_and_revoke_write_audit_without_secrets(client, db):
    session = await signup(client)
    created = await _create(client, session)
    await client.delete(f"{KEYS}/{created['id']}", headers=csrf(session))

    made = await _audit(db, "api_key_created")
    revoked = await _audit(db, "api_key_revoked")

    for rows in (made, revoked):
        assert len(rows) == 1
        row = rows[0]
        assert str(row["tenant_id"]) == session["tenant"]["id"]
        assert str(row["user_id"]) == session["user"]["id"]
        assert row["target_type"] == "api_key"
        assert row["target_id"] == created["id"]
        assert created["token"] not in (row["details"] or "")


async def test_keys_require_authentication(client):
    assert (await client.get(KEYS)).status_code == 401
