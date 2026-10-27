from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest

from backend.api.pagination import encode_cursor
from backend.db.repos import members, tenants, users
from tests.helpers.auth import PASSWORD, csrf, signup

pytestmark = pytest.mark.pg

BASE = "/api/v1/tenants/current"
COLLECTIONS = "/api/v1/collections"
KEYS = "/api/v1/api-keys"
ME = "/api/v1/auth/me"


async def _users(client, client2, client3):
    """Alice, Bob and Carol, each owning their own tenant."""
    alice = await signup(client)
    bob = await signup(client2, email="bob@example.com", display_name="Bob")
    carol = await signup(client3, email="carol@example.com", display_name="Carol")
    return alice, bob, carol


async def _join(db, client, owner_session, session, role):
    """Add the user of `session` to the owner's tenant and switch the client to it."""
    async with db.system_tx() as conn:
        await tenants.add_membership(
            conn, owner_session["tenant"]["id"], session["user"]["id"], role
        )
    switched = await client.post(
        "/api/v1/auth/switch-tenant",
        json={"tenant_id": owner_session["tenant"]["id"]},
        headers=csrf(session),
    )
    assert switched.status_code == 200, switched.text
    return {**session, "tenant": switched.json()["tenant"]}


async def _add(client, session, email, role="member"):
    return await client.post(
        f"{BASE}/members", json={"email": email, "role": role}, headers=csrf(session)
    )


async def test_owner_adds_and_lists_member(client, client2, client3):
    alice, bob, _carol = await _users(client, client2, client3)

    added = await _add(client, alice, "  BOB@Example.com ", "admin")
    listed = await client.get(f"{BASE}/members")

    assert added.status_code == 201, added.text
    assert added.json()["user_id"] == bob["user"]["id"]
    assert added.json()["email"] == "bob@example.com"
    assert added.json()["display_name"] == "Bob"
    assert added.json()["role"] == "admin"
    assert "created_at" in added.json()
    assert listed.status_code == 200
    assert {(m["email"], m["role"]) for m in listed.json()} == {
        ("alice@example.com", "owner"),
        ("bob@example.com", "admin"),
    }


async def test_admin_cannot_add_owner(client, client2, client3, db):
    alice, bob, _carol = await _users(client, client2, client3)
    bob = await _join(db, client2, alice, bob, "admin")

    response = await _add(client2, bob, "carol@example.com", "owner")
    listed = await client.get(f"{BASE}/members")

    assert response.status_code == 403
    assert response.json()["code"] == "insufficient_permission"
    assert {m["email"] for m in listed.json()} == {"alice@example.com", "bob@example.com"}


async def test_member_cannot_add(client, client2, client3, db):
    alice, bob, _carol = await _users(client, client2, client3)
    bob = await _join(db, client2, alice, bob, "member")

    response = await _add(client2, bob, "carol@example.com")

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


async def test_unknown_email_404(client):
    alice = await signup(client)

    response = await _add(client, alice, "nobody@example.com")

    assert response.status_code == 404
    assert response.json()["code"] == "user_not_found"


async def test_already_member_409(client, client2, client3):
    alice, _bob, _carol = await _users(client, client2, client3)

    first = await _add(client, alice, "bob@example.com")
    second = await _add(client, alice, "bob@example.com")
    owner_again = await _add(client, alice, "alice@example.com")

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["code"] == "already_member"
    assert owner_again.status_code == 409
    assert owner_again.json()["code"] == "already_member"


async def test_last_owner_cannot_be_removed(client, client2, client3):
    alice, _bob, _carol = await _users(client, client2, client3)
    await _add(client, alice, "bob@example.com", "admin")

    response = await client.delete(
        f"{BASE}/members/{alice['user']['id']}", headers=csrf(alice)
    )
    listed = await client.get(f"{BASE}/members")

    assert response.status_code == 409
    assert response.json()["code"] == "last_owner"
    assert "alice@example.com" in {m["email"] for m in listed.json()}


async def test_owner_can_be_removed_when_another_owner_exists(client, client2, client3, db):
    alice, bob, _carol = await _users(client, client2, client3)
    bob = await _join(db, client2, alice, bob, "owner")

    response = await client.delete(
        f"{BASE}/members/{alice['user']['id']}", headers=csrf(alice)
    )
    listed = await client2.get(f"{BASE}/members")

    assert response.status_code == 204
    assert response.content == b""
    assert [m["email"] for m in listed.json()] == ["bob@example.com"]


async def test_admin_cannot_remove_owner(client, client2, client3, db):
    alice, bob, _carol = await _users(client, client2, client3)
    bob = await _join(db, client2, alice, bob, "admin")

    response = await client2.delete(
        f"{BASE}/members/{alice['user']['id']}", headers=csrf(bob)
    )

    assert response.status_code == 403
    assert response.json()["code"] == "insufficient_permission"


async def test_remove_non_member_404(client, client2, client3):
    alice, bob, _carol = await _users(client, client2, client3)

    response = await client.delete(f"{BASE}/members/{bob['user']['id']}", headers=csrf(alice))

    assert response.status_code == 404
    assert response.json()["code"] == "member_not_found"


async def test_removed_member_loses_session_and_grants(client, client2, client3, db):
    alice, bob, _carol = await _users(client, client2, client3)
    bob = await _join(db, client2, alice, bob, "member")
    created = await client.post(COLLECTIONS, json={"name": "Reports"}, headers=csrf(alice))
    collection_id = created.json()["id"]
    granted = await client.put(
        f"{COLLECTIONS}/{collection_id}/grants",
        json={
            "grants": [
                {"user_id": alice["user"]["id"], "permission": "admin"},
                {"user_id": bob["user"]["id"], "permission": "read"},
            ]
        },
        headers=csrf(alice),
    )
    assert granted.status_code == 200, granted.text
    assert (await client2.get("/api/v1/auth/me")).status_code == 200

    removed = await client.delete(f"{BASE}/members/{bob['user']['id']}", headers=csrf(alice))

    assert removed.status_code == 204
    assert (await client2.get("/api/v1/auth/me")).status_code == 401
    async with db.tenant_tx(UUID(alice["tenant"]["id"])) as conn:
        cur = await conn.execute(
            "select user_id from collection_grants where collection_id = %s", (collection_id,)
        )
        remaining = [str(row["user_id"]) for row in await cur.fetchall()]
    assert remaining == [alice["user"]["id"]]
    audit = await client.get(f"{BASE}/audit")
    actions = [item["action"] for item in audit.json()["items"]]
    assert "member_removed" in actions


async def test_audit_lists_only_own_tenant(client, client2, client3):
    alice, _bob, carol = await _users(client, client2, client3)
    await _add(client, alice, "bob@example.com", "admin")
    await client3.post(COLLECTIONS, json={"name": "Secret"}, headers=csrf(carol))

    alice_audit = await client.get(f"{BASE}/audit")
    carol_audit = await client3.get(f"{BASE}/audit")

    assert alice_audit.status_code == 200
    items = alice_audit.json()["items"]
    actions = [item["action"] for item in items]
    assert "member_added" in actions
    assert "collection_created" not in actions
    assert carol["user"]["id"] not in {item["user_id"] for item in items}
    added = next(item for item in items if item["action"] == "member_added")
    assert added["user_id"] == alice["user"]["id"]
    assert added["target_type"] == "user"
    assert added["details"] == {"role": "admin"}
    assert "collection_created" in [i["action"] for i in carol_audit.json()["items"]]


async def test_audit_is_newest_first_and_paginates(client, client2, client3):
    alice, _bob, _carol = await _users(client, client2, client3)
    await _add(client, alice, "bob@example.com")
    await _add(client, alice, "carol@example.com")

    first = await client.get(f"{BASE}/audit", params={"limit": 1})
    second = await client.get(
        f"{BASE}/audit", params={"limit": 1, "cursor": first.json()["next_cursor"]}
    )
    everything = await client.get(f"{BASE}/audit")

    full = [item["id"] for item in everything.json()["items"]]
    assert full == sorted(full, reverse=True)
    assert len(full) >= 2
    assert [first.json()["items"][0]["id"], second.json()["items"][0]["id"]] == full[:2]


@pytest.mark.parametrize("key", ["not-an-integer", "-1", "99999999999999999999"])
async def test_audit_bad_cursor_key_400(client, key):
    await signup(client)
    cursor = encode_cursor(datetime.now(UTC), key)

    response = await client.get(f"{BASE}/audit", params={"cursor": cursor})

    assert response.status_code == 400
    assert response.json()["code"] == "bad_cursor"


async def test_member_cannot_read_audit(client, client2, client3, db):
    alice, bob, _carol = await _users(client, client2, client3)
    bob = await _join(db, client2, alice, bob, "member")

    response = await client2.get(f"{BASE}/audit")

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


async def _mint_key(client, session, name="ci"):
    response = await client.post(KEYS, json={"name": name}, headers=csrf(session))
    assert response.status_code == 201, response.text
    return response.json()["token"]


async def _bearer_request(api_app, method, url, token, **kwargs):
    """Send a request from a client that holds no cookies."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://test"
    ) as bearer_client:
        return await bearer_client.request(
            method, url, headers={"Authorization": f"Bearer {token}"}, **kwargs
        )


async def test_bearer_key_cannot_add_member(client, client2, client3, api_app):
    alice, _bob, _carol = await _users(client, client2, client3)
    token = await _mint_key(client, alice)

    response = await _bearer_request(
        api_app,
        "POST",
        f"{BASE}/members",
        token,
        json={"email": "bob@example.com", "role": "owner"},
    )
    listed = await client.get(f"{BASE}/members")

    assert response.status_code == 403
    assert response.json()["code"] == "session_required"
    assert response.json()["title"] == "This action requires signing in with a session"
    assert [m["email"] for m in listed.json()] == ["alice@example.com"]


async def test_bearer_key_cannot_remove_member(client, client2, client3, api_app, db):
    alice, bob, _carol = await _users(client, client2, client3)
    await _join(db, client2, alice, bob, "member")
    token = await _mint_key(client, alice)

    response = await _bearer_request(
        api_app, "DELETE", f"{BASE}/members/{bob['user']['id']}", token
    )
    listed = await client.get(f"{BASE}/members")

    assert response.status_code == 403
    assert response.json()["code"] == "session_required"
    assert response.json()["title"] == "This action requires signing in with a session"
    assert {m["email"] for m in listed.json()} == {"alice@example.com", "bob@example.com"}


async def test_bearer_key_can_list_members_and_audit(client, client2, client3, api_app):
    alice, _bob, _carol = await _users(client, client2, client3)
    token = await _mint_key(client, alice)

    members_response = await _bearer_request(api_app, "GET", f"{BASE}/members", token)
    audit_response = await _bearer_request(api_app, "GET", f"{BASE}/audit", token)

    assert members_response.status_code == 200
    assert audit_response.status_code == 200


async def test_member_cannot_remove(client, client2, client3, db):
    alice, bob, _carol = await _users(client, client2, client3)
    bob = await _join(db, client2, alice, bob, "member")

    response = await client2.delete(f"{BASE}/members/{alice['user']['id']}", headers=csrf(bob))

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


async def test_removed_member_keys_stay_revoked_after_readd(client, client2, client3, api_app, db):
    alice, bob, _carol = await _users(client, client2, client3)
    own_key = await _mint_key(client2, bob, name="own tenant")
    bob = await _join(db, client2, alice, bob, "member")
    joined_key = await _mint_key(client2, bob, name="alice tenant")
    other_session = await client3.post(
        "/api/v1/auth/login", json={"email": "bob@example.com", "password": PASSWORD}
    )
    assert other_session.status_code == 200, other_session.text
    assert (await _bearer_request(api_app, "GET", ME, joined_key)).status_code == 200

    removed = await client.delete(f"{BASE}/members/{bob['user']['id']}", headers=csrf(alice))
    readded = await _add(client, alice, "bob@example.com", "member")

    assert removed.status_code == 204
    assert readded.status_code == 201
    assert (await _bearer_request(api_app, "GET", ME, joined_key)).status_code == 401
    assert (await client2.get(ME)).status_code == 401
    # Credentials of the same user in other tenants are untouched.
    assert (await _bearer_request(api_app, "GET", ME, own_key)).status_code == 200
    assert (await client3.get(ME)).status_code == 200


async def test_audit_cursor_key_too_long_400(client):
    await signup(client)
    cursor = encode_cursor(datetime.now(UTC), "9" * 6720)

    response = await client.get(f"{BASE}/audit", params={"cursor": cursor})

    assert response.status_code == 400
    assert response.json()["code"] == "bad_cursor"


async def test_owner_locks_are_taken_in_user_id_order(db):
    async with db.system_tx() as conn:
        tenant = await tenants.insert_tenant(conn, "Locks")
        for index in range(8):
            user = await users.insert_user(conn, f"owner{index}@example.com", "x", f"Owner {index}")
            await tenants.add_membership(conn, tenant["id"], user["id"], "owner")
        owner_ids = await members.lock_owner_ids(conn, tenant["id"])

    assert len(owner_ids) == 8
    assert owner_ids == sorted(owner_ids)


async def test_owner_who_lost_ownership_mid_request_cannot_remove(
    client, client2, client3, api_app, db, monkeypatch
):
    alice, bob, _carol = await _users(client, client2, client3)
    await _join(db, client2, alice, bob, "owner")
    original = api_app.state.auth.resolve_session

    async def resolve_then_lose_ownership(token):
        principal = await original(token)
        if principal is not None and str(principal.user_id) == alice["user"]["id"]:
            async with db.system_tx() as conn:
                await members.delete_membership(
                    conn, principal.tenant_id, UUID(alice["user"]["id"])
                )
        return principal

    monkeypatch.setattr(api_app.state.auth, "resolve_session", resolve_then_lose_ownership)

    response = await client.delete(f"{BASE}/members/{bob['user']['id']}", headers=csrf(alice))

    assert response.status_code == 403
    assert response.json()["code"] == "insufficient_permission"
    async with db.system_tx() as conn:
        assert await tenants.get_membership(conn, alice["tenant"]["id"], bob["user"]["id"])
