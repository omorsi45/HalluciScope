from datetime import UTC, datetime

import pytest

from backend.api.pagination import encode_cursor
from backend.db.repos import tenants
from tests.helpers.auth import csrf, signup

pytestmark = pytest.mark.pg

BASE = "/api/v1/tenants/current"
COLLECTIONS = "/api/v1/collections"


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
    async with db.tenant_tx(alice["tenant"]["id"]) as conn:
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
