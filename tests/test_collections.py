from uuid import UUID

import pytest

from backend.api.errors import ApiError
from backend.auth.service import Principal
from backend.authz.policy import readable_collection_ids, require_collection
from backend.db.repos import tenants
from tests.helpers.auth import csrf, signup

pytestmark = pytest.mark.pg

COLLECTIONS = "/api/v1/collections"


async def _team(client, client2, client3, db):
    """Alice (owner) and Bob (member) share a tenant; Carol owns another one."""
    alice = await signup(client)
    bob = await signup(client2, email="bob@example.com", display_name="Bob")
    carol = await signup(client3, email="carol@example.com", display_name="Carol")
    async with db.system_tx() as conn:
        await tenants.add_membership(conn, alice["tenant"]["id"], bob["user"]["id"], "member")
    switched = await client2.post(
        "/api/v1/auth/switch-tenant",
        json={"tenant_id": alice["tenant"]["id"]},
        headers=csrf(bob),
    )
    assert switched.status_code == 200, switched.text
    bob = {**bob, "tenant": switched.json()["tenant"]}
    return alice, bob, carol


async def _create(client, session, name="Reports", description=""):
    response = await client.post(
        COLLECTIONS, json={"name": name, "description": description}, headers=csrf(session)
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _put_grants(client, session, collection_id, grants):
    return await client.put(
        f"{COLLECTIONS}/{collection_id}/grants",
        json={"grants": grants},
        headers=csrf(session),
    )


async def test_owner_creates_and_lists(client, client2, client3, db):
    alice, _bob, _carol = await _team(client, client2, client3, db)

    created = await _create(client, alice, description="Quarterly")
    listed = await client.get(COLLECTIONS)
    grants = await client.get(f"{COLLECTIONS}/{created['id']}/grants")

    assert created["name"] == "Reports"
    assert created["kind"] == "normal"
    assert created["permission"] == "admin"
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == [created["id"]]
    assert listed.json()["items"][0]["permission"] == "admin"
    assert listed.json()["next_cursor"] is None
    assert [(g["user_id"], g["permission"]) for g in grants.json()] == [
        (alice["user"]["id"], "admin")
    ]
    assert grants.json()[0]["email"] == "alice@example.com"


async def test_member_without_grant_cannot_see(client, client2, client3, db):
    alice, bob, _carol = await _team(client, client2, client3, db)
    created = await _create(client, alice)

    got = await client2.get(f"{COLLECTIONS}/{created['id']}")
    listed = await client2.get(COLLECTIONS)

    assert got.status_code == 404
    assert got.json()["code"] == "collection_not_found"
    assert listed.json()["items"] == []
    assert bob["tenant"]["role"] == "member"


async def test_read_grant_allows_read_not_write(client, client2, client3, db):
    alice, bob, _carol = await _team(client, client2, client3, db)
    created = await _create(client, alice)

    put = await _put_grants(
        client,
        alice,
        created["id"],
        [
            {"user_id": alice["user"]["id"], "permission": "admin"},
            {"user_id": bob["user"]["id"], "permission": "read"},
        ],
    )
    got = await client2.get(f"{COLLECTIONS}/{created['id']}")
    patched = await client2.patch(
        f"{COLLECTIONS}/{created['id']}", json={"name": "Renamed"}, headers=csrf(bob)
    )
    listed = await client2.get(COLLECTIONS)

    assert put.status_code == 200, put.text
    assert got.status_code == 200
    assert got.json()["permission"] == "read"
    assert patched.status_code == 403
    assert patched.json()["code"] == "insufficient_permission"
    assert [item["permission"] for item in listed.json()["items"]] == ["read"]


async def test_cross_tenant_is_404(client, client2, client3, db):
    alice, _bob, carol = await _team(client, client2, client3, db)
    created = await _create(client, alice)
    target = f"{COLLECTIONS}/{created['id']}"

    responses = [
        await client3.get(target),
        await client3.patch(target, json={"name": "x"}, headers=csrf(carol)),
        await client3.delete(target, headers=csrf(carol)),
        await client3.get(f"{target}/grants"),
        await _put_grants(client3, carol, created["id"], []),
    ]

    assert [r.status_code for r in responses] == [404] * 5
    assert {r.json()["code"] for r in responses} == {"collection_not_found"}


async def test_duplicate_name_409(client, client2, client3, db):
    alice, _bob, carol = await _team(client, client2, client3, db)
    await _create(client, alice, name="Reports")

    duplicate = await client.post(COLLECTIONS, json={"name": "Reports"}, headers=csrf(alice))
    other_tenant = await client3.post(COLLECTIONS, json={"name": "Reports"}, headers=csrf(carol))

    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "collection_name_taken"
    assert other_tenant.status_code == 201


async def test_grant_to_non_member_422(client, client2, client3, db):
    alice, _bob, carol = await _team(client, client2, client3, db)
    created = await _create(client, alice)

    response = await _put_grants(
        client, alice, created["id"], [{"user_id": carol["user"]["id"], "permission": "read"}]
    )
    duplicates = await _put_grants(
        client,
        alice,
        created["id"],
        [
            {"user_id": alice["user"]["id"], "permission": "read"},
            {"user_id": alice["user"]["id"], "permission": "admin"},
        ],
    )
    grants = await client.get(f"{COLLECTIONS}/{created['id']}/grants")

    assert response.status_code == 422
    assert response.json()["code"] == "not_a_member"
    assert carol["user"]["id"] not in response.text
    assert duplicates.status_code == 422
    assert duplicates.json()["code"] == "validation_error"
    assert [g["user_id"] for g in grants.json()] == [alice["user"]["id"]]


async def test_pagination_walks_all_pages(client, client2, client3, db):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    created_ids = [(await _create(client, alice, name=f"c{i:02d}"))["id"] for i in range(25)]

    seen, sizes, cursor = [], [], None
    while True:
        params = {"limit": 10, **({"cursor": cursor} if cursor else {})}
        page = (await client.get(COLLECTIONS, params=params)).json()
        sizes.append(len(page["items"]))
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert sizes == [10, 10, 5]
    assert len(set(seen)) == 25
    assert seen == list(reversed(created_ids))


async def test_bad_cursor_400(client, client2, client3, db):
    await _team(client, client2, client3, db)

    garbage = await client.get(COLLECTIONS, params={"cursor": "not-a-cursor"})
    bad_id = await client.get(
        COLLECTIONS,
        params={"cursor": "WyIyMDI2LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwgIm5vdC1hLXV1aWQiXQ=="},
    )

    assert garbage.status_code == 400
    assert garbage.json()["code"] == "bad_cursor"
    assert bad_id.status_code == 400
    assert bad_id.json()["code"] == "bad_cursor"


async def test_readable_collection_ids_rejects_unreadable(client, client2, client3, db):
    alice, bob, _carol = await _team(client, client2, client3, db)
    granted = await _create(client, alice, name="granted")
    hidden = await _create(client, alice, name="hidden")
    await _put_grants(
        client,
        alice,
        granted["id"],
        [
            {"user_id": alice["user"]["id"], "permission": "admin"},
            {"user_id": bob["user"]["id"], "permission": "read"},
        ],
    )
    async with db.tenant_tx(UUID(alice["tenant"]["id"])) as conn:
        await conn.execute(
            "insert into collections (tenant_id, name, kind) values (%s, 'scratch', 'quick')",
            (alice["tenant"]["id"],),
        )
        quick_id = (
            await (await conn.execute("select id from collections where kind = 'quick'")).fetchone()
        )["id"]
        await conn.execute(
            "insert into collection_grants (tenant_id, collection_id, user_id, permission) "
            "values (%s, %s, %s, 'read')",
            (alice["tenant"]["id"], quick_id, bob["user"]["id"]),
        )
    principal = Principal(
        user_id=UUID(bob["user"]["id"]),
        email="bob@example.com",
        display_name="Bob",
        tenant_id=UUID(alice["tenant"]["id"]),
        role="member",
        auth="session",
    )

    async with db.tenant_tx(principal.tenant_id) as conn:
        everything = await readable_collection_ids(conn, principal, None)
        with_quick = await readable_collection_ids(conn, principal, None, kinds=("normal", "quick"))
        with pytest.raises(ApiError) as excinfo:
            await readable_collection_ids(conn, principal, [UUID(hidden["id"])])
        explicit = await readable_collection_ids(conn, principal, [UUID(granted["id"])])

    assert everything == (UUID(granted["id"]),)
    assert set(with_quick) == {UUID(granted["id"]), quick_id}
    assert excinfo.value.status == 404
    assert excinfo.value.code == "collection_not_found"
    assert explicit == (UUID(granted["id"]),)


async def test_lone_surrogate_in_collection_name_is_422(client):
    session = await signup(client)
    raw = b'{"name": "bad\\ud800name", "description": ""}'

    response = await client.post(
        COLLECTIONS,
        content=raw,
        headers={**csrf(session), "Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


@pytest.mark.parametrize("kind", ["quick", "benchmark"])
async def test_by_id_routes_do_not_expose_non_normal_collections(client, db, kind):
    alice = await signup(client)
    tenant_id = UUID(alice["tenant"]["id"])
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "insert into collections (tenant_id, name, kind) values (%s, 'Scratch', %s) "
            "returning id",
            (tenant_id, kind),
        )
        collection_id = (await cur.fetchone())["id"]
    url = f"{COLLECTIONS}/{collection_id}"
    headers = csrf(alice)

    responses = [
        await client.get(url),
        await client.patch(url, json={"name": "Renamed"}, headers=headers),
        await client.get(f"{url}/grants"),
        await _put_grants(client, alice, collection_id, []),
        await client.delete(url, headers=headers),
    ]

    assert [r.status_code for r in responses] == [404] * 5
    assert {r.json()["code"] for r in responses} == {"collection_not_found"}
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select name from collections where id = %s", (collection_id,))
        assert (await cur.fetchone())["name"] == "Scratch"


async def test_require_collection_kinds_keyword_opts_into_other_kinds(client, db):
    alice = await signup(client)
    tenant_id = UUID(alice["tenant"]["id"])
    principal = Principal(
        UUID(alice["user"]["id"]), "alice@example.com", "Alice", tenant_id, "owner", "session"
    )
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "insert into collections (tenant_id, name, kind) values (%s, 'Scratch', 'quick') "
            "returning id",
            (tenant_id,),
        )
        collection_id = (await cur.fetchone())["id"]

        row = await require_collection(conn, principal, collection_id, "read", kinds=("quick",))
        with pytest.raises(ApiError) as info:
            await require_collection(conn, principal, collection_id, "read")

    assert (row["kind"], row["permission"]) == ("quick", "admin")
    assert (info.value.status, info.value.code) == (404, "collection_not_found")
