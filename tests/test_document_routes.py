"""Document, version, and upload routes under /api/v1."""

import base64
import json
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from backend.config import Settings
from backend.ingest.parsers import parse
from tests.helpers.auth import csrf
from tests.pg_fixtures import drain
from tests.test_collections import COLLECTIONS, _create, _put_grants, _team
from tests.test_ingest_upload import json_logs

pytestmark = pytest.mark.pg

DOCUMENTS = "/api/v1/documents"
UPLOADS = "/api/v1/uploads"
V1 = b"# Refunds\n\nRefunds are issued within 30 days. Keep the receipt.\n"
V2 = b"# Refunds\n\nRefunds are issued within 14 days. Keep the receipt and the packaging.\n"
MAX_TEXT_CHARS = 100_000
LONG_TEXT = " ".join(f"Entry {i} states a fact." for i in range(5_000)).encode()
DOCUMENT_KEYS = {
    "id",
    "collection_id",
    "title",
    "filename",
    "mime_type",
    "current_version_number",
    "version_count",
    "created_at",
}
SUMMARY_KEYS = {
    "id",
    "version_number",
    "effective_from",
    "created_at",
    "char_count",
    "n_passages",
    "parser",
    "is_current",
}


async def post_document(client, session, collection_id, data=V1, filename="policy.md", **form):
    return await client.post(
        f"{COLLECTIONS}/{collection_id}/documents",
        files={"file": (filename, data, "application/octet-stream")},
        data=form,
        headers=csrf(session),
    )


async def post_version(client, session, document_id, data=V2, filename="policy.md", **form):
    return await client.post(
        f"{DOCUMENTS}/{document_id}/versions",
        files={"file": (filename, data, "application/octet-stream")},
        data=form,
        headers=csrf(session),
    )


async def ingest(client, session, collection_id, db, services, data=V1, filename="policy.md"):
    """Upload a new document through the API, run the worker, return the accepted body."""
    response = await post_document(client, session, collection_id, data, filename)
    assert response.status_code == 202, response.text
    await drain(db, services)
    return response.json()


async def ingest_version(client, session, document_id, db, services, data=V2):
    response = await post_version(client, session, document_id, data)
    assert response.status_code == 202, response.text
    await drain(db, services)
    return response.json()


async def alice_with_document(client, client2, client3, db, services):
    alice, bob, carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    accepted = await ingest(client, alice, collection["id"], db, services)
    return alice, bob, carol, collection, accepted


async def test_upload_drain_and_read_back(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)

    response = await post_document(client, alice, collection["id"], title="Refund policy")
    accepted = response.json()
    queued = (await client.get(f"{UPLOADS}/{accepted['upload_id']}")).json()
    await drain(db, services)
    upload = (await client.get(f"{UPLOADS}/{accepted['upload_id']}")).json()
    detail = (await client.get(f"{DOCUMENTS}/{accepted['document_id']}")).json()
    text = (await client.get(f"{DOCUMENTS}/{accepted['document_id']}/versions/1/text")).json()
    version = (await client.get(f"{DOCUMENTS}/{accepted['document_id']}/versions/1")).json()

    parsed = parse(V1, "policy.md")
    assert response.status_code == 202
    assert set(accepted) == {"document_id", "upload_id", "status"}
    assert accepted["status"] == "queued"
    assert (queued["status"], queued["version_id"], queued["version_number"]) == (
        "queued",
        None,
        None,
    )
    assert queued["document_id"] == accepted["document_id"]
    assert set(upload) == {
        "id",
        "document_id",
        "status",
        "version_id",
        "version_number",
        "error_code",
        "stats",
        "created_at",
        "updated_at",
    }
    assert (upload["status"], upload["version_number"], upload["error_code"]) == (
        "created_version",
        1,
        None,
    )
    assert upload["stats"]["n_passages"] >= 1
    assert set(detail) == DOCUMENT_KEYS | {"versions"}
    assert (detail["title"], detail["filename"], detail["mime_type"]) == (
        "Refund policy",
        "policy.md",
        "text/markdown",
    )
    assert detail["collection_id"] == collection["id"]
    assert (detail["current_version_number"], detail["version_count"]) == (1, 1)
    (summary,) = detail["versions"]
    assert set(summary) == SUMMARY_KEYS
    assert summary["id"] == upload["version_id"]
    assert (summary["version_number"], summary["is_current"]) == (1, True)
    assert summary["char_count"] == len(parsed.text)
    assert summary["n_passages"] == upload["stats"]["n_passages"]
    assert summary["parser"] == parsed.parser
    assert text == {
        "start": 0,
        "end": len(parsed.text),
        "char_count": len(parsed.text),
        "text": parsed.text,
    }
    assert set(version) == SUMMARY_KEYS | {"page_map", "sections", "metadata", "parser_version"}
    assert version["id"] == summary["id"]
    assert version["page_map"] == [[page.page, page.start, page.end] for page in parsed.pages]
    assert version["sections"] == [[s.start, s.end, s.path] for s in parsed.sections]
    assert version["metadata"] == parsed.metadata
    assert version["parser_version"] == parsed.parser_version


async def test_text_range_validation(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    accepted = await ingest(client, alice, collection["id"], db, services, LONG_TEXT, "long.txt")
    char_count = len(parse(LONG_TEXT, "long.txt").text)
    assert char_count > MAX_TEXT_CHARS + 1
    url = f"{DOCUMENTS}/{accepted['document_id']}/versions/1/text"

    too_wide = await client.get(url, params={"start": 0, "end": MAX_TEXT_CHARS + 1})
    past_end = await client.get(url, params={"start": 0, "end": char_count + 1})
    negative = await client.get(url, params={"start": -1, "end": 10})
    backwards = await client.get(url, params={"start": 10, "end": 5})
    start_past_end = await client.get(url, params={"start": char_count + 1})
    widest = await client.get(url, params={"start": 0, "end": MAX_TEXT_CHARS})

    for rejected in (too_wide, past_end, negative, backwards, start_past_end):
        assert rejected.status_code == 422, rejected.text
        assert rejected.json()["code"] == "bad_range"
    assert widest.status_code == 200
    assert len(widest.json()["text"]) == MAX_TEXT_CHARS


async def test_text_range_defaults_and_empty_slice(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    accepted = await ingest(client, alice, collection["id"], db, services, LONG_TEXT, "long.txt")
    text = parse(LONG_TEXT, "long.txt").text
    url = f"{DOCUMENTS}/{accepted['document_id']}/versions/1/text"

    first = (await client.get(url)).json()
    tail = (await client.get(url, params={"start": MAX_TEXT_CHARS})).json()
    empty = (await client.get(url, params={"start": 7, "end": 7})).json()
    at_end = (await client.get(url, params={"start": len(text)})).json()

    assert (first["start"], first["end"]) == (0, MAX_TEXT_CHARS)
    assert first["text"] == text[:MAX_TEXT_CHARS]
    assert (tail["start"], tail["end"], tail["char_count"]) == (
        MAX_TEXT_CHARS,
        len(text),
        len(text),
    )
    assert tail["text"] == text[MAX_TEXT_CHARS:]
    assert (empty["start"], empty["end"], empty["text"]) == (7, 7, "")
    assert (at_end["start"], at_end["end"], at_end["text"]) == (len(text), len(text), "")


async def test_text_slices_use_code_point_offsets(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    raw = "Cafe\u0301 \U0001f600 \u6771\u4eac tower. Re\u0301sume\u0301 filed on time.\n".encode()
    accepted = await ingest(client, alice, collection["id"], db, services, raw, "mixed.txt")
    text = parse(raw, "mixed.txt").text
    url = f"{DOCUMENTS}/{accepted['document_id']}/versions/1/text"

    sliced = (await client.get(url, params={"start": 3, "end": 12})).json()

    assert sliced["text"] == text[3:12]
    assert len(sliced["text"]) == 9
    assert sliced["char_count"] == len(text)


async def test_passage_boundaries_cover_text(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    accepted = await ingest(client, alice, collection["id"], db, services, LONG_TEXT, "long.txt")
    base = f"{DOCUMENTS}/{accepted['document_id']}/versions/1"
    text = parse(LONG_TEXT, "long.txt").text

    response = await client.get(f"{base}/passages")
    detail = (await client.get(f"{DOCUMENTS}/{accepted['document_id']}")).json()

    passages = response.json()
    assert response.status_code == 200
    assert len(passages) > 1
    assert detail["versions"][0]["n_passages"] == len(passages)
    assert {frozenset(p) for p in passages} == {
        frozenset({"id", "ordinal", "start_char", "end_char", "page_start", "page_end", "section"})
    }
    assert [p["ordinal"] for p in passages] == list(range(len(passages)))
    assert all(0 <= p["start_char"] < p["end_char"] <= len(text) for p in passages)
    starts = [p["start_char"] for p in passages]
    assert starts == sorted(starts)
    covered = bytearray(len(text))
    for passage in passages:
        covered[passage["start_char"] : passage["end_char"]] = b"\x01" * (
            passage["end_char"] - passage["start_char"]
        )
    assert all(covered[i] for i, char in enumerate(text) if not char.isspace())


async def test_new_version_route_and_listing(client, client2, client3, db, services):
    alice, _bob, _carol, collection, first = await alice_with_document(
        client, client2, client3, db, services
    )

    response = await post_version(client, alice, first["document_id"])
    accepted = response.json()
    await drain(db, services)
    upload = (await client.get(f"{UPLOADS}/{accepted['upload_id']}")).json()
    listed = (await client.get(DOCUMENTS, params={"collection_id": collection["id"]})).json()
    detail = (await client.get(f"{DOCUMENTS}/{first['document_id']}")).json()
    second_text = (await client.get(f"{DOCUMENTS}/{first['document_id']}/versions/2/text")).json()
    first_text = (await client.get(f"{DOCUMENTS}/{first['document_id']}/versions/1/text")).json()

    assert response.status_code == 202
    assert accepted["document_id"] == first["document_id"]
    assert accepted["upload_id"] != first["upload_id"]
    assert accepted["status"] == "queued"
    assert (upload["status"], upload["version_number"]) == ("created_version", 2)
    (item,) = listed["items"]
    assert set(item) == DOCUMENT_KEYS
    assert item["id"] == first["document_id"]
    assert (item["current_version_number"], item["version_count"]) == (2, 2)
    assert [v["version_number"] for v in detail["versions"]] == [1, 2]
    assert [v["is_current"] for v in detail["versions"]] == [False, True]
    assert all(
        isinstance(v["n_passages"], int) and v["n_passages"] >= 1 for v in detail["versions"]
    )
    assert "14 days" in second_text["text"]
    assert "30 days" in first_text["text"]


async def test_reuploading_current_bytes_is_a_duplicate(client, client2, client3, db, services):
    alice, _bob, _carol, _collection, first = await alice_with_document(
        client, client2, client3, db, services
    )

    response = await post_version(client, alice, first["document_id"], data=V1)
    upload = (await client.get(f"{UPLOADS}/{response.json()['upload_id']}")).json()
    detail = (await client.get(f"{DOCUMENTS}/{first['document_id']}")).json()

    assert response.status_code == 202
    assert response.json()["status"] == "duplicate"
    assert (upload["status"], upload["version_number"]) == ("duplicate", 1)
    assert detail["version_count"] == 1
    assert await drain(db, services) == 0


async def test_effective_from_is_validated(client, client2, client3, db, services):
    alice, _bob, _carol, collection, first = await alice_with_document(
        client, client2, client3, db, services
    )
    soon = (datetime.now(UTC) + timedelta(hours=1)).isoformat()

    naive = await post_version(
        client, alice, first["document_id"], effective_from="2026-01-01T00:00:00"
    )
    future = await post_document(client, alice, collection["id"], effective_from=soon)
    garbage = await post_version(client, alice, first["document_id"], effective_from="yesterday")
    valid = await post_version(
        client, alice, first["document_id"], effective_from=datetime.now(UTC).isoformat()
    )

    assert (naive.status_code, naive.json()["code"]) == (422, "validation_error")
    assert (future.status_code, future.json()["code"]) == (422, "effective_from_in_future")
    assert (garbage.status_code, garbage.json()["code"]) == (422, "validation_error")
    assert valid.status_code == 202, valid.text
    assert (await client.get(DOCUMENTS)).json()["items"][0]["version_count"] == 1


async def test_upload_rejections(client, client2, client3, db, api_app, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    url = f"{COLLECTIONS}/{collection['id']}/documents"

    unsupported = await post_document(client, alice, collection["id"], filename="notes.exe")
    missing_file = await client.post(url, data={"title": "x"}, headers=csrf(alice))
    long_title = await post_document(client, alice, collection["id"], title="T" * 301)
    blank_title = await post_document(client, alice, collection["id"], title="   ")
    api_app.state.settings = Settings(max_upload_bytes=10, dev_insecure_cookies=True)
    too_large = await post_document(
        client, alice, collection["id"], data=b"x" * 11, filename="a.txt"
    )
    listed = await client.get(DOCUMENTS)

    assert (unsupported.status_code, unsupported.json()["code"]) == (415, "unsupported_type")
    assert (missing_file.status_code, missing_file.json()["code"]) == (422, "validation_error")
    assert (long_title.status_code, long_title.json()["code"]) == (422, "validation_error")
    assert (blank_title.status_code, blank_title.json()["code"]) == (422, "validation_error")
    assert (too_large.status_code, too_large.json()["code"]) == (413, "too_large")
    assert listed.json()["items"] == []


async def test_title_with_nul_character_is_422(client, client2, client3, db):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)

    response = await post_document(client, alice, collection["id"], title="bad\x00title")

    assert (response.status_code, response.json()["code"]) == (422, "validation_error")


@pytest.mark.parametrize("who", ["carol", "bob"])
async def test_unreadable_documents_are_404(client, client2, client3, db, services, who):
    _alice, bob, carol, collection, accepted = await alice_with_document(
        client, client2, client3, db, services
    )
    outsider, session = (client3, carol) if who == "carol" else (client2, bob)
    document_id, upload_id = accepted["document_id"], accepted["upload_id"]
    random_id = uuid4()

    async def hit(subject, document, upload, collection_id):
        files = {"file": ("policy.md", V2, "application/octet-stream")}
        headers = csrf(session)
        return [
            await subject.post(
                f"{COLLECTIONS}/{collection_id}/documents", files=files, headers=headers
            ),
            await subject.post(f"{DOCUMENTS}/{document}/versions", files=files, headers=headers),
            await subject.get(f"{UPLOADS}/{upload}"),
            await subject.get(DOCUMENTS, params={"collection_id": collection_id}),
            await subject.get(f"{DOCUMENTS}/{document}"),
            await subject.delete(f"{DOCUMENTS}/{document}", headers=headers),
            await subject.get(f"{DOCUMENTS}/{document}/versions/1"),
            await subject.get(f"{DOCUMENTS}/{document}/versions/1/text"),
            await subject.get(f"{DOCUMENTS}/{document}/versions/1/passages"),
        ]

    real = await hit(outsider, document_id, upload_id, collection["id"])
    unknown_ids = await hit(outsider, random_id, uuid4(), uuid4())
    listing = await outsider.get(DOCUMENTS)
    still_there = await client.get(f"{DOCUMENTS}/{document_id}")

    assert [r.status_code for r in real] == [404] * 9
    assert [r.json()["code"] for r in real] == [
        "collection_not_found",
        "document_not_found",
        "upload_not_found",
        "collection_not_found",
        "document_not_found",
        "document_not_found",
        "document_not_found",
        "document_not_found",
        "document_not_found",
    ]
    assert [r.json() for r in real] == [r.json() for r in unknown_ids]
    assert (listing.status_code, listing.json()["items"]) == (200, [])
    assert still_there.status_code == 200
    assert (await client.get(f"{DOCUMENTS}/{document_id}/versions/1")).status_code == 200
    assert document_id not in "".join(r.text for r in real)


async def test_read_only_member_cannot_upload(client, client2, client3, db, services):
    alice, bob, _carol, collection, accepted = await alice_with_document(
        client, client2, client3, db, services
    )
    await _put_grants(
        client,
        alice,
        collection["id"],
        [
            {"user_id": alice["user"]["id"], "permission": "admin"},
            {"user_id": bob["user"]["id"], "permission": "read"},
        ],
    )
    document_id = accepted["document_id"]

    new_document = await post_document(client2, bob, collection["id"])
    new_version = await post_version(client2, bob, document_id)
    deleted = await client2.delete(f"{DOCUMENTS}/{document_id}", headers=csrf(bob))
    reads = [
        await client2.get(f"{DOCUMENTS}/{document_id}"),
        await client2.get(f"{DOCUMENTS}/{document_id}/versions/1"),
        await client2.get(f"{DOCUMENTS}/{document_id}/versions/1/text"),
        await client2.get(f"{DOCUMENTS}/{document_id}/versions/1/passages"),
        await client2.get(f"{UPLOADS}/{accepted['upload_id']}"),
        await client2.get(DOCUMENTS, params={"collection_id": collection["id"]}),
    ]
    await _put_grants(
        client,
        alice,
        collection["id"],
        [
            {"user_id": alice["user"]["id"], "permission": "admin"},
            {"user_id": bob["user"]["id"], "permission": "write"},
        ],
    )
    writer = await post_version(client2, bob, document_id)

    for rejected in (new_document, new_version, deleted):
        assert rejected.status_code == 403
        assert rejected.json()["code"] == "insufficient_permission"
    assert [r.status_code for r in reads] == [200] * 6
    assert writer.status_code == 202
    detail = (await client.get(f"{DOCUMENTS}/{document_id}")).json()
    assert detail["version_count"] == 1


async def test_soft_deleted_document_404(client, client2, client3, db, services):
    alice, _bob, _carol, collection, accepted = await alice_with_document(
        client, client2, client3, db, services
    )
    document_id = accepted["document_id"]

    deleted = await client.delete(f"{DOCUMENTS}/{document_id}", headers=csrf(alice))
    again = await client.delete(f"{DOCUMENTS}/{document_id}", headers=csrf(alice))
    after = [
        await client.get(f"{DOCUMENTS}/{document_id}"),
        await client.get(f"{DOCUMENTS}/{document_id}/versions/1"),
        await client.get(f"{DOCUMENTS}/{document_id}/versions/1/text"),
        await client.get(f"{DOCUMENTS}/{document_id}/versions/1/passages"),
        await post_version(client, alice, document_id),
    ]
    upload = await client.get(f"{UPLOADS}/{accepted['upload_id']}")
    listed = await client.get(DOCUMENTS, params={"collection_id": collection["id"]})
    everything = await client.get(DOCUMENTS)
    async with db.system_tx() as conn:
        cur = await conn.execute("select * from audit_log where action = 'document_deleted'")
        (entry,) = await cur.fetchall()

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert (again.status_code, again.json()["code"]) == (404, "document_not_found")
    assert [r.status_code for r in after] == [404] * 5
    assert {r.json()["code"] for r in after} == {"document_not_found"}
    assert (upload.status_code, upload.json()["code"]) == (404, "upload_not_found")
    assert listed.json()["items"] == []
    assert everything.json()["items"] == []
    assert str(entry["user_id"]) == alice["user"]["id"]
    assert str(entry["tenant_id"]) == alice["tenant"]["id"]
    assert (entry["target_type"], entry["target_id"]) == ("document", document_id)
    assert entry["details"] is None


async def test_collection_with_documents_cannot_be_deleted(client, client2, client3, db, services):
    alice, _bob, _carol, collection, accepted = await alice_with_document(
        client, client2, client3, db, services
    )
    url = f"{COLLECTIONS}/{collection['id']}"

    live = await client.delete(url, headers=csrf(alice))
    await client.delete(f"{DOCUMENTS}/{accepted['document_id']}", headers=csrf(alice))
    # Soft deletion keeps the document row, its versions and its passages, and the
    # collection's foreign keys still point at them, so the collection stays.
    history = await client.delete(url, headers=csrf(alice))

    for blocked in (live, history):
        assert (blocked.status_code, blocked.json()["code"]) == (409, "collection_not_empty")
    assert (await client.get(url)).status_code == 200
    assert (await client.get(DOCUMENTS, params={"collection_id": collection["id"]})).json()[
        "items"
    ] == []


async def test_empty_collection_can_still_be_deleted(client, client2, client3, db):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)

    deleted = await client.delete(f"{COLLECTIONS}/{collection['id']}", headers=csrf(alice))

    assert deleted.status_code == 204


async def test_missing_version_number_404(client, client2, client3, db, services):
    _alice, _bob, _carol, _collection, accepted = await alice_with_document(
        client, client2, client3, db, services
    )
    base = f"{DOCUMENTS}/{accepted['document_id']}/versions"

    responses = [
        await client.get(f"{base}/{number}{suffix}")
        for number in (0, 2, -1, 99999999999999999999)
        for suffix in ("", "/text", "/passages")
    ]

    assert [r.status_code for r in responses] == [404] * 12
    assert {r.json()["code"] for r in responses} == {"version_not_found"}


async def test_listing_paginates_and_filters_by_permission(client, client2, client3, db, services):
    alice, bob, _carol = await _team(client, client2, client3, db)
    shared = await _create(client, alice, name="shared")
    private = await _create(client, alice, name="private")
    await _put_grants(
        client,
        alice,
        shared["id"],
        [
            {"user_id": alice["user"]["id"], "permission": "admin"},
            {"user_id": bob["user"]["id"], "permission": "read"},
        ],
    )
    shared_ids = []
    for index in range(3):
        data = f"Document {index} says refunds take {index + 10} days.\n".encode()
        accepted = await ingest(client, alice, shared["id"], db, services, data, f"s{index}.txt")
        shared_ids.append(accepted["document_id"])
    private_doc = await ingest(client, alice, private["id"], db, services, V2, "private.md")

    seen, sizes, cursor = [], [], None
    while True:
        params = {"collection_id": shared["id"], "limit": 2}
        if cursor:
            params["cursor"] = cursor
        page = (await client.get(DOCUMENTS, params=params)).json()
        sizes.append(len(page["items"]))
        seen.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    alice_all = (await client.get(DOCUMENTS)).json()["items"]
    bob_all = (await client2.get(DOCUMENTS)).json()["items"]
    bob_private = await client2.get(DOCUMENTS, params={"collection_id": private["id"]})
    bad_cursor = await client.get(DOCUMENTS, params={"cursor": "not-a-cursor"})
    non_uuid_key = base64.urlsafe_b64encode(
        json.dumps([datetime.now(UTC).isoformat(), "not-a-uuid"]).encode()
    ).decode()
    bad_key = await client.get(DOCUMENTS, params={"cursor": non_uuid_key})
    # The listing contract is "created_at desc, id desc"; ask the database for that order
    # instead of assuming the clock never stands still between two uploads.
    async with db.tenant_tx(UUID(alice["tenant"]["id"])) as conn:
        cur = await conn.execute(
            "select id from documents where collection_id = %s order by created_at desc, id desc",
            (shared["id"],),
        )
        expected = [str(row["id"]) for row in await cur.fetchall()]

    assert sizes == [2, 1]
    assert seen == expected
    assert set(seen) == set(shared_ids)
    assert {item["id"] for item in alice_all} == {*shared_ids, private_doc["document_id"]}
    assert {item["id"] for item in bob_all} == set(shared_ids)
    assert (bob_private.status_code, bob_private.json()["code"]) == (404, "collection_not_found")
    assert (bad_cursor.status_code, bad_cursor.json()["code"]) == (400, "bad_cursor")
    assert (bad_key.status_code, bad_key.json()["code"]) == (400, "bad_cursor")


async def test_non_normal_collections_are_not_reachable(client, client2, client3, db, services):
    alice, _bob, _carol, _collection, accepted = await alice_with_document(
        client, client2, client3, db, services
    )
    tenant_id = UUID(alice["tenant"]["id"])
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "insert into collections (tenant_id, name, kind) values (%s, 'Scratch', 'quick') "
            "returning id",
            (tenant_id,),
        )
        quick_id = (await cur.fetchone())["id"]
        await conn.execute(
            "update documents set collection_id = %s where id = %s",
            (quick_id, accepted["document_id"]),
        )
    document = f"{DOCUMENTS}/{accepted['document_id']}"

    responses = [
        await client.get(document),
        await client.get(f"{document}/versions/1"),
        await client.get(f"{UPLOADS}/{accepted['upload_id']}"),
        await client.delete(document, headers=csrf(alice)),
        await post_version(client, alice, accepted["document_id"]),
        await post_document(client, alice, quick_id),
    ]
    listing = await client.get(DOCUMENTS)

    assert [r.status_code for r in responses] == [404] * 6
    assert listing.json()["items"] == []


async def test_routes_require_a_session_and_csrf(client, client2, client3, db, services):
    _alice, _bob, _carol, collection, accepted = await alice_with_document(
        client, client2, client3, db, services
    )
    document = f"{DOCUMENTS}/{accepted['document_id']}"
    files = {"file": ("policy.md", V2, "application/octet-stream")}
    requests = [
        ("POST", f"{COLLECTIONS}/{collection['id']}/documents", {"files": files}),
        ("POST", f"{document}/versions", {"files": files}),
        ("GET", f"{UPLOADS}/{accepted['upload_id']}", {}),
        ("GET", DOCUMENTS, {}),
        ("GET", document, {}),
        ("DELETE", document, {}),
        ("GET", f"{document}/versions/1", {}),
        ("GET", f"{document}/versions/1/text", {}),
        ("GET", f"{document}/versions/1/passages", {}),
    ]

    client3.cookies.clear()  # Carol signed up in _team; this request set is anonymous
    anonymous = [await client3.request(method, url, **kw) for method, url, kw in requests]
    no_csrf = [await client.request(method, url, **kw) for method, url, kw in requests[:2]]
    no_csrf.append(await client.delete(document))

    assert [r.status_code for r in anonymous] == [401] * 9
    assert {r.json()["code"] for r in anonymous} == {"unauthenticated"}
    assert [r.status_code for r in no_csrf] == [403] * 3
    assert {r.json()["code"] for r in no_csrf} == {"csrf_failed"}
    assert (await client.get(document)).status_code == 200


async def test_api_key_can_upload_and_read(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    created = await client.post("/api/v1/api-keys", json={"name": "ci"}, headers=csrf(alice))
    bearer = {"Authorization": f"Bearer {created.json()['token']}"}

    response = await client3.post(
        f"{COLLECTIONS}/{collection['id']}/documents",
        files={"file": ("policy.md", V1, "application/octet-stream")},
        headers=bearer,
    )
    await drain(db, services)
    upload = await client3.get(f"{UPLOADS}/{response.json()['upload_id']}", headers=bearer)

    assert response.status_code == 202, response.text
    assert upload.json()["status"] == "created_version"


async def test_routes_never_log_filenames_or_text(client, client2, client3, db, services):
    alice, _bob, _carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    secret = b"SECRET-TEXT is in this file."

    with json_logs() as stream:
        accepted = await ingest(
            client, alice, collection["id"], db, services, secret, "SECRET-NAME.txt"
        )
        await client.get(f"{UPLOADS}/{accepted['upload_id']}")
        await client.get(f"{DOCUMENTS}/{accepted['document_id']}")
        await client.get(f"{DOCUMENTS}/{accepted['document_id']}/versions/1/text")
        await client.delete(f"{DOCUMENTS}/{accepted['document_id']}", headers=csrf(alice))

    assert "SECRET" not in stream.getvalue()
