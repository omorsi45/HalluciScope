"""Sentence-level version diff and its route."""

import random
from itertools import pairwise
from uuid import uuid4

import pytest

from backend.ingest.diff import DiffHunk, _merge_adjacent, diff_texts
from tests.helpers.auth import csrf
from tests.test_collections import _create, _put_grants, _team
from tests.test_document_routes import DOCUMENTS, ingest, ingest_version

OLD = "Refunds take 30 days. Shipping is free."
NEW = "Refunds take 14 days. Shipping is free."


def test_identical_texts_single_equal():
    assert diff_texts(OLD, OLD) == [DiffHunk("equal", 0, len(OLD), 0, len(OLD), "", "")]


def test_identical_one_sentence_text_single_equal():
    text = "Refunds take 30 days."

    assert diff_texts(text, text) == [DiffHunk("equal", 0, len(text), 0, len(text), "", "")]


def test_replace_one_sentence():
    hunks = diff_texts(OLD, NEW)

    assert [h.op for h in hunks] == ["replace", "equal"]
    replaced, kept = hunks
    assert OLD[replaced.from_start : replaced.from_end] == "Refunds take 30 days."
    assert NEW[replaced.to_start : replaced.to_end] == "Refunds take 14 days."
    assert replaced.from_text == "Refunds take 30 days."
    assert replaced.to_text == "Refunds take 14 days."
    assert (kept.from_text, kept.to_text) == ("", "")
    assert OLD[kept.from_start : kept.from_end] == "Shipping is free."
    assert NEW[kept.to_start : kept.to_end] == "Shipping is free."


def test_insert_appended_sentence_sits_at_end_of_old():
    old = "Refunds take 30 days. "
    new = old + "Shipping is free."

    hunks = diff_texts(old, new)

    assert [h.op for h in hunks] == ["equal", "insert"]
    inserted = hunks[1]
    assert inserted.from_start == inserted.from_end == len(old)
    assert (inserted.from_text, inserted.to_text) == ("", "Shipping is free.")
    assert new[inserted.to_start : inserted.to_end] == "Shipping is free."


def test_delete_trailing_sentence_sits_at_end_of_new():
    new = "Refunds take 30 days. "
    old = new + "Shipping is free."

    hunks = diff_texts(old, new)

    assert [h.op for h in hunks] == ["equal", "delete"]
    deleted = hunks[1]
    assert deleted.to_start == deleted.to_end == len(new)
    assert (deleted.from_text, deleted.to_text) == ("Shipping is free.", "")
    assert old[deleted.from_start : deleted.from_end] == "Shipping is free."


def test_insert_at_beginning_sits_at_start_of_first_old_sentence():
    old = "  Shipping is free."
    new = "Refunds take 30 days. Shipping is free."

    hunks = diff_texts(old, new)

    assert [h.op for h in hunks] == ["insert", "equal"]
    inserted = hunks[0]
    assert inserted.from_start == inserted.from_end == old.index("Shipping")
    assert inserted.to_text == "Refunds take 30 days."


def test_delete_at_beginning_sits_at_start_of_first_new_sentence():
    old = "Refunds take 30 days. Shipping is free."
    new = "  Shipping is free."

    hunks = diff_texts(old, new)

    assert [h.op for h in hunks] == ["delete", "equal"]
    deleted = hunks[0]
    assert deleted.to_start == deleted.to_end == new.index("Shipping")
    assert deleted.from_text == "Refunds take 30 days."


def test_insert_in_middle_sits_at_end_of_preceding_old_sentence():
    old = "One is here.   Three is here."
    new = "One is here. Two is new. Three is here."

    hunks = diff_texts(old, new)

    assert [h.op for h in hunks] == ["equal", "insert", "equal"]
    inserted = hunks[1]
    assert inserted.from_start == inserted.from_end == old.index("   Three")
    assert inserted.to_text == "Two is new."


def test_delete_in_middle_sits_at_end_of_preceding_new_sentence():
    old = "One is here. Two is old. Three is here."
    new = "One is here.   Three is here."

    hunks = diff_texts(old, new)

    assert [h.op for h in hunks] == ["equal", "delete", "equal"]
    deleted = hunks[1]
    assert deleted.to_start == deleted.to_end == new.index("   Three")
    assert deleted.from_text == "Two is old."


def test_equal_hunk_spans_the_equal_sentences_in_each_text():
    old = "Keep one. Keep two. Old three."
    new = "Keep one.\n\nKeep two. New three."

    hunks = diff_texts(old, new)

    assert [h.op for h in hunks] == ["equal", "replace"]
    kept = hunks[0]
    assert old[kept.from_start : kept.from_end] == "Keep one. Keep two."
    assert new[kept.to_start : kept.to_end] == "Keep one.\n\nKeep two."
    assert (kept.from_text, kept.to_text) == ("", "")


def test_non_ascii_offsets_are_code_points():
    old = "Cafe\u0301 \U0001f600 opens. Tokyo \u6771\u4eac closes."
    new = "Cafe\u0301 \U0001f600 opens. Tokyo \u6771\u4eac opens."

    hunks = diff_texts(old, new)

    replaced = hunks[1]
    assert [h.op for h in hunks] == ["equal", "replace"]
    assert old[replaced.from_start : replaced.from_end] == "Tokyo \u6771\u4eac closes."
    assert new[replaced.to_start : replaced.to_end] == "Tokyo \u6771\u4eac opens."
    assert replaced.to_start == len("Cafe\u0301 \U0001f600 opens. ")


def test_adjacent_hunks_with_the_same_op_merge():
    first = DiffHunk("delete", 0, 4, 0, 0, "abcd", "")
    second = DiffHunk("delete", 4, 9, 0, 0, "efghi", "")
    kept = DiffHunk("equal", 9, 12, 0, 3, "", "")

    merged = _merge_adjacent([first, second, kept], "abcdefghijkl", "xyz")

    assert merged == [DiffHunk("delete", 0, 9, 0, 0, "abcdefghi", ""), kept]


@pytest.mark.parametrize("seed", range(20))
def test_offsets_slice_exactly(seed):
    rng = random.Random(seed)
    old_sentences = [f"Sentence {i} states fact number {i * 7}." for i in range(40)]
    new_sentences = []
    for sentence in old_sentences:
        roll = rng.random()
        if roll < 0.15:
            continue
        new_sentences.append(sentence if roll < 0.6 else sentence.replace("fact", "claim"))
        if rng.random() < 0.15:
            new_sentences.append(f"Inserted sentence {rng.randrange(1000)} appears here.")
    old, new = " ".join(old_sentences), " ".join(new_sentences)

    hunks = diff_texts(old, new)

    changed = [h for h in hunks if h.op != "equal"]
    assert changed
    for h in hunks:
        assert 0 <= h.from_start <= h.from_end <= len(old)
        assert 0 <= h.to_start <= h.to_end <= len(new)
        if h.op == "equal":
            assert (h.from_text, h.to_text) == ("", "")
            assert old[h.from_start : h.from_end] == new[h.to_start : h.to_end]
        else:
            assert old[h.from_start : h.from_end] == h.from_text
            assert new[h.to_start : h.to_end] == h.to_text
        if h.op == "insert":
            assert h.from_start == h.from_end
        if h.op == "delete":
            assert h.to_start == h.to_end
    assert [h.from_start for h in hunks] == sorted(h.from_start for h in hunks)
    assert [h.to_start for h in hunks] == sorted(h.to_start for h in hunks)
    assert all(a.op != b.op for a, b in pairwise(hunks))


async def _two_versions(client, client2, client3, db, services):
    alice, bob, carol = await _team(client, client2, client3, db)
    collection = await _create(client, alice)
    first = await ingest(client, alice, collection["id"], db, services, OLD.encode())
    await ingest_version(client, alice, first["document_id"], db, services, NEW.encode())
    return alice, bob, carol, collection, f"{DOCUMENTS}/{first['document_id']}"


@pytest.mark.pg
async def test_diff_route(client, client2, client3, db, services):
    _alice, _bob, _carol, _collection, document = await _two_versions(
        client, client2, client3, db, services
    )

    response = await client.get(f"{document}/diff", params={"from": 1, "to": 2})
    unknown_to = await client.get(f"{document}/diff", params={"from": 1, "to": 3})
    unknown_from = await client.get(f"{document}/diff", params={"from": 0, "to": 2})
    old_text = (await client.get(f"{document}/versions/1/text")).json()["text"]
    new_text = (await client.get(f"{document}/versions/2/text")).json()["text"]

    body = response.json()
    assert response.status_code == 200, response.text
    assert set(body) == {"from_version", "to_version", "hunks", "stats"}
    assert (body["from_version"], body["to_version"]) == (1, 2)
    assert body["stats"] == {"equal": 1, "insert": 0, "delete": 0, "replace": 1}
    replaced = next(h for h in body["hunks"] if h["op"] == "replace")
    assert set(replaced) == {
        "op",
        "from_start",
        "from_end",
        "to_start",
        "to_end",
        "from_text",
        "to_text",
    }
    assert old_text[replaced["from_start"] : replaced["from_end"]] == "Refunds take 30 days."
    assert new_text[replaced["to_start"] : replaced["to_end"]] == "Refunds take 14 days."
    assert (replaced["from_text"], replaced["to_text"]) == (
        "Refunds take 30 days.",
        "Refunds take 14 days.",
    )
    for rejected in (unknown_to, unknown_from):
        assert rejected.status_code == 404
        assert rejected.json()["code"] == "version_not_found"


@pytest.mark.pg
async def test_diff_route_same_version_is_one_equal_hunk(client, client2, client3, db, services):
    _alice, _bob, _carol, _collection, document = await _two_versions(
        client, client2, client3, db, services
    )
    text = (await client.get(f"{document}/versions/2/text")).json()["text"]

    response = await client.get(f"{document}/diff", params={"from": 2, "to": 2})
    unknown = await client.get(f"{document}/diff", params={"from": 7, "to": 7})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "from_version": 2,
        "to_version": 2,
        "hunks": [
            {
                "op": "equal",
                "from_start": 0,
                "from_end": len(text),
                "to_start": 0,
                "to_end": len(text),
                "from_text": "",
                "to_text": "",
            }
        ],
        "stats": {"equal": 1, "insert": 0, "delete": 0, "replace": 0},
    }
    assert (unknown.status_code, unknown.json()["code"]) == (404, "version_not_found")


@pytest.mark.pg
async def test_diff_route_requires_both_numbers(client, client2, client3, db, services):
    _alice, _bob, _carol, _collection, document = await _two_versions(
        client, client2, client3, db, services
    )

    responses = [
        await client.get(f"{document}/diff"),
        await client.get(f"{document}/diff", params={"from": 1}),
        await client.get(f"{document}/diff", params={"to": 2}),
        await client.get(f"{document}/diff", params={"from": "x", "to": 2}),
    ]

    assert [r.status_code for r in responses] == [422] * 4
    assert {r.json()["code"] for r in responses} == {"validation_error"}


@pytest.mark.pg
@pytest.mark.parametrize("who", ["carol", "bob"])
async def test_diff_route_unreadable_document_is_404(client, client2, client3, db, services, who):
    _alice, _bob, _carol, _collection, document = await _two_versions(
        client, client2, client3, db, services
    )
    outsider = client3 if who == "carol" else client2
    params = {"from": 1, "to": 2}

    real = await outsider.get(f"{document}/diff", params=params)
    unknown = await outsider.get(f"{DOCUMENTS}/{uuid4()}/diff", params=params)

    assert (real.status_code, real.json()["code"]) == (404, "document_not_found")
    assert real.json() == unknown.json()
    assert document.rsplit("/", 1)[1] not in real.text


@pytest.mark.pg
async def test_diff_route_read_permission_is_enough(client, client2, client3, db, services):
    alice, bob, _carol, collection, document = await _two_versions(
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

    response = await client2.get(f"{document}/diff", params={"from": 1, "to": 2})

    assert response.status_code == 200, response.text
    assert response.json()["stats"]["replace"] == 1


@pytest.mark.pg
async def test_diff_route_soft_deleted_document_is_404(client, client2, client3, db, services):
    alice, _bob, _carol, _collection, document = await _two_versions(
        client, client2, client3, db, services
    )

    await client.delete(document, headers=csrf(alice))
    response = await client.get(f"{document}/diff", params={"from": 1, "to": 2})

    assert (response.status_code, response.json()["code"]) == (404, "document_not_found")
