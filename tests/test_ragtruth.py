"""RAGTruth harness: loading, source rendering, span-aligned units, splits, and ingestion."""

import hashlib
import json
from pathlib import Path

import pytest

from backend.db.repos import tenants
from backend.ingest.parsers.text import parse_text
from evaluation import data, ragtruth
from evaluation.data import DatasetFile
from evaluation.ragtruth import RTResponse, RTSource, RTSpan

pg = pytest.mark.pg

# Source ids whose R1 bucket (sha256 modulo 100) sits on a boundary or deep inside a role.
FIT_LOW, FIT_EDGE = "188", "235"  # buckets 0 and 69
SELECT_EDGE_LOW, SELECT_EDGE_HIGH = "18", "159"  # buckets 70 and 84
RISK_EDGE, RISK_HIGH = "78", "40"  # buckets 85 and 99

TEXT = "A is 1. B is 2. C is 3."  # sentences: [0, 7), [8, 15), [16, 23)


def make_response(
    text: str, labels: tuple[RTSpan, ...], *, response_id: str = "r1", split: str = "train"
) -> RTResponse:
    return RTResponse(response_id, "s1", "gpt-4", split, text, labels)


def make_source(task_type: str = "Summary") -> RTSource:
    return RTSource("s1", task_type, "CNN/DM", "source text", None)


def golds(units) -> list[str]:
    return [unit.gold for unit in units]


def test_unitize_overlap_rules():
    labels = (
        RTSpan(13, 14, "2", "Evident Conflict"),
        RTSpan(16, 20, "C is", "Subtle Baseless Info"),
    )

    units, stats = ragtruth.unitize(make_response(TEXT, labels), make_source(), "fit")

    assert golds(units) == ["supported", "contradicted", "insufficient"]
    assert [(u.start, u.end, u.text) for u in units] == [
        (0, 7, "A is 1."),
        (8, 15, "B is 2."),
        (16, 23, "C is 3."),
    ]
    assert stats == {"n_spans": 2, "n_span_text_mismatch": 0, "n_spans_without_unit": 0}


def test_unit_fields_come_from_the_response_source_and_role():
    units, _ = ragtruth.unitize(make_response(TEXT, ()), make_source("QA"), "risk")

    assert [u.unit_id for u in units] == ["r1:0", "r1:1", "r1:2"]
    assert {(u.response_id, u.source_id, u.task_type, u.role) for u in units} == {
        ("r1", "s1", "QA", "risk")
    }


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (7, 8),  # the space between the first two sentences, outside every unit
        (1, 2),  # the space inside the first sentence, a whitespace-only overlap with it
    ],
)
def test_whitespace_only_overlap_ignored(start, end):
    labels = (RTSpan(start, end, " ", "Evident Conflict"),)

    units, stats = ragtruth.unitize(make_response(TEXT, labels), make_source(), "fit")

    assert golds(units) == ["supported", "supported", "supported"]
    assert stats["n_spans_without_unit"] == 1


def test_span_across_sentences_marks_every_overlapped_unit():
    # "1. B" crosses the sentence boundary, so both neighbouring units carry the label.
    labels = (RTSpan(5, 9, "1. B", "Subtle Baseless Info"),)

    units, stats = ragtruth.unitize(make_response(TEXT, labels), make_source(), "fit")

    assert golds(units) == ["insufficient", "insufficient", "supported"]
    assert stats["n_spans_without_unit"] == 0


@pytest.mark.parametrize("conflict", ["Evident Conflict", "Subtle Conflict"])
def test_conflict_beats_baseless(conflict):
    labels = (
        RTSpan(16, 20, "C is", "Evident Baseless Info"),
        RTSpan(21, 22, "3", conflict),
    )

    units, _ = ragtruth.unitize(make_response(TEXT, labels), make_source(), "fit")

    assert golds(units) == ["supported", "supported", "contradicted"]


def test_span_text_mismatch_counted_and_offsets_still_used():
    labels = (
        RTSpan(13, 14, "wrong text", "Evident Conflict"),
        RTSpan(16, 20, "C is", "Subtle Baseless Info"),
    )

    units, stats = ragtruth.unitize(make_response(TEXT, labels), make_source(), "fit")

    assert stats == {"n_spans": 2, "n_span_text_mismatch": 1, "n_spans_without_unit": 0}
    assert golds(units) == ["supported", "contradicted", "insufficient"]


def test_unitize_empty_response_has_no_units():
    units, stats = ragtruth.unitize(make_response("", ()), make_source(), "fit")

    assert units == []
    assert stats == {"n_spans": 0, "n_span_text_mismatch": 0, "n_spans_without_unit": 0}


def test_render_summary_has_no_question():
    assert ragtruth.render_source("Summary", "The article text.") == ("The article text.", None)


def test_render_qa_returns_question():
    info = {"passages": "passage 1: a\n\npassage 2: b", "question": "what is a?"}

    assert ragtruth.render_source("QA", info) == ("passage 1: a\n\npassage 2: b", "what is a?")


def test_render_data2txt_literal():
    info = {
        "name": "X",
        "hours": {"Mon": "9-5"},
        "tags": ["a", "b"],
        "x": None,
        "ok": True,
    }

    text, question = ragtruth.render_source("Data2txt", info)

    assert text == "name: X\nhours.Mon: 9-5\ntags.0: a\ntags.1: b\nok: true"
    assert question is None


def test_render_data2txt_numbers_booleans_and_nested_lists():
    info = {
        "count": 3,
        "rating": 4.5,
        "closed": False,
        "reviews": [{"text": None}, {"text": "good", "stars": 5}],
        "empty": [],
    }

    text, _ = ragtruth.render_source("Data2txt", info)

    # A skipped None leaves its list index unused; empty containers add no line.
    assert text == "count: 3\nrating: 4.5\nclosed: false\nreviews.1.text: good\nreviews.1.stars: 5"


def test_render_unknown_task_type_raises():
    with pytest.raises(ValueError, match="task type"):
        ragtruth.render_source("Poetry", "x")


@pytest.mark.parametrize(
    ("source_id", "split", "role"),
    [
        (FIT_LOW, "train", "fit"),
        (FIT_EDGE, "train", "fit"),
        (SELECT_EDGE_LOW, "train", "select"),
        (SELECT_EDGE_HIGH, "train", "select"),
        (RISK_EDGE, "train", "risk"),
        (RISK_HIGH, "train", "risk"),
        (FIT_LOW, "test", "test"),
        (RISK_HIGH, "test", "test"),
    ],
)
def test_split_roles_deterministic(source_id, split, role):
    assert ragtruth.split_role(source_id, split) == role
    assert ragtruth.split_role(source_id, split) == role  # the same call, the same answer


def test_split_role_buckets_follow_the_sha256_rule():
    for number in range(300):
        source_id = str(number)
        bucket = int(hashlib.sha256(source_id.encode()).hexdigest(), 16) % 100
        expected = "fit" if bucket < 70 else "select" if bucket < 85 else "risk"
        assert ragtruth.split_role(source_id, "train") == expected


def make_corpus(
    n_sources: int = 400,
) -> tuple[dict[str, RTSource], list[RTResponse]]:
    """Unequal task populations; every fifth source belongs to the test split."""
    sources: dict[str, RTSource] = {}
    responses: list[RTResponse] = []
    for number in range(n_sources):
        source_id = str(number)
        task_type = ["Summary", "Summary", "Summary", "Summary", "QA", "QA", "Data2txt"][number % 7]
        sources[source_id] = RTSource(source_id, task_type, "src", f"text {number}", None)
        split = "test" if number % 5 == 0 else "train"
        for model in ("m1", "m2"):
            response_id = f"{source_id}-{model}"
            responses.append(RTResponse(response_id, source_id, model, split, "x", ()))
    return sources, responses


SHA = {"response.jsonl": "a" * 64, "source_info.jsonl": "b" * 64}
SIZES = {"fit": 40, "select": 15, "risk": 9, "test": None}


def populations(sources, responses) -> dict[str, dict[str, int]]:
    """Sources per role and task type, computed straight from the split rule."""
    split_of = {r.source_id: r.split for r in responses}
    result: dict[str, dict[str, int]] = {}
    for source_id, source in sources.items():
        role = ragtruth.split_role(source_id, split_of[source_id])
        tasks = result.setdefault(role, {})
        tasks[source.task_type] = tasks.get(source.task_type, 0) + 1
    return result


def test_manifest_stratified_and_disjoint():
    sources, responses = make_corpus()
    population = populations(sources, responses)
    split_of = {r.source_id: r.split for r in responses}

    manifest = ragtruth.build_manifest(sources, responses, sizes=SIZES, data_sha256=SHA)

    roles = manifest["roles"]
    assert set(roles) == {"fit", "select", "risk", "test"}
    for role, size in SIZES.items():
        total = sum(population[role].values())
        wanted = total if size is None else size
        assert len(roles[role]) == wanted, role
        assert roles[role] == sorted(roles[role]), role
        for task, available in population[role].items():
            chosen = sum(1 for sid in roles[role] if sources[sid].task_type == task)
            target = wanted * available / total
            assert abs(chosen - target) < 1, (role, task)
            assert manifest["counts"][role]["sources"][task] == chosen
    everything = [sid for ids in roles.values() for sid in ids]
    assert len(everything) == len(set(everything))  # roles are disjoint
    for role, ids in roles.items():
        assert {ragtruth.split_role(sid, split_of[sid]) for sid in ids} <= {role}
    # None keeps every source of the role.
    assert len(roles["test"]) == sum(population["test"].values())


def test_manifest_records_inputs_and_counts_responses_of_chosen_sources():
    sources, responses = make_corpus(60)

    manifest = ragtruth.build_manifest(
        sources, responses, sizes={"fit": 3, "select": 2, "risk": 1, "test": 4}, data_sha256=SHA
    )

    assert manifest["dataset"] == "ragtruth"
    assert manifest["commit"] == "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
    assert manifest["seed"] == 20261026
    assert manifest["sizes"] == {"fit": 3, "select": 2, "risk": 1, "test": 4}
    assert manifest["data_sha256"] == SHA
    assert set(manifest) == {
        "dataset",
        "commit",
        "seed",
        "sizes",
        "roles",
        "counts",
        "data_sha256",
    }
    for role, ids in manifest["roles"].items():
        # Two responses per source in this corpus.
        assert manifest["counts"][role]["responses"] == 2 * len(ids)
        assert sum(manifest["counts"][role]["sources"].values()) == len(ids)


def test_manifest_same_seed_same_manifest():
    sources, responses = make_corpus()

    first = ragtruth.build_manifest(sources, responses, sizes=SIZES, data_sha256=SHA)
    again = ragtruth.build_manifest(sources, responses, sizes=SIZES, data_sha256=SHA)
    other = ragtruth.build_manifest(sources, responses, sizes=SIZES, seed=7, data_sha256=SHA)

    assert again == first
    assert ragtruth.manifest_hash(again) == ragtruth.manifest_hash(first)
    assert other["roles"]["fit"] != first["roles"]["fit"]
    assert ragtruth.manifest_hash(other) != ragtruth.manifest_hash(first)
    assert other["seed"] == 7


def test_manifest_does_not_depend_on_input_order():
    sources, responses = make_corpus()

    forward = ragtruth.build_manifest(sources, responses, sizes=SIZES, data_sha256=SHA)
    shuffled = ragtruth.build_manifest(
        dict(reversed(sources.items())), responses[::-1], sizes=SIZES, data_sha256=SHA
    )

    assert shuffled == forward


def test_manifest_hash_covers_roles_only():
    roles = {"fit": ["1", "2"], "select": [], "risk": ["3"], "test": ["4"]}
    canonical = json.dumps(roles, sort_keys=True, separators=(",", ":"))

    one = {"roles": roles, "seed": 1, "data_sha256": {"a": "x"}}
    two = {"roles": roles, "seed": 2, "data_sha256": {"a": "y"}}

    assert ragtruth.manifest_hash(one) == hashlib.sha256(canonical.encode()).hexdigest()
    assert ragtruth.manifest_hash(two) == ragtruth.manifest_hash(one)
    assert ragtruth.manifest_hash({"roles": {**roles, "fit": ["1"]}}) != ragtruth.manifest_hash(one)


def test_manifest_rejects_a_source_split_across_train_and_test():
    sources, responses = make_corpus(30)
    leaked = responses[1]  # the second response of source "0", a test source
    responses[1] = RTResponse(leaked.response_id, leaked.source_id, leaked.model, "train", "x", ())

    with pytest.raises(ValueError, match="split"):
        ragtruth.build_manifest(
            sources, responses, sizes={"fit": 1, "select": 1, "risk": 1, "test": 1}, data_sha256=SHA
        )


def test_manifest_rejects_a_size_above_the_population():
    sources, responses = make_corpus(30)

    with pytest.raises(ValueError, match="exceeds"):
        ragtruth.build_manifest(sources, responses, sizes={**SIZES, "fit": 10_000}, data_sha256=SHA)


def test_manifest_requires_a_size_for_every_role():
    sources, responses = make_corpus(30)

    with pytest.raises(ValueError, match="roles"):
        ragtruth.build_manifest(sources, responses, sizes={"fit": 1}, data_sha256=SHA)


def test_allocate_largest_remainder_gives_the_remainder_to_the_largest_fractions():
    # Quotas 2.0, 1.2, 0.8: floors 2, 1, 0 leave one seat, which the 0.8 takes.
    assert ragtruth.allocate({"a": 5, "b": 3, "c": 2}, 4) == {"a": 2, "b": 1, "c": 1}


def test_allocate_breaks_ties_by_task_name():
    assert ragtruth.allocate({"b": 1, "a": 1}, 1) == {"a": 1, "b": 0}


def test_allocate_hits_every_total_exactly():
    counts = {"Summary": 943, "QA": 989, "Data2txt": 1033}
    for size in (0, 1, 2, 7, 100, 999, 2964, 2965):
        allocation = ragtruth.allocate(counts, size)
        assert sum(allocation.values()) == size
        assert all(0 <= allocation[task] <= counts[task] for task in counts)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def label(start: int, end: int, text: str, label_type: str, **flags: bool) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        "label_type": label_type,
        "meta": "note",
        "implicit_true": flags.get("implicit_true", False),
        "due_to_null": flags.get("due_to_null", False),
    }


def source_row(source_id: str, task_type: str, info) -> dict:
    return {
        "source_id": source_id,
        "task_type": task_type,
        "source": f"name-{task_type}",
        "source_info": info,
        "prompt": "unused prompt",
    }


def response_row(response_id: str, source_id: str, split: str, labels: list[dict]) -> dict:
    return {
        "id": response_id,
        "source_id": source_id,
        "model": "gpt-4",
        "temperature": 0.7,
        "labels": labels,
        "split": split,
        "quality": "good",
        "response": "Alpha is wrong. Beta is fine.",
    }


def write_ragtruth(root: Path) -> None:
    """Seven sources spread over every role, two responses each, in the fetched file layout."""
    root.mkdir(parents=True, exist_ok=True)
    ids = [FIT_LOW, FIT_EDGE, SELECT_EDGE_LOW, SELECT_EDGE_HIGH, RISK_EDGE, RISK_HIGH, "7"]
    infos = [
        ("Summary", "Alpha is the first body."),
        ("QA", {"passages": "passage one", "question": "what is one?"}),
        ("Data2txt", {"name": "X", "hours": {"Mon": "9-5"}, "ok": True}),
    ]
    write_jsonl(
        root / "source_info.jsonl",
        [source_row(sid, *infos[i % 3]) for i, sid in enumerate(ids)],
    )
    rows = []
    for sid in ids:
        split = "test" if sid == "7" else "train"
        rows.append(
            response_row(f"{sid}-a", sid, split, [label(0, 5, "Alpha", "Evident Conflict")])
        )
        rows.append(response_row(f"{sid}-b", sid, split, []))
    write_jsonl(root / "response.jsonl", rows)


def test_load_ragtruth_reads_sources_and_every_response(tmp_path):
    write_ragtruth(tmp_path)
    # A refused, low-quality response with implicit and null-caused labels must still load.
    bad = response_row(
        "extra",
        FIT_LOW,
        "train",
        [
            label(6, 14, "is wrong", "Subtle Baseless Info", implicit_true=True),
            label(15, 19, "Beta", "Subtle Conflict", due_to_null=True),
        ],
    )
    bad["quality"] = "incorrect_refusal"
    with (tmp_path / "response.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(bad) + "\n")

    sources, responses = ragtruth.load_ragtruth(tmp_path)

    assert list(sources) == [
        FIT_LOW,
        FIT_EDGE,
        SELECT_EDGE_LOW,
        SELECT_EDGE_HIGH,
        RISK_EDGE,
        RISK_HIGH,
        "7",
    ]
    assert sources[FIT_LOW] == RTSource(
        FIT_LOW, "Summary", "name-Summary", "Alpha is the first body.", None
    )
    assert sources[FIT_EDGE] == RTSource(FIT_EDGE, "QA", "name-QA", "passage one", "what is one?")
    assert sources[SELECT_EDGE_LOW] == RTSource(
        SELECT_EDGE_LOW, "Data2txt", "name-Data2txt", "name: X\nhours.Mon: 9-5\nok: true", None
    )
    assert len(responses) == 15
    first, last = responses[0], responses[-1]
    assert (first.response_id, first.source_id, first.model, first.split) == (
        f"{FIT_LOW}-a",
        FIT_LOW,
        "gpt-4",
        "train",
    )
    assert first.response == "Alpha is wrong. Beta is fine."
    assert first.labels == (RTSpan(0, 5, "Alpha", "Evident Conflict"),)
    assert last.response_id == "extra"
    assert last.labels == (
        RTSpan(6, 14, "is wrong", "Subtle Baseless Info"),
        RTSpan(15, 19, "Beta", "Subtle Conflict"),
    )
    assert {r.split for r in responses} == {"train", "test"}


def test_parse_sizes_accepts_numbers_and_all():
    parsed = ragtruth.parse_sizes("fit=5,select=all,risk=0,test=all")

    assert parsed == {"fit": 5, "select": None, "risk": 0, "test": None}


@pytest.mark.parametrize(
    "text",
    [
        "fit=1",
        "fit=1,select=1,risk=1",
        "fit=1,select=1,risk=1,test=x",
        "fit=-1,select=1,risk=1,test=1",
    ],
)
def test_parse_sizes_rejects_bad_input(text):
    with pytest.raises(ValueError, match="sizes"):
        ragtruth.parse_sizes(text)


def test_manifest_command_writes_canonical_json(tmp_path, monkeypatch):
    root = tmp_path / "data"
    write_ragtruth(root)
    shas = {}
    for registry_name, filename in (
        ("ragtruth_response", "response.jsonl"),
        ("ragtruth_source", "source_info.jsonl"),
    ):
        shas[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        monkeypatch.setitem(
            data.REGISTRY,
            registry_name,
            DatasetFile(registry_name, f"https://example.test/{filename}", shas[filename], "file"),
        )
    out = tmp_path / "manifests" / "ragtruth_splits.json"

    exit_code = ragtruth.main(
        [
            "manifest",
            "--sizes",
            "fit=1,select=all,risk=1,test=all",
            "--out",
            str(out),
            "--data-root",
            str(root),
        ]
    )

    assert exit_code == 0
    text = out.read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert text == json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    assert manifest["roles"]["select"] == sorted([SELECT_EDGE_LOW, SELECT_EDGE_HIGH])
    assert manifest["roles"]["test"] == ["7"]
    assert len(manifest["roles"]["fit"]) == 1
    assert len(manifest["roles"]["risk"]) == 1
    assert manifest["sizes"] == {"fit": 1, "select": None, "risk": 1, "test": None}
    assert manifest["data_sha256"] == shas


async def make_tenant(db):
    async with db.system_tx() as conn:
        return (await tenants.insert_tenant(conn, "bench"))["id"]


RT_SOURCES = {
    "s1": RTSource("s1", "Summary", "CNN/DM", "Alpha is the first body.", None),
    "s2": RTSource("s2", "QA", "MARCO", "Beta is the second body.", "what is beta?"),
    "s3": RTSource("s3", "Data2txt", "yelp", "Gamma is the third body.", None),
}


async def rows(db, tenant_id) -> dict[str, list[tuple]]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select name, kind, id from collections order by name")
        collections = [(r["name"], r["kind"], r["id"]) for r in await cur.fetchall()]
        cur = await conn.execute(
            "select c.name as collection, d.filename, d.title, v.text "
            "from documents d join collections c on c.id = d.collection_id "
            "join document_versions v on v.id = d.current_version_id order by d.filename"
        )
        documents = [
            (r["collection"], r["filename"], r["title"], r["text"]) for r in await cur.fetchall()
        ]
        cur = await conn.execute("select count(*) as n from document_versions")
        versions = [(await cur.fetchone())["n"]]
    return {"collections": collections, "documents": documents, "versions": versions}


@pg
async def test_ingest_ragtruth_idempotent(db, services):
    tenant_id = await make_tenant(db)

    first = await ragtruth.ingest_ragtruth(services, RT_SOURCES, ["s1", "s2"], tenant_id=tenant_id)
    before = await rows(db, tenant_id)
    again = await ragtruth.ingest_ragtruth(services, RT_SOURCES, ["s1", "s2"], tenant_id=tenant_id)
    after = await rows(db, tenant_id)
    later = await ragtruth.ingest_ragtruth(services, RT_SOURCES, ["s2", "s3"], tenant_id=tenant_id)

    assert set(first) == {"s1", "s2"}
    assert first["s1"] != first["s2"]
    assert again == first
    # The rerun created no collection, document, or version.
    assert after == before
    assert [(name, kind) for name, kind, _ in before["collections"]] == [
        ("rt-s1", "benchmark"),
        ("rt-s2", "benchmark"),
    ]
    assert before["documents"] == [
        ("rt-s1", "rt-s1.txt", "rt-s1", "Alpha is the first body."),
        ("rt-s2", "rt-s2.txt", "rt-s2", "Beta is the second body."),
    ]
    assert before["versions"] == [2]
    # A later run reuses s2 and adds only s3.
    assert set(later) == {"s2", "s3"}
    assert later["s2"] == first["s2"]
    final = await rows(db, tenant_id)
    assert [name for name, _, _ in final["collections"]] == ["rt-s1", "rt-s2", "rt-s3"]
    assert final["versions"] == [3]


@pg
async def test_ingest_ragtruth_stores_the_rendered_source_text(db, services):
    tenant_id = await make_tenant(db)
    text = "name: X\nhours.Mon: 9-5\n\nSecond paragraph."
    sources = {"s9": RTSource("s9", "Data2txt", "yelp", text, None)}

    await ragtruth.ingest_ragtruth(services, sources, ["s9"], tenant_id=tenant_id)

    stored = (await rows(db, tenant_id))["documents"]
    assert [row[3] for row in stored] == [parse_text(text.encode()).text]


@pg
async def test_ingest_ragtruth_refuses_a_collection_that_is_not_a_benchmark(db, services):
    tenant_id = await make_tenant(db)
    async with db.tenant_tx(tenant_id) as conn:
        await conn.execute(
            "insert into collections (tenant_id, name) values (%s, 'rt-s1')", (tenant_id,)
        )

    with pytest.raises(ValueError, match="benchmark"):
        await ragtruth.ingest_ragtruth(services, RT_SOURCES, ["s1"], tenant_id=tenant_id)

    assert (await rows(db, tenant_id))["documents"] == []


@pg
async def test_ingest_ragtruth_rejects_a_source_without_text_and_creates_nothing(db, services):
    tenant_id = await make_tenant(db)
    sources = {"e1": RTSource("e1", "Summary", "CNN/DM", " \n ", None)}

    with pytest.raises(ValueError, match="no text"):
        await ragtruth.ingest_ragtruth(services, sources, ["e1"], tenant_id=tenant_id)

    after = await rows(db, tenant_id)
    assert (after["collections"], after["documents"]) == ([], [])
