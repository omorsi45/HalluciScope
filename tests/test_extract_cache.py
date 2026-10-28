"""Signal cache, RAGTruth extraction runner, throughput measurement, and compute-budget sizing."""

import hashlib
import json
import time
from pathlib import Path

import pytest

from backend.verify.types import Claim, EvidenceSignal, NLIResult
from evaluation import data, extract, ragtruth
from evaluation.cache import SignalCache, config_key
from evaluation.data import DatasetFile
from evaluation.ragtruth import RTResponse, RTSource
from tests.fakes import HashEmbedder, KeywordReranker, RuleNLIScorer
from tests.test_ragtruth import (
    FIT_EDGE,
    FIT_LOW,
    RISK_EDGE,
    RISK_HIGH,
    SELECT_EDGE_HIGH,
    SELECT_EDGE_LOW,
    write_ragtruth,
)

pg = pytest.mark.pg


def signal(
    passage_id: str, text: str = "A premise.", *, rerank: float | None = 1.5
) -> EvidenceSignal:
    # Values that are not exact in binary, so a lossy round trip would show.
    return EvidenceSignal(passage_id, text, NLIResult(0.7, 0.2, 0.1), rerank, 0.1234567890123)


def signals_path(root: Path, key: str) -> Path:
    return root / key / "signals.jsonl"


def test_cache_roundtrip_and_resume(tmp_path):
    first = [signal("p1", "Caf\u00e9 body."), signal("p2", rerank=None)]
    SignalCache(tmp_path, "k1").put_many([("u1", first), ("u2", [])])
    # A crash in the middle of writing the third record leaves half a line.
    with signals_path(tmp_path, "k1").open("ab") as handle:
        handle.write(b'{"unit_id": "u3", "signals": [{"passage_id": "p9", "te')

    resumed = SignalCache(tmp_path, "k1")

    assert "u1" in resumed
    assert "u2" in resumed
    assert "u3" not in resumed
    assert resumed.get("u1") == first
    assert resumed.get("u2") == []
    assert resumed.get("u3") is None
    assert resumed.load_all() == {"u1": first, "u2": []}


def test_cache_appends_after_a_crash_without_gluing_records(tmp_path):
    SignalCache(tmp_path, "k").put_many([("u1", [signal("p1")])])
    with signals_path(tmp_path, "k").open("ab") as handle:
        handle.write(b'{"unit_id": "u2", "sig')

    SignalCache(tmp_path, "k").put_many([("u2", [signal("p2")]), ("u3", [])])

    lines = signals_path(tmp_path, "k").read_bytes().split(b"\n")
    assert [json.loads(line)["unit_id"] for line in lines[:-1]] == ["u1", "u2", "u3"]
    assert lines[-1] == b""
    assert SignalCache(tmp_path, "k").load_all() == {
        "u1": [signal("p1")],
        "u2": [signal("p2")],
        "u3": [],
    }


def test_cache_serves_units_written_through_the_same_instance(tmp_path):
    cache = SignalCache(tmp_path, "k")
    cache.put_many([("u1", [signal("p1"), signal("p2")]), ("u2", [signal("p3")])])
    cache.put_many([("u3", [signal("p4", "Later body.")])])

    assert cache.get("u2") == [signal("p3")]
    assert cache.get("u3") == [signal("p4", "Later body.")]
    assert cache.get("u1") == [signal("p1"), signal("p2")]


def test_cache_notices_what_another_instance_appended(tmp_path):
    stale = SignalCache(tmp_path, "k")
    SignalCache(tmp_path, "k").put_many([("u1", [signal("p1")])])

    # The stale instance has not seen u1: it must neither store it again (this copy differs)
    # nor cut the file back to the empty state it remembers.
    stale.put_many([("u1", [signal("different")]), ("u2", [signal("p2")])])

    assert stale.get("u2") == [signal("p2")]
    assert SignalCache(tmp_path, "k").load_all() == {"u1": [signal("p1")], "u2": [signal("p2")]}


def test_cache_does_not_store_a_unit_twice(tmp_path):
    cache = SignalCache(tmp_path, "k")
    cache.put_many([("u1", [signal("p1")])])
    before = signals_path(tmp_path, "k").read_bytes()

    cache.put_many([("u1", [signal("p1")]), ("u1", [signal("other")])])  # same batch and later
    SignalCache(tmp_path, "k").put_many([("u1", [signal("p1")])])  # another process, a rerun

    assert signals_path(tmp_path, "k").read_bytes() == before


def test_cache_line_layout(tmp_path):
    signals = [
        EvidenceSignal("p1", "Premise text.", NLIResult(0.75, 0.125, 0.125), 2.5, 0.5),
        EvidenceSignal("p2", "Other.", NLIResult(0.25, 0.5, 0.25), None, -0.5),
    ]

    SignalCache(tmp_path, "k").put_many([("188-a:0", signals)])

    [line] = signals_path(tmp_path, "k").read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {
        "unit_id": "188-a:0",
        "signals": [
            {
                "passage_id": "p1",
                "text": "Premise text.",
                "nli": {"entail": 0.75, "contra": 0.125, "neutral": 0.125},
                "rerank": 2.5,
                "dense_cos": 0.5,
            },
            {
                "passage_id": "p2",
                "text": "Other.",
                "nli": {"entail": 0.25, "contra": 0.5, "neutral": 0.25},
                "rerank": None,
                "dense_cos": -0.5,
            },
        ],
    }


def test_cache_rejects_a_corrupt_complete_line(tmp_path):
    SignalCache(tmp_path, "k").put_many([("u1", [])])
    with signals_path(tmp_path, "k").open("ab") as handle:
        handle.write(b"this is not json\n")

    with pytest.raises(ValueError, match="line 2"):
        SignalCache(tmp_path, "k")


def test_cache_writes_its_config_next_to_the_signals(tmp_path):
    config = {"models": {"nli": "n"}, "premise_mode": "passage"}

    SignalCache(tmp_path, "abc", config=config)

    written = json.loads((tmp_path / "abc" / "config.json").read_text(encoding="utf-8"))
    assert written == config


def test_opening_a_missing_cache_creates_nothing(tmp_path):
    cache = SignalCache(tmp_path, "missing")

    assert "u1" not in cache
    assert cache.get("u1") is None
    assert cache.load_all() == {}
    assert list(tmp_path.iterdir()) == []


MODELS = {"nli": "n", "embedder": "e", "reranker": "r"}
INPUTS = {"premise_mode": "passage", "depth": 8, "candidates": 20, "manifest_hash": "m" * 64}


def test_config_key_is_a_stable_digest_of_the_canonical_inputs():
    canonical = (
        '{"candidates":20,"depth":8,"manifest_hash":"' + "m" * 64 + '",'
        '"models":{"embedder":"e","nli":"n","reranker":"r"},"premise_mode":"passage"}'
    )

    key = config_key(MODELS, **INPUTS)

    assert key == hashlib.sha256(canonical.encode()).hexdigest()[:16]
    assert config_key(dict(reversed(MODELS.items())), **INPUTS) == key


@pytest.mark.parametrize(
    "change",
    [
        {"models": {**MODELS, "nli": "other"}},
        {"models": {**MODELS, "embedder": "other"}},
        {"models": {**MODELS, "reranker": "other"}},
        {"premise_mode": "window"},
        {"depth": 6},
        {"candidates": 30},
        {"manifest_hash": "x" * 64},
    ],
)
def test_config_key_changes_with_inputs(change):
    inputs = {"models": MODELS, **INPUTS, **change}

    assert config_key(inputs.pop("models"), **inputs) != config_key(MODELS, **INPUTS)


POPULATION = {"fit": 1724, "select": 400, "risk": 391, "test": 450}
TEN_SECONDS = {"fit": 10.0, "select": 10.0, "risk": 10.0, "test": 10.0}


@pytest.mark.parametrize(
    ("budget_s", "expected"),
    [
        # Everything fits: all sources, and the fit role capped at 600.
        (1e9, {"fit": 600, "select": 400, "risk": 391, "test": 450}),
        # Start 18,410 s. The fit role gives up 150 sources first (16,910 s; 460 would be 17,010).
        (17_000.0, {"fit": 450, "select": 400, "risk": 391, "test": 450}),
        # Fit is at its floor of 200 (14,410 s); select gives up 50 (13,910 s; 360 is 14,010).
        (14_000.0, {"fit": 200, "select": 350, "risk": 391, "test": 450}),
        # Select is at 150 (11,910 s); risk gives up 100 (10,910 s; 301 would be 11,010).
        (11_000.0, {"fit": 200, "select": 150, "risk": 291, "test": 450}),
        # Risk is at 150 (9,500 s); test gives up 50 and lands exactly on the budget.
        (9_000.0, {"fit": 200, "select": 150, "risk": 150, "test": 400}),
        # Not even the floors fit: 9,500 s > 1,000 s.
        (1_000.0, {"fit": 200, "select": 150, "risk": 150, "test": 150, "over_budget": True}),
    ],
)
def test_plan_sizes_priority(budget_s, expected):
    assert extract.plan_sizes(TEN_SECONDS, POPULATION, budget_s) == expected


def test_plan_sizes_uses_a_budget_of_four_hours_by_default():
    # 18,410 s at the start; fit to its floor leaves 14,410 s, so select gives up 10 more.
    assert extract.plan_sizes(TEN_SECONDS, POPULATION) == {
        "fit": 200,
        "select": 390,
        "risk": 391,
        "test": 450,
    }


@pytest.mark.parametrize(
    ("budget_s", "expected"),
    [
        # Start 3,400 s (600 + 400 + 800 + 1,600). Fit gives up 400 s to reach 3,000 s.
        (3_000.0, {"fit": 200, "select": 200, "risk": 200, "test": 200}),
        # Select saves 20 s per step: 5 steps reach 150 and 2,900 s.
        (2_900.0, {"fit": 200, "select": 150, "risk": 200, "test": 200}),
        # Risk saves 40 s per step: 3 steps give 2,780 s (2 would leave 2,820 s).
        (2_800.0, {"fit": 200, "select": 150, "risk": 170, "test": 200}),
    ],
)
def test_plan_sizes_weighs_each_role_by_its_own_seconds_per_source(budget_s, expected):
    seconds = {"fit": 1.0, "select": 2.0, "risk": 4.0, "test": 8.0}
    population = {"fit": 700, "select": 200, "risk": 200, "test": 200}

    assert extract.plan_sizes(seconds, population, budget_s) == expected


def test_plan_sizes_never_asks_for_more_sources_than_exist():
    # Populations below the floors: every role keeps all it has, and the budget is still blown.
    population = {"fit": 120, "select": 100, "risk": 100, "test": 100}

    assert extract.plan_sizes(TEN_SECONDS, population, 1.0) == {
        "fit": 120,
        "select": 100,
        "risk": 100,
        "test": 100,
        "over_budget": True,
    }


def make_response(response_id: str, source_id: str, split: str, text: str) -> RTResponse:
    return RTResponse(response_id, source_id, "m", split, text, ())


def make_source(source_id: str) -> RTSource:
    return RTSource(source_id, "Summary", "n", "source text", None)


ONE, TWO, THREE = "A is 1.", "A is 1. B is 2.", "A is 1. B is 2. C is 3."  # 1, 2, 3 sentences
TEST_SOURCE = "7"
SOURCES = {
    sid: make_source(sid) for sid in (FIT_LOW, FIT_EDGE, SELECT_EDGE_LOW, RISK_EDGE, TEST_SOURCE)
}
RESPONSES = [
    make_response(f"{FIT_LOW}-a", FIT_LOW, "train", THREE),
    make_response(f"{FIT_LOW}-b", FIT_LOW, "train", ONE),
    make_response(f"{FIT_EDGE}-a", FIT_EDGE, "train", TWO),
    make_response(f"{SELECT_EDGE_LOW}-a", SELECT_EDGE_LOW, "train", THREE),
    make_response(f"{RISK_EDGE}-a", RISK_EDGE, "train", ONE),
    make_response(f"{TEST_SOURCE}-a", TEST_SOURCE, "test", TWO),
    make_response(f"{TEST_SOURCE}-b", TEST_SOURCE, "test", TWO),
]
PLAN_MANIFEST = {
    "roles": {
        "fit": [FIT_LOW, FIT_EDGE],
        "select": [SELECT_EDGE_LOW],
        "risk": [RISK_EDGE],
        "test": [TEST_SOURCE],
    }
}


def test_role_profile_averages_units_over_the_sources_of_each_role():
    mean_units, population = extract.role_profile(SOURCES, RESPONSES)

    # fit: (3 + 1) + 2 units over 2 sources; test: 2 + 2 units over 1 source.
    assert mean_units == {"fit": 3.0, "select": 3.0, "risk": 1.0, "test": 4.0}
    assert population == {"fit": 2, "select": 1, "risk": 1, "test": 1}


def test_plan_units_follows_role_manifest_and_file_order():
    plan = extract.plan_units(SOURCES, RESPONSES, PLAN_MANIFEST, ("select", "fit"))

    assert [(item.response.response_id, [u.unit_id for u in item.units]) for item in plan] == [
        (f"{FIT_LOW}-a", [f"{FIT_LOW}-a:0", f"{FIT_LOW}-a:1", f"{FIT_LOW}-a:2"]),
        (f"{FIT_LOW}-b", [f"{FIT_LOW}-b:0"]),
        (f"{FIT_EDGE}-a", [f"{FIT_EDGE}-a:0", f"{FIT_EDGE}-a:1"]),
        (f"{SELECT_EDGE_LOW}-a", [f"{SELECT_EDGE_LOW}-a:{i}" for i in range(3)]),
    ]
    assert [u.role for item in plan for u in item.units] == ["fit"] * 6 + ["select"] * 3


def test_plan_units_limit_takes_a_prefix_of_the_units():
    plan = extract.plan_units(SOURCES, RESPONSES, PLAN_MANIFEST, ("fit",), limit=5)

    assert [(item.response.response_id, len(item.units)) for item in plan] == [
        (f"{FIT_LOW}-a", 3),
        (f"{FIT_LOW}-b", 1),
        (f"{FIT_EDGE}-a", 1),
    ]


def test_plan_units_refuses_a_test_source_listed_under_another_role():
    # A manifest that put test data in the fit role would leak it past the --final guard.
    manifest = {"roles": {**PLAN_MANIFEST["roles"], "fit": [FIT_LOW, TEST_SOURCE]}}

    with pytest.raises(ValueError, match="split"):
        extract.plan_units(SOURCES, RESPONSES, manifest, ("fit",))


def test_plan_units_refuses_a_manifest_source_missing_from_the_data():
    manifest = {"roles": {**PLAN_MANIFEST["roles"], "fit": [FIT_LOW, "999999"]}}

    with pytest.raises(ValueError, match="not in the data"):
        extract.plan_units(SOURCES, RESPONSES, manifest, ("fit",))


def make_fit_corpus(n_sources: int = 60):
    sources = {str(n): make_source(str(n)) for n in range(n_sources)}
    responses = [
        make_response(f"{n}-{suffix}", str(n), "train", THREE)
        for suffix in "ab"
        for n in range(n_sources)
    ]
    return sources, responses


def test_sample_units_draws_n_units_of_the_role_from_many_sources():
    sources, responses = make_fit_corpus()
    fit_ids = {sid for sid in sources if ragtruth.split_role(sid, "train") == "fit"}

    plan = extract.sample_units(sources, responses, role="fit", n_units=10)

    units = [unit for item in plan for unit in item.units]
    assert len(units) == 10
    assert {unit.role for unit in units} == {"fit"}
    assert {unit.source_id for unit in units} <= fit_ids
    # Three units per response: 4 responses (3 + 3 + 3 + 1). Every source's first response
    # comes before any second one, so the 4 responses come from 4 different sources.
    assert len({unit.source_id for unit in units}) == 4


def test_sample_units_is_seeded():
    sources, responses = make_fit_corpus()

    first = extract.sample_units(sources, responses, role="fit", n_units=10)
    again = extract.sample_units(sources, responses, role="fit", n_units=10)
    other = extract.sample_units(sources, responses, role="fit", n_units=10, seed=7)

    assert again == first
    assert other != first


def test_sample_units_refuses_more_units_than_the_role_holds():
    with pytest.raises(ValueError, match="only 6"):  # 3 + 1 + 2 fit units
        extract.sample_units(SOURCES, RESPONSES, role="fit", n_units=10)


def test_test_role_requires_final(tmp_path, capsys):
    argv = [
        "--roles",
        "fit,test",
        "--premise-mode",
        "passage",
        "--embedder",
        "fake",
        "--nli",
        "fake",
        "--manifest",
        str(tmp_path / "missing.json"),
    ]

    with pytest.raises(SystemExit) as exit_info:
        extract.main(argv)

    assert exit_info.value.code == 2
    assert "--final" in capsys.readouterr().err
    # With --final the same roles are accepted.
    assert extract.parse_args([*argv, "--final"]).roles == ("fit", "test")


@pytest.mark.parametrize("roles", ["fit,train", ""])
def test_unknown_roles_are_rejected(roles, capsys):
    with pytest.raises(SystemExit) as exit_info:
        extract.parse_args(["--roles", roles, "--premise-mode", "passage"])

    assert exit_info.value.code == 2
    assert "roles" in capsys.readouterr().err


def prepare(tmp_path, monkeypatch):
    """Tiny RAGTruth files pinned in the registry, their manifest, and a private cache root."""
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
    sources, responses = ragtruth.load_ragtruth(root)
    manifest = ragtruth.build_manifest(
        sources, responses, sizes=dict.fromkeys(ragtruth.ROLES), data_sha256=shas
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(extract, "CACHE_ROOT", tmp_path / "cache")
    return root, manifest_path, manifest


def spy_on_collect(monkeypatch) -> list[dict]:
    """Record each collect_signals call of the runner, then run the real function."""
    calls: list[dict] = []
    real = extract.collect_signals

    async def spy(retriever, reranker, nli, embedder, claims, scope, **kwargs):
        calls.append({"claims": list(claims), "scope": scope, "kwargs": kwargs})
        return await real(retriever, reranker, nli, embedder, claims, scope, **kwargs)

    monkeypatch.setattr(extract, "collect_signals", spy)
    return calls


def extract_args(root, manifest_path, database_url, *extra: str):
    return extract.parse_args(
        [
            "--premise-mode",
            "window",
            "--embedder",
            "fake",
            "--nli",
            "fake",
            "--database-url",
            database_url,
            "--data-root",
            str(root),
            "--manifest",
            str(manifest_path),
            *extra,
        ]
    )


def all_unit_ids(source_ids) -> set[str]:
    # Two responses per source (-a, -b), two sentences each.
    return {f"{sid}-{r}:{i}" for sid in source_ids for r in "ab" for i in (0, 1)}


ALL_SOURCES = [
    FIT_LOW,
    FIT_EDGE,
    SELECT_EDGE_LOW,
    SELECT_EDGE_HIGH,
    RISK_EDGE,
    RISK_HIGH,
    TEST_SOURCE,
]


@pg
async def test_extract_smoke_fakes(db, pg_urls, tmp_path, monkeypatch, capsys):
    root, manifest_path, manifest = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(extract, "PROGRESS_EVERY", 10)
    calls = spy_on_collect(monkeypatch)
    args = extract_args(
        root, manifest_path, pg_urls["app"], "--roles", "fit,select,risk,test", "--final"
    )

    summary = await extract.run(args)

    assert (summary["n_units"], summary["n_extracted"], summary["n_skipped"]) == (28, 28, 0)
    cache_dir = tmp_path / "cache" / summary["cache_key"]
    assert summary["cache_key"] == config_key(
        {"nli": "fake-rule-nli", "embedder": "fake-hash-384", "reranker": "fake-keyword"},
        premise_mode="window",
        depth=8,
        candidates=20,
        manifest_hash=ragtruth.manifest_hash(manifest),
    )
    assert json.loads((cache_dir / "config.json").read_text(encoding="utf-8")) == {
        "candidates": 20,
        "depth": 8,
        "manifest_hash": ragtruth.manifest_hash(manifest),
        "models": {"embedder": "fake-hash-384", "nli": "fake-rule-nli", "reranker": "fake-keyword"},
        "premise_mode": "window",
    }

    # One collect_signals call per response, with the claims of its sentences.
    assert len(calls) == 14
    assert calls[0]["claims"] == [
        Claim(0, "Alpha is wrong.", 0, 15, "sentence"),
        Claim(1, "Beta is fine.", 16, 29, "sentence"),
    ]
    assert all(
        call["kwargs"] == {"depth": 8, "candidates": 20, "premise_mode": "window"} for call in calls
    )
    assert all(len(call["scope"].collection_ids) == 1 for call in calls)

    # Every unit is cached, and its signals come from its own source's collection only.
    cached = SignalCache(tmp_path / "cache", summary["cache_key"]).load_all()
    assert set(cached) == all_unit_ids(ALL_SOURCES)
    async with db.system_tx() as conn:
        cur = await conn.execute("select id from tenants where name = 'ragtruth-bench'")
        tenant_id = (await cur.fetchone())["id"]
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select p.id, p.text, c.name as collection from passages p "
            "join documents d on d.id = p.document_id "
            "join collections c on c.id = d.collection_id where c.name like 'rt-%'"
        )
        passages = {str(r["id"]): (r["collection"], r["text"]) for r in await cur.fetchall()}
    assert len(passages) == 7  # one tiny document, one passage per source
    for unit_id, signals in cached.items():
        source_id = unit_id.split("-")[0]
        assert len(signals) == 1, unit_id
        for item in signals:
            assert passages[item.passage_id] == (f"rt-{source_id}", item.text), unit_id
            assert item.rerank is not None
    assert {call["scope"].tenant_id for call in calls} == {tenant_id}
    assert len({call["scope"].collection_ids for call in calls}) == 7  # one collection per source

    # units.jsonl carries the unit metadata and the gold label.
    rows = [json.loads(x) for x in (cache_dir / "units.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 28
    assert rows[0] == {
        "unit_id": f"{FIT_LOW}-a:0",
        "response_id": f"{FIT_LOW}-a",
        "source_id": FIT_LOW,
        "task_type": "Summary",
        "role": "fit",
        "start": 0,
        "end": 15,
        "text": "Alpha is wrong.",
        "gold": "contradicted",
    }
    assert sorted(row["gold"] for row in rows).count("contradicted") == 7
    assert {row["role"] for row in rows if row["source_id"] == TEST_SOURCE} == {"test"}

    # Progress goes to stderr every 10 units here (every 100 by default).
    assert capsys.readouterr().err.count("units/s") == 2

    # A rerun collects nothing and changes nothing: not a byte, not even a rewrite.
    names = ("signals.jsonl", "units.jsonl")
    before = {
        name: ((cache_dir / name).read_bytes(), (cache_dir / name).stat().st_mtime_ns)
        for name in names
    }
    calls.clear()
    again = await extract.run(args)

    assert (again["n_units"], again["n_extracted"], again["n_skipped"]) == (28, 0, 28)
    assert calls == []
    assert {
        name: ((cache_dir / name).read_bytes(), (cache_dir / name).stat().st_mtime_ns)
        for name in names
    } == before


@pg
async def test_extract_resumes_after_a_crash(db, pg_urls, tmp_path, monkeypatch):
    root, manifest_path, _ = prepare(tmp_path, monkeypatch)
    calls = spy_on_collect(monkeypatch)
    args = extract_args(root, manifest_path, pg_urls["app"], "--roles", "fit,select,risk")

    interrupted = await extract.run(
        extract_args(
            root, manifest_path, pg_urls["app"], "--roles", "fit,select,risk", "--limit-units", "5"
        )
    )
    signals_file = tmp_path / "cache" / interrupted["cache_key"] / "signals.jsonl"
    with signals_file.open("ab") as handle:
        handle.write(b'{"unit_id": "kill-9", "signals": [{"passage_id"')  # half-written record
    calls.clear()

    resumed = await extract.run(args)

    assert interrupted["n_extracted"] == 5
    assert resumed["cache_key"] == interrupted["cache_key"]
    assert (resumed["n_units"], resumed["n_extracted"], resumed["n_skipped"]) == (24, 19, 5)
    # Only the 19 missing units were collected: the cached five are not scored again.
    assert sum(len(call["claims"]) for call in calls) == 19
    cached = SignalCache(tmp_path / "cache", resumed["cache_key"]).load_all()
    assert set(cached) == all_unit_ids(ALL_SOURCES[:-1])  # no unit of the test source
    for line in signals_file.read_bytes().splitlines():
        json.loads(line)  # no record was glued onto the half-written one
    assert "kill-9" not in cached


@pg
async def test_measure_times_collect_signals_and_sizes_each_role(
    db, pg_urls, tmp_path, monkeypatch
):
    root, _, _ = prepare(tmp_path, monkeypatch)
    out = tmp_path / "results" / "throughput.json"
    args = extract.parse_measure_args(
        [
            "--units",
            "6",
            "--database-url",
            pg_urls["app"],
            "--data-root",
            str(root),
            "--out",
            str(out),
        ]
    )

    class SpyNLI(RuleNLIScorer):
        def __init__(self):
            self.n_calls = 0
            self.n_pairs = 0

        def score(self, pairs):
            time.sleep(0.02)
            self.n_calls += 1
            self.n_pairs += len(pairs)
            return super().score(pairs)

    nli = SpyNLI()

    result = await extract.measure(
        args, models=extract.Models(HashEmbedder(), KeywordReranker(), nli)
    )

    assert json.loads(out.read_text(encoding="utf-8")) == result
    assert result["n_units"] == 6
    assert result["n_nli_pairs"] == nli.n_pairs > 0
    assert result["nli_seconds"] >= 0.015 * nli.n_calls  # each scorer call sleeps 20 ms
    assert result["collect_seconds"] >= result["nli_seconds"]
    assert result["units_per_s"] == pytest.approx(6 / result["collect_seconds"])
    assert result["nli_pairs_per_s"] == pytest.approx(nli.n_pairs / result["nli_seconds"])
    # Every role of the tiny data has 4 units per source; the test role has one source.
    assert result["mean_units_per_source"] == {"fit": 4.0, "select": 4.0, "risk": 4.0, "test": 4.0}
    assert result["population"] == {"fit": 2, "select": 2, "risk": 2, "test": 1}
    assert result["seconds_per_source"] == {
        role: pytest.approx(4.0 / result["units_per_s"]) for role in ragtruth.ROLES
    }
    assert result["models"] == {
        "nli": "fake-rule-nli",
        "embedder": "fake-hash-384",
        "reranker": "fake-keyword",
    }
    assert (result["premise_mode"], result["depth"], result["candidates"]) == ("passage", 8, 20)
    assert {"git_commit", "cpu", "python", "postgres"} <= set(result["env"])
    # The sizes plan_sizes needs are in the file as they are.
    assert extract.plan_sizes(result["seconds_per_source"], result["population"], 1e9) == {
        "fit": 2,
        "select": 2,
        "risk": 2,
        "test": 1,
    }
