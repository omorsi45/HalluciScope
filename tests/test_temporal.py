"""The synthetic temporal and provenance benchmark: generator, metrics, runner and report.

The generator tests check the labels by construction against an independent rule. The runner
smoke test uses the fake models and the test calibrator, so it asserts the shape of the output
and the integrity checks, never the quality of the verdicts.
"""

import json
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from backend.calibration.artifact import Calibrator
from backend.verify.claims import sentence_claims
from evaluation import report, run_temporal, synthetic_temporal
from evaluation.synthetic_temporal import (
    FamilySpec,
    TemporalClaim,
    claim_value,
    generate,
    render_document,
)
from tests.fakes import make_test_calibrator
from tests.test_schema_analyses import (
    insert_analysis_row,
    insert_claim_row,
    insert_link_row,
    make_world,
)

pg = pytest.mark.pg

CLASSES = ("supported", "contradicted", "insufficient")
KINDS = ("true_v1", "true_latest", "never_true", "absent")


@pytest.fixture(scope="module")
def corpus():
    return generate()


def family_of(corpus, claim: TemporalClaim) -> FamilySpec:
    return next(f for f in corpus.families if f.family_id == claim.family_id)


def expected_gold(family: FamilySpec, claim: TemporalClaim) -> tuple[str, ...]:
    """The gold rule, written out independently of the generator."""
    value = claim_value(claim, family)
    gold = []
    for version in family.versions:
        stored = version.get(claim.fact_key)
        if stored is None:
            gold.append("insufficient")
        elif stored == value:
            gold.append("supported")
        else:
            gold.append("contradicted")
    return tuple(gold)


# --- generator ---------------------------------------------------------------------------------


def test_generate_deterministic():
    first, again = generate(), generate()
    other = generate(seed=7)

    assert first == again
    assert first.families[0].versions != other.families[0].versions
    assert [c.text for c in first.claims] != [c.text for c in other.claims]


def test_generate_sizes_and_structure(corpus):
    assert len(corpus.families) == 40
    assert len(corpus.distractors) == 40
    assert len(corpus.foreign) == 10
    domains = [f.domain for f in corpus.families]
    assert {d: domains.count(d) for d in set(domains)} == {
        "product_policy": 10,
        "hr_policy": 10,
        "api_docs": 10,
        "service_terms": 10,
    }
    assert sum(len(f.versions) == 3 for f in corpus.families) == 10
    assert all(len(f.versions) in (2, 3) for f in corpus.families)
    for family in corpus.families:
        assert len(family.effective) == len(family.versions)
        assert all(set(v) == set(family.versions[0]) for v in family.versions)
        assert 6 <= sum(v is not None for v in family.versions[0].values()) <= 10


def test_effective_dates(corpus):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for index, family in enumerate(corpus.families):
        v1 = base + timedelta(days=index)
        expected = [v1, v1 + timedelta(days=90), v1 + timedelta(days=180)]
        assert list(family.effective) == expected[: len(family.versions)], family.family_id


def test_v2_changes_and_removals(corpus):
    for family in corpus.families:
        v1, v2 = family.versions[0], family.versions[1]
        changed = [k for k in v1 if v1[k] is not None and v2[k] is not None and v1[k] != v2[k]]
        dropped = [k for k in v1 if v1[k] is not None and v2[k] is None]
        added = [k for k in v1 if v1[k] is None and v2[k] is not None]
        assert 2 <= len(changed) <= 4, family.family_id
        assert len(dropped) == 1, family.family_id
        assert len(added) == 1, family.family_id


def test_v3_changes_one_or_two_more_facts(corpus):
    for family in corpus.families:
        if len(family.versions) < 3:
            continue
        v2, v3 = family.versions[1], family.versions[2]
        changed = [k for k in v2 if v2[k] is not None and v3[k] is not None and v2[k] != v3[k]]
        assert 1 <= len(changed) <= 2, family.family_id
        assert {k for k in v2 if v2[k] is None} == {k for k in v3 if v3[k] is None}


def test_gold_by_construction(corpus):
    assert corpus.claims
    for claim in corpus.claims:
        family = family_of(corpus, claim)
        assert len(claim.gold) == len(family.versions), claim.claim_id
        assert claim.gold == expected_gold(family, claim), claim.claim_id
        assert set(claim.gold) <= set(CLASSES)


def test_claim_kinds_have_the_gold_the_name_promises(corpus):
    assert {c.kind for c in corpus.claims} == set(KINDS)
    for claim in corpus.claims:
        if claim.kind == "true_v1":
            assert claim.gold[0] == "supported"
        elif claim.kind == "true_latest":
            assert claim.gold[-1] == "supported"
            assert claim.gold[0] != "supported"
        elif claim.kind == "never_true":
            assert "supported" not in claim.gold
        else:
            assert set(claim.gold) == {"insufficient"}


def test_every_family_has_claims_within_the_extraction_limits(corpus):
    ids = [c.claim_id for c in corpus.claims]
    assert len(ids) == len(set(ids))
    for family in corpus.families:
        mine = [c for c in corpus.claims if c.family_id == family.family_id]
        assert 1 <= len(mine) <= 50, family.family_id
        assert {"true_v1", "never_true", "absent"} <= {c.kind for c in mine}, family.family_id
        assert all(3 <= len(c.text) <= 400 for c in mine)


def test_claims_survive_sentence_extraction_as_written(corpus):
    # The runner verifies the claims of a family as the sentences of one answer, and relies on
    # getting exactly those claims back, in order.
    for family in corpus.families:
        mine = [c.text for c in corpus.claims if c.family_id == family.family_id]
        answer = " ".join(mine)
        assert len(answer) <= 20_000
        extracted = sentence_claims(answer, method="sentence")
        assert [claim.text for claim in extracted] == mine, family.family_id


def test_render_contains_facts_once(corpus):
    for family in corpus.families:
        documents = [render_document(family, j) for j in range(len(family.versions))]
        assert len(set(documents)) == len(documents), family.family_id
        for claim in (c for c in corpus.claims if c.family_id == family.family_id):
            for j, document in enumerate(documents):
                expected = 1 if claim.gold[j] == "supported" else 0
                assert document.count(claim.text) == expected, (claim.claim_id, j)


def test_render_document_has_a_title_and_two_fillers_per_present_fact(corpus):
    fillers = synthetic_temporal.FILLERS
    assert len(set(fillers)) == len(fillers) >= 8
    # Fillers carry no figures, so they cannot be mistaken for a fact value.
    assert not any(ch.isdigit() for filler in fillers for ch in filler)

    for family in corpus.families[:8]:
        for j, version in enumerate(family.versions):
            text = render_document(family, j)
            present = [k for k, v in version.items() if v is not None]
            assert render_document(family, j) == text
            assert text.splitlines()[0].startswith(family.entity)
            assert sum(text.count(filler) for filler in fillers) == 2 * len(present)
            assert "\u2014" not in text


def test_fact_tables_are_well_formed():
    tables = synthetic_temporal.DOMAIN_FACTS

    assert set(tables) == {"product_policy", "hr_policy", "api_docs", "service_terms"}
    for domain, facts in tables.items():
        keys = [fact.key for fact in facts]
        # A family keeps up to 10 facts, adds one in v2, and still has one never used.
        assert len(keys) == len(set(keys)) >= 12, domain
        assert {fact.kind for fact in facts} == {"number", "duration", "date", "name", "boolean"}
        for fact in facts:
            assert fact.template.count("{value}") == 1, fact.key
            assert fact.template.count("{entity}") == 1, fact.key
            assert fact.template.endswith("."), fact.key


def test_distractors_share_templates_but_not_values_or_entities(corpus):
    for family, distractor in zip(corpus.families, corpus.distractors, strict=True):
        assert distractor.domain == family.domain
        assert distractor.entity != family.entity
        assert distractor.family_id != family.family_id
        assert len(distractor.versions) == 1
        mine = {k for k, v in family.versions[0].items() if v is not None}
        theirs = {k for k, v in distractor.versions[0].items() if v is not None}
        assert theirs == mine
        assert all(distractor.versions[0][k] != family.versions[0][k] for k in mine)
    entities = [f.entity for f in (*corpus.families, *corpus.distractors)]
    assert len(entities) == len(set(entities))


def test_foreign_copies_conflict_with_every_value(corpus):
    assert [f.family_id for f in corpus.foreign] == [f.family_id for f in corpus.families[:10]]
    for copy, original in zip(corpus.foreign, corpus.families, strict=False):
        assert copy.entity == original.entity
        assert copy.effective == original.effective
        assert len(copy.versions) == len(original.versions)
        for j, (theirs, ours) in enumerate(zip(copy.versions, original.versions, strict=True)):
            for key, value in ours.items():
                if value is None:
                    assert theirs[key] is None
                else:
                    assert theirs[key] is not None
                    assert theirs[key] != value, (copy.family_id, j, key)
        documents = [render_document(copy, j) for j in range(len(copy.versions))]
        assert len(set(documents)) == len(documents), copy.family_id


def test_generate_scales_down_for_a_smoke_run():
    small = generate(n_families=4)

    assert [f.domain for f in small.families] == [
        "product_policy",
        "hr_policy",
        "api_docs",
        "service_terms",
    ]
    assert len(small.distractors) == 4
    assert len(small.foreign) == 4
    assert sum(len(f.versions) == 3 for f in small.families) == 1


# --- metrics -----------------------------------------------------------------------------------


def test_change_points_and_temporal_correct():
    s, c, i = CLASSES
    assert run_temporal.change_points([s, s, c, c, s]) == (2, 4)
    assert run_temporal.temporal_correct([s, c, c], [s, c, c])
    assert run_temporal.temporal_correct([s, s], [s, s])
    # The same set of verdicts, changing at another place, is wrong.
    assert not run_temporal.temporal_correct([s, c, c], [s, s, c])
    # A change where the gold has none is wrong.
    assert not run_temporal.temporal_correct([i, i], [i, s])
    # Only where the verdict changes matters, not what it changes to.
    assert run_temporal.temporal_correct([s, c], [s, i])


def test_temporal_correctness_counts_claims():
    s, c, i = CLASSES
    gold = {"a": [s, c], "b": [i, i], "c": [s, s, c], "d": [s, c, c]}
    pred = {"a": [s, i], "b": [i, s], "c": [s, c, c], "d": [s, c, c]}

    assert run_temporal.temporal_correctness(gold, pred) == {
        "n_claims": 4,
        "n_correct": 2,
        "rate": 0.5,
    }


def test_version_scores_hand_computed():
    s, c, i = CLASSES
    scores = run_temporal.version_scores([s, c, i, s], [s, c, s, s])

    assert scores["n"] == 4
    assert scores["accuracy"] == 0.75
    # supported: tp 2, fp 1, fn 0 -> 0.8; contradicted: 1.0; insufficient: tp 0, fn 1 -> 0.0
    assert scores["macro_f1"] == pytest.approx((0.8 + 1.0 + 0.0) / 3)
    # Two gold-unsupported claims, one of them predicted supported.
    assert scores["false_supported_rate"] == 0.5


def test_version_scores_without_unsupported_gold_has_no_false_supported_rate():
    s, _, _ = CLASSES
    assert run_temporal.version_scores([s, s], [s, s])["false_supported_rate"] is None
    assert run_temporal.version_scores([], [])["accuracy"] is None


def test_source_change_scores_hand_computed():
    scores = run_temporal.source_change_scores(
        [True, True, False, False, True], [True, False, True, False, False]
    )

    assert scores["tp"] == 1
    assert scores["n_gold_changed"] == 3
    assert scores["n_predicted_changed"] == 2
    assert scores["precision"] == 0.5
    assert scores["recall"] == pytest.approx(1 / 3)


def test_gold_changed_compares_the_last_version_with_the_first():
    s, c, i = CLASSES
    assert run_temporal.gold_changed([s, s, c])
    assert run_temporal.gold_changed([s, c, i])
    # A fact that goes back to its first value is no change between version 1 and the last.
    assert not run_temporal.gold_changed([s, c, s])
    assert not run_temporal.gold_changed([i, i])


def test_predicted_changed_compares_the_latest_verdict_with_the_original():
    s, c, _ = CLASSES
    assert not run_temporal.predicted_changed(s, None)  # never re-checked
    assert run_temporal.predicted_changed(s, c)
    assert not run_temporal.predicted_changed(s, s)  # re-checked and confirmed


def test_source_change_scores_undefined_ratios_are_none():
    scores = run_temporal.source_change_scores([False, False], [False, False])

    assert (scores["precision"], scores["recall"]) == (None, None)


def test_span_check_needs_the_right_version_and_the_value_in_the_span():
    text = "See below. The refund window is 30 days. Thanks."
    start = text.index("The refund")
    end = text.index(" Thanks")
    item = {
        "version_id": "v-valid",
        "text": text,
        "start_char": 1000,
        "span_start": 1000 + start,
        "span_end": 1000 + end,
    }

    assert run_temporal.span_check(item, "v-valid", "30 days") == (True, True)
    assert run_temporal.span_check(item, "v-other", "30 days") == (False, True)
    assert run_temporal.span_check(item, "v-valid", "14 days") == (True, False)
    assert run_temporal.span_check(None, "v-valid", "30 days") == (False, False)


def test_render_sample_one_row_per_case_with_pipes_escaped():
    cases = [
        {
            "claim_id": f"f:{n}",
            "claim": f"Claim {n} | with a pipe.",
            "as_of": "2026-04-02T00:00:00+00:00",
            "version_index": 0,
            "gold": "supported",
            "prediction": "insufficient",
            "evidence_span": "Span\nacross lines | and a pipe.",
            "document": "Aurora Router product policy",
            "version": 1,
        }
        for n in range(3)
    ]
    cases.append({**cases[0], "claim_id": "f:none", "evidence_span": None, "version": None})

    text = run_temporal.render_sample(cases)
    rows = [line for line in text.splitlines() if line.startswith("| ") and "---" not in line]

    assert len(rows) == 1 + len(cases)  # header row plus one per case
    assert "Claim 1 \\| with a pipe." in text
    assert "Span across lines \\| and a pipe." in text
    assert "\r" not in text
    assert "\u2014" not in text


# --- command line ------------------------------------------------------------------------------


def test_parse_args_defaults(monkeypatch):
    monkeypatch.delenv("HALLUCISCOPE_BENCH_DATABASE_URL", raising=False)

    args = run_temporal.parse_args([])

    assert args.database_url == "postgresql://eg_app@localhost:54329/evidencegraph_bench"
    assert (args.families, args.embedder, args.nli) == (40, "real", "real")
    assert str(args.out).replace("\\", "/") == "evaluation/results/temporal.json"
    assert str(args.sample_out).replace("\\", "/") == "docs/results/temporal-sample.md"
    assert str(args.artifact).replace("\\", "/") == "backend/calibration/artifacts/default.json"


def test_parse_args_reads_the_environment_and_flags(monkeypatch):
    monkeypatch.setenv("HALLUCISCOPE_BENCH_DATABASE_URL", "postgresql://x@h/db")

    assert run_temporal.parse_args([]).database_url == "postgresql://x@h/db"
    args = run_temporal.parse_args(["--families", "4", "--embedder", "fake", "--nli", "fake"])
    assert (args.families, args.embedder, args.nli) == (4, "fake", "fake")
    with pytest.raises(SystemExit):
        run_temporal.parse_args(["--families", "0"])


@pytest.mark.parametrize(("links", "expected"), [(0, 0), (1, 1), (7, 1)])
def test_main_exits_nonzero_on_cross_tenant_links(monkeypatch, links, expected):
    async def fake_run(args):
        return {"cross_tenant_links": links}

    monkeypatch.setattr(run_temporal, "run", fake_run)

    assert run_temporal.main([]) == expected


# --- cross-tenant count (Postgres) -------------------------------------------------------------


@pg
async def test_count_cross_tenant_links_sees_a_link_into_the_other_tenant(db, pg_urls):
    ours = await make_world(db, "alpha")
    theirs = await make_world(db, "beta")
    async with db.tenant_tx(ours.tenant_id) as conn:
        analysis_id = await insert_analysis_row(conn, ours)
        claim_id = await insert_claim_row(conn, ours.tenant_id, analysis_id)
        await insert_link_row(conn, ours.tenant_id, claim_id, ours.passages[0], ours.version_id)

    assert await run_temporal.count_cross_tenant_links(db, ours.tenant_id, theirs.tenant_id) == 0

    # The composite foreign keys refuse this row; a superuser in replica mode can store it, the
    # way a faulty repair could. A zero from the check must mean "none", not "could not see".
    url = make_conninfo(pg_urls["owner"], user="postgres")
    async with await psycopg.AsyncConnection.connect(
        url, autocommit=True, connect_timeout=5
    ) as conn:
        await conn.execute("set session_replication_role = replica")
        await conn.execute(
            "insert into evidence_links (tenant_id, claim_id, passage_id, version_id, relation, "
            "rank, span_start, span_end) values (%s, %s, %s, %s, 'supports', 2, 0, 7)",
            (ours.tenant_id, claim_id, theirs.passages[0], theirs.version_id),
        )

    assert await run_temporal.count_cross_tenant_links(db, ours.tenant_id, theirs.tenant_id) == 1
    assert await run_temporal.count_cross_tenant_links(db, theirs.tenant_id, ours.tenant_id) == 0


# --- runner smoke (Postgres, fake models) ------------------------------------------------------


@pg
async def test_run_temporal_smoke_fakes(db, pg_urls, tmp_path):
    artifact = tmp_path / "calibrator.json"
    make_test_calibrator().save(artifact)
    out = tmp_path / "results" / "temporal.json"
    sample = tmp_path / "docs" / "temporal-sample.md"
    args = run_temporal.parse_args(
        [
            "--database-url",
            pg_urls["app"],
            "--families",
            "4",
            "--embedder",
            "fake",
            "--nli",
            "fake",
            "--artifact",
            str(artifact),
            "--out",
            str(out),
            "--sample-out",
            str(sample),
        ]
    )

    result = await run_temporal.run(args)

    expected = generate(n_families=4)
    assert set(run_temporal.METRIC_KEYS) <= set(result)
    assert result["cross_tenant_links"] == 0
    validity = result["evidence_validity"]
    assert validity["n_links"] > 0
    assert validity["n_valid_links"] == validity["n_links"]

    assert result["corpus"]["families"] == 4
    assert result["corpus"]["claims"] == len(expected.claims)
    n_versions = sum(len(f.versions) for f in expected.families)
    assert result["corpus"]["versions"] == n_versions
    assert result["corpus"]["analyses"] == 4 + n_versions
    assert result["live"]["n"] == len(expected.claims)
    first = result["per_version"][0]
    assert (first["version"], first["n"]) == (1, len(expected.claims))
    assert [row["version"] for row in result["per_version"]] == [1, 2, 3]
    assert result["temporal_correctness"]["n_claims"] == len(expected.claims)
    assert 0.0 <= result["temporal_correctness"]["rate"] <= 1.0

    # As-of scoping: the supporting item of every claim predicted supported sits in the
    # version that was valid at that as-of time (not in a newer one, not in a distractor).
    spans = result["span_correctness"]
    assert spans["n_supported"] > 0
    assert spans["n_version_correct"] == spans["n_supported"]

    # Re-verification really ran: one job per version after the first, in each of the two
    # tenants (the conflicting copies have the same versions), and claims were checked.
    assert result["reverify"]["jobs"] == 2 * (n_versions - len(expected.families))
    assert result["source_change"]["n_claims"] == len(expected.claims)
    assert result["source_change"]["n_checked"] > 0
    # The versions really change verdicts, and re-verification finds some of the changes.
    assert result["source_change"]["n_gold_changed"] > 0
    assert result["source_change"]["n_predicted_changed"] > 0
    assert result["source_change"]["tp"] > 0

    assert result["models"] == {
        "nli": "fake-rule-nli",
        "embedder": "fake-hash-384",
        "reranker": "fake-keyword",
    }
    assert result["calibrator"]["id"] == Calibrator.load(artifact).artifact_id
    assert json.loads(out.read_text(encoding="utf-8")) == result

    assert len(result["sample"]) == 30
    assert len({(c["claim_id"], c["version_index"]) for c in result["sample"]}) == 30
    assert {c["gold"] for c in result["sample"]} <= set(CLASSES)
    assert {c["prediction"] for c in result["sample"]} <= set(CLASSES)
    text = sample.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 1 + 30
    assert "\r" not in text


# --- report ------------------------------------------------------------------------------------


def temporal_results(**overrides) -> dict:
    result = {
        "seed": 20261026,
        "corpus": {
            "families": 40,
            "distractors": 40,
            "foreign": 10,
            "versions": 90,
            "claims": 812,
            "claims_by_kind": {
                "true_v1": 320,
                "true_latest": 150,
                "never_true": 320,
                "absent": 22,
            },
            "analyses": 130,
        },
        "live": {
            "n": 812,
            "accuracy": 0.81234,
            "macro_f1": 0.7,
            "false_supported_rate": 0.0456,
        },
        "per_version": [
            {
                "version": 1,
                "n": 812,
                "accuracy": 0.8,
                "macro_f1": 0.75,
                "false_supported_rate": 0.05,
            },
            {
                "version": 2,
                "n": 812,
                "accuracy": 0.7,
                "macro_f1": 0.65,
                "false_supported_rate": None,
            },
            {
                "version": 3,
                "n": 200,
                "accuracy": 0.6,
                "macro_f1": 0.55,
                "false_supported_rate": 0.2,
            },
        ],
        "temporal_correctness": {"n_claims": 812, "n_correct": 600, "rate": 0.7389},
        "evidence_validity": {
            "n_analyses": 130,
            "n_links": 9000,
            "n_valid_links": 9000,
            "link_rate": 1.0,
            "n_citations": 0,
            "n_valid_citations": 0,
            "citation_rate": None,
        },
        "span_correctness": {
            "n_supported": 400,
            "n_version_correct": 380,
            "n_span_contains_value": 390,
            "n_correct": 370,
            "rate": 0.925,
        },
        "source_change": {
            "n_claims": 812,
            "n_checked": 500,
            "n_gold_changed": 300,
            "n_predicted_changed": 250,
            "tp": 200,
            "precision": 0.8,
            "recall": 0.6667,
        },
        "cross_tenant_links": 0,
        "reverify": {"jobs": 50},
        "models": {"nli": "nli-x", "embedder": "embed-x", "reranker": "rerank-x"},
        "calibrator": {"id": "cal-abc", "premise_mode": "passage", "evidence_k": 4},
        "env": {
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "git_dirty": False,
            "cpu": "Example CPU 9000",
            "ram_gb": 16.0,
            "python": "3.14.9",
            "torch": "2.14.1+cpu",
            "postgres": "17.11",
            "pgvector": "0.8.7",
        },
        "started_at": "2026-10-27T12:00:00+00:00",
        "finished_at": "2026-10-27T13:00:00+00:00",
        "runtime_seconds": 3600.0,
        "sample": [],
    }
    result.update(overrides)
    return result


def test_render_temporal_golden():
    text = report.render_temporal(temporal_results())

    assert text.startswith("# Temporal and provenance benchmark (synthetic)\n")
    assert "out-of-distribution" in text
    assert "labeled by construction" in text
    assert "human review" in text
    assert "temporal-sample.md" in text
    assert "Example CPU 9000" in text
    assert "0123456789abcdef0123456789abcdef01234567" in text
    assert "cal-abc" in text
    assert "nli-x" in text
    assert "- Families: 40" in text
    assert "claims: 812" in text
    assert "| As-of | Claims | Accuracy | Macro-F1 | False-supported |" in text
    assert "| live (before new versions) | 812 | 0.8123 | 0.7000 | 0.0456 |" in text
    assert "| v1 | 812 | 0.8000 | 0.7500 | 0.0500 |" in text
    assert "| v2 | 812 | 0.7000 | 0.6500 | n/a |" in text
    assert "| v3 | 200 | 0.6000 | 0.5500 | 0.2000 |" in text
    assert "| 812 | 600 | 0.7389 |" in text
    assert "9000 of 9000 (1.0000)" in text
    assert "370 of 400 (0.9250)" in text
    assert "| 300 | 250 | 200 | 0.8000 | 0.6667 |" in text
    assert "cross_tenant_links: 0 (must be 0)" in text
    assert "\u2014" not in text
    assert "None" not in text
    assert "\r" not in text


def test_render_temporal_marks_undefined_source_change_ratios_and_unknown_git():
    source_change = {
        "n_claims": 10,
        "n_checked": 0,
        "n_gold_changed": 0,
        "n_predicted_changed": 0,
        "tp": 0,
        "precision": None,
        "recall": None,
    }
    results = temporal_results(source_change=source_change)
    results["env"] = {**results["env"], "git_commit": None, "git_dirty": None}

    text = report.render_temporal(results)

    assert "| 0 | 0 | 0 | n/a | n/a |" in text
    assert "unknown" in text
    assert "None" not in text


def test_report_cli_writes_the_temporal_file(tmp_path):
    source = tmp_path / "temporal.json"
    source.write_text(json.dumps(temporal_results()), encoding="utf-8")
    out = tmp_path / "docs" / "temporal.md"

    exit_code = report.main(["temporal", str(source), "--out", str(out)])

    assert exit_code == 0
    written = out.read_text(encoding="utf-8")
    assert "| v1 | 812 | 0.8000 | 0.7500 | 0.0500 |" in written
    assert "\r" not in written
