"""Factuality evaluation: the runner (systems, intervals, the test-role guard and log), ablations,
the atomic-claim subset, and the markdown report with its figures.

All data is synthetic and seeded. The cache tests build caches like the extraction runner writes
them, with a test role that a select run must never read.
"""

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import numpy as np
import pytest

from backend.calibration.artifact import Calibrator, CalibratorMismatch
from backend.calibration.metrics import reliability_bins
from backend.calibration.risk import clopper_pearson_upper
from backend.config import Settings
from backend.llm.client import OllamaClient
from backend.verify.types import Claim, EvidenceSignal, Verdict
from evaluation import extract, ragtruth, report, run_factuality, train_calibrator
from evaluation.baselines import (
    choose_threshold,
    legacy_as_shipped_score,
    legacy_fixed_score,
    legacy_predictions,
)
from evaluation.cache import SignalCache, config_dict, config_key
from evaluation.metrics.classification import (
    accepted_risk,
    detection_prf,
    false_supported_rate,
    macro_f1,
)
from evaluation.ragtruth import RTResponse, RTUnit
from evaluation.train_calibrator import FEATURE_SETS, build_matrix, train
from tests.fakes import (
    HashEmbedder,
    KeywordReranker,
    RuleNLIScorer,
    ScriptedLLM,
    make_test_calibrator,
)
from tests.test_baselines_training import CLASSES, MODELS, PATTERN, SEED, make_corpus
from tests.test_extract_cache import extract_args, prepare

CACHE_ROOT = Path("evaluation/results/cache")
TEST_LOG = Path("evaluation/results/test_log.jsonl")
TASKS = ("QA", "Summary", "Data2txt")
pg = pytest.mark.pg
SYSTEMS = ("legacy_as_shipped", "legacy_fixed", "similarity_only", "nli_only", "evidencegraph")


@pytest.fixture(autouse=True)
def small_bootstrap(monkeypatch):
    """The default 1,000 resamples make a run take seconds; the tests need only the shape."""
    monkeypatch.setattr(run_factuality, "N_RESAMPLES", 200)


# --- synthetic caches -------------------------------------------------------------------------


class Synthetic(NamedTuple):
    keys: list[str]  # one cache key per premise mode, in the order of `modes`
    corpora: dict[str, tuple[list[RTUnit], dict[str, list[EvidenceSignal]]]]
    manifest_hash: str
    artifact: Calibrator


def regroup(
    units: list[RTUnit], signals: dict[str, list[EvidenceSignal]], rng: np.random.Generator
) -> tuple[list[RTUnit], dict[str, list[EvidenceSignal]]]:
    """Two responses per source, one task type per source (three in all), deeper evidence.

    Each unit gets four more signals of a random pattern ranked below the two it had, so that
    the evidence depth `evidence_k` changes the features.
    """
    new_units: list[RTUnit] = []
    new_signals: dict[str, list[EvidenceSignal]] = {}
    for unit in units:
        index = int(unit.unit_id.rsplit(":", 1)[1])
        response_id = f"{unit.source_id}-r{index % 2}"
        unit_id = f"{response_id}:{index}"
        task_type = TASKS[int(unit.source_id.rsplit("s", 1)[1]) % len(TASKS)]
        new_units.append(
            dataclasses.replace(unit, unit_id=unit_id, response_id=response_id, task_type=task_type)
        )
        deeper = [
            EvidenceSignal(
                f"{unit_id}-p{depth}", "Other text.", PATTERN[CLASSES[int(rng.integers(3))]],
                1.5 - 0.5 * depth, 0.3,
            )
            for depth in range(2, 6)
        ]  # fmt: skip
        new_signals[unit_id] = [*signals[unit.unit_id], *deeper]
    return new_units, new_signals


def perturb(unit_id: str, signals: list[EvidenceSignal]) -> list[EvidenceSignal]:
    """The evidence of another premise mode, as far as the tests care.

    Half of the units lose the NLI result of their best passage, which is not a rescaling: a
    model retrained on it learns something else.
    """
    if int(unit_id.rsplit(":", 1)[1]) % 2:
        return signals
    return [dataclasses.replace(signals[0], nli=PATTERN["insufficient"]), *signals[1:]]


def write_cache(corpora, mode: str, manifest_hash: str, *, other: bool = False) -> str:
    inputs = {
        "premise_mode": mode,
        "depth": 8,
        "candidates": 20,
        "manifest_hash": manifest_hash,
    }
    key = config_key(MODELS, **inputs)
    cache = SignalCache(CACHE_ROOT, key, config=config_dict(MODELS, **inputs))
    cache.put_many(
        (unit_id, perturb(unit_id, s) if other else s)
        for _, signals in corpora.values()
        for unit_id, s in signals.items()
    )
    every_unit = [u for units, _ in corpora.values() for u in units]
    (cache.directory / "units.jsonl").write_text(
        "".join(json.dumps(asdict(u)) + "\n" for u in every_unit), encoding="utf-8"
    )
    return key


def write_synthetic(modes: tuple[str, ...] = ("passage",)) -> Synthetic:
    """Caches for `modes`, `manifest.json` and a trained `artifact.json` in the working folder.

    The artifact is trained on the first mode's cache (the others hold perturbed signals).
    """
    rng = np.random.default_rng(SEED)
    sizes = (("fit", 60), ("select", 20), ("risk", 40), ("test", 20))
    corpora = {role: regroup(*make_corpus(role, n, rng), rng) for role, n in sizes}
    manifest = {
        "commit": ragtruth.COMMIT,
        "roles": {role: sorted({u.source_id for u in us}) for role, (us, _) in corpora.items()},
    }
    Path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = ragtruth.manifest_hash(manifest)
    keys = [
        write_cache(corpora, mode, manifest_hash, other=number > 0)
        for number, mode in enumerate(modes)
    ]
    fit, select, risk = (build_matrix(*corpora[role], 4) for role in ("fit", "select", "risk"))
    artifact, _ = train(
        fit,
        select,
        risk,
        feature_set="nli",
        evidence_k=4,
        premise_mode=modes[0],
        models=MODELS,
        metadata={"manifest_hash": manifest_hash},
    )
    artifact.save("artifact.json")
    return Synthetic(keys, corpora, manifest_hash, artifact)


def argv(key: str, split: str = "select", *extra: str, out: str = "out/results.json") -> list[str]:
    return [
        "--cache-key", key,
        "--manifest", "manifest.json",
        "--artifact", "artifact.json",
        "--split", split,
        "--out", out,
        *extra,
    ]  # fmt: skip


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def requested_units(monkeypatch):
    """Spy on the cache: every unit id asked for is recorded; whole-file reads are forbidden."""
    requested: list[str] = []
    original_get = SignalCache.get

    def get(self, unit_id):
        requested.append(unit_id)
        return original_get(self, unit_id)

    def load_all(self):
        raise AssertionError("load_all would read the test role")

    monkeypatch.setattr(SignalCache, "get", get)
    monkeypatch.setattr(SignalCache, "load_all", load_all)
    return requested


# --- the test-role guard and log --------------------------------------------------------------


def test_test_split_requires_final(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # Neither file exists: touching the data would raise FileNotFoundError, not exit with 2.
    arguments = argv("k", "test")

    with pytest.raises(SystemExit) as exit_info:
        run_factuality.main(arguments)

    assert exit_info.value.code == 2
    assert "--final" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
    assert run_factuality.parse_args([*arguments, "--final"]).final is True


def test_atomic_subset_needs_the_test_split(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        run_factuality.main(argv("k", "select", "--atomic-subset", "5"))

    assert exit_info.value.code == 2
    assert "--split test" in capsys.readouterr().err


def test_select_split_never_reads_the_test_role(tmp_path, monkeypatch, requested_units):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys

    def no_test_loader(*args, **kwargs):
        raise AssertionError("the test loader must not run for --split select")

    monkeypatch.setattr(run_factuality, "_read_test_units", no_test_loader)

    assert run_factuality.main(argv(key, "select")) == 0

    assert requested_units  # signals were read, and only for the three training roles
    assert not [unit_id for unit_id in requested_units if unit_id.startswith("test-")]
    assert not TEST_LOG.exists()
    assert read_json("out/results.json")["test_evaluations"]["count"] == 0


def test_test_split_logs_every_run_and_reports_the_count(tmp_path, monkeypatch, requested_units):
    monkeypatch.chdir(tmp_path)
    synthetic = write_synthetic()
    (key,) = synthetic.keys

    assert run_factuality.main(argv(key, "test", "--final")) == 0

    # The test units were read now, and the log holds one entry for this run.
    assert any(unit_id.startswith("test-") for unit_id in requested_units)
    [line] = TEST_LOG.read_text(encoding="utf-8").splitlines()
    entry = json.loads(line)
    assert set(entry) == {"at", "git_commit", "artifact_id", "args"}
    assert entry["artifact_id"] == synthetic.artifact.artifact_id
    assert entry["args"]["split"] == "test"
    assert entry["args"]["final"] is True
    assert entry["args"]["cache_key"] == key
    first = read_json("out/results.json")
    assert first["split"] == "test"
    assert first["test_evaluations"] == {
        "artifact_id": synthetic.artifact.artifact_id,
        "count": 1,
        "log": TEST_LOG.as_posix(),
    }

    # A second run is counted too; a select run adds nothing.
    run_factuality.main(argv(key, "test", "--final", out="out/second.json"))
    run_factuality.main(argv(key, "select", out="out/select.json"))

    assert len(TEST_LOG.read_text(encoding="utf-8").splitlines()) == 2
    assert read_json("out/second.json")["test_evaluations"]["count"] == 2
    assert read_json("out/select.json")["test_evaluations"]["count"] == 2


def test_the_count_only_covers_the_artifact_that_was_evaluated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    TEST_LOG.write_text(
        json.dumps({"at": "t", "git_commit": None, "artifact_id": "other", "args": {}}) + "\n",
        encoding="utf-8",
    )

    run_factuality.main(argv(key, "test", "--final"))

    results = read_json("out/results.json")
    assert results["test_evaluations"]["count"] == 1  # the entry of the other artifact is not ours
    assert len(TEST_LOG.read_text(encoding="utf-8").splitlines()) == 2


def test_a_test_run_that_fails_after_reading_started_is_still_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys

    def broken(*args, **kwargs):
        raise RuntimeError("disk error")

    monkeypatch.setattr(run_factuality, "_read_test_units", broken)

    with pytest.raises(RuntimeError, match="disk error"):
        run_factuality.main(argv(key, "test", "--final"))

    assert len(TEST_LOG.read_text(encoding="utf-8").splitlines()) == 1


def test_a_run_rejected_before_any_test_unit_is_read_is_not_logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys
    stranger = dataclasses.replace(
        Calibrator.load("artifact.json"), metadata={"manifest_hash": "0" * 64}
    )
    stranger.save("artifact.json")

    with pytest.raises(ValueError, match="manifest"):
        run_factuality.main(argv(key, "test", "--final"))

    assert not TEST_LOG.exists()
    assert not Path("out/results.json").exists()


def test_the_artifact_must_match_the_cache_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys
    other = dataclasses.replace(
        Calibrator.load("artifact.json"), models={**MODELS, "nli": "another-nli"}
    )
    other.save("artifact.json")

    with pytest.raises(ValueError, match="models"):
        run_factuality.main(argv(key, "select"))


def test_an_artifact_for_a_missing_premise_mode_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic(("passage",)).keys
    window = dataclasses.replace(Calibrator.load("artifact.json"), premise_mode="window")
    window.save("artifact.json")

    with pytest.raises(ValueError, match="premise"):
        run_factuality.main(argv(key, "select"))


# --- systems and metrics ----------------------------------------------------------------------

# (unit id, source, task type, gold, evidence pattern shown). The evidence is one signal with a
# cosine of 0.8, so the make_test_calibrator verdict equals the pattern shown.
LITERAL = [
    ("a-r0:0", "select-a", "QA", "supported", "supported"),
    ("a-r0:1", "select-a", "QA", "contradicted", "contradicted"),
    ("a-r0:2", "select-a", "QA", "insufficient", "supported"),
    ("b-r0:0", "select-b", "Summary", "supported", "supported"),
    ("b-r0:1", "select-b", "Summary", "insufficient", "insufficient"),
    ("b-r1:0", "select-b", "Summary", "contradicted", "insufficient"),
    ("b-r1:1", "select-b", "Summary", "supported", "insufficient"),
    ("b-r2:0", "select-b", "Summary", "supported", "supported"),
]
FILLER = ("fit", "risk", "test")  # one supported unit each, so that every role has data


def write_literal() -> str:
    """The eight units above as the select role, plus one filler unit for fit and for risk."""
    rows = [(*row, "select") for row in LITERAL]
    rows += [(f"{role}-r0:0", f"{role}-s", "QA", "supported", "supported", role) for role in FILLER]
    unit_rows = []
    signals: dict[str, list[EvidenceSignal]] = {}
    for unit_id, source, task, gold, shown, role in rows:
        response_id = unit_id.split(":")[0]
        unit_rows.append(RTUnit(unit_id, response_id, source, task, role, 0, 4, "x", gold))
        signals[unit_id] = [EvidenceSignal("p", "text", PATTERN[shown], 2.0, 0.8)]
    manifest = {
        "commit": ragtruth.COMMIT,
        "roles": {
            "fit": ["fit-s"],
            "select": ["select-a", "select-b"],
            "risk": ["risk-s"],
            "test": ["test-s"],
        },
    }
    Path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    inputs = {
        "premise_mode": "passage",
        "depth": 8,
        "candidates": 20,
        "manifest_hash": ragtruth.manifest_hash(manifest),
    }
    models = make_test_calibrator().models
    key = config_key(models, **inputs)
    cache = SignalCache(CACHE_ROOT, key, config=config_dict(models, **inputs))
    cache.put_many(signals.items())
    (cache.directory / "units.jsonl").write_text(
        "".join(json.dumps(asdict(u)) + "\n" for u in unit_rows), encoding="utf-8"
    )
    artifact = dataclasses.replace(
        make_test_calibrator(), metadata={"manifest_hash": inputs["manifest_hash"]}
    )
    artifact.save("artifact.json")
    return key


def test_literal_metrics_of_the_systems_on_the_select_role(tmp_path, monkeypatch):
    # Every number below was derived by hand from the table in LITERAL.
    monkeypatch.chdir(tmp_path)
    key = write_literal()

    run_factuality.main(argv(key, "select"))

    results = read_json("out/results.json")
    systems = results["systems"]
    assert list(systems) == list(SYSTEMS)

    # evidencegraph predicts the pattern shown: S C S S I I I S against gold S C I S I C S S.
    overall = systems["evidencegraph"]["overall"]
    assert overall["n_units"] == 8
    assert overall["detection"] == {
        "precision": 0.75, "recall": 0.75, "f1": 0.75, "tp": 3, "fp": 1, "fn": 1, "tn": 3,
    }  # fmt: skip
    assert overall["macro_f1"] == pytest.approx(109 / 180)
    assert overall["false_supported_rate"] == 0.25
    assert overall["accepted_risk"] == {
        "risk": 0.25, "k": 1, "n": 4, "cp_upper": clopper_pearson_upper(1, 4, 0.05),
    }  # fmt: skip
    assert overall["supported_coverage"] == 0.5
    # Four responses, the first three with an unsupported gold sentence and a flagged one.
    assert overall["response_level"] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 3, "fp": 0, "fn": 0, "tn": 1,
    }  # fmt: skip

    # The shipped scorer reads the entailment column as neutral: it accepts every unit whose
    # evidence is neutral, so its binary decision flags only the strong contradiction, and its
    # tiers give macro-F1 3/7. The fixed order has the same binary decision at 0.5 but its tiers
    # reproduce the verdicts of evidencegraph here.
    shipped = systems["legacy_as_shipped"]["overall"]
    assert shipped["detection"]["tp"] == 1
    assert shipped["detection"]["f1"] == pytest.approx(0.4)
    assert shipped["false_supported_rate"] == 0.75
    assert shipped["accepted_risk"]["k"] == 3
    assert shipped["accepted_risk"]["n"] == 7
    assert shipped["macro_f1"] == pytest.approx(3 / 7)
    fixed = systems["legacy_fixed"]["overall"]
    assert fixed["detection"]["f1"] == pytest.approx(0.4)
    assert fixed["macro_f1"] == pytest.approx(109 / 180)

    # Single-signal baselines get the smallest grid threshold with the best select F1.
    assert results["thresholds"] == {"nli_only": 0.06, "similarity_only": 0.81}

    # The risk-coverage curve has 50 points and ends at full coverage, where the risk is the
    # share of unsupported units (4 of 8).
    curve = systems["evidencegraph"]["risk_coverage"]
    assert len(curve["coverage"]) == len(curve["risk"]) == 50
    assert curve["coverage"][0] == pytest.approx(0.02)
    assert curve["coverage"][-1] == 1.0
    assert curve["risk"][-1] == 0.5
    assert 0.0 <= curve["aurc"] <= 1.0


def test_runner_on_synthetic_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    synthetic = write_synthetic()
    (key,) = synthetic.keys

    code = run_factuality.main(argv(key, "test", "--final"))

    assert code == 0
    results = read_json("out/results.json")
    assert list(results["systems"]) == list(SYSTEMS)

    test_units, test_signals = synthetic.corpora["test"]
    gold = [u.gold for u in test_units]
    # evidencegraph, computed again here from the artifact and the matrix of the test units.
    X, _, _ = build_matrix(test_units, test_signals, synthetic.artifact.evidence_k)
    probs = synthetic.artifact.predict_proba(X)
    pred = [v.value for v in synthetic.artifact.decide(probs)]
    eg = results["systems"]["evidencegraph"]["overall"]
    assert eg["n_units"] == len(test_units) == 100
    assert eg["detection"]["f1"] == pytest.approx(detection_prf(gold, pred)["f1"])
    assert eg["macro_f1"] == pytest.approx(macro_f1(gold, pred))
    assert eg["false_supported_rate"] == pytest.approx(false_supported_rate(gold, pred))
    assert eg["accepted_risk"]["risk"] == pytest.approx(accepted_risk(gold, pred)[0])

    # legacy_as_shipped: binary predictions for detection, tiers for macro-F1.
    ordered = [test_signals[u.unit_id] for u in test_units]
    scores = [legacy_as_shipped_score(s) for s in ordered]
    shipped = results["systems"]["legacy_as_shipped"]["overall"]
    assert shipped["detection"]["f1"] == pytest.approx(
        detection_prf(gold, legacy_predictions(scores, "binary"))["f1"]
    )
    assert shipped["macro_f1"] == pytest.approx(macro_f1(gold, legacy_predictions(scores, "tiers")))
    fixed_scores = [legacy_fixed_score(s) for s in ordered]
    assert results["systems"]["legacy_fixed"]["overall"]["macro_f1"] == pytest.approx(
        macro_f1(gold, legacy_predictions(fixed_scores, "tiers"))
    )

    # The single-signal thresholds come from the select role, not from these labels.
    select_units, select_signals = synthetic.corpora["select"]
    select_gold = [u.gold for u in select_units]
    nli_support = [max(x.nli.entail for x in select_signals[u.unit_id]) for u in select_units]
    assert results["thresholds"]["nli_only"] == choose_threshold(nli_support, select_gold)

    # Every metric is a probability or None, and every interval brackets its point.
    checked = 0
    for name, system in results["systems"].items():
        blocks = [system["overall"], *system["by_task"].values()]
        for block in blocks:
            for field in ("precision", "recall", "f1"):
                assert 0.0 <= block["detection"][field] <= 1.0
                assert 0.0 <= block["response_level"][field] <= 1.0
            for value in (
                block["macro_f1"],
                block["false_supported_rate"],
                block["accepted_risk"]["risk"],
                block["accepted_risk"]["cp_upper"],
                block["supported_coverage"],
            ):
                assert value is None or 0.0 <= value <= 1.0, name
            points = {
                "detection_f1": block["detection"]["f1"],
                "macro_f1": block["macro_f1"],
                "false_supported_rate": block["false_supported_rate"],
                "accepted_risk": block["accepted_risk"]["risk"],
            }
            assert set(block["ci"]) == set(points)
            for metric, point in points.items():
                lo, hi = block["ci"][metric]
                if point is None:
                    assert lo is None and hi is None
                else:
                    assert 0.0 <= lo <= point + 1e-12, (name, metric)
                    assert point - 1e-12 <= hi <= 1.0, (name, metric)
                    checked += 1
    assert checked > 40

    # Per task type: three types, whose units add up to the overall count.
    for system in results["systems"].values():
        assert set(system["by_task"]) == set(TASKS)
        assert sum(b["n_units"] for b in system["by_task"].values()) == 100

    # Probabilistic metrics belong to evidencegraph alone.
    calibration = results["systems"]["evidencegraph"]["calibration"]
    assert {"ece", "classwise_ece", "brier", "nll", "reliability_bins"} <= set(calibration)
    assert sum(b["count"] for b in calibration["reliability_bins"]) == 100
    assert all("calibration" not in results["systems"][s] for s in SYSTEMS[:-1])
    # Every system with a support score has a curve and an AURC.
    for system in results["systems"].values():
        assert len(system["risk_coverage"]["coverage"]) == 50
        assert 0.0 <= system["risk_coverage"]["aurc"] <= 1.0

    # The run records what it evaluated.
    assert results["artifact"]["id"] == synthetic.artifact.artifact_id
    assert results["artifact"]["tau_s"] == synthetic.artifact.tau_s
    assert results["data"]["manifest_hash"] == synthetic.manifest_hash
    assert results["data"]["dataset_commit"] == ragtruth.COMMIT
    assert results["models"] == MODELS
    assert results["bootstrap"] == {"n_resamples": 200, "seed": SEED, "ci": 0.95, "by": "source"}
    assert results["cp_delta"] == 0.05


def test_interval_statistics_agree_with_the_metric_functions():
    # The bootstrap works on confusion matrices; the point estimates come from the metric
    # functions. They must be the same numbers, including the undefined cases.
    rng = np.random.default_rng(SEED)
    cases = [(rng.integers(0, 3, 40), rng.integers(0, 3, 40)) for _ in range(30)]
    cases.append((np.zeros(5, dtype=int), np.array([0, 0, 1, 2, 0])))  # nothing unsupported
    cases.append((np.array([0, 1, 2, 1]), np.array([1, 1, 2, 2])))  # nothing predicted supported
    names = ("supported", "contradicted", "insufficient")
    for gold, pred in cases:
        cm = run_factuality._confusion(gold, pred)
        g, p = [names[i] for i in gold], [names[i] for i in pred]
        assert run_factuality._detection_f1(cm) == pytest.approx(detection_prf(g, p)["f1"])
        assert run_factuality._macro_f1(cm) == pytest.approx(macro_f1(g, p))
        assert run_factuality._false_supported(cm) == false_supported_rate(g, p)
        assert run_factuality._accepted_risk(cm) == accepted_risk(g, p)[0]


# --- ablations --------------------------------------------------------------------------------

ROW_FIELDS = {
    "kind", "name", "premise_mode", "evidence_k", "feature_set", "use_temperature",
    "detection_f1", "macro_f1", "false_supported_rate", "accepted_risk", "nll", "ece",
    "tau_s", "tau_c",
}  # fmt: skip


def rows_of(ablations: dict, kind: str) -> list[dict]:
    return [row for row in ablations["rows"] if row["kind"] == kind]


def test_ablations_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    synthetic = write_synthetic(("passage", "window"))
    passage, window = synthetic.keys

    run_factuality.main(argv(passage, "select", "--ablations", "--alt-cache-key", window))

    ablations = read_json("out/results_ablations.json")
    results = read_json("out/results.json")
    assert ablations["split"] == "select"
    assert ablations["artifact_id"] == synthetic.artifact.artifact_id
    # One row per feature set, per evidence_k, the temperature switch, and the other premise mode.
    assert [row["name"] for row in rows_of(ablations, "feature_set")] == list(FEATURE_SETS)
    assert [row["evidence_k"] for row in rows_of(ablations, "evidence_k")] == [1, 2, 4, 8]
    assert [row["use_temperature"] for row in rows_of(ablations, "temperature")] == [False]
    assert [row["premise_mode"] for row in rows_of(ablations, "premise_mode")] == ["window"]
    assert len(ablations["rows"]) == 11
    for row in ablations["rows"]:
        assert set(row) == ROW_FIELDS
        for field in ("detection_f1", "macro_f1", "false_supported_rate", "accepted_risk", "ece"):
            assert row[field] is None or 0.0 <= row[field] <= 1.0, (row["kind"], row["name"])
        assert row["nll"] > 0.0

    # The row of the artifact's own configuration retrains the artifact exactly.
    base = next(row for row in rows_of(ablations, "feature_set") if row["name"] == "nli")
    assert (base["tau_s"], base["tau_c"]) == (synthetic.artifact.tau_s, synthetic.artifact.tau_c)
    assert base["detection_f1"] == results["systems"]["evidencegraph"]["overall"]["detection"]["f1"]
    assert base["nll"] == results["systems"]["evidencegraph"]["calibration"]["nll"]
    # Each switch really changes the model: no temperature fits the select role worse, other
    # evidence depths and the evidence of another premise mode give other NLLs.
    (no_temperature,) = rows_of(ablations, "temperature")
    assert no_temperature["nll"] > base["nll"]
    (window_row,) = rows_of(ablations, "premise_mode")
    assert abs(window_row["nll"] - base["nll"]) > 0.01
    assert len({row["nll"] for row in rows_of(ablations, "evidence_k")}) == 4


def test_ablations_without_an_alt_cache_skip_the_premise_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys

    run_factuality.main(argv(key, "select", "--ablations"))

    ablations = read_json("out/results_ablations.json")
    assert len(ablations["rows"]) == 10
    assert rows_of(ablations, "premise_mode") == []


def test_no_ablation_file_without_the_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys

    run_factuality.main(argv(key, "select"))

    assert not Path("out/results_ablations.json").exists()


def test_select_ablations_never_read_the_test_role(tmp_path, monkeypatch, requested_units):
    monkeypatch.chdir(tmp_path)
    passage, window = write_synthetic(("passage", "window")).keys

    def no_test_loader(*args, **kwargs):
        raise AssertionError("the test loader must not run for --split select")

    monkeypatch.setattr(run_factuality, "_read_test_units", no_test_loader)

    run_factuality.main(argv(passage, "select", "--ablations", "--alt-cache-key", window))

    assert requested_units
    assert not [unit_id for unit_id in requested_units if unit_id.startswith("test-")]


def test_test_ablations_use_the_test_units_of_every_cache_and_log_one_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    synthetic = write_synthetic(("passage", "window"))
    passage, window = synthetic.keys

    run_factuality.main(argv(passage, "test", "--final", "--ablations", "--alt-cache-key", window))

    ablations = read_json("out/results_ablations.json")
    results = read_json("out/results.json")
    assert ablations["split"] == "test"
    assert len(TEST_LOG.read_text(encoding="utf-8").splitlines()) == 1
    # The artifact's own configuration, retrained and scored on the 100 test units.
    base = next(row for row in rows_of(ablations, "feature_set") if row["name"] == "nli")
    assert base["detection_f1"] == results["systems"]["evidencegraph"]["overall"]["detection"]["f1"]
    assert results["systems"]["evidencegraph"]["overall"]["n_units"] == 100


def test_ablations_need_an_artifact_made_of_a_known_feature_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys
    custom = dataclasses.replace(
        Calibrator.load("artifact.json"),
        feature_names=("max_entail", "max_contra"),
        mean=np.zeros(2),
        std=np.ones(2),
        W=np.zeros((3, 2)),
    )
    custom.save("artifact.json")

    run_factuality.main(argv(key, "select"))  # the plain evaluation does not care
    with pytest.raises(ValueError, match="feature set"):
        run_factuality.main(argv(key, "select", "--ablations"))


def test_alt_cache_must_differ_only_in_the_premise_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_synthetic().keys

    with pytest.raises(ValueError, match="premise modes"):
        run_factuality.main(argv(key, "select", "--alt-cache-key", key))


# --- the report and the figures ---------------------------------------------------------------


def make_block(*, p, r, f1, f1_ci, macro, fsr, fsr_ci, risk, k, n, cp, coverage) -> dict:
    return {
        "n_units": 200,
        "n_sources": 20,
        "detection": {"precision": p, "recall": r, "f1": f1, "tp": 1, "fp": 1, "fn": 1, "tn": 1},
        "macro_f1": macro,
        "false_supported_rate": fsr,
        "accepted_risk": {"risk": risk, "k": k, "n": n, "cp_upper": cp},
        "supported_coverage": coverage,
        "response_level": {
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "tp": 8,
            "fp": 1,
            "fn": 2,
            "tn": 9,
        },
        "ci": {
            "detection_f1": f1_ci,
            "macro_f1": [macro - 0.05, macro + 0.05],
            "false_supported_rate": fsr_ci,
            "accepted_risk": [None, None] if risk is None else [risk - 0.01, risk + 0.01],
        },
    }


def curve(slope: float) -> dict:
    coverage = [j / 50 for j in range(1, 51)]
    return {"coverage": coverage, "risk": [slope * c for c in coverage], "aurc": slope / 2}


def fixture_results() -> dict:
    confidence = np.array([0.55, 0.65, 0.75, 0.85, 0.95, 0.95])
    correct = np.array([True, False, True, True, True, False])
    return {
        "split": "test",
        "created_at": "2026-10-28T12:00:00+00:00",
        "env": {
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "git_dirty": False,
            "python": "3.14.9",
            "platform": "Linux-test",
        },
        "data": {
            "dataset": "ragtruth",
            "dataset_commit": "c103204b9ce28d6bbad859304bf30de72b8ed8fe",
            "manifest_hash": "ab" * 32,
            "seed": 20261026,
            "role_sizes": {
                "fit": {"sources": 600, "units": 4000, "responses": 3000},
                "select": {"sources": 150, "units": 900, "responses": 700},
                "risk": {"sources": 150, "units": 910, "responses": 720},
                "test": {"sources": 200, "units": 1500, "responses": 1100},
            },
            "evaluated": {"role": "test", "units": 1500, "sources": 200, "responses": 1100},
        },
        "models": {"nli": "nli-x", "embedder": "embed-y", "reranker": "rerank-z"},
        "artifact": {
            "id": "cd" * 32,
            "premise_mode": "passage",
            "evidence_k": 4,
            "feature_set": "nli_retrieval",
            "temperature": 1.25,
            "tau_s": 0.81,
            "tau_c": None,
            "alpha_s": 0.1,
            "alpha_c": 0.05,
            "delta": 0.05,
        },
        "thresholds": {"nli_only": 0.62, "similarity_only": 0.77},
        "bootstrap": {"n_resamples": 1000, "seed": 20261026, "ci": 0.95, "by": "source"},
        "cp_delta": 0.05,
        "test_evaluations": {
            "artifact_id": "cd" * 32,
            "count": 3,
            "log": "evaluation/results/test_log.jsonl",
        },
        "systems": {
            "legacy_as_shipped": {
                "score": "1 - legacy score",
                "overall": make_block(
                    p=0.5,
                    r=0.25,
                    f1=0.3333,
                    f1_ci=[0.2, 0.4],
                    macro=0.3,
                    fsr=None,
                    fsr_ci=[None, None],
                    risk=None,
                    k=0,
                    n=0,
                    cp=1.0,
                    coverage=0.0,
                ),
                "by_task": {},
                "risk_coverage": curve(0.8),
            },
            "evidencegraph": {
                "score": "p_supported",
                "overall": make_block(
                    p=0.8123,
                    r=0.7011,
                    f1=0.7527,
                    f1_ci=[0.7101, 0.7899],
                    macro=0.6544,
                    fsr=0.1234,
                    fsr_ci=[0.0912, 0.1676],
                    risk=0.0876,
                    k=12,
                    n=137,
                    cp=0.1321,
                    coverage=0.5432,
                ),
                "by_task": {
                    "QA": make_block(
                        p=0.6,
                        r=0.5,
                        f1=0.5455,
                        f1_ci=[0.5, 0.6111],
                        macro=0.45,
                        fsr=0.2,
                        fsr_ci=[0.15, 0.25],
                        risk=0.1,
                        k=5,
                        n=50,
                        cp=0.2,
                        coverage=0.4,
                    ),
                    "Summary": make_block(
                        p=0.7,
                        r=0.6,
                        f1=0.6462,
                        f1_ci=[0.55, 0.74],
                        macro=0.55,
                        fsr=0.3,
                        fsr_ci=[0.2, 0.4],
                        risk=0.15,
                        k=6,
                        n=40,
                        cp=0.28,
                        coverage=0.3,
                    ),
                },
                "risk_coverage": {**curve(0.4), "aurc": 0.1111},
                "calibration": {
                    "ece": 0.0421,
                    "classwise_ece": 0.0333,
                    "brier": 0.5678,
                    "nll": 0.8765,
                    "reliability_bins": reliability_bins(confidence, correct),
                },
            },
        },
    }


def fixture_ablations() -> dict:
    row = {
        "kind": "feature_set", "name": "nli", "premise_mode": "passage", "evidence_k": 4,
        "feature_set": "nli", "use_temperature": True, "detection_f1": 0.7001,
        "macro_f1": 0.6002, "false_supported_rate": 0.1003, "accepted_risk": 0.0804,
        "nll": 0.9005, "ece": 0.0306, "tau_s": 0.82, "tau_c": None,
    }  # fmt: skip
    return {"split": "test", "artifact_id": "cd" * 32, "rows": [row]}


def fixture_atomic() -> dict:
    return {
        "n_requested": 60,
        "n_responses": 60,
        "n_units": 400,
        "n_claims": 913,
        "n_claims_unaligned": 2,
        "seed": 20261026,
        "llm_model": "llama3.2:3b",
        "by_task": {"QA": 20, "Summary": 20, "Data2txt": 20},
        "methods": {"llm_atomic": 880, "sentence": 21, "sentence_fallback": 12},
        "atomic": {
            "detection": {
                "precision": 0.61,
                "recall": 0.72,
                "f1": 0.6612,
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 1,
            },
            "false_supported_rate": 0.1717,
        },
        "sentence": {
            "detection": {
                "precision": 0.66,
                "recall": 0.62,
                "f1": 0.6394,
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 1,
            },
            "false_supported_rate": 0.2121,
        },
    }


MAIN_HEADER = (
    "| System | Det. P | Det. R | Det. F1 [95% CI] | Macro-F1 | False-supported [95% CI] "
    "| Accepted risk (CP upper) | Coverage | AURC |"
)


def test_render_factuality_golden():
    text = report.render_factuality(fixture_results(), fixture_ablations(), fixture_atomic())

    # The main table: the header of the brief, and rows formatted from the results alone.
    assert MAIN_HEADER in text
    assert (
        "| evidencegraph | 0.8123 | 0.7011 | 0.7527 [0.7101, 0.7899] | 0.6544 "
        "| 0.1234 [0.0912, 0.1676] | 0.0876 (0.1321) | 0.5432 | 0.1111 |"
    ) in text
    # Undefined numbers print as n/a instead of None.
    assert (
        "| legacy_as_shipped | 0.5000 | 0.2500 | 0.3333 [0.2000, 0.4000] | 0.3000 "
        "| n/a [n/a] | n/a (1.0000) | 0.0000 | 0.4000 |"
    ) in text
    assert "None" not in text

    # The run lines.
    assert "0123456789abcdef0123456789abcdef01234567" in text
    assert "c103204b9ce28d6bbad859304bf30de72b8ed8fe" in text
    assert "ab" * 32 in text
    assert "nli-x" in text and "embed-y" in text and "rerank-z" in text
    assert "cd" * 32 in text
    assert "0.62" in text and "0.77" in text  # the baseline thresholds from the select role
    assert "test evaluations logged for this artifact: 3" in text.lower()

    # Per task type, calibration, ablations, atomic subset.
    assert "### QA" in text and "### Summary" in text
    assert "| evidencegraph | 0.6000 | 0.5000 | 0.5455 [0.5000, 0.6111] | 0.4500 " in text
    assert "| 0.0421 | 0.0333 | 0.5678 | 0.8765 |" in text
    assert (
        "| feature_set: nli | 0.7001 | 0.6002 | 0.1003 | 0.0804 | 0.9005 | 0.0306 | 0.82 | none |"
        in text
    )
    assert "913" in text and "llm_atomic: 880" in text and "sentence_fallback: 12" in text
    assert "| Atomic claims | 0.6100 | 0.7200 | 0.6612 | 0.1717 |" in text
    assert "| Sentence units | 0.6600 | 0.6200 | 0.6394 | 0.2121 |" in text

    # Fixed methodology notes, with the guarantee stated for what it covers.
    lowered = text.lower()
    assert "labeled by span overlap" in lowered
    assert "sentence units" in lowered
    assert "per-source false-supported rate" in lowered
    assert "same evidence" in lowered
    assert "nothing was tuned on test" in lowered
    assert "\u2014" not in text
    assert (
        "![" in text and "figures/reliability.png" in text and "figures/risk_coverage.png" in text
    )


def test_render_factuality_leaves_out_missing_sections():
    text = report.render_factuality(fixture_results(), None, None)

    assert MAIN_HEADER in text
    assert "## Ablations" not in text
    assert "## Atomic" not in text
    assert "## Methodology" in text


def test_render_factuality_marks_the_select_split_as_development(tmp_path):
    results = fixture_results()
    results["split"] = "select"
    results["data"]["evaluated"]["role"] = "select"
    results["test_evaluations"]["count"] = 0

    text = report.render_factuality(results, None, None)

    assert "select split" in text.lower()
    assert "test evaluations logged for this artifact: 0" in text.lower()


def test_render_factuality_accepts_what_the_runner_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    passage, window = write_synthetic(("passage", "window")).keys
    run_factuality.main(argv(passage, "test", "--final", "--ablations", "--alt-cache-key", window))

    text = report.render_factuality(
        read_json("out/results.json"), read_json("out/results_ablations.json"), None
    )

    for system in SYSTEMS:
        assert f"| {system} |" in text
    assert "None" not in text
    assert "nan" not in text.lower()
    assert "premise_mode: window" in text
    figures = report.plot_factuality(read_json("out/results.json"), tmp_path / "figures")
    assert [path.name for path in figures] == ["reliability.png", "risk_coverage.png"]


def test_plots_written(tmp_path):
    figures = report.plot_factuality(fixture_results(), tmp_path / "figures")

    assert [path.name for path in figures] == ["reliability.png", "risk_coverage.png"]
    for path in figures:
        assert path.parent == tmp_path / "figures"
        assert path.stat().st_size > 1000
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figures[0].read_bytes() != figures[1].read_bytes()


def test_report_cli_writes_the_markdown_and_the_figures(tmp_path):
    inputs = {}
    for name, value in (
        ("results", fixture_results()),
        ("ablations", fixture_ablations()),
        ("atomic", fixture_atomic()),
    ):
        inputs[name] = tmp_path / f"{name}.json"
        inputs[name].write_text(json.dumps(value), encoding="utf-8")
    out = tmp_path / "docs" / "factuality.md"

    code = report.main(
        [
            "factuality", str(inputs["results"]),
            "--ablations", str(inputs["ablations"]),
            "--atomic", str(inputs["atomic"]),
            "--out", str(out),
            "--figures", str(tmp_path / "docs" / "figures"),
        ]
    )  # fmt: skip

    assert code == 0
    written = out.read_text(encoding="utf-8")
    assert MAIN_HEADER in written
    assert "## Ablations" in written and "## Atomic" in written
    assert "\r" not in written
    assert (tmp_path / "docs" / "figures" / "reliability.png").stat().st_size > 0
    assert (tmp_path / "docs" / "figures" / "risk_coverage.png").stat().st_size > 0
    # The images are linked relative to the markdown file.
    assert "](figures/reliability.png)" in written


# --- the atomic-claim subset ------------------------------------------------------------------


def make_responses(plan: dict[str, dict[str, int]]) -> tuple[list[RTUnit], list[RTResponse]]:
    """One unit per response. `plan` maps a task type to {source id: number of responses}."""
    units, responses = [], []
    for task, per_source in plan.items():
        for source_id, count in per_source.items():
            for number in range(count):
                response_id = f"{source_id}-{number}"
                responses.append(RTResponse(response_id, source_id, "m", "test", "text", ()))
                units.append(
                    RTUnit(
                        f"{response_id}:0",
                        response_id,
                        source_id,
                        task,
                        "test",
                        0,
                        4,
                        "text",
                        "supported",
                    )
                )
    return units, responses


SAMPLE_PLAN = {
    "QA": {"q1": 2, "q2": 2, "q3": 2, "q4": 2},
    "Summary": {"s1": 3, "s2": 3},
    "Data2txt": {"d1": 2},
}


def test_sample_responses_is_stratified_seeded_and_spread_over_sources():
    units, responses = make_responses(SAMPLE_PLAN)
    task_of = {u.response_id: u.task_type for u in units}

    sample = run_factuality.sample_responses(units, responses, 8)

    # 16 responses over three task types: the quota follows each type's share.
    assert [task_of[r.response_id] for r in sample].count("QA") == 4
    assert [task_of[r.response_id] for r in sample].count("Summary") == 3
    assert [task_of[r.response_id] for r in sample].count("Data2txt") == 1
    # Every source of a task type gives a first response before any gives a second.
    qa_sources = [r.source_id for r in sample if task_of[r.response_id] == "QA"]
    assert sorted(qa_sources) == ["q1", "q2", "q3", "q4"]
    summary_sources = [r.source_id for r in sample if task_of[r.response_id] == "Summary"]
    assert sorted(set(summary_sources)) == ["s1", "s2"]
    # Seeded: the same call gives the same sample, another seed another one.
    assert run_factuality.sample_responses(units, responses, 8) == sample
    assert run_factuality.sample_responses(units, responses, 8, seed=7) != sample


def test_sample_responses_skips_responses_without_units_and_refuses_too_many():
    units, responses = make_responses(SAMPLE_PLAN)
    responses.append(RTResponse("orphan-0", "orphan", "m", "test", "text", ()))

    everything = run_factuality.sample_responses(units, responses, 16)

    assert len(everything) == 16
    assert "orphan-0" not in {r.response_id for r in everything}
    with pytest.raises(ValueError, match="only 16"):
        run_factuality.sample_responses(units, responses, 17)


def unit_at(unit_id: str, start: int, end: int) -> RTUnit:
    return RTUnit(unit_id, "r", "s", "QA", "test", start, end, "x", "supported")


def claim_at(ordinal: int, start: int, end: int) -> Claim:
    return Claim(ordinal, "claim text", start, end, "llm_atomic")


def test_align_claims_to_the_sentence_units_they_overlap_most():
    units = [unit_at("r:0", 0, 10), unit_at("r:1", 11, 25), unit_at("r:2", 26, 40)]
    units.append(unit_at("r:3", 41, 50))  # no claim reaches it
    claims = [
        claim_at(0, 0, 10),
        claim_at(1, 11, 25),
        claim_at(2, 11, 25),
        claim_at(3, 20, 35),  # 5 characters in r:1, 9 in r:2
        claim_at(4, 100, 120),  # in no sentence
    ]
    verdicts = [
        Verdict.SUPPORTED,
        Verdict.SUPPORTED,
        Verdict.CONTRADICTED,
        Verdict.INSUFFICIENT,
        Verdict.SUPPORTED,
    ]

    aligned, unaligned = run_factuality.align_claims(units, claims, verdicts)

    # A sentence is not supported when any of its claims is not; contradicted outranks the rest.
    assert aligned == {"r:0": "supported", "r:1": "contradicted", "r:2": "insufficient"}
    assert unaligned == 1


def test_a_claim_between_two_sentences_equally_goes_to_the_earlier_one():
    units = [unit_at("r:0", 0, 10), unit_at("r:1", 10, 20)]

    aligned, unaligned = run_factuality.align_claims(
        units, [claim_at(0, 5, 15)], [Verdict.CONTRADICTED]
    )

    assert aligned == {"r:0": "contradicted"}
    assert unaligned == 0


def test_default_llm_client_is_ollama_with_the_configured_model():
    settings = Settings()

    client = run_factuality._ollama_client(settings, httpx.AsyncClient())

    assert isinstance(client, OllamaClient)
    assert client.model_id == settings.llm_model


SOURCE_CLAIM = "Alpha is the first body"  # entailed by the benchmark source of test response 7-a


def reply(*claims: tuple[int, str]) -> str:
    return json.dumps({"claims": [{"sentence": s, "text": text} for s, text in claims]})


async def prepared_final_run(pg_urls, tmp_path, monkeypatch, *extra: str):
    """Extract the tiny benchmark with the fakes, then run the final evaluation on it."""
    monkeypatch.chdir(tmp_path)
    root, manifest_path, manifest = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(train_calibrator, "CACHE_ROOT", tmp_path / "cache")
    summary = await extract.run(
        extract_args(
            root, manifest_path, pg_urls["app"], "--roles", "fit,select,risk,test", "--final"
        )
    )
    artifact = dataclasses.replace(
        make_test_calibrator(),
        premise_mode="window",
        metadata={"manifest_hash": ragtruth.manifest_hash(manifest)},
    )
    artifact.save("artifact.json")
    args = run_factuality.parse_args(
        [
            "--cache-key", summary["cache_key"],
            "--manifest", str(manifest_path),
            "--artifact", "artifact.json",
            "--split", "test", "--final",
            "--atomic-subset", "2",
            "--database-url", pg_urls["app"],
            "--data-root", str(root),
            "--out", "out/results.json",
            *extra,
        ]
    )  # fmt: skip
    return args, run_factuality.run(args)


FAKE_MODELS = (HashEmbedder, KeywordReranker, RuleNLIScorer)


@pg
async def test_atomic_subset_with_fake_models(db, pg_urls, tmp_path, monkeypatch):
    args, outcome = await prepared_final_run(pg_urls, tmp_path, monkeypatch)
    llm = ScriptedLLM()
    # Response 7-a: both sentences get a claim the source entails. Response 7-b: its first
    # sentence gets one entailed and one unsupported claim; the second gets none from the LLM,
    # so its own text becomes the claim.
    llm.responses = [
        reply((0, SOURCE_CLAIM), (1, SOURCE_CLAIM)),
        reply((0, SOURCE_CLAIM), (0, "Alpha is wrong")),
    ]
    models = extract.Models(*(each() for each in FAKE_MODELS))

    atomic = await run_factuality.run_atomic(
        args, outcome, client_factory=lambda settings, http: llm, models=models
    )

    # The LLM saw the sentences of the real response text, at temperature 0.
    assert [c["messages"][-1]["content"] for c in llm.calls] == [
        "0. Alpha is wrong.\n1. Beta is fine."
    ] * 2
    assert {c["temperature"] for c in llm.calls} == {0.0}
    assert atomic["n_requested"] == atomic["n_responses"] == 2
    assert (atomic["n_units"], atomic["n_claims"], atomic["n_claims_unaligned"]) == (4, 5, 0)
    assert atomic["methods"] == {"llm_atomic": 4, "sentence": 1, "sentence_fallback": 0}
    assert atomic["by_task"] == {"Summary": 2}
    assert (atomic["llm_model"], atomic["seed"]) == ("fake-llm", 20261026)
    # Gold: 7-a:0 contradicted, the other three supported. The sentence-level artifact finds
    # every sentence insufficient. Atomic: 7-a is all supported (the contradicted sentence slips
    # through), 7-b has one claim per sentence that fails, so both its sentences are flagged.
    assert atomic["sentence"]["detection"] == {
        "precision": 0.25, "recall": 1.0, "f1": 0.4, "tp": 1, "fp": 3, "fn": 0, "tn": 0,
    }  # fmt: skip
    assert atomic["sentence"]["false_supported_rate"] == 0.0
    assert atomic["atomic"]["detection"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 2, "fn": 1, "tn": 1,
    }  # fmt: skip
    assert atomic["atomic"]["false_supported_rate"] == 1.0
    # The result is also written next to the results.
    assert read_json("out/results_atomic.json") == atomic
    assert outcome.results["split"] == "test"


@pg
async def test_atomic_subset_refuses_models_other_than_the_artifacts(
    db, pg_urls, tmp_path, monkeypatch
):
    args, outcome = await prepared_final_run(pg_urls, tmp_path, monkeypatch)
    llm = ScriptedLLM()
    other = dataclasses.replace(
        outcome.artifact, models={**outcome.artifact.models, "nli": "another-nli"}
    )
    models = extract.Models(*(each() for each in FAKE_MODELS))

    with pytest.raises(CalibratorMismatch):
        await run_factuality.run_atomic(
            args,
            dataclasses.replace(outcome, artifact=other),
            client_factory=lambda settings, http: llm,
            models=models,
        )

    assert llm.calls == []
