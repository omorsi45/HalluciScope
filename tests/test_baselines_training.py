"""Factuality baselines, calibrator training from cached signals, and design selection.

All data is synthetic and seeded. The corpora follow the rule-NLI pattern of the test fakes
(a supported claim has an entailing passage, a contradicted one a contradicting passage, an
insufficient one a neutral passage) with some label noise.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from backend.calibration.artifact import Calibrator
from backend.calibration.metrics import nll
from backend.calibration.risk import select_thresholds
from backend.verify.features import FEATURE_NAMES
from backend.verify.types import EvidenceSignal, NLIResult
from evaluation import ragtruth, train_calibrator
from evaluation.baselines import (
    choose_threshold,
    legacy_as_shipped_score,
    legacy_fixed_score,
    legacy_predictions,
    nli_support,
    similarity_support,
)
from evaluation.cache import SignalCache, config_dict, config_key
from evaluation.ragtruth import RTUnit
from evaluation.train_calibrator import FEATURE_SETS, build_matrix, select_design, train

SEED = 20261026
CLASSES = ("supported", "contradicted", "insufficient")
MODELS = {"nli": "fake-rule-nli", "embedder": "fake-hash-384", "reranker": "fake-keyword"}
PATTERN = {
    "supported": NLIResult(entail=0.9, contra=0.05, neutral=0.05),
    "contradicted": NLIResult(entail=0.05, contra=0.9, neutral=0.05),
    "insufficient": NLIResult(entail=0.05, contra=0.05, neutral=0.9),
}
CLAIM = "The warranty lasts 24 months."
COMMON = {
    "feature_set": "nli",
    "evidence_k": 4,
    "premise_mode": "passage",
    "models": MODELS,
    "metadata": {"manifest_hash": "abc123"},
}


def signal(nli: NLIResult, cos: float = 0.5, rerank: float | None = None) -> EvidenceSignal:
    return EvidenceSignal("p", "A passage.", nli, rerank, cos)


def unit(unit_id: str, source_id: str, gold: str, role: str = "fit", text: str = CLAIM) -> RTUnit:
    return RTUnit(unit_id, unit_id.split(":")[0], source_id, "QA", role, 0, len(text), text, gold)


def make_corpus(
    role: str, n_sources: int, rng: np.random.Generator, *, per_source: int = 5, noise: float = 0.1
) -> tuple[list[RTUnit], dict[str, list[EvidenceSignal]]]:
    """Units and cached signals of one role.

    A unit's evidence shows the pattern of its gold class, except for a `noise` share of units
    that shows the pattern of one of the other two classes.
    """
    units: list[RTUnit] = []
    signals: dict[str, list[EvidenceSignal]] = {}
    for s in range(n_sources):
        source_id = f"{role}-s{s}"
        for i in range(per_source):
            gold = CLASSES[int(rng.choice(3, p=[0.5, 0.2, 0.3]))]
            shown = gold
            if rng.random() < noise:
                others = [c for c in CLASSES if c != gold]
                shown = others[int(rng.integers(2))]
            unit_id = f"{source_id}-r0:{i}"
            units.append(unit(unit_id, source_id, gold, role))
            signals[unit_id] = [
                EvidenceSignal(f"{unit_id}-p0", CLAIM, PATTERN[shown], 2.0, 0.8),
                EvidenceSignal(
                    f"{unit_id}-p1", "Shipping text.", PATTERN["insufficient"], -1.0, 0.1
                ),
            ]
    return units, signals


def make_splits(seed: int, evidence_k: int = 4):
    """(fit, select, risk), each `(X, y, groups)` over the full feature vector."""
    rng = np.random.default_rng(seed)
    sizes = (("fit", 80), ("select", 30), ("risk", 60))
    corpora = {role: make_corpus(role, n, rng) for role, n in sizes}
    return tuple(build_matrix(*corpora[role], evidence_k) for role, _ in sizes)


@pytest.fixture(scope="module")
def splits():
    return make_splits(SEED)


# --- baselines --------------------------------------------------------------------------------


def test_legacy_score_literals():
    signals = [signal(NLIResult(entail=0.9, contra=0.05, neutral=0.05), cos=0.8)]

    assert legacy_as_shipped_score(signals) == pytest.approx(0.41428571428571426, abs=1e-12)
    assert legacy_fixed_score(signals) == pytest.approx(0.11071428571428571, abs=1e-12)


def test_legacy_scores_take_the_min_nli_term_and_the_max_similarity():
    # Each mode reaches its minimum on a different signal; the similarity comes from the
    # signal with the larger cosine (0.5), and a negative cosine is clipped away.
    signals = [
        signal(NLIResult(entail=0.6, contra=0.1, neutral=0.3), cos=-0.2),
        signal(NLIResult(entail=0.0, contra=0.3, neutral=0.7), cos=0.5),
    ]
    # as shipped: min(0.1 + 0.3, 0.3 + 0.0) = 0.3; fixed: min(0.1 + 0.15, 0.3 + 0.35) = 0.25
    assert legacy_as_shipped_score(signals) == pytest.approx((5 * 0.3 + 2 * 0.5) / 7, abs=1e-12)
    assert legacy_fixed_score(signals) == pytest.approx((5 * 0.25 + 2 * 0.5) / 7, abs=1e-12)


def test_scores_without_signals_and_cosine_clipping():
    assert legacy_as_shipped_score([]) == 1.0
    assert legacy_fixed_score([]) == 1.0
    assert similarity_support([]) == 0.0
    assert nli_support([]) == 0.0

    neutral = NLIResult(entail=0.2, contra=0.1, neutral=0.7)
    assert similarity_support([signal(neutral, cos=-0.3), signal(neutral, cos=1.2)]) == 1.0
    assert similarity_support([signal(neutral, cos=-0.3), signal(neutral, cos=-0.1)]) == 0.0
    assert similarity_support([signal(neutral, cos=0.25), signal(neutral, cos=0.4)]) == 0.4
    assert nli_support([signal(neutral), signal(NLIResult(0.7, 0.1, 0.2))]) == 0.7


def test_legacy_prediction_modes():
    scores = [0.0, 0.19, 0.2, 0.49, 0.5, 0.6, 0.61, 1.0]

    assert legacy_predictions(scores, "binary") == [
        "supported",
        "supported",
        "supported",
        "supported",
        "insufficient",
        "insufficient",
        "insufficient",
        "insufficient",
    ]
    assert legacy_predictions(scores, "tiers") == [
        "supported",
        "supported",
        "insufficient",
        "insufficient",
        "insufficient",
        "insufficient",
        "contradicted",
        "contradicted",
    ]
    with pytest.raises(ValueError, match="mode"):
        legacy_predictions(scores, "other")  # type: ignore[arg-type]


def test_choose_threshold_ties_to_smaller():
    support = [0.2, 0.4, 0.6, 0.8]
    gold = ["insufficient", "insufficient", "supported", "supported"]

    # Every t in (0.4, 0.6] separates the classes perfectly; the smallest grid point wins.
    assert choose_threshold(support, gold) == 0.41
    # The grid order must not matter.
    assert choose_threshold(support, gold, grid=(0.6, 0.5, 0.45)) == 0.45
    assert choose_threshold(support, gold, grid=(0.45, 0.5, 0.6)) == 0.45


# --- build_matrix -----------------------------------------------------------------------------


def test_build_matrix_shapes_and_missing_unit():
    units = [
        unit("r1:0", "s1", "supported"),
        unit("r1:1", "s1", "contradicted"),
        unit("r2:0", "s2", "insufficient"),
    ]
    signals = {
        "r1:0": [signal(NLIResult(0.9, 0.05, 0.05))],
        "r1:1": [signal(NLIResult(0.05, 0.9, 0.05))],
        "r2:0": [],
    }

    X, y, groups = build_matrix(units, signals, 4)

    assert X.shape == (3, len(FEATURE_NAMES))
    assert y.tolist() == [0, 1, 2]
    assert groups.tolist() == ["s1", "s1", "s2"]
    assert X[0, :2].tolist() == [0.9, 0.05]  # max_entail, max_contra
    assert X[1, :2].tolist() == [0.05, 0.9]
    assert X[2, 3] == 1.0  # no evidence: neutral_at_best is 1

    with pytest.raises(KeyError, match="r9:0"):
        build_matrix([unit("r9:0", "s9", "supported")], signals, 4)


def test_build_matrix_passes_evidence_k_to_the_features():
    # Signal A entails strongly but ranks last; B ranks first. k=1 sees only B.
    signals = {
        "r1:0": [
            signal(NLIResult(0.9, 0.05, 0.05), rerank=-1.0),
            signal(NLIResult(0.1, 0.1, 0.8), rerank=3.0),
        ]
    }
    units = [unit("r1:0", "s1", "supported")]

    assert build_matrix(units, signals, 1)[0][0, 0] == 0.1
    assert build_matrix(units, signals, 2)[0][0, 0] == 0.9


def test_feature_sets_group_the_columns_for_the_ablation():
    assert FEATURE_SETS["nli"] == (
        "max_entail",
        "max_contra",
        "entail_minus_contra",
        "neutral_at_best",
        "n_entail_ge_half",
    )
    assert FEATURE_SETS["nli_retrieval"] == (*FEATURE_SETS["nli"], "top_rerank")
    # The lexical group skips max_dense_cos; the similarity group adds it back.
    assert FEATURE_SETS["nli_retrieval_lexical"] == (
        *FEATURE_SETS["nli_retrieval"],
        "number_coverage",
        "has_numbers",
        "content_coverage",
        "claim_len",
    )
    assert "max_dense_cos" not in FEATURE_SETS["nli_retrieval_lexical"]
    assert FEATURE_SETS["nli_retrieval_lexical_similarity"] == FEATURE_NAMES[:11]
    assert "max_dense_cos" in FEATURE_SETS["nli_retrieval_lexical_similarity"]
    assert FEATURE_SETS["full"] == FEATURE_NAMES


# --- train ------------------------------------------------------------------------------------


def test_train_end_to_end_synthetic(splits):
    fit, select, risk = splits

    calibrator, report = train(fit, select, risk, **COMMON)

    # The artifact is valid and the settings are the chosen ones.
    assert Calibrator.from_dict(calibrator.to_dict()).to_dict() == calibrator.to_dict()
    assert calibrator.feature_names == (
        "max_entail",
        "max_contra",
        "entail_minus_contra",
        "neutral_at_best",
        "n_entail_ge_half",
    )
    assert (calibrator.unit, calibrator.premise_mode, calibrator.evidence_k) == (
        "sentence",
        "passage",
        4,
    )
    assert calibrator.candidates == 20
    assert calibrator.models == MODELS
    assert (calibrator.alpha_s, calibrator.alpha_c, calibrator.delta) == (0.10, 0.05, 0.05)
    # Standardization comes from the fit role.
    columns = [FEATURE_NAMES.index(name) for name in calibrator.feature_names]
    np.testing.assert_allclose(calibrator.mean, fit[0][:, columns].mean(axis=0), atol=1e-12)

    # Held-out quality and a threshold for SUPPORTED.
    assert report["select"]["macro_f1"] > 0.8
    assert calibrator.tau_s is not None

    # The report agrees with the artifact's own prediction path (columns picked by name).
    assert report["select"]["nll"] == pytest.approx(
        nll(calibrator.predict_proba(select[0]), select[1]), abs=1e-12
    )
    expected = select_thresholds(calibrator.predict_proba(risk[0]), risk[1], risk[2])
    assert (calibrator.tau_s, calibrator.tau_c) == (expected.tau_s, expected.tau_c)
    assert report["risk"]["trace_s"] == expected.trace_s
    assert report["risk"]["trace_c"] == expected.trace_c

    # The metadata keeps what the caller passed plus the split sizes, select metrics, traces.
    assert calibrator.metadata["manifest_hash"] == "abc123"
    assert calibrator.metadata["split_sizes"]["fit"] == {"units": 400, "sources": 80}
    assert calibrator.metadata["split_sizes"]["select"] == {"units": 150, "sources": 30}
    assert calibrator.metadata["split_sizes"]["risk"] == {"units": 300, "sources": 60}
    assert calibrator.metadata["select"] == report["select"]
    assert calibrator.metadata["risk"]["trace_s"] == expected.trace_s


def test_each_role_feeds_only_its_own_step(splits):
    fit, select, risk = splits
    other_fit, other_select, other_risk = make_splits(SEED + 1)

    base, _ = train(fit, select, risk, **COMMON)
    new_risk, _ = train(fit, select, other_risk, **COMMON)
    new_select, _ = train(fit, other_select, risk, **COMMON)
    new_fit, _ = train(other_fit, select, risk, **COMMON)

    # The risk role only moves the thresholds.
    for name in ("mean", "std", "W", "b"):
        np.testing.assert_array_equal(getattr(new_risk, name), getattr(base, name))
    assert new_risk.temperature == base.temperature
    # The select role moves the temperature and nothing in the model.
    for name in ("mean", "std", "W", "b"):
        np.testing.assert_array_equal(getattr(new_select, name), getattr(base, name))
    assert new_select.temperature != base.temperature
    # The fit role moves the model.
    assert not np.array_equal(new_fit.W, base.W)


def test_train_without_temperature_scales_by_one(splits):
    fit, select, risk = splits

    plain, plain_report = train(fit, select, risk, use_temperature=False, **COMMON)
    scaled, scaled_report = train(fit, select, risk, **COMMON)

    assert plain.temperature == 1.0
    assert scaled.temperature != 1.0
    # Temperature scaling fits the select NLL, so it can only lower it.
    assert scaled_report["select"]["nll"] < plain_report["select"]["nll"]


# --- select_design ----------------------------------------------------------------------------


def grouped(n: int, per_source: int = 10) -> np.ndarray:
    return np.array([f"s{i // per_source}" for i in range(n)])


def number_split(rng: np.random.Generator, n: int):
    """Entailment cannot tell supported from contradicted; number_coverage can."""
    y = rng.choice(3, size=n, p=[0.4, 0.4, 0.2])
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    X[:, FEATURE_NAMES.index("max_entail")] = np.where(y == 2, 0.2, 0.8) + rng.normal(0, 0.05, n)
    X[:, FEATURE_NAMES.index("number_coverage")] = np.where(
        y == 0, 1.0, np.where(y == 1, 0.0, rng.random(n))
    )
    return X, y, grouped(n)


def test_select_design_prefers_informative_features():
    rng = np.random.default_rng(SEED)
    config = {
        ("passage", 4): (number_split(rng, 600), number_split(rng, 300), number_split(rng, 300))
    }

    result = select_design(config)

    choice = result["choice"]
    assert "number_coverage" in FEATURE_SETS[choice["feature_set"]]
    assert (choice["premise_mode"], choice["evidence_k"]) == ("passage", 4)
    # The comparison table has a row per feature set, and the NLI-only set is clearly worse.
    rows = {row["feature_set"]: row for row in result["table"]}
    assert set(rows) == set(FEATURE_SETS)
    assert rows["nli"]["nll"] > choice["nll"] + 0.3
    assert choice["nll"] == min(row["nll"] for row in result["table"])


def tied_split(rng: np.random.Generator, n: int = 200):
    y = rng.integers(0, 3, n)
    X = np.zeros((n, len(FEATURE_NAMES)))
    X[:, FEATURE_NAMES.index("max_entail")] = y + rng.normal(0, 0.5, n)
    return X, y, grouped(n)


NARROW = ("max_entail",)
WIDE = ("max_entail", "claim_len")  # claim_len is constant zero in tied_split: same model


@pytest.mark.parametrize(
    ("keys", "feature_sets", "expected"),
    [
        pytest.param(
            [("passage", 4)],
            {"wide": WIDE, "narrow": NARROW},
            ("passage", 4, "narrow"),
            id="fewer-features",
        ),
        pytest.param(
            [("passage", 8), ("passage", 2), ("passage", 4)],
            {"only": NARROW},
            ("passage", 2, "only"),
            id="smaller-evidence-k",
        ),
        pytest.param(
            [("window", 2), ("passage", 2)],
            {"only": NARROW},
            ("passage", 2, "only"),
            id="passage-before-window",
        ),
        pytest.param(
            [("window", 1), ("passage", 2)],
            {"only": NARROW},
            ("window", 1, "only"),
            id="smaller-evidence-k-beats-premise-mode",
        ),
    ],
)
def test_select_design_tie_break(keys, feature_sets, expected):
    shared = tied_split(np.random.default_rng(SEED))
    config = {key: (shared, shared, shared) for key in keys}

    choice = select_design(config, feature_sets=feature_sets)["choice"]

    assert (choice["premise_mode"], choice["evidence_k"], choice["feature_set"]) == expected


def test_select_design_treats_rounding_noise_as_a_tie():
    # Same data, except that k=8 sees a 1e-10 nudge along the label: its NLL is lower, but only
    # by far less than the tolerance, so the smaller evidence_k must still win.
    shared = tied_split(np.random.default_rng(SEED))
    X, y, groups = shared
    nudged = X.copy()
    nudged[:, FEATURE_NAMES.index("max_entail")] += 1e-10 * y
    config = {("passage", 8): ((nudged, y, groups),) * 3, ("passage", 2): (shared,) * 3}

    result = select_design(config, feature_sets={"only": NARROW})

    nlls = {row["evidence_k"]: row["nll"] for row in result["table"]}
    assert 0 < nlls[2] - nlls[8] < 1e-9
    assert result["choice"]["evidence_k"] == 2


def test_select_design_prefers_fewer_features_over_smaller_evidence_k():
    # At k=2 only max_contra carries the signal; at k=8 only max_entail does. The one-feature
    # set at k=8 and the two-feature set at k=2 end up with the same model and the same NLL.
    rng = np.random.default_rng(SEED)
    n = 200
    y = rng.integers(0, 3, n)
    signal_column = y + rng.normal(0, 0.5, n)
    noise_column = rng.normal(size=n)

    def split(informative: str, other: str):
        X = np.zeros((n, len(FEATURE_NAMES)))
        X[:, FEATURE_NAMES.index(informative)] = signal_column
        X[:, FEATURE_NAMES.index(other)] = noise_column
        return X, y, grouped(n)

    config = {
        ("passage", 2): (split("max_contra", "max_entail"),) * 3,
        ("passage", 8): (split("max_entail", "max_contra"),) * 3,
    }

    choice = select_design(
        config, feature_sets={"two": ("max_contra", "claim_len"), "one": ("max_entail",)}
    )["choice"]

    assert (choice["evidence_k"], choice["feature_set"]) == (8, "one")


# --- command line -----------------------------------------------------------------------------

CACHE_ROOT = Path("evaluation/results/cache")


def write_caches(modes: tuple[str, ...], manifest_hash_override: str | None = None):
    """Caches for `modes` under the relative default cache root, plus a matching manifest.

    The test role has units and cached signals too, as after a final run.
    """
    rng = np.random.default_rng(SEED)
    corpora = {
        role: make_corpus(role, n, rng) for role, n in (("fit", 40), ("select", 15), ("risk", 30))
    }
    corpora["test"] = make_corpus("test", 10, rng)
    manifest = {
        "roles": {
            role: sorted({u.source_id for u in units}) for role, (units, _) in corpora.items()
        }
    }
    Path("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    real_hash = ragtruth.manifest_hash(manifest)
    keys = []
    for mode in modes:
        inputs = {
            "premise_mode": mode,
            "depth": 8,
            "candidates": 20,
            "manifest_hash": manifest_hash_override or real_hash,
        }
        key = config_key(MODELS, **inputs)
        cache = SignalCache(CACHE_ROOT, key, config=config_dict(MODELS, **inputs))
        every_unit = [u for units, _ in corpora.values() for u in units]
        cache.put_many((uid, s) for _, signals in corpora.values() for uid, s in signals.items())
        (cache.directory / "units.jsonl").write_text(
            "".join(json.dumps(asdict(u)) + "\n" for u in every_unit), encoding="utf-8"
        )
        keys.append(key)
    return keys


@pytest.fixture
def forbid_test_role(monkeypatch):
    """Spy on the cache: a test-role unit id or a whole-file read fails the test."""
    requested: list[str] = []
    original_get = SignalCache.get

    def get(self, unit_id):
        requested.append(unit_id)
        assert not unit_id.startswith("test-"), "the test role must not be read"
        return original_get(self, unit_id)

    def load_all(self):
        raise AssertionError("load_all would read the test role")

    monkeypatch.setattr(SignalCache, "get", get)
    monkeypatch.setattr(SignalCache, "load_all", load_all)
    return requested


def test_cli_trains_from_both_caches_and_leaves_the_test_role_alone(
    tmp_path, monkeypatch, forbid_test_role
):
    monkeypatch.chdir(tmp_path)
    passage_key, window_key = write_caches(("passage", "window"))
    out = tmp_path / "artifact" / "calibrator.json"

    code = train_calibrator.main(
        [
            "--cache-key", passage_key,
            "--alt-cache-key", window_key,
            "--manifest", "manifest.json",
            "--out", str(out),
        ]
    )  # fmt: skip

    assert code == 0
    assert forbid_test_role  # signals were read, and only for the three training roles
    calibrator = Calibrator.load(out)
    report = json.loads(Path("evaluation/results/training_report.json").read_text("utf-8"))
    # 2 premise modes x 4 evidence_k x 5 feature sets, and the artifact is the chosen row.
    assert len(report["design"]["table"]) == 40
    choice = report["design"]["choice"]
    assert (calibrator.premise_mode, calibrator.evidence_k) == (
        choice["premise_mode"],
        choice["evidence_k"],
    )
    assert calibrator.feature_names == FEATURE_SETS[choice["feature_set"]]
    assert (calibrator.unit, calibrator.candidates, calibrator.models) == ("sentence", 20, MODELS)
    # Metadata: the manifest, split sizes of the three roles only, git, time, select, risk.
    metadata = calibrator.metadata
    manifest = json.loads(Path("manifest.json").read_text("utf-8"))
    assert metadata["manifest_hash"] == ragtruth.manifest_hash(manifest)
    assert metadata["split_sizes"]["fit"]["units"] == 200
    assert set(metadata["split_sizes"]) == {"fit", "select", "risk"}
    assert {"git_commit", "created_at", "select", "risk"} <= set(metadata)
    assert report["trained"]["select"] == metadata["select"]


def test_cli_with_one_cache_compares_only_its_premise_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (window_key,) = write_caches(("window",))

    train_calibrator.main(
        ["--cache-key", window_key, "--manifest", "manifest.json", "--out", "out.json"]
    )

    report = json.loads(Path("evaluation/results/training_report.json").read_text("utf-8"))
    assert {row["premise_mode"] for row in report["design"]["table"]} == {"window"}
    assert len(report["design"]["table"]) == 20
    assert Calibrator.load("out.json").premise_mode == "window"


def test_cli_rejects_a_manifest_that_did_not_produce_the_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (key,) = write_caches(("passage",), manifest_hash_override="0" * 64)

    with pytest.raises(ValueError, match="manifest"):
        train_calibrator.main(
            ["--cache-key", key, "--manifest", "manifest.json", "--out", "out.json"]
        )

    assert not Path("out.json").exists()
