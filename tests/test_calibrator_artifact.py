import copy
import dataclasses
import hashlib
import json
import math
import re

import numpy as np
import pytest

from backend.calibration.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    Calibrator,
    CalibratorError,
    CalibratorMismatch,
)
from backend.verify.features import FEATURE_NAMES
from backend.verify.types import Verdict
from tests.fakes import HashEmbedder, KeywordReranker, RuleNLIScorer, make_test_calibrator

SERVING = {
    "nli_model": "fake-rule-nli",
    "embedder_model": "fake-hash-384",
    "reranker_model": "fake-keyword",
}


def _features(**values: float) -> np.ndarray:
    """A full-width feature vector of zeros with the named features set."""
    vector = np.zeros(len(FEATURE_NAMES))
    for name, value in values.items():
        vector[FEATURE_NAMES.index(name)] = value
    return vector


def _softmax(logits: list[float]) -> list[float]:
    exps = [math.exp(x) for x in logits]
    return [e / sum(exps) for e in exps]


def _subset_calibrator() -> Calibrator:
    """Two features, listed in the opposite order to FEATURE_NAMES, with non-trivial scaling."""
    return dataclasses.replace(
        make_test_calibrator(),
        feature_names=("max_contra", "max_entail"),
        mean=np.array([0.1, 0.2]),
        std=np.array([0.5, 0.25]),
        W=np.array([[0.0, 2.0], [1.0, 0.0], [0.0, 0.0]]),
        b=np.array([0.0, 0.0, 1.0]),
        temperature=2.0,
    )


# --- save, load, identity ---------------------------------------------------------------------


def test_roundtrip_save_load(tmp_path):
    rng = np.random.default_rng(7)
    original = dataclasses.replace(
        make_test_calibrator(),
        feature_names=("max_contra", "top_rerank", "claim_len"),
        mean=rng.normal(size=3),
        std=rng.uniform(0.5, 2.0, size=3),
        W=rng.normal(size=(3, 3)),
        b=rng.normal(size=3),
        temperature=1.37,
        tau_s=None,
        tau_c=0.83,
        premise_mode="window",
        metadata={
            "manifest_sha256": "ab12",
            "split_sizes": {"fit": 70, "select": 15, "risk": 15},
            "risk_metrics": [0.1, None],
        },
    )
    path = tmp_path / "calibrator.json"

    original.save(path)
    loaded = Calibrator.load(path)

    assert loaded.to_dict() == original.to_dict()
    # Arrays come back bit-identical, not merely close.
    for name in ("mean", "std", "W", "b"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(original, name))
    assert loaded.feature_names == original.feature_names
    assert loaded.tau_s is None
    assert loaded.artifact_id == original.artifact_id


def test_save_writes_canonical_json(tmp_path):
    path = tmp_path / "calibrator.json"

    make_test_calibrator().save(path)

    text = path.read_bytes().decode("utf-8")
    assert text.endswith("}\n")
    assert "\r" not in text
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"


def test_artifact_id_hashes_canonical_content():
    calibrator = make_test_calibrator()
    canonical = json.dumps(calibrator.to_dict(), sort_keys=True, separators=(",", ":"))

    assert calibrator.artifact_id == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    changed = dataclasses.replace(calibrator, temperature=1.5)
    assert changed.artifact_id != calibrator.artifact_id


def test_to_dict_is_json_ready_and_independent_of_the_calibrator():
    calibrator = make_test_calibrator()

    data = calibrator.to_dict()

    assert data["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert json.loads(json.dumps(data)) == data
    data["W"][0][0] = 99.0
    data["models"]["nli"] = "mutated"
    assert calibrator.W[0, 0] == 8.0
    assert calibrator.models["nli"] == "fake-rule-nli"


# --- predict and decide -----------------------------------------------------------------------


def test_feature_subset_selection():
    calibrator = _subset_calibrator()
    X = np.full((2, len(FEATURE_NAMES)), 7.0)  # the unused columns carry junk
    X[0, FEATURE_NAMES.index("max_entail")] = 0.9
    X[0, FEATURE_NAMES.index("max_contra")] = 0.1
    X[1, FEATURE_NAMES.index("max_entail")] = 0.2
    X[1, FEATURE_NAMES.index("max_contra")] = 0.6

    probs = calibrator.predict_proba(X)

    # Standardized (max_contra, max_entail): row 0 is (0, 2.8), row 1 is (1, 0).
    # Logits W @ x + b: row 0 is (5.6, 0, 1), row 1 is (0, 1, 1); divided by T = 2.
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs[0], _softmax([2.8, 0.0, 0.5]), rtol=1e-12)
    np.testing.assert_allclose(probs[1], _softmax([0.0, 0.5, 0.5]), rtol=1e-12)


def test_predict_proba_rejects_matrices_that_are_not_full_feature_width():
    calibrator = make_test_calibrator()

    with pytest.raises(ValueError, match="features"):
        calibrator.predict_proba(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="features"):
        calibrator.predict_proba(np.zeros(len(FEATURE_NAMES)))


def test_test_calibrator_semantics():
    calibrator = make_test_calibrator()
    X = np.stack([_features(max_entail=0.9), _features(max_contra=0.9), _features()])

    probs = calibrator.predict_proba(X)
    verdicts = calibrator.decide(probs)

    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert verdicts == [Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT]


PROBS = np.array([[0.96, 0.02, 0.02], [0.3, 0.6, 0.1], [0.4, 0.3, 0.3]])


def test_decide_rule():
    calibrator = make_test_calibrator(tau_s=0.9, tau_c=0.5)

    verdicts = calibrator.decide(PROBS)

    assert verdicts == [Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT]
    assert all(isinstance(v, Verdict) for v in verdicts)


def test_decide_without_tau_s_never_supports():
    calibrator = dataclasses.replace(make_test_calibrator(tau_c=0.5), tau_s=None)

    verdicts = calibrator.decide(np.vstack([PROBS, [0.99, 0.0, 0.01]]))

    assert verdicts == [
        Verdict.INSUFFICIENT,
        Verdict.CONTRADICTED,
        Verdict.INSUFFICIENT,
        Verdict.INSUFFICIENT,
    ]


def test_decide_without_tau_c_never_contradicts():
    calibrator = dataclasses.replace(make_test_calibrator(tau_s=0.9), tau_c=None)

    verdicts = calibrator.decide(PROBS)

    assert verdicts == [Verdict.SUPPORTED, Verdict.INSUFFICIENT, Verdict.INSUFFICIENT]


def test_decide_thresholds_are_inclusive():
    calibrator = make_test_calibrator(tau_s=0.5, tau_c=0.5)

    verdicts = calibrator.decide(np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2]]))

    assert verdicts == [Verdict.SUPPORTED, Verdict.CONTRADICTED]


def test_decide_supported_takes_priority_over_contradicted():
    calibrator = make_test_calibrator(tau_s=0.5, tau_c=0.5)

    verdicts = calibrator.decide(np.array([[0.5, 0.5, 0.0]]))

    assert verdicts == [Verdict.SUPPORTED]


# --- fake calibrator --------------------------------------------------------------------------


def test_test_calibrator_defaults_and_overrides():
    calibrator = make_test_calibrator()

    assert calibrator.feature_names == FEATURE_NAMES
    assert (calibrator.evidence_k, calibrator.candidates) == (4, 20)
    assert calibrator.premise_mode == "passage"
    assert calibrator.models == {
        "nli": "fake-rule-nli",
        "embedder": "fake-hash-384",
        "reranker": "fake-keyword",
    }
    # The fake is itself a valid artifact.
    assert Calibrator.from_dict(calibrator.to_dict()).to_dict() == calibrator.to_dict()

    tuned = make_test_calibrator(tau_s=0.7, tau_c=0.2, evidence_k=2, models={"nli": "other"})
    assert (tuned.tau_s, tuned.tau_c, tuned.evidence_k) == (0.7, 0.2, 2)
    assert tuned.models["nli"] == "other"
    assert tuned.models["embedder"] == "fake-hash-384"


# --- validation -------------------------------------------------------------------------------


def _set(field, value):
    def mutate(data):
        data[field] = value

    return mutate


def _drop(field):
    def mutate(data):
        del data[field]

    return mutate


def _nan_in_w(data):
    data["W"][0][0] = float("nan")


INVALID = [
    ("schema_version", _set("schema_version", 2)),
    ("schema_version", _set("schema_version", True)),
    ("feature_names", _set("feature_names", [*FEATURE_NAMES[:-1], "bogus"])),
    ("feature_names", _set("feature_names", [*FEATURE_NAMES[:-1], FEATURE_NAMES[0]])),
    ("feature_names", _set("feature_names", [])),
    ("feature_names", _set("feature_names", "max_entail")),
    ("feature_names", _set("feature_names", [1, 2])),
    ("mean", _set("mean", [0.0] * 11)),
    ("std", _set("std", [1.0] * 11)),
    ("std", _set("std", [0.0] + [1.0] * 11)),
    ("W", _set("W", [[0.0] * 12] * 2)),
    ("W", _set("W", [[0.0] * 11] * 3)),
    ("W", _nan_in_w),
    ("W", _set("W", [[0.0] * 12, [0.0] * 12, ["x"] * 12])),
    ("b", _set("b", [0.0, 0.0])),
    ("temperature", _set("temperature", 0.0)),
    ("temperature", _set("temperature", -1.0)),
    ("temperature", _set("temperature", float("inf"))),
    ("temperature", _set("temperature", "1.0")),
    ("tau_s", _set("tau_s", 1.5)),
    ("tau_c", _set("tau_c", -0.1)),
    ("alpha_s", _set("alpha_s", 1.5)),
    ("alpha_c", _set("alpha_c", float("nan"))),
    ("delta", _set("delta", -0.05)),
    ("unit", _set("unit", "document")),
    ("premise_mode", _set("premise_mode", "paragraph")),
    ("evidence_k", _set("evidence_k", 0)),
    ("evidence_k", _set("evidence_k", 9)),
    ("evidence_k", _set("evidence_k", 4.5)),
    ("candidates", _set("candidates", 0)),
    ("models", _set("models", {"nli": "a", "embedder": "b"})),
    ("models", _set("models", {"nli": "a", "embedder": "b", "reranker": 3})),
    ("models", _set("models", ["a", "b", "c"])),
    ("metadata", _set("metadata", [])),
    *[
        (field, _drop(field))
        for field in (
            "schema_version",
            "feature_names",
            "mean",
            "std",
            "W",
            "b",
            "temperature",
            "tau_s",
            "tau_c",
            "alpha_s",
            "alpha_c",
            "delta",
            "unit",
            "premise_mode",
            "evidence_k",
            "candidates",
            "models",
            "metadata",
        )
    ],
]


@pytest.mark.parametrize(("field", "mutate"), INVALID)
def test_invalid_artifacts_rejected(field, mutate):
    data = copy.deepcopy(make_test_calibrator().to_dict())
    mutate(data)

    with pytest.raises(CalibratorError, match=rf"^{re.escape(field)}:"):
        Calibrator.from_dict(data)


def test_artifact_that_is_not_an_object_is_rejected():
    with pytest.raises(CalibratorError, match=r"^artifact:"):
        Calibrator.from_dict([1, 2, 3])


def test_load_rejects_invalid_artifact_file(tmp_path):
    data = make_test_calibrator().to_dict()
    data["temperature"] = 0.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CalibratorError, match=r"^temperature:"):
        Calibrator.load(path)


# --- train/serve parity -----------------------------------------------------------------------


def test_parity_accepts_the_models_the_fakes_report():
    calibrator = make_test_calibrator()

    calibrator.check_parity(
        nli_model=RuleNLIScorer.model_id,
        embedder_model=HashEmbedder.model_id,
        reranker_model=KeywordReranker.model_id,
    )


@pytest.mark.parametrize(
    ("field", "serving_key", "actual"),
    [
        ("nli", "nli_model", "cross-encoder/nli-deberta-v3-large"),
        ("embedder", "embedder_model", "BAAI/bge-small-en-v1.5"),
        ("reranker", "reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2"),
    ],
)
def test_parity_mismatch(field, serving_key, actual):
    calibrator = make_test_calibrator()

    with pytest.raises(CalibratorMismatch) as raised:
        calibrator.check_parity(**{**SERVING, serving_key: actual})

    assert raised.value.field == field
    assert raised.value.expected == calibrator.models[field]
    assert raised.value.actual == actual
    assert field in str(raised.value)


def test_parity_reports_the_first_differing_model_in_nli_embedder_reranker_order():
    calibrator = make_test_calibrator()

    with pytest.raises(CalibratorMismatch) as raised:
        calibrator.check_parity(nli_model="x", embedder_model="y", reranker_model="z")
    assert raised.value.field == "nli"

    with pytest.raises(CalibratorMismatch) as raised:
        calibrator.check_parity(**{**SERVING, "embedder_model": "y", "reranker_model": "z"})
    assert raised.value.field == "embedder"
