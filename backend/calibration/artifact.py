"""The versioned calibrator artifact: a multinomial logistic model over claim features.

The artifact is one small JSON file. It carries the feature columns it was trained on, the
standardization and weights, the temperature, the decision thresholds, the evidence settings
(`evidence_k`, `candidates`, `premise_mode`) and the ids of the three models that produced the
features. Serving refuses to run when those model ids differ from the ones it loaded
(`check_parity`), because features from other models mean something else to these weights.

Verdict columns follow `VERDICT_ORDER`: supported, contradicted, insufficient.
"""

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from backend.calibration.metrics import softmax
from backend.verify.features import FEATURE_NAMES
from backend.verify.types import VERDICT_ORDER, Verdict

ARTIFACT_SCHEMA_VERSION = 1

MODEL_KEYS = ("nli", "embedder", "reranker")  # the order check_parity compares them in
PREMISE_MODES = ("passage", "window")
MAX_EVIDENCE_K = 8  # the evidence cache depth

_SUPPORTED = VERDICT_ORDER.index(Verdict.SUPPORTED)
_CONTRADICTED = VERDICT_ORDER.index(Verdict.CONTRADICTED)


class CalibratorError(ValueError):
    """The artifact is malformed. The message starts with `<field>:` for the offending field."""


class CalibratorMismatch(RuntimeError):
    """A serving model differs from the one the artifact was trained with."""

    def __init__(self, field: str, expected: Any, actual: Any) -> None:
        super().__init__(f"{field}: artifact expects {expected!r}, serving uses {actual!r}")
        self.field = field
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class Calibrator:
    feature_names: tuple[str, ...]
    mean: np.ndarray  # (d,)
    std: np.ndarray  # (d,)
    W: np.ndarray  # (3, d)
    b: np.ndarray  # (3,)
    temperature: float
    tau_s: float | None
    tau_c: float | None
    alpha_s: float
    alpha_c: float
    delta: float
    unit: str
    premise_mode: Literal["passage", "window"]
    evidence_k: int
    candidates: int
    models: dict[str, str]  # keys "nli", "embedder", "reranker"
    metadata: dict[str, Any]  # manifest hash, split sizes, select/risk metrics, commit, created_at

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Verdict probabilities, shape (n, 3), from full `FEATURE_NAMES` vectors, shape (n, 12).

        The artifact's own columns are picked by name, standardized, and passed through
        `softmax((Xs @ W.T + b) / temperature)`.
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape (n, {len(FEATURE_NAMES)}), got {x.shape}")
        columns = [FEATURE_NAMES.index(name) for name in self.feature_names]
        standardized = (x[:, columns] - self.mean) / self.std
        return softmax((standardized @ self.W.T + self.b) / self.temperature)

    def decide(self, probs: np.ndarray) -> list[Verdict]:
        """SUPPORTED if p_s >= tau_s, else CONTRADICTED if p_c >= tau_c, else INSUFFICIENT.

        A threshold that is None never fires.
        """
        verdicts: list[Verdict] = []
        for row in probs:
            if self.tau_s is not None and row[_SUPPORTED] >= self.tau_s:
                verdicts.append(Verdict.SUPPORTED)
            elif self.tau_c is not None and row[_CONTRADICTED] >= self.tau_c:
                verdicts.append(Verdict.CONTRADICTED)
            else:
                verdicts.append(Verdict.INSUFFICIENT)
        return verdicts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "temperature": float(self.temperature),
            "tau_s": None if self.tau_s is None else float(self.tau_s),
            "tau_c": None if self.tau_c is None else float(self.tau_c),
            "alpha_s": float(self.alpha_s),
            "alpha_c": float(self.alpha_c),
            "delta": float(self.delta),
            "unit": self.unit,
            "premise_mode": self.premise_mode,
            "evidence_k": int(self.evidence_k),
            "candidates": int(self.candidates),
            "models": dict(self.models),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Calibrator":
        """Build a calibrator from parsed JSON; every field is validated.

        Raises `CalibratorError` with a message starting `<field>:` on the first problem.
        """
        if not isinstance(data, Mapping):
            raise _invalid("artifact", "must be a JSON object")

        version = _field(data, "schema_version")
        if isinstance(version, bool) or version != ARTIFACT_SCHEMA_VERSION:
            raise _invalid("schema_version", f"must be {ARTIFACT_SCHEMA_VERSION}")

        names = _feature_names(_field(data, "feature_names"))
        d = len(names)
        mean = _array(_field(data, "mean"), "mean", (d,))
        std = _array(_field(data, "std"), "std", (d,))
        if not (std > 0).all():
            raise _invalid("std", "must be positive")
        W = _array(_field(data, "W"), "W", (len(VERDICT_ORDER), d))
        b = _array(_field(data, "b"), "b", (len(VERDICT_ORDER),))

        temperature = _number(_field(data, "temperature"), "temperature")
        if temperature <= 0:
            raise _invalid("temperature", "must be positive")
        tau_s = _threshold(_field(data, "tau_s"), "tau_s")
        tau_c = _threshold(_field(data, "tau_c"), "tau_c")
        alpha_s = _probability(_field(data, "alpha_s"), "alpha_s")
        alpha_c = _probability(_field(data, "alpha_c"), "alpha_c")
        delta = _probability(_field(data, "delta"), "delta")

        unit = _field(data, "unit")
        if unit != "sentence":
            raise _invalid("unit", "must be 'sentence'")
        premise_mode = _field(data, "premise_mode")
        if premise_mode not in PREMISE_MODES:
            raise _invalid("premise_mode", f"must be one of {list(PREMISE_MODES)}")
        evidence_k = _integer(_field(data, "evidence_k"), "evidence_k")
        if not 1 <= evidence_k <= MAX_EVIDENCE_K:
            raise _invalid("evidence_k", f"must be between 1 and {MAX_EVIDENCE_K}")
        candidates = _integer(_field(data, "candidates"), "candidates")
        if candidates < 1:
            raise _invalid("candidates", "must be at least 1")

        models = _models(_field(data, "models"))
        metadata = _field(data, "metadata")
        if not isinstance(metadata, Mapping):
            raise _invalid("metadata", "must be an object")

        return cls(
            feature_names=names,
            mean=mean,
            std=std,
            W=W,
            b=b,
            temperature=temperature,
            tau_s=tau_s,
            tau_c=tau_c,
            alpha_s=alpha_s,
            alpha_c=alpha_c,
            delta=delta,
            unit=unit,
            premise_mode=premise_mode,
            evidence_k=evidence_k,
            candidates=candidates,
            models=models,
            metadata=copy.deepcopy(dict(metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        """Read and validate an artifact file. A missing file raises `FileNotFoundError`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        """Write canonical JSON: sorted keys, two-space indent, LF line ends, trailing newline."""
        text = json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"
        Path(path).write_text(text, encoding="utf-8", newline="\n")

    @property
    def artifact_id(self) -> str:
        """SHA-256 hex digest of the compact canonical JSON of `to_dict()`."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def check_parity(self, *, nli_model: str, embedder_model: str, reranker_model: str) -> None:
        """Raise `CalibratorMismatch` for the first of nli, embedder, reranker that differs."""
        serving = {"nli": nli_model, "embedder": embedder_model, "reranker": reranker_model}
        for key in MODEL_KEYS:
            if serving[key] != self.models[key]:
                raise CalibratorMismatch(key, self.models[key], serving[key])


def _invalid(field: str, problem: str) -> CalibratorError:
    return CalibratorError(f"{field}: {problem}")


def _field(data: Mapping[str, Any], field: str) -> Any:
    if field not in data:
        raise _invalid(field, "missing")
    return data[field]


def _feature_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _invalid("feature_names", "must be a list of feature names")
    names = tuple(value)
    if not names:
        raise _invalid("feature_names", "must not be empty")
    for name in names:
        if not isinstance(name, str) or name not in FEATURE_NAMES:
            raise _invalid("feature_names", f"unknown feature {name!r}")
    if len(set(names)) != len(names):
        raise _invalid("feature_names", "must not contain duplicates")
    return names


def _array(value: Any, field: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        raise _invalid(field, "must be a rectangular array of numbers") from None
    if array.dtype.kind not in "iuf":
        raise _invalid(field, "must be an array of numbers")
    if array.shape != shape:
        raise _invalid(field, f"expected shape {shape}, got {array.shape}")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise _invalid(field, "must be finite")
    return array


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _invalid(field, "must be a finite number")
    return float(value)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(field, "must be an integer")
    return value


def _probability(value: Any, field: str) -> float:
    number = _number(value, field)
    if not 0.0 <= number <= 1.0:
        raise _invalid(field, "must be in [0, 1]")
    return number


def _threshold(value: Any, field: str) -> float | None:
    if value is None:
        return None
    number = _number(value, field)
    if not 0.0 <= number <= 1.0:
        raise _invalid(field, "must be in [0, 1] or null")
    return number


def _models(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise _invalid("models", "must be an object")
    for key in MODEL_KEYS:
        model_id = value.get(key)
        if not isinstance(model_id, str) or not model_id:
            raise _invalid("models", f"{key} must be a non-empty string")
    return dict(value)
