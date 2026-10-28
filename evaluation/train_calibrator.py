"""Calibrator training from cached evidence signals, and design selection on the select role.

    python -m evaluation.train_calibrator --cache-key KEY [--alt-cache-key KEY_WINDOW] \\
        --manifest evaluation/manifests/ragtruth_splits.json \\
        --out backend/calibration/artifacts/default.json

The roles follow spec ruling R1. The logistic regression is fit on the fit role only (its
standardization too), the temperature on the select role, and the Learn-then-Test thresholds on
the risk role. The design (premise mode, ``evidence_k``, feature set) is chosen by the lowest
select NLL. The test role is never read: its units are dropped when ``units.jsonl`` is loaded,
and the signal view the matrices are built from only holds fit, select and risk unit ids.
"""

import argparse
import dataclasses
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.calibration.artifact import PREMISE_MODES, Calibrator
from backend.calibration.logreg import LogRegModel, fit_logreg, predict_logits
from backend.calibration.metrics import brier_multiclass, ece_top_label, nll, softmax
from backend.calibration.risk import select_thresholds
from backend.calibration.temperature import fit_temperature
from backend.verify.features import FEATURE_NAMES, compute_features
from backend.verify.types import VERDICT_ORDER, EvidenceSignal, Verdict
from evaluation.cache import SignalCache
from evaluation.extract import CACHE_ROOT, CANDIDATES
from evaluation.metrics.classification import macro_f1
from evaluation.ragtruth import RTUnit, manifest_hash
from evaluation.run_retrieval import git_info

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "nli": FEATURE_NAMES[:5],
    "nli_retrieval": FEATURE_NAMES[:6],
    "nli_retrieval_lexical": (
        *FEATURE_NAMES[:6],
        "number_coverage",
        "has_numbers",
        "content_coverage",
        "claim_len",
    ),
    "nli_retrieval_lexical_similarity": FEATURE_NAMES[:11],
    "full": FEATURE_NAMES,
}

EVIDENCE_KS = (1, 2, 4, 8)
TRAIN_ROLES = ("fit", "select", "risk")  # the test role is deliberately absent
L2 = 1.0
TIE_TOLERANCE = 1e-9  # select NLLs closer than this are a tie
REPORT_PATH = Path("evaluation/results/training_report.json")
_REPO_ROOT = Path(__file__).resolve().parent.parent

Split = tuple[np.ndarray, np.ndarray, np.ndarray]  # features (n, 12), label indices, source ids


def build_matrix(
    units: Sequence[RTUnit],
    signals: Mapping[str, Sequence[EvidenceSignal]],
    evidence_k: int,
) -> Split:
    """Features, label indices (in `VERDICT_ORDER`) and source ids of `units`.

    The features use the top `evidence_k` cached signals of each unit. A unit with no entry in
    `signals` raises `KeyError`: a unit without a cached record is not the same as a unit with
    no evidence (an empty list).
    """
    X = np.zeros((len(units), len(FEATURE_NAMES)))
    for row, unit in enumerate(units):
        try:
            unit_signals = signals[unit.unit_id]
        except KeyError:
            raise KeyError(f"no cached signals for unit {unit.unit_id}") from None
        X[row] = compute_features(unit.text, unit_signals, evidence_k)
    y = np.array([VERDICT_ORDER.index(Verdict(unit.gold)) for unit in units], dtype=np.int64)
    groups = np.array([unit.source_id for unit in units])
    return X, y, groups


def _columns(names: Sequence[str]) -> list[int]:
    return [FEATURE_NAMES.index(name) for name in names]


def _verdicts(indices: Sequence[int]) -> list[str]:
    return [VERDICT_ORDER[i].value for i in indices]


def _metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """NLL, top-label ECE, multiclass Brier and macro-F1 of the argmax verdicts."""
    return {
        "nll": nll(probs, y),
        "ece": ece_top_label(probs, y),
        "brier": brier_multiclass(probs, y),
        "macro_f1": macro_f1(_verdicts(y), _verdicts(probs.argmax(axis=1))),
    }


def _fit_scaled(
    fit: Split, select: Split, columns: list[int], use_temperature: bool
) -> tuple[LogRegModel, float, np.ndarray]:
    """The model fit on the fit role, its temperature from select, and the select probabilities."""
    model = fit_logreg(fit[0][:, columns], fit[1], l2=L2)
    logits = predict_logits(model, select[0][:, columns])
    temperature = fit_temperature(logits, select[1]) if use_temperature else 1.0
    return model, temperature, softmax(logits / temperature)


def train(
    fit: Split,
    select: Split,
    risk: Split,
    *,
    feature_set: str,
    evidence_k: int,
    premise_mode: str,
    use_temperature: bool = True,
    alpha_s: float = 0.10,
    alpha_c: float = 0.05,
    delta: float = 0.05,
    models: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> tuple[Calibrator, dict[str, Any]]:
    """Fit on `fit`, scale the temperature on `select`, choose the thresholds on `risk`.

    Each split is `(X, y, groups)` over the full feature vector; the artifact keeps only the
    columns of `FEATURE_SETS[feature_set]`. Returns the calibrator and a report with the split
    sizes, the select metrics (`macro_f1` of the argmax verdicts, `macro_f1_decision` of the
    verdicts under the chosen thresholds) and the risk-split threshold traces. The same pieces
    go into the artifact's metadata next to the caller's `metadata`.
    """
    names = FEATURE_SETS[feature_set]
    columns = _columns(names)
    model, temperature, select_probs = _fit_scaled(fit, select, columns, use_temperature)

    risk_probs = softmax(predict_logits(model, risk[0][:, columns]) / temperature)
    thresholds = select_thresholds(
        risk_probs, risk[1], risk[2], alpha_s=alpha_s, alpha_c=alpha_c, delta=delta
    )
    calibrator = Calibrator(
        feature_names=names,
        mean=model.mean,
        std=model.std,
        W=model.W,
        b=model.b,
        temperature=temperature,
        tau_s=thresholds.tau_s,
        tau_c=thresholds.tau_c,
        alpha_s=alpha_s,
        alpha_c=alpha_c,
        delta=delta,
        unit="sentence",
        premise_mode=premise_mode,  # type: ignore[arg-type]  # validated on load
        evidence_k=evidence_k,
        candidates=CANDIDATES,
        models=dict(models),
        metadata={},
    )

    decided = [verdict.value for verdict in calibrator.decide(select_probs)]
    select_metrics = {
        **_metrics(select_probs, select[1]),
        "macro_f1_decision": macro_f1(_verdicts(select[1]), decided),
    }
    split_sizes = {
        role: {"units": len(y), "sources": len(np.unique(groups))}
        for role, (_, y, groups) in zip(TRAIN_ROLES, (fit, select, risk), strict=True)
    }
    risk_report = {
        "tau_s": thresholds.tau_s,
        "tau_c": thresholds.tau_c,
        "n_sources_s": thresholds.n_sources_s,
        "n_sources_c": thresholds.n_sources_c,
        "trace_s": thresholds.trace_s,
        "trace_c": thresholds.trace_c,
    }
    calibrator = dataclasses.replace(
        calibrator,
        metadata={
            **metadata,
            "feature_set": feature_set,
            "split_sizes": split_sizes,
            "select": select_metrics,
            "risk": risk_report,
        },
    )
    report = {
        "feature_set": feature_set,
        "premise_mode": premise_mode,
        "evidence_k": evidence_k,
        "use_temperature": use_temperature,
        "temperature": temperature,
        "n_iter": model.n_iter,
        "converged": model.converged,
        "split_sizes": split_sizes,
        "select": select_metrics,
        "risk": risk_report,
    }
    return calibrator, report


def select_design(
    splits_by_config: Mapping[tuple[str, int], tuple],
    *,
    feature_sets: Mapping[str, Sequence[str]] = FEATURE_SETS,
) -> dict[str, Any]:
    """Compare every (premise mode, evidence_k, feature set) by its select NLL.

    `splits_by_config` maps `(premise_mode, evidence_k)` to `(fit, select, risk)`, the arguments
    of `train`; only fit and select are read. Each row is fit on the fit role, with the
    temperature fitted on select, and scored on select. The choice is the lowest NLL; NLLs
    within `TIE_TOLERANCE` tie, and ties go to fewer features, then the smaller `evidence_k`,
    then `passage` before `window`. Returns `{"choice": row, "table": [rows]}`.
    """
    table: list[dict[str, Any]] = []
    for (premise_mode, evidence_k), (fit, select, *_) in splits_by_config.items():
        for feature_set, names in feature_sets.items():
            _, temperature, probs = _fit_scaled(fit, select, _columns(names), True)
            table.append(
                {
                    "premise_mode": premise_mode,
                    "evidence_k": evidence_k,
                    "feature_set": feature_set,
                    "n_features": len(names),
                    "temperature": temperature,
                    **_metrics(probs, select[1]),
                }
            )
    best = min(row["nll"] for row in table)
    tied = [row for row in table if row["nll"] <= best + TIE_TOLERANCE]
    choice = min(
        tied,
        key=lambda row: (
            row["n_features"],
            row["evidence_k"],
            PREMISE_MODES.index(row["premise_mode"]),
        ),
    )
    return {"choice": dict(choice), "table": table}


class _CachedSignals(Mapping[str, list[EvidenceSignal]]):
    """The signals of the given units, read from the cache one record at a time.

    Any other id is not in the mapping, so the test role cannot be reached through it.
    """

    def __init__(self, cache: SignalCache, unit_ids: Sequence[str]) -> None:
        self._cache = cache
        self._ids = frozenset(unit_ids)

    def __getitem__(self, unit_id: str) -> list[EvidenceSignal]:
        signals = self._cache.get(unit_id) if unit_id in self._ids else None
        if signals is None:
            raise KeyError(unit_id)
        return signals

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


@dataclass(frozen=True)
class _CacheData:
    key: str
    config: dict[str, Any]
    cache: SignalCache
    units: dict[str, list[RTUnit]]  # by role, fit/select/risk only


def _read_units(directory: Path) -> dict[str, list[RTUnit]]:
    """The fit, select and risk units of ``units.jsonl``; test rows are dropped unread."""
    by_role: dict[str, list[RTUnit]] = {role: [] for role in TRAIN_ROLES}
    for line in (directory / "units.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["role"] in by_role:
                by_role[row["role"]].append(RTUnit(**row))
    for role, units in by_role.items():
        if not units:
            raise ValueError(f"the cache has no {role} units")
    return by_role


def _open_cache(key: str, expected_manifest_hash: str) -> _CacheData:
    directory = CACHE_ROOT / key
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config["manifest_hash"] != expected_manifest_hash:
        raise ValueError(f"cache {key} was not extracted with this manifest")
    if config["candidates"] != CANDIDATES:
        raise ValueError(f"cache {key} used {config['candidates']} candidates, not {CANDIDATES}")
    return _CacheData(key, config, SignalCache(CACHE_ROOT, key), _read_units(directory))


def _split_matrices(data: _CacheData, evidence_k: int) -> tuple[Split, Split, Split]:
    unit_ids = [unit.unit_id for units in data.units.values() for unit in units]
    signals = _CachedSignals(data.cache, unit_ids)
    fit, select, risk = (
        build_matrix(data.units[role], signals, evidence_k) for role in TRAIN_ROLES
    )
    return fit, select, risk


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Select the design, train the chosen configuration, and write the artifact and report."""
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_hash = manifest_hash(manifest)
    caches = [_open_cache(args.cache_key, expected_hash)]
    if args.alt_cache_key is not None:
        caches.append(_open_cache(args.alt_cache_key, expected_hash))

    def settings(data: _CacheData) -> dict[str, Any]:
        return {k: v for k, v in data.config.items() if k != "premise_mode"}

    if len({data.config["premise_mode"] for data in caches}) != len(caches):
        raise ValueError("the caches must have different premise modes")
    if any(settings(data) != settings(caches[0]) for data in caches):
        raise ValueError("the caches differ in more than the premise mode")

    splits_by_config = {
        (data.config["premise_mode"], evidence_k): _split_matrices(data, evidence_k)
        for data in caches
        for evidence_k in EVIDENCE_KS
    }
    design = select_design(splits_by_config)
    choice = design["choice"]
    chosen = next(d for d in caches if d.config["premise_mode"] == choice["premise_mode"])
    git_commit, git_dirty = git_info(_REPO_ROOT)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

    calibrator, trained = train(
        *splits_by_config[(choice["premise_mode"], choice["evidence_k"])],
        feature_set=choice["feature_set"],
        evidence_k=choice["evidence_k"],
        premise_mode=choice["premise_mode"],
        models=chosen.config["models"],
        metadata={
            "manifest_hash": expected_hash,
            "cache_key": chosen.key,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "created_at": created_at,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(args.out)

    report = {
        "created_at": created_at,
        "manifest_hash": expected_hash,
        "cache_keys": [data.key for data in caches],
        "artifact": str(args.out),
        "design": design,
        "trained": trained,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evaluation.train_calibrator")
    parser.add_argument("--cache-key", required=True, help="the signal cache to train from")
    parser.add_argument(
        "--alt-cache-key", help="a second cache with the other premise mode, to compare"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="where to write the artifact")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run(args)
    choice = report["design"]["choice"]
    trained = report["trained"]
    print(
        f"wrote {args.out} and {REPORT_PATH} design={choice['premise_mode']}/"
        f"k{choice['evidence_k']}/{choice['feature_set']} select_nll={choice['nll']:.4f} "
        f"tau_s={trained['risk']['tau_s']} tau_c={trained['risk']['tau_c']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
