"""Factuality evaluation of five systems on the RAGTruth select or test role.

    python -m evaluation.run_factuality --cache-key KEY --manifest PATH --artifact PATH \\
        --split select|test [--final] [--ablations] [--alt-cache-key KEY] \\
        [--atomic-subset N] [--out PATH]

The systems are ``legacy_as_shipped``, ``legacy_fixed``, ``similarity_only``, ``nli_only`` and
``evidencegraph`` (the calibrator artifact). All of them are scored on the cached evidence
signals of the same sentence units; the single-signal baselines take their thresholds from the
select role. Per system the results hold the detection, macro-F1, false-supported and
accepted-risk numbers with 95% cluster-bootstrap intervals by source, the same per task type,
and a risk-coverage curve (50 coverage points: the point ``c`` takes the risk of the ``ceil(c n)``
highest-scoring units) with its AURC. ``evidencegraph`` also gets the probabilistic metrics.

The test role is read only with ``--split test --final``. Its units come from their own loader
(``_read_test_units``); the training loader of ``train_calibrator`` drops them by design and the
select split never calls this one. Every test-split run appends a line to
``evaluation/results/test_log.jsonl`` once its inputs are validated and before the first test
unit is read, and the results report how many entries the log holds for the artifact.
"""

import argparse
import asyncio
import json
import os
import platform
import random
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from backend.calibration.artifact import Calibrator
from backend.calibration.metrics import (
    aurc,
    brier_multiclass,
    classwise_ece,
    ece_top_label,
    nll,
    reliability_bins,
    risk_coverage,
)
from backend.calibration.risk import clopper_pearson_upper
from backend.config import Settings
from backend.db.pool import Database
from backend.llm.client import LLMClient, OllamaClient
from backend.retrieval.types import Retriever, SearchScope
from backend.services import Services
from backend.verify.claims import llm_atomic_claims
from backend.verify.engine import collect_signals
from backend.verify.features import FEATURE_NAMES, compute_features
from backend.verify.types import VERDICT_ORDER, Claim, EvidenceSignal, Verdict
from evaluation import baselines, data, extract, ragtruth
from evaluation.beir import ensure_bench_scope
from evaluation.metrics.bootstrap import cluster_bootstrap
from evaluation.metrics.classification import (
    accepted_risk,
    detection_prf,
    false_supported_rate,
    macro_f1,
    response_level,
    supported_coverage,
)
from evaluation.ragtruth import RTResponse, RTUnit, manifest_hash
from evaluation.run_retrieval import DEFAULT_DATABASE_URL, git_info
from evaluation.train_calibrator import (
    EVIDENCE_KS,
    FEATURE_SETS,
    Split,
    _CacheData,
    _CachedSignals,
    _open_cache,
    _split_matrices,
    build_matrix,
    train,
)

SYSTEMS = ("legacy_as_shipped", "legacy_fixed", "similarity_only", "nli_only", "evidencegraph")
SCORE_NAMES = {
    "legacy_as_shipped": "1 - legacy score",
    "legacy_fixed": "1 - legacy score",
    "similarity_only": "similarity support",
    "nli_only": "NLI support",
    "evidencegraph": "p_supported",
}

SEED = ragtruth.SEED
N_RESAMPLES = 1000
CI = 0.95
CP_DELTA = 0.05  # the Clopper-Pearson upper bound of the accepted risk is one-sided at 5%
CURVE_POINTS = 50

RESULTS_DIR = Path("evaluation/results")
TEST_LOG_PATH = RESULTS_DIR / "test_log.jsonl"
_REPO_ROOT = Path(__file__).resolve().parent.parent

_NAMES = tuple(verdict.value for verdict in VERDICT_ORDER)
_INDEX = {name: i for i, name in enumerate(_NAMES)}
SUPPORTED = _INDEX[Verdict.SUPPORTED]
INSUFFICIENT = _INDEX[Verdict.INSUFFICIENT]


# --- loading ----------------------------------------------------------------------------------


def _read_test_units(directory: Path) -> list[RTUnit]:
    """The test units of ``units.jsonl``. Only ``--split test --final`` reaches this."""
    units: list[RTUnit] = []
    for line in (directory / "units.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["role"] == "test":
                units.append(RTUnit(**row))
    if not units:
        raise ValueError("the cache has no test units")
    return units


def _artifact_cache(caches: Sequence[_CacheData], artifact: Calibrator) -> _CacheData:
    """The cache that holds the evidence the artifact was trained on."""
    for data_ in caches:
        if data_.config["premise_mode"] == artifact.premise_mode:
            if data_.config["models"] != artifact.models:
                raise ValueError(f"the artifact's models differ from those of cache {data_.key}")
            return data_
    raise ValueError(f"no cache has the artifact's premise mode {artifact.premise_mode!r}")


def _check_caches(caches: Sequence[_CacheData]) -> None:
    if len({data_.config["premise_mode"] for data_ in caches}) != len(caches):
        raise ValueError("the caches must have different premise modes")
    settings = [{k: v for k, v in d.config.items() if k != "premise_mode"} for d in caches]
    if any(each != settings[0] for each in settings):
        raise ValueError("the caches differ in more than the premise mode")


# --- the test log -----------------------------------------------------------------------------


def _log_test_run(args: argparse.Namespace, artifact_id: str, git_commit: str | None) -> None:
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "artifact_id": artifact_id,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    TEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEST_LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def count_test_runs(artifact_id: str) -> int:
    """How many entries of the test log belong to the artifact."""
    if not TEST_LOG_PATH.exists():
        return 0
    lines = TEST_LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    return sum(1 for entry in entries if entry["artifact_id"] == artifact_id)


# --- scores and predictions -------------------------------------------------------------------


@dataclass(frozen=True)
class _UnitScores:
    """What every system needs from the cached signals of a list of units."""

    gold: np.ndarray  # (n,) verdict indices
    legacy_shipped: np.ndarray  # hallucination scores, low means supported
    legacy_fixed: np.ndarray
    nli: np.ndarray  # support scores
    similarity: np.ndarray
    features: np.ndarray  # (n, 12), computed with the artifact's evidence_k


def _unit_scores(
    units: Sequence[RTUnit],
    signals: Mapping[str, Sequence[EvidenceSignal]],
    evidence_k: int,
) -> _UnitScores:
    """One pass over the cache: each unit's signals are read once and only scores are kept."""
    n = len(units)
    gold = np.array([_INDEX[unit.gold] for unit in units], dtype=np.int64)
    shipped, fixed, nli, similarity = (np.zeros(n) for _ in range(4))
    features = np.zeros((n, len(FEATURE_NAMES)))
    for row, unit in enumerate(units):
        try:
            found = signals[unit.unit_id]
        except KeyError:
            raise KeyError(f"no cached signals for unit {unit.unit_id}") from None
        shipped[row] = baselines.legacy_as_shipped_score(found)
        fixed[row] = baselines.legacy_fixed_score(found)
        nli[row] = baselines.nli_support(found)
        similarity[row] = baselines.similarity_support(found)
        features[row] = compute_features(unit.text, found, evidence_k)
    return _UnitScores(gold, shipped, fixed, nli, similarity, features)


@dataclass(frozen=True)
class _Predictions:
    support: np.ndarray  # higher means more supported
    verdicts: np.ndarray  # verdict indices: detection, false-supported, accepted risk, coverage
    macro_verdicts: np.ndarray  # verdict indices for macro-F1 (the legacy tiers)


def _indices(labels: Sequence[str]) -> np.ndarray:
    return np.array([_INDEX[label] for label in labels], dtype=np.int64)


def _labels(indices: Sequence[int]) -> list[str]:
    return [_NAMES[i] for i in indices]


def _predict(
    scores: _UnitScores, artifact: Calibrator, thresholds: Mapping[str, float]
) -> tuple[dict[str, _Predictions], np.ndarray]:
    """The predictions of the five systems, and the artifact's verdict probabilities."""
    predictions: dict[str, _Predictions] = {}
    for name, hallucination in (
        ("legacy_as_shipped", scores.legacy_shipped),
        ("legacy_fixed", scores.legacy_fixed),
    ):
        values = hallucination.tolist()
        predictions[name] = _Predictions(
            1.0 - hallucination,
            _indices(baselines.legacy_predictions(values, "binary")),
            _indices(baselines.legacy_predictions(values, "tiers")),
        )
    for name, support in (("similarity_only", scores.similarity), ("nli_only", scores.nli)):
        verdicts = np.where(support >= thresholds[name], SUPPORTED, INSUFFICIENT)
        predictions[name] = _Predictions(support, verdicts, verdicts)
    probs = artifact.predict_proba(scores.features)
    decided = np.array([VERDICT_ORDER.index(v) for v in artifact.decide(probs)], dtype=np.int64)
    predictions["evidencegraph"] = _Predictions(probs[:, SUPPORTED], decided, decided)
    return predictions, probs


# --- metrics ----------------------------------------------------------------------------------
#
# The intervals resample thousands of times, so their statistics work on the 3x3 confusion
# matrix (rows gold, columns predicted, both in VERDICT_ORDER, class 0 = supported) instead of
# label lists. The point estimates come from ``evaluation.metrics.classification``.


def _confusion(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    size = len(VERDICT_ORDER)
    return np.bincount(gold * size + pred, minlength=size * size).reshape(size, size)


def _detection_f1(cm: np.ndarray) -> float:
    tp, fp, fn = cm[1:, 1:].sum(), cm[0, 1:].sum(), cm[1:, 0].sum()
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm)
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    seen = tp + fp + fn > 0
    return float(np.mean(2 * tp[seen] / (2 * tp + fp + fn)[seen])) if seen.any() else 0.0


def _false_supported(cm: np.ndarray) -> float | None:
    n = cm[1:, :].sum()
    return float(cm[1:, SUPPORTED].sum() / n) if n else None


def _accepted_risk(cm: np.ndarray) -> float | None:
    n = cm[:, SUPPORTED].sum()
    return float(cm[1:, SUPPORTED].sum() / n) if n else None


# metric name -> (statistic of a confusion matrix, whether it uses the macro-F1 predictions)
_INTERVALS: dict[str, tuple[Callable[[np.ndarray], float | None], bool]] = {
    "detection_f1": (_detection_f1, False),
    "macro_f1": (_macro_f1, True),
    "false_supported_rate": (_false_supported, False),
    "accepted_risk": (_accepted_risk, False),
}


def _block(
    gold: np.ndarray,
    predictions: _Predictions,
    groups: np.ndarray,
    responses: np.ndarray,
) -> dict[str, Any]:
    """All metrics of one system on a set of units, with 95% intervals by source."""
    pred, macro_pred = predictions.verdicts, predictions.macro_verdicts
    gold_labels, pred_labels = _labels(gold), _labels(pred)
    risk, k, n = accepted_risk(gold_labels, pred_labels)

    intervals = {}
    for metric, (statistic, uses_macro) in _INTERVALS.items():
        chosen = macro_pred if uses_macro else pred
        _, lo, hi = cluster_bootstrap(
            groups,
            lambda rows, statistic=statistic, chosen=chosen: statistic(
                _confusion(gold[rows], chosen[rows])
            ),
            n_resamples=N_RESAMPLES,
            seed=SEED,
            ci=CI,
        )
        intervals[metric] = [lo, hi]
    return {
        "n_units": len(gold),
        "n_sources": len(np.unique(groups)),
        "detection": detection_prf(gold_labels, pred_labels),
        "macro_f1": macro_f1(gold_labels, _labels(macro_pred)),
        "false_supported_rate": false_supported_rate(gold_labels, pred_labels),
        "accepted_risk": {
            "risk": risk,
            "k": k,
            "n": n,
            "cp_upper": clopper_pearson_upper(k, n, CP_DELTA),
        },
        "supported_coverage": supported_coverage(pred_labels),
        "response_level": response_level(gold_labels, pred_labels, list(responses)),
        "ci": intervals,
    }


def _curve(support: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
    """Risk at 50 evenly spaced coverages, and the AURC of the full curve."""
    _, risks = risk_coverage(support, gold == SUPPORTED)
    n = len(gold)
    # Coverage c = j / 50 takes the ceil(c n) highest-scoring units (integer arithmetic).
    taken = [-(-j * n // CURVE_POINTS) for j in range(1, CURVE_POINTS + 1)]
    return {
        "coverage": [j / CURVE_POINTS for j in range(1, CURVE_POINTS + 1)],
        "risk": [float(risks[count - 1]) for count in taken],
        "aurc": aurc(risks),
    }


def _calibration(probs: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
    return {
        "ece": ece_top_label(probs, gold),
        "classwise_ece": classwise_ece(probs, gold),
        "brier": brier_multiclass(probs, gold),
        "nll": nll(probs, gold),
        "reliability_bins": reliability_bins(probs.max(axis=1), probs.argmax(axis=1) == gold),
    }


def evaluate(
    units: Sequence[RTUnit],
    gold: np.ndarray,
    predictions: Mapping[str, _Predictions],
    probs: np.ndarray,
) -> dict[str, Any]:
    """Every system's overall and per-task metrics, risk-coverage curve, and calibration."""
    groups = np.array([unit.source_id for unit in units])
    responses = np.array([unit.response_id for unit in units])
    tasks = np.array([unit.task_type for unit in units])

    def subset(system: _Predictions, rows: np.ndarray) -> _Predictions:
        return _Predictions(
            system.support[rows], system.verdicts[rows], system.macro_verdicts[rows]
        )

    systems: dict[str, Any] = {}
    for name, system in predictions.items():
        everything = np.arange(len(units))
        by_task = {}
        for task in sorted({unit.task_type for unit in units}):
            rows = np.flatnonzero(tasks == task)
            by_task[task] = _block(gold[rows], subset(system, rows), groups[rows], responses[rows])
        systems[name] = {
            "score": SCORE_NAMES[name],
            "overall": _block(gold, subset(system, everything), groups, responses),
            "by_task": by_task,
            "risk_coverage": _curve(system.support, gold),
        }
    systems["evidencegraph"]["calibration"] = _calibration(probs, gold)
    return systems


# --- ablations --------------------------------------------------------------------------------


def _ablation_rows(
    artifact: Calibrator,
    caches: Sequence[_CacheData],
    split: str,
    test_units: Mapping[str, Sequence[RTUnit]],
) -> list[dict[str, Any]]:
    """Retrain on fit, select and risk for each configuration; score it on the split.

    The configurations change one thing of the artifact at a time: the feature set, the
    evidence depth, the temperature (off), and the premise mode (when another cache is given).
    """
    feature_set = _feature_set_of(artifact)
    if feature_set is None:
        raise ValueError("the ablations need an artifact made of a known feature set")
    base = {
        "premise_mode": artifact.premise_mode,
        "evidence_k": artifact.evidence_k,
        "feature_set": feature_set,
        "use_temperature": True,
    }
    configs = [("feature_set", name, {**base, "feature_set": name}) for name in FEATURE_SETS]
    configs += [("evidence_k", str(k), {**base, "evidence_k": k}) for k in EVIDENCE_KS]
    configs.append(("temperature", "off", {**base, "use_temperature": False}))
    by_mode = {each.config["premise_mode"]: each for each in caches}
    configs += [
        ("premise_mode", mode, {**base, "premise_mode": mode})
        for mode in by_mode
        if mode != artifact.premise_mode
    ]

    matrices: dict[tuple[str, int], tuple[tuple[Split, Split, Split], Split]] = {}

    def matrices_of(each: _CacheData, evidence_k: int) -> tuple[tuple[Split, Split, Split], Split]:
        """The training matrices of a cache and the matrix of the split to score."""
        if (each.key, evidence_k) not in matrices:
            training = _split_matrices(each, evidence_k)
            if split == "select":
                scored = training[1]
            else:
                units = test_units[each.key]
                signals = _CachedSignals(each.cache, [unit.unit_id for unit in units])
                scored = build_matrix(units, signals, evidence_k)
            matrices[each.key, evidence_k] = (training, scored)
        return matrices[each.key, evidence_k]

    rows = []
    for kind, name, config in configs:
        each = by_mode[config["premise_mode"]]
        (fit, select, risk), (X, y, _) = matrices_of(each, config["evidence_k"])
        calibrator, _ = train(
            fit, select, risk, **config, models=each.config["models"], metadata={}
        )
        probs = calibrator.predict_proba(X)
        gold = _labels(y)
        pred = _labels([VERDICT_ORDER.index(v) for v in calibrator.decide(probs)])
        rows.append(
            {
                "kind": kind,
                "name": name,
                **config,
                "detection_f1": detection_prf(gold, pred)["f1"],
                "macro_f1": macro_f1(gold, pred),
                "false_supported_rate": false_supported_rate(gold, pred),
                "accepted_risk": accepted_risk(gold, pred)[0],
                "nll": nll(probs, y),
                "ece": ece_top_label(probs, y),
                "tau_s": calibrator.tau_s,
                "tau_c": calibrator.tau_c,
            }
        )
    return rows


# --- the run ----------------------------------------------------------------------------------


def _feature_set_of(artifact: Calibrator) -> str | None:
    """The name in ``FEATURE_SETS`` of the artifact's columns, if there is one."""
    for name, names in FEATURE_SETS.items():
        if names == artifact.feature_names:
            return name
    return None


@dataclass(frozen=True)
class Outcome:
    """What a run leaves for the atomic-claim subset, next to the written results."""

    results: dict[str, Any]
    artifact: Calibrator
    units: list[RTUnit]  # the evaluated units
    sentence_verdicts: dict[str, str]  # unit id to the artifact's sentence-level verdict


def run(args: argparse.Namespace) -> Outcome:
    """Evaluate the artifact and the baselines on the split; write the results (and ablations)."""
    if args.split == "test" and not args.final:
        raise ValueError("the test split needs --final")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_hash = manifest_hash(manifest)
    artifact = Calibrator.load(args.artifact)
    if artifact.metadata.get("manifest_hash") != expected_hash:
        raise ValueError("the artifact was not trained with this manifest")
    caches = [_open_cache(args.cache_key, expected_hash)]
    if args.alt_cache_key is not None:
        caches.append(_open_cache(args.alt_cache_key, expected_hash))
    _check_caches(caches)
    cache = _artifact_cache(caches, artifact)
    git_commit, git_dirty = git_info(_REPO_ROOT)

    select_units = cache.units["select"]
    select_signals = _CachedSignals(cache.cache, [unit.unit_id for unit in select_units])
    select_scores = _unit_scores(select_units, select_signals, artifact.evidence_k)
    select_gold = _labels(select_scores.gold)
    thresholds = {
        "nli_only": baselines.choose_threshold(select_scores.nli.tolist(), select_gold),
        "similarity_only": baselines.choose_threshold(
            select_scores.similarity.tolist(), select_gold
        ),
    }

    role_sizes = {role: {"sources": len(manifest["roles"][role])} for role in ragtruth.ROLES}
    for role, units in cache.units.items():
        role_sizes[role].update(_size(units))
    test_units: dict[str, list[RTUnit]] = {}
    if args.split == "test":
        # The log line comes first: a run that crashes while reading is still a test evaluation.
        _log_test_run(args, artifact.artifact_id, git_commit)
        for each in caches if args.ablations else [cache]:
            test_units[each.key] = _read_test_units(each.cache.directory)
        units = test_units[cache.key]
        role_sizes["test"].update(_size(units))
        scores = _unit_scores(
            units,
            _CachedSignals(cache.cache, [unit.unit_id for unit in units]),
            artifact.evidence_k,
        )
    else:
        units, scores = select_units, select_scores

    predictions, probs = _predict(scores, artifact, thresholds)
    results: dict[str, Any] = {
        "split": args.split,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "env": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "data": {
            "dataset": manifest.get("dataset", "ragtruth"),
            "dataset_commit": manifest["commit"],
            "manifest_hash": expected_hash,
            "seed": manifest.get("seed", SEED),
            "role_sizes": role_sizes,
            "evaluated": {"role": args.split, **_size(units)},
        },
        "models": dict(artifact.models),
        "artifact": {
            "id": artifact.artifact_id,
            "premise_mode": artifact.premise_mode,
            "evidence_k": artifact.evidence_k,
            "feature_set": _feature_set_of(artifact),
            "temperature": artifact.temperature,
            "tau_s": artifact.tau_s,
            "tau_c": artifact.tau_c,
            "alpha_s": artifact.alpha_s,
            "alpha_c": artifact.alpha_c,
            "delta": artifact.delta,
        },
        "thresholds": thresholds,
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": SEED, "ci": CI, "by": "source"},
        "cp_delta": CP_DELTA,
        "test_evaluations": {
            "artifact_id": artifact.artifact_id,
            "count": count_test_runs(artifact.artifact_id),
            "log": TEST_LOG_PATH.as_posix(),
        },
        "systems": evaluate(units, scores.gold, predictions, probs),
    }
    _write_json(args.out, results)
    if args.ablations:
        ablations = {
            "split": args.split,
            "artifact_id": artifact.artifact_id,
            "rows": _ablation_rows(artifact, caches, args.split, test_units),
        }
        _write_json(_sibling(args.out, "ablations"), ablations)
    sentence_verdicts = {
        unit.unit_id: _NAMES[verdict]
        for unit, verdict in zip(units, predictions["evidencegraph"].verdicts, strict=True)
    }
    return Outcome(results, artifact, list(units), sentence_verdicts)


def _size(units: Sequence[RTUnit]) -> dict[str, Any]:
    return {
        "units": len(units),
        "sources": len({unit.source_id for unit in units}),
        "responses": len({unit.response_id for unit in units}),
    }


def _sibling(out: Path, suffix: str) -> Path:
    """``results.json`` and ``ablations`` give ``results_ablations.json`` beside it."""
    return out.with_name(f"{out.stem}_{suffix}{out.suffix}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


# --- the atomic-claim subset ------------------------------------------------------------------


def sample_responses(
    units: Sequence[RTUnit], responses: Sequence[RTResponse], n: int, seed: int = SEED
) -> list[RTResponse]:
    """``n`` of the responses that have units, stratified by task type and sampled by source.

    The quota of a task type follows its share of those responses (largest remainder). Within
    a task type the sources are shuffled with the seed, and every source gives a first response
    before any gives a second, so the sample spreads over as many sources as it can.
    """
    task_of = {unit.response_id: unit.task_type for unit in units}
    by_task: dict[str, dict[str, list[RTResponse]]] = {}
    for response in responses:
        task = task_of.get(response.response_id)
        if task is not None:
            by_task.setdefault(task, {}).setdefault(response.source_id, []).append(response)
    counts = {
        task: sum(len(found) for found in per_source.values())
        for task, per_source in by_task.items()
    }
    total = sum(counts.values())
    if n > total:
        raise ValueError(f"only {total} responses have units, fewer than the {n} asked")
    quota = ragtruth.allocate(counts, n)

    sample: list[RTResponse] = []
    for task in sorted(by_task):
        per_source = by_task[task]
        source_ids = sorted(per_source)
        random.Random(f"{seed}:{task}").shuffle(source_ids)
        rounds = max(len(found) for found in per_source.values())
        order = [
            per_source[source_id][number]
            for number in range(rounds)
            for source_id in source_ids
            if number < len(per_source[source_id])
        ]
        sample += order[: quota[task]]
    return sample


def align_claims(
    units: Sequence[RTUnit], claims: Sequence[Claim], verdicts: Sequence[Verdict]
) -> tuple[dict[str, str], int]:
    """The atomic verdict of each sentence unit that has claims, and the claims left over.

    A claim belongs to the unit its sentence span overlaps most (the earliest on a tie); a claim
    that overlaps none is unaligned. A unit is contradicted when any of its claims is, else
    insufficient when any is, else supported: it is not supported when any claim is not.
    """
    found: dict[str, list[Verdict]] = {}
    unaligned = 0
    for claim, verdict in zip(claims, verdicts, strict=True):
        best, best_overlap = None, 0
        for unit in units:
            overlap = min(claim.answer_end, unit.end) - max(claim.answer_start, unit.start)
            if overlap > best_overlap:
                best, best_overlap = unit, overlap
        if best is None:
            unaligned += 1
        else:
            found.setdefault(best.unit_id, []).append(verdict)
    aligned: dict[str, str] = {}
    for unit_id, unit_verdicts in found.items():
        aligned[unit_id] = Verdict.SUPPORTED.value
        for worse in (Verdict.CONTRADICTED, Verdict.INSUFFICIENT):
            if worse in unit_verdicts:
                aligned[unit_id] = worse.value
                break
    return aligned, unaligned


def _ollama_client(settings: Settings, http: httpx.AsyncClient) -> LLMClient:
    return OllamaClient(
        http, settings.ollama_base_url, settings.llm_model, timeout_s=settings.llm_timeout_s
    )


async def _verify_claims(
    claims: Sequence[Claim],
    scope: SearchScope,
    artifact: Calibrator,
    retriever: Retriever,
    models: extract.Models,
) -> list[Verdict]:
    """The artifact's verdict per claim: ``collect_signals`` evidence, then the calibrator."""
    collected = await collect_signals(
        retriever,
        models.reranker,
        models.nli,
        models.embedder,
        claims,
        scope,
        depth=extract.DEPTH,
        candidates=artifact.candidates,
        premise_mode=artifact.premise_mode,
    )
    if not collected:
        return []
    features = np.stack(
        [
            compute_features(claim.text, signals, artifact.evidence_k)
            for claim, (_, signals) in zip(claims, collected, strict=True)
        ]
    )
    return artifact.decide(artifact.predict_proba(features))


async def run_atomic(
    args: argparse.Namespace,
    outcome: Outcome,
    *,
    client_factory: Callable[[Settings, httpx.AsyncClient], LLMClient] = _ollama_client,
    models: extract.Models | None = None,
) -> dict[str, Any]:
    """Verify the atomic claims of ``args.atomic_subset`` test responses; write and return them.

    The claims come from the LLM (``llm_atomic_claims``), the evidence from the response's own
    source, ingested as in the extraction. The result compares the atomic predictions with the
    sentence predictions of the run on the same responses. ``client_factory`` and ``models``
    replace the Ollama client and the real models (for tests).
    """
    settings = Settings()
    models = models if models is not None else extract.load_models("real", "real", settings)
    artifact = outcome.artifact
    artifact.check_parity(
        nli_model=models.nli.model_id,
        embedder_model=models.embedder.model_id,
        reranker_model=models.reranker.model_id,
    )
    sources, responses = ragtruth.load_ragtruth(_fetch_ragtruth(args.data_root))
    sample = sample_responses(outcome.units, responses, args.atomic_subset)
    units_of: dict[str, list[RTUnit]] = {}
    for unit in outcome.units:
        units_of.setdefault(unit.response_id, []).append(unit)

    aligned: dict[str, str] = {}
    methods: Counter[str] = Counter()
    n_unaligned = 0
    db = Database(args.database_url, max_size=4)
    await db.open()
    try:
        tenant_id, _ = await ensure_bench_scope(db, extract.TENANT_NAME, extract.BENCH_COLLECTION)
        services = Services(settings, db, models.embedder, models.reranker, nli=models.nli)
        collection_ids = await ragtruth.ingest_ragtruth(
            services, sources, [response.source_id for response in sample], tenant_id=tenant_id
        )
        retriever = services.retriever()
        async with httpx.AsyncClient() as http:
            llm = client_factory(settings, http)
            for response in sample:
                claims = await llm_atomic_claims(llm, response.response)
                scope = SearchScope(tenant_id, (collection_ids[response.source_id],))
                verdicts = await _verify_claims(claims, scope, artifact, retriever, models)
                found, left_over = align_claims(units_of[response.response_id], claims, verdicts)
                aligned.update(found)
                n_unaligned += left_over
                methods.update(claim.method for claim in claims)
    finally:
        await db.close()

    sampled = [unit for response in sample for unit in units_of[response.response_id]]
    gold = [unit.gold for unit in sampled]
    sentence = [outcome.sentence_verdicts[unit.unit_id] for unit in sampled]
    # A sentence without claims keeps its sentence-unit prediction.
    atomic = [aligned.get(unit.unit_id, kept) for unit, kept in zip(sampled, sentence, strict=True)]
    task_of = {unit.response_id: unit.task_type for unit in sampled}
    result = {
        "n_requested": args.atomic_subset,
        "n_responses": len(sample),
        "n_units": len(sampled),
        "n_claims": sum(methods.values()),
        "n_claims_unaligned": n_unaligned,
        "seed": SEED,
        "llm_model": llm.model_id,
        "by_task": dict(Counter(task_of[response.response_id] for response in sample)),
        "methods": {
            method: methods[method] for method in ("llm_atomic", "sentence", "sentence_fallback")
        },
        "atomic": {
            "detection": detection_prf(gold, atomic),
            "false_supported_rate": false_supported_rate(gold, atomic),
        },
        "sentence": {
            "detection": detection_prf(gold, sentence),
            "false_supported_rate": false_supported_rate(gold, sentence),
        },
    }
    _write_json(_sibling(args.out, "atomic"), result)
    return result


def _fetch_ragtruth(data_root: Path) -> Path:
    """The folder of the pinned RAGTruth files, each verified against its sha256."""
    for name in ("ragtruth_response", "ragtruth_source"):
        data.fetch(name, data_root)
    return data_root


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run_factuality")
    parser.add_argument("--cache-key", required=True, help="the signal cache of the artifact")
    parser.add_argument(
        "--alt-cache-key", help="a second cache with the other premise mode, for the ablations"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=["select", "test"], required=True)
    parser.add_argument(
        "--final",
        action="store_true",
        help="allow the test split (the final evaluation, logged in the test log)",
    )
    parser.add_argument("--ablations", action="store_true", help="retrain per configuration")
    parser.add_argument(
        "--atomic-subset",
        type=_positive_int,
        metavar="N",
        help="verify atomic claims of N test responses (needs --split test --final and Ollama)",
    )
    parser.add_argument(
        "--out", type=Path, help="default: evaluation/results/factuality_<split>.json"
    )
    parser.add_argument(
        "--database-url",
        help="for --atomic-subset; default: $HALLUCISCOPE_BENCH_DATABASE_URL, else "
        + DEFAULT_DATABASE_URL,
    )
    parser.add_argument("--data-root", type=Path, default=data.DATA_DIR, help="for --atomic-subset")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final:
        parser.error("--split test needs --final: the test split is evaluated at the end only")
    if args.atomic_subset is not None and args.split != "test":
        parser.error("--atomic-subset needs --split test --final")
    if args.out is None:
        args.out = RESULTS_DIR / f"factuality_{args.split}.json"
    if args.database_url is None:
        args.database_url = os.environ.get("HALLUCISCOPE_BENCH_DATABASE_URL", DEFAULT_DATABASE_URL)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    outcome = run(args)
    print(f"wrote {args.out} split={outcome.results['split']}")
    if args.atomic_subset is not None:
        atomic = asyncio.run(run_atomic(args, outcome))
        print(f"wrote {_sibling(args.out, 'atomic')} claims={atomic['n_claims']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
