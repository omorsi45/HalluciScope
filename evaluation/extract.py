"""Evidence signal extraction for the RAGTruth factuality benchmark.

    python -m evaluation.extract --roles fit,select,risk --premise-mode passage
    python -m evaluation.extract --roles test --final --premise-mode passage
    python -m evaluation.extract measure --units 200

Every response sentence of the sources a manifest names is verified against its own source: the
sources are ingested as benchmark collections of one tenant, and ``collect_signals`` (the code
path of production verification) gathers and scores the evidence of each sentence within that
source's collection alone. The signals go to a ``SignalCache``; a rerun skips the cached units,
so a crashed run resumes where it stopped. ``units.jsonl`` next to the cache holds the unit
metadata and gold labels.

``measure`` times ``collect_signals`` with the real models and writes what ``plan_sizes`` needs
to fit the extraction into a compute budget.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.db.pool import Database
from backend.models.loader import load_embedder, load_nli, load_reranker
from backend.models.protocols import Embedder, Reranker
from backend.retrieval.types import SearchScope
from backend.services import Services
from backend.verify.engine import collect_signals
from backend.verify.types import Claim, NLIResult, NLIScorer
from evaluation import data, ragtruth
from evaluation.beir import ensure_bench_scope
from evaluation.cache import (
    DEFAULT_ROOT,
    SignalCache,
    config_dict,
    config_key,
    write_atomic,
)
from evaluation.ragtruth import ROLES, RTResponse, RTSource, RTUnit
from evaluation.run_retrieval import DEFAULT_DATABASE_URL, collect_env

TENANT_NAME = "ragtruth-bench"
BENCH_COLLECTION = "ragtruth"  # ensure_bench_scope makes it; only the tenant is used

DEPTH = 8  # evidence passages kept per claim
CANDIDATES = 20  # passages retrieved per claim before reranking
PROGRESS_EVERY = 100  # units between progress lines

CACHE_ROOT = DEFAULT_ROOT
THROUGHPUT_PATH = Path("evaluation/results/throughput.json")
DEFAULT_MANIFEST = Path("evaluation/manifests/ragtruth_splits.json")

# plan_sizes: the fit role starts at 600 sources, and no role goes below its floor.
FIT_START = 600
FLOORS = {"fit": 200, "select": 150, "risk": 150, "test": 150}
SHRINK_STEP = 10
SHRINK_ORDER = ("fit", "select", "risk", "test")


@dataclass(frozen=True)
class ResponseUnits:
    """The units of one response that a run covers (a prefix when `limit` cut it short)."""

    response: RTResponse
    units: tuple[RTUnit, ...]


@dataclass(frozen=True)
class Models:
    embedder: Embedder
    reranker: Reranker
    nli: NLIScorer

    @property
    def ids(self) -> dict[str, str]:
        return {
            "nli": self.nli.model_id,
            "embedder": self.embedder.model_id,
            "reranker": self.reranker.model_id,
        }


def load_models(embedder: str, nli: str, settings: Settings) -> Models:
    """The models of a run: ``real`` loads the configured checkpoints, ``fake`` the test fakes."""
    if embedder == "fake":
        # The deterministic stand-ins of the test suite: for smoke runs, never for results.
        from tests.fakes import HashEmbedder, KeywordReranker

        embedding_model: Embedder = HashEmbedder()
        reranking_model: Reranker = KeywordReranker()
    else:
        embedding_model, reranking_model = load_embedder(settings), load_reranker(settings)
    if nli == "fake":
        from tests.fakes import RuleNLIScorer

        scorer: NLIScorer = RuleNLIScorer()
    else:
        scorer = load_nli(settings)
    return Models(embedding_model, reranking_model, scorer)


def plan_sizes(
    seconds_per_source: Mapping[str, float],
    population: Mapping[str, int],
    budget_s: float = 14_400.0,
) -> dict[str, int]:
    """How many sources of each role to extract so that the estimate stays within the budget.

    Every role starts with all its sources, except fit with at most 600. While the estimated
    total (sources times seconds per source, over the roles) exceeds the budget, sources are
    dropped 10 at a time, never below the role's floor (fit 200, select 150, risk 150, test 150,
    or the whole role when it is smaller): first from fit, then select, risk, and test. If even
    the floors exceed the budget they are returned with ``"over_budget": True`` added; the key
    is absent otherwise, so the result can go straight to ``build_manifest`` as its sizes.
    """
    sizes = {role: population[role] for role in ROLES}
    sizes["fit"] = min(FIT_START, population["fit"])

    def estimate() -> float:
        return sum(sizes[role] * seconds_per_source[role] for role in ROLES)

    for role in SHRINK_ORDER:
        while estimate() > budget_s and sizes[role] > FLOORS[role]:
            sizes[role] = max(FLOORS[role], sizes[role] - SHRINK_STEP)
    if estimate() > budget_s:
        return {**sizes, "over_budget": True}
    return sizes


def _responses_by_source(responses: Sequence[RTResponse]) -> dict[str, list[RTResponse]]:
    grouped: dict[str, list[RTResponse]] = {}
    for response in responses:
        grouped.setdefault(response.source_id, []).append(response)
    return grouped


def _source_roles(responses: Sequence[RTResponse]) -> dict[str, str]:
    """The role of each source, from its id and split; all its responses must agree."""
    roles: dict[str, str] = {}
    for response in responses:
        role = ragtruth.split_role(response.source_id, response.split)
        if roles.setdefault(response.source_id, role) != role:
            raise ValueError(f"source {response.source_id} has responses in more than one split")
    return roles


def role_profile(
    sources: Mapping[str, RTSource], responses: Sequence[RTResponse]
) -> tuple[dict[str, float], dict[str, int]]:
    """Mean units per source and number of sources, per role, over the whole data."""
    roles = _source_roles(responses)
    population = Counter(roles.values())
    n_units: Counter[str] = Counter()
    for response in responses:
        role = roles[response.source_id]
        n_units[role] += len(ragtruth.unitize(response, sources[response.source_id], role)[0])
    mean_units = {
        role: n_units[role] / population[role] if population[role] else 0.0 for role in ROLES
    }
    return mean_units, {role: population[role] for role in ROLES}


def plan_units(
    sources: Mapping[str, RTSource],
    responses: Sequence[RTResponse],
    manifest: Mapping[str, Any],
    roles: Sequence[str],
    *,
    limit: int | None = None,
) -> list[ResponseUnits]:
    """The units of ``roles`` in run order: roles as in ``ROLES``, then the manifest's source
    order, then file order of the responses. ``limit`` keeps only the first N units.

    A source must sit in the role the manifest gives it. Its split decides that role too, so a
    manifest cannot slip test sources into a role that needs no ``--final``.
    """
    by_source = _responses_by_source(responses)
    plan: list[ResponseUnits] = []
    remaining = limit
    for role in ROLES:
        if role not in roles:
            continue
        for source_id in manifest["roles"][role]:
            if source_id not in sources or source_id not in by_source:
                raise ValueError(f"manifest source {source_id} is not in the data")
            for response in by_source[source_id]:
                if ragtruth.split_role(source_id, response.split) != role:
                    raise ValueError(
                        f"source {source_id} is listed under {role} but its split puts it in "
                        "another role"
                    )
                units = ragtruth.unitize(response, sources[source_id], role)[0]
                if remaining is not None:
                    units = units[:remaining]
                    remaining -= len(units)
                if units:
                    plan.append(ResponseUnits(response, tuple(units)))
                if remaining == 0:
                    return plan
    return plan


def sample_units(
    sources: Mapping[str, RTSource],
    responses: Sequence[RTResponse],
    *,
    role: str,
    n_units: int,
    seed: int = ragtruth.SEED,
) -> list[ResponseUnits]:
    """``n_units`` units of ``role`` from a seeded shuffle of its sources, no manifest needed.

    The first responses of all sources come before any second response, so the sample spreads
    over as many sources (and task types) as it can.
    """
    roles = _source_roles(responses)
    by_source = _responses_by_source(responses)
    source_ids = sorted(source_id for source_id, found in roles.items() if found == role)
    random.Random(seed).shuffle(source_ids)
    rounds = max((len(by_source[source_id]) for source_id in source_ids), default=0)
    order = [
        by_source[source_id][number]
        for number in range(rounds)
        for source_id in source_ids
        if number < len(by_source[source_id])
    ]

    plan: list[ResponseUnits] = []
    remaining = n_units
    for response in order:
        units = ragtruth.unitize(response, sources[response.source_id], role)[0][:remaining]
        remaining -= len(units)
        if units:
            plan.append(ResponseUnits(response, tuple(units)))
        if remaining == 0:
            return plan
    raise ValueError(
        f"the data holds only {n_units - remaining} {role} units, fewer than the {n_units} asked"
    )


def write_units(path: Path, plan: Sequence[ResponseUnits]) -> None:
    """Add the units of ``plan`` that ``path`` lacks (all of its fields), written atomically."""
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["unit_id"]] = row
    known = len(rows)
    for item in plan:
        for unit in item.units:
            rows.setdefault(unit.unit_id, asdict(unit))
    if len(rows) > known:
        write_atomic(path, "".join(json.dumps(row) + "\n" for row in rows.values()))


def claims_of(item: ResponseUnits, skip: SignalCache | None = None) -> list[tuple[RTUnit, Claim]]:
    """A claim per unit of the response (those not in ``skip``), numbered by sentence index."""
    return [
        (unit, Claim(index, unit.text, unit.start, unit.end, "sentence"))
        for index, unit in enumerate(item.units)
        if skip is None or unit.unit_id not in skip
    ]


class _Progress:
    """A line on stderr each time another PROGRESS_EVERY units are done."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._done = 0
        self._reported = 0
        self._started = time.perf_counter()

    def advance(self, n_units: int) -> None:
        self._done += n_units
        if self._done // PROGRESS_EVERY > self._reported:
            self._reported = self._done // PROGRESS_EVERY
            rate = self._done / (time.perf_counter() - self._started)
            print(f"{self._done}/{self._total} units, {rate:.2f} units/s", file=sys.stderr)


def _load_ragtruth(data_root: Path) -> tuple[dict[str, RTSource], list[RTResponse]]:
    # fetch verifies the pinned sha256 (and downloads only when no verified copy exists).
    for name in ("ragtruth_response", "ragtruth_source"):
        data.fetch(name, data_root)
    return ragtruth.load_ragtruth(data_root)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Extract the signals of the units not cached yet; return what was done."""
    clock = time.perf_counter()
    sources, responses = _load_ragtruth(args.data_root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = plan_units(sources, responses, manifest, args.roles, limit=args.limit_units)

    settings = Settings()
    models = load_models(args.embedder, args.nli, settings)
    inputs = {
        "premise_mode": args.premise_mode,
        "depth": DEPTH,
        "candidates": CANDIDATES,
        "manifest_hash": ragtruth.manifest_hash(manifest),
    }
    key = config_key(models.ids, **inputs)
    cache = SignalCache(CACHE_ROOT, key, config=config_dict(models.ids, **inputs))
    # The units go to disk before their signals, so every cached unit has its metadata.
    write_units(cache.directory / "units.jsonl", plan)

    todo = [(item, claims) for item in plan if (claims := claims_of(item, skip=cache))]
    n_units = sum(len(item.units) for item in plan)
    n_extracted = sum(len(claims) for _, claims in todo)

    if todo:
        db = Database(args.database_url, max_size=4)
        await db.open()
        try:
            tenant_id, _ = await ensure_bench_scope(db, TENANT_NAME, BENCH_COLLECTION)
            services = Services(settings, db, models.embedder, models.reranker, nli=models.nli)
            pending_sources = list(dict.fromkeys(item.response.source_id for item, _ in todo))
            print(f"ingesting {len(pending_sources)} sources", file=sys.stderr)
            collection_ids = await ragtruth.ingest_ragtruth(
                services, sources, pending_sources, tenant_id=tenant_id
            )
            retriever = services.retriever()
            progress = _Progress(n_extracted)
            for item, claims in todo:
                collected = await collect_signals(
                    retriever,
                    models.reranker,
                    models.nli,
                    models.embedder,
                    [claim for _, claim in claims],
                    SearchScope(tenant_id, (collection_ids[item.response.source_id],)),
                    depth=DEPTH,
                    candidates=CANDIDATES,
                    premise_mode=args.premise_mode,
                )
                cache.put_many(
                    (unit.unit_id, signals)
                    for (unit, _), (_, signals) in zip(claims, collected, strict=True)
                )
                progress.advance(len(claims))
        finally:
            await db.close()

    return {
        "cache_key": key,
        "cache_dir": str(cache.directory),
        "n_units": n_units,
        "n_extracted": n_extracted,
        "n_skipped": n_units - n_extracted,
        "seconds": time.perf_counter() - clock,
    }


class _TimedNLI:
    """An NLI scorer that counts the pairs it scores and the time it spends on them."""

    def __init__(self, inner: NLIScorer) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.n_pairs = 0
        self.seconds = 0.0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        started = time.perf_counter()
        try:
            return self._inner.score(pairs)
        finally:
            self.seconds += time.perf_counter() - started
            self.n_pairs += len(pairs)


def _rate(count: float, seconds: float) -> float:
    if seconds <= 0:
        raise ValueError("nothing was timed")
    return count / seconds


async def measure(args: argparse.Namespace, models: Models | None = None) -> dict[str, Any]:
    """Time ``collect_signals`` on ``args.units`` fit units and write the result to ``args.out``.

    Ingesting the sources is not timed. ``models`` replaces the real models (for tests); the
    command line always measures the real ones. ``seconds_per_source`` is the mean units per
    source of a role (over the full data) divided by ``units_per_s``.
    """
    started_at = datetime.now(UTC)
    sources, responses = _load_ragtruth(args.data_root)
    mean_units, population = role_profile(sources, responses)
    sample = sample_units(sources, responses, role="fit", n_units=args.units)

    settings = Settings()
    models = models if models is not None else load_models("real", "real", settings)
    nli = _TimedNLI(models.nli)
    db = Database(args.database_url, max_size=4)
    await db.open()
    try:
        tenant_id, _ = await ensure_bench_scope(db, TENANT_NAME, BENCH_COLLECTION)
        services = Services(settings, db, models.embedder, models.reranker, nli=nli)
        sample_sources = list(dict.fromkeys(item.response.source_id for item in sample))
        ingest_started = time.perf_counter()
        collection_ids = await ragtruth.ingest_ragtruth(
            services, sources, sample_sources, tenant_id=tenant_id
        )
        print(
            f"ingested {len(sample_sources)} sources in "
            f"{time.perf_counter() - ingest_started:.1f}s (not timed)",
            file=sys.stderr,
        )
        retriever = services.retriever()
        progress = _Progress(args.units)
        collect_seconds = 0.0
        for item in sample:
            claims = [claim for _, claim in claims_of(item)]
            scope = SearchScope(tenant_id, (collection_ids[item.response.source_id],))
            started = time.perf_counter()
            await collect_signals(
                retriever,
                models.reranker,
                nli,
                models.embedder,
                claims,
                scope,
                depth=DEPTH,
                candidates=CANDIDATES,
                premise_mode=args.premise_mode,
            )
            collect_seconds += time.perf_counter() - started
            progress.advance(len(claims))
        env = await collect_env(db)
    finally:
        await db.close()

    n_units = sum(len(item.units) for item in sample)
    units_per_s = _rate(n_units, collect_seconds)
    result: dict[str, Any] = {
        "units_per_s": units_per_s,
        "nli_pairs_per_s": _rate(nli.n_pairs, nli.seconds),
        "n_units": n_units,
        "n_nli_pairs": nli.n_pairs,
        "collect_seconds": collect_seconds,
        "nli_seconds": nli.seconds,
        "mean_units_per_source": mean_units,
        "population": population,
        "seconds_per_source": {role: mean_units[role] / units_per_s for role in ROLES},
        "premise_mode": args.premise_mode,
        "depth": DEPTH,
        "candidates": CANDIDATES,
        "models": models.ids,
        "env": env,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _roles(text: str) -> tuple[str, ...]:
    chosen = [name.strip() for name in text.split(",") if name.strip()]
    if not chosen or set(chosen) - set(ROLES):
        raise argparse.ArgumentTypeError(f"roles must be among {','.join(ROLES)}")
    return tuple(dict.fromkeys(chosen))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database-url",
        help="default: $HALLUCISCOPE_BENCH_DATABASE_URL, else " + DEFAULT_DATABASE_URL,
    )
    parser.add_argument("--data-root", type=Path, default=data.DATA_DIR)


def _default_database_url(args: argparse.Namespace) -> None:
    if args.database_url is None:
        args.database_url = os.environ.get("HALLUCISCOPE_BENCH_DATABASE_URL", DEFAULT_DATABASE_URL)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evaluation.extract")
    parser.add_argument("--roles", type=_roles, required=True, help="comma separated roles")
    parser.add_argument(
        "--final", action="store_true", help="allow the test role (the final evaluation only)"
    )
    parser.add_argument("--premise-mode", choices=["passage", "window"], required=True)
    parser.add_argument("--limit-units", type=_positive_int, help="use the first N units only")
    parser.add_argument("--embedder", choices=["real", "fake"], default="real")
    parser.add_argument("--nli", choices=["real", "fake"], default="real")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    _add_common_arguments(parser)
    args = parser.parse_args(argv)
    if "test" in args.roles and not args.final:
        parser.error("the test role needs --final: the test split is evaluated once, at the end")
    _default_database_url(args)
    return args


def parse_measure_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evaluation.extract measure")
    parser.add_argument("--units", type=_positive_int, default=200, help="fit units to time")
    parser.add_argument("--premise-mode", choices=["passage", "window"], default="passage")
    parser.add_argument("--out", type=Path, default=THROUGHPUT_PATH)
    _add_common_arguments(parser)
    args = parser.parse_args(argv)
    _default_database_url(args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["measure"]:
        measure_args = parse_measure_args(arguments[1:])
        result = asyncio.run(measure(measure_args))
        print(
            f"wrote {measure_args.out} units_per_s={result['units_per_s']:.3f} "
            f"nli_pairs_per_s={result['nli_pairs_per_s']:.3f}"
        )
        return 0
    args = parse_args(arguments)
    summary = asyncio.run(run(args))
    print(
        f"{summary['cache_dir']} units={summary['n_units']} "
        f"extracted={summary['n_extracted']} skipped={summary['n_skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
