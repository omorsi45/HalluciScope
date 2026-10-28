"""Synthetic temporal and provenance benchmark (spec section 20.3).

    python -m evaluation.run_temporal --families 40
    python -m evaluation.report temporal evaluation/results/temporal.json

The corpus comes from ``evaluation.synthetic_temporal`` and every gold verdict follows from the
generated versions. The run, in order:

1. Two tenants, each with a system principal: ``A`` holds the families and their distractors in
   one collection, ``B`` holds the conflicting copies. Both are new for every run, so a rerun
   never meets the documents of an earlier one.
2. Version 1 of every document is uploaded through ``intake_upload`` and ``process_upload``
   with its ``effective_from``, then one live ``verify`` analysis per family checks all its
   claims as sentences.
3. Versions 2 and 3 are uploaded the same way, round by round; each upload queues a
   ``reverify_document`` job and the worker runs them all before the next round.
4. A ``verify`` analysis per family and version, with ``as_of`` the version's effective time
   plus one day.

The calibrator is used unchanged. ``known_at`` and ``created_at`` come from the database and
are never compared with Python's clock.

``cross_tenant_links`` counts evidence links of tenant A that point at a passage of tenant B.
Row level security is forced on every tenant table and ``eg_owner`` has no ``BYPASSRLS``, so an
owner-level count without a tenant sees no rows at all and would report a meaningless 0. The
count instead reads B's passage ids inside B's transaction and looks for them among A's links
inside A's, so each side is read where it is visible. The run exits non-zero when it is not 0.

Source-change detection compares the verdict a live claim ends with (that of its latest
re-check, else its own) with the verdict it started with, which is also how the gold change is
defined (last version against version 1). For a claim with one re-check this is the same as the
latest ``claim_checks.changed``; when a version 3 re-check follows a version 2 one,
``changed`` only compares version 3 with version 2.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from backend.analysis.service import create_analysis_request, run_analysis
from backend.auth.service import Principal, principal_for_user
from backend.calibration.artifact import Calibrator
from backend.config import Settings
from backend.db.pool import Database
from backend.db.repos import analyses, tenants, users
from backend.ingest.handlers import process_upload
from backend.ingest.intake import intake_upload
from backend.jobs.registry import build_failure_handlers, build_handlers
from backend.jobs.worker import Worker
from backend.provenance.citations import CitationReport, validate_analysis
from backend.services import Services
from evaluation.beir import ensure_bench_scope
from evaluation.extract import load_models
from evaluation.metrics.classification import false_supported_rate, macro_f1
from evaluation.run_retrieval import DEFAULT_DATABASE_URL, collect_env
from evaluation.synthetic_temporal import (
    SEED,
    SUPPORTED,
    FamilySpec,
    TemporalClaim,
    claim_value,
    document_title,
    generate,
    render_document,
)

COLLECTION_NAME = "temporal"
BENCH_KINDS = ("benchmark",)
# The hash of the CLI's system user: no password verifies against it, so nobody can log in as
# the bench users. They are not disabled, because re-verification runs as the analysis's
# creator and skips a disabled one.
LOCKED_PASSWORD_HASH = "!"
AS_OF_OFFSET = timedelta(days=1)
SAMPLE_SIZE = 30
# Generous limits for the real models; a benchmark has no latency requirement.
ANALYSIS_TIMEOUT_S = 3600.0
JOB_TIMEOUT_S = 3600.0

METRIC_KEYS = (
    "live",
    "per_version",
    "temporal_correctness",
    "evidence_validity",
    "span_correctness",
    "source_change",
    "cross_tenant_links",
)
_SUPPORT_RELATION = {"supported": "supports", "contradicted": "contradicts"}


# --- metrics -----------------------------------------------------------------------------------


def _ratio(count: int, total: int) -> float | None:
    return count / total if total else None


def version_scores(gold: Sequence[str], pred: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro-F1 and false-supported rate of one as-of time (None when empty)."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    if not gold:
        return {"n": 0, "accuracy": None, "macro_f1": None, "false_supported_rate": None}
    return {
        "n": len(gold),
        "accuracy": sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold),
        "macro_f1": macro_f1(gold, pred),
        "false_supported_rate": false_supported_rate(gold, pred),
    }


def change_points(sequence: Sequence[str]) -> tuple[int, ...]:
    """The indices at which a verdict sequence differs from the one before."""
    return tuple(i for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1])


def temporal_correct(gold: Sequence[str], pred: Sequence[str]) -> bool:
    """True when the predicted verdict changes exactly where the gold verdict does."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    return change_points(gold) == change_points(pred)


def temporal_correctness(
    gold: Mapping[str, Sequence[str]], pred: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """The share of claims whose predicted sequence changes exactly where the gold one does."""
    correct = sum(temporal_correct(gold[claim_id], pred[claim_id]) for claim_id in gold)
    return {"n_claims": len(gold), "n_correct": correct, "rate": _ratio(correct, len(gold))}


def gold_changed(gold: Sequence[str]) -> bool:
    """Whether the gold verdict at the last version differs from the one at version 1."""
    return gold[-1] != gold[0]


def predicted_changed(original: str, rechecked: str | None) -> bool:
    """Whether a claim's latest verdict differs from its original one.

    ``rechecked`` is the verdict of its newest re-check, None for a claim never re-checked.
    """
    return (original if rechecked is None else rechecked) != original


def source_change_scores(gold: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    """Precision and recall of the predicted source changes against the gold ones."""
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted differ in length")
    tp = sum(g and p for g, p in zip(gold, predicted, strict=True))
    n_gold, n_predicted = sum(gold), sum(predicted)
    return {
        "n_gold_changed": n_gold,
        "n_predicted_changed": n_predicted,
        "tp": tp,
        "precision": _ratio(tp, n_predicted),
        "recall": _ratio(tp, n_gold),
    }


def span_text(item: Mapping[str, Any]) -> str:
    """The text of an evidence item's span, cut from its passage."""
    start = item["span_start"] - item["start_char"]
    return item["text"][start : item["span_end"] - item["start_char"]]


def span_check(
    item: Mapping[str, Any] | None, expected_version_id: object, value: str
) -> tuple[bool, bool]:
    """(the item is in the expected version, its span text contains the claim's value)."""
    if item is None:
        return False, False
    return item["version_id"] == expected_version_id, value in span_text(item)


def span_scores(checks: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    correct = sum(version and contains for version, contains in checks)
    return {
        "n_supported": len(checks),
        "n_version_correct": sum(version for version, _ in checks),
        "n_span_contains_value": sum(contains for _, contains in checks),
        "n_correct": correct,
        "rate": _ratio(correct, len(checks)),
    }


async def count_cross_tenant_links(db: Database, tenant_id: UUID, other_tenant_id: UUID) -> int:
    """Evidence links of ``tenant_id`` that point at a passage of ``other_tenant_id``."""
    async with db.tenant_tx(other_tenant_id) as conn:
        cur = await conn.execute("select id from passages")
        foreign_ids = [row["id"] for row in await cur.fetchall()]
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select count(*) as n from evidence_links where passage_id = any(%s::uuid[])",
            (foreign_ids,),
        )
        return (await cur.fetchone())["n"]


# --- sample ------------------------------------------------------------------------------------


def _cell(text: object) -> str:
    """A table cell: one line, pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def render_sample(cases: Sequence[Mapping[str, Any]]) -> str:
    """The sampled cases as a markdown table for human review (format only)."""
    lines = [
        "# Temporal benchmark: sampled cases",
        "",
        f"{len(cases)} cases drawn with seed {SEED} from the as-of analyses of the synthetic "
        "temporal benchmark, for human review. The gold verdict follows from the generated "
        "versions; the evidence is the span the verdict was reached on. Generated by "
        "`python -m evaluation.run_temporal`; do not edit by hand.",
        "",
        "| Claim | As-of | Gold | Prediction | Evidence span | Document | Version |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for case in cases:
        span = case["evidence_span"]
        lines.append(
            f"| {_cell(case['claim'])} "
            f"| v{case['version_index'] + 1} ({case['as_of'][:10]}) "
            f"| {case['gold']} | {case['prediction']} "
            f"| {'(no evidence)' if span is None else _cell(span)} "
            f"| {'-' if case['document'] is None else _cell(case['document'])} "
            f"| {'-' if case['version'] is None else case['version']} |"
        )
    return "\n".join(lines) + "\n"


# --- the run -----------------------------------------------------------------------------------


@dataclass
class _Tenant:
    principal: Principal
    collection_id: UUID
    documents: dict[str, UUID] = field(default_factory=dict)  # family id -> document id

    @property
    def tenant_id(self) -> UUID:
        return self.principal.tenant_id


@dataclass
class _Tally:
    """What the analyses measured, collected as they finish."""

    golds: defaultdict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    preds: defaultdict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    sequences: defaultdict[str, dict[int, str]] = field(default_factory=lambda: defaultdict(dict))
    spans: list[tuple[bool, bool]] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    n_analyses: int = 0
    n_links: int = 0
    n_valid_links: int = 0
    n_citations: int = 0
    n_valid_citations: int = 0


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def _make_tenant(db: Database, name: str) -> _Tenant:
    """A tenant with a benchmark collection and an owner that nobody can log in as."""
    tenant_id, collection_id = await ensure_bench_scope(db, name, COLLECTION_NAME)
    async with db.system_tx() as conn:
        user = await users.insert_user(
            conn, f"{name}@localhost", LOCKED_PASSWORD_HASH, "Temporal bench"
        )
        await tenants.add_membership(conn, tenant_id, user["id"], "owner")
    principal = await principal_for_user(db, tenant_id, user["id"])
    if principal is None:
        raise RuntimeError("the benchmark user cannot act for its tenant")
    return _Tenant(principal, collection_id)


async def _upload(
    services: Services, tenant: _Tenant, family: FamilySpec, version_index: int
) -> None:
    """Upload one version of a family's document and ingest it in this process."""
    document_id = tenant.documents.get(family.family_id)
    target = (
        {"collection_id": tenant.collection_id}
        if document_id is None
        else {"document_id": document_id}
    )
    intake = await intake_upload(
        services.db,
        tenant.principal,
        data=render_document(family, version_index).encode(),
        filename=f"{family.family_id}.txt",
        title=document_title(family),
        effective_from=family.effective[version_index],
        settings=services.settings,
        collection_kinds=BENCH_KINDS,
        enqueue_job=False,
        **target,
    )
    created = await process_upload(services, tenant.tenant_id, intake.upload_id)
    if created is None or created.status != "created_version":
        raise RuntimeError("an upload did not create a version")
    tenant.documents[family.family_id] = intake.document_id


async def _drain_reverification(services: Services) -> int:
    """Run every queued ``reverify_document`` job; return how many ran."""
    handlers = {
        kind: handler
        for kind, handler in build_handlers(services).items()
        if kind == "reverify_document"
    }
    failure_handlers = {
        kind: handler
        for kind, handler in build_failure_handlers(services).items()
        if kind in handlers
    }
    worker = Worker(
        services.db,
        handlers,
        failure_handlers=failure_handlers,
        services=services,
        timeouts_s={"reverify_document": JOB_TIMEOUT_S},
    )
    ran = 0
    while await asyncio.wait_for(worker.run_once(), JOB_TIMEOUT_S + 60):
        ran += 1
    return ran


async def _require_reverification(db: Database, tenant_ids: Sequence[UUID], expected: int) -> None:
    """Every re-verification job of the run must have succeeded: none missing, none failed."""
    async with db.system_tx() as conn:
        cur = await conn.execute(
            "select status, count(*) as n from jobs "
            "where kind = 'reverify_document' and tenant_id = any(%s::uuid[]) group by status",
            (list(tenant_ids),),
        )
        counts = {row["status"]: row["n"] for row in await cur.fetchall()}
    if counts != ({"succeeded": expected} if expected else {}):
        raise RuntimeError(f"re-verification jobs did not all succeed: {counts}")


async def _run_verify(
    services: Services,
    tenant: _Tenant,
    claims: Sequence[TemporalClaim],
    as_of: datetime | None,
) -> UUID:
    """Create a verify analysis whose answer is the claims as sentences, and run it here."""
    analysis_id = await create_analysis_request(
        services.db,
        tenant.principal,
        kind="verify",
        question=None,
        answer=" ".join(claim.text for claim in claims),
        collection_ids=[tenant.collection_id],
        as_of=as_of,
        options={"extraction": "sentence"},
        settings=services.settings,
        collection_kinds=BENCH_KINDS,
        enqueue=False,
    )
    await run_analysis(services, tenant.tenant_id, analysis_id)
    return analysis_id


async def _read_analysis(
    db: Database, tenant_id: UUID, analysis_id: UUID, claims: Sequence[TemporalClaim]
) -> tuple[list[dict[str, Any]], dict[UUID, list[dict[str, Any]]], CitationReport]:
    """The stored claims, their evidence by claim, and the provenance audit of one analysis."""
    async with db.tenant_tx(tenant_id) as conn:
        rows = await analyses.get_claims(conn, analysis_id)
        evidence = await analyses.get_evidence(conn, [row["id"] for row in rows])
        audit = await validate_analysis(conn, analysis_id)
    if len(rows) != len(claims) or any(
        row["text"] != claim.text for row, claim in zip(rows, claims, strict=True)
    ):
        raise RuntimeError("the analysis did not keep the claims as they were written")
    by_claim: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        by_claim[item["claim_id"]].append(item)
    return rows, by_claim, audit


def _add_audit(tally: _Tally, audit: CitationReport) -> None:
    tally.n_analyses += 1
    tally.n_links += audit.n_links
    tally.n_valid_links += audit.n_valid_links
    tally.n_citations += audit.n_citations
    tally.n_valid_citations += audit.n_valid_citations


def _shown_item(items: Sequence[dict[str, Any]], verdict: str) -> dict[str, Any] | None:
    """The evidence item that explains a verdict: the first with its relation, else the first."""
    relation = _SUPPORT_RELATION.get(verdict)
    matching = [item for item in items if item["relation"] == relation]
    return (matching or list(items) or [None])[0]


def _record_as_of(
    tally: _Tally,
    family: FamilySpec,
    claims: Sequence[TemporalClaim],
    version_index: int,
    as_of: datetime,
    expected_version_id: UUID,
    rows: Sequence[dict[str, Any]],
    by_claim: Mapping[UUID, Sequence[dict[str, Any]]],
) -> None:
    for claim, row in zip(claims, rows, strict=True):
        verdict, items = row["verdict"], by_claim.get(row["id"], ())
        tally.golds[version_index].append(claim.gold[version_index])
        tally.preds[version_index].append(verdict)
        tally.sequences[claim.claim_id][version_index] = verdict
        if verdict == SUPPORTED:
            support = next((item for item in items if item["relation"] == "supports"), None)
            tally.spans.append(span_check(support, expected_version_id, claim_value(claim, family)))
        shown = _shown_item(items, verdict)
        tally.cases.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "kind": claim.kind,
                "as_of": as_of.isoformat(),
                "version_index": version_index,
                "gold": claim.gold[version_index],
                "prediction": verdict,
                "evidence_span": None if shown is None else span_text(shown),
                "document": None if shown is None else shown["document_title"],
                "version": None if shown is None else shown["version_number"],
            }
        )


async def _version_ids(
    db: Database, tenant: _Tenant, families: Sequence[FamilySpec]
) -> dict[str, list[UUID]]:
    """The version ids of each family's document in version order."""
    async with db.tenant_tx(tenant.tenant_id) as conn:
        cur = await conn.execute(
            "select document_id, id from document_versions "
            "where document_id = any(%s::uuid[]) order by document_id, version_number",
            (list(tenant.documents.values()),),
        )
        rows = await cur.fetchall()
    by_document: dict[UUID, list[UUID]] = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row["id"])
    found = {f.family_id: by_document[tenant.documents[f.family_id]] for f in families}
    if any(len(found[f.family_id]) != len(f.versions) for f in families):
        raise RuntimeError("a document does not have the versions that were uploaded")
    return found


def _models_and_calibrator(args: argparse.Namespace, settings: Settings):
    return load_models(args.embedder, args.nli, settings), Calibrator.load(args.artifact)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the benchmark on the database, write the results and the sample, return the results."""
    started_at = datetime.now(UTC)
    clock = time.perf_counter()
    corpus = generate(n_families=args.families)
    claims_of: dict[str, list[TemporalClaim]] = defaultdict(list)
    for claim in corpus.claims:
        claims_of[claim.family_id].append(claim)

    settings = Settings(analysis_timeout_s=ANALYSIS_TIMEOUT_S)
    models = load_models(args.embedder, args.nli, settings)
    calibrator = Calibrator.load(args.artifact)
    db = Database(args.database_url, max_size=4)
    await db.open()
    try:
        services = Services(
            settings, db, models.embedder, models.reranker, nli=models.nli, calibrator=calibrator
        )
        label = uuid4().hex[:8]
        ours = await _make_tenant(db, f"temporal-{label}-a")
        theirs = await _make_tenant(db, f"temporal-{label}-b")

        _log("uploading version 1")
        for family in (*corpus.families, *corpus.distractors):
            await _upload(services, ours, family, 0)
        for family in corpus.foreign:
            await _upload(services, theirs, family, 0)

        tally = _Tally()
        live: dict[str, list[tuple[UUID, str]]] = {}  # family id -> (claim row id, verdict)
        live_gold: list[str] = []
        live_pred: list[str] = []
        for family in corpus.families:
            claims = claims_of[family.family_id]
            analysis_id = await _run_verify(services, ours, claims, None)
            rows, _, audit = await _read_analysis(db, ours.tenant_id, analysis_id, claims)
            _add_audit(tally, audit)
            live[family.family_id] = [(row["id"], row["verdict"]) for row in rows]
            live_gold += [claim.gold[0] for claim in claims]
            live_pred += [row["verdict"] for row in rows]
        _log(f"live analyses done: {len(corpus.families)}")

        n_rounds = max(len(f.versions) for f in corpus.families)
        n_reverify = reverify_ran = 0
        for version_index in range(1, n_rounds):
            _log(f"uploading version {version_index + 1}")
            for family in corpus.families:
                if version_index < len(family.versions):
                    await _upload(services, ours, family, version_index)
                    n_reverify += 1
            for family in corpus.foreign:
                if version_index < len(family.versions):
                    await _upload(services, theirs, family, version_index)
                    n_reverify += 1
            reverify_ran += await _drain_reverification(services)
        await _require_reverification(db, [ours.tenant_id, theirs.tenant_id], n_reverify)
        if reverify_ran != n_reverify:
            raise RuntimeError("the worker did not run every re-verification job")
        _log(f"re-verification jobs done: {reverify_ran}")

        version_ids = await _version_ids(db, ours, corpus.families)
        for family in corpus.families:
            claims = claims_of[family.family_id]
            for version_index, effective in enumerate(family.effective):
                as_of = effective + AS_OF_OFFSET
                analysis_id = await _run_verify(services, ours, claims, as_of)
                rows, by_claim, audit = await _read_analysis(
                    db, ours.tenant_id, analysis_id, claims
                )
                _add_audit(tally, audit)
                _record_as_of(
                    tally,
                    family,
                    claims,
                    version_index,
                    as_of,
                    version_ids[family.family_id][version_index],
                    rows,
                    by_claim,
                )
        _log(f"as-of analyses done: {sum(len(f.versions) for f in corpus.families)}")

        async with db.tenant_tx(ours.tenant_id) as conn:
            latest = await analyses.latest_checks(
                conn, [row_id for rows in live.values() for row_id, _ in rows]
            )
        cross_tenant_links = await count_cross_tenant_links(db, ours.tenant_id, theirs.tenant_id)
        env = await collect_env(db)
    finally:
        await db.close()

    gold_sequences = {claim.claim_id: claim.gold for claim in corpus.claims}
    predicted_sequences = {
        claim_id: [tally.sequences[claim_id][j] for j in range(len(gold))]
        for claim_id, gold in gold_sequences.items()
    }
    gold_flags: list[bool] = []
    predicted_flags: list[bool] = []
    for family in corpus.families:
        for claim, (row_id, original) in zip(
            claims_of[family.family_id], live[family.family_id], strict=True
        ):
            rechecked = latest[row_id]["verdict"] if row_id in latest else None
            gold_flags.append(gold_changed(claim.gold))
            predicted_flags.append(predicted_changed(original, rechecked))

    sample = random.Random(SEED).sample(tally.cases, min(SAMPLE_SIZE, len(tally.cases)))
    sample.sort(key=lambda case: (case["claim_id"], case["version_index"]))
    n_versions = sum(len(f.versions) for f in corpus.families)
    result: dict[str, Any] = {
        "seed": SEED,
        "corpus": {
            "families": len(corpus.families),
            "distractors": len(corpus.distractors),
            "foreign": len(corpus.foreign),
            "versions": n_versions,
            "claims": len(corpus.claims),
            "claims_by_kind": dict(Counter(claim.kind for claim in corpus.claims)),
            "analyses": len(corpus.families) + n_versions,
        },
        "live": version_scores(live_gold, live_pred),
        "per_version": [
            {"version": index + 1, **version_scores(tally.golds[index], tally.preds[index])}
            for index in sorted(tally.golds)
        ],
        "temporal_correctness": temporal_correctness(gold_sequences, predicted_sequences),
        "evidence_validity": {
            "n_analyses": tally.n_analyses,
            "n_links": tally.n_links,
            "n_valid_links": tally.n_valid_links,
            "link_rate": _ratio(tally.n_valid_links, tally.n_links),
            "n_citations": tally.n_citations,
            "n_valid_citations": tally.n_valid_citations,
            "citation_rate": _ratio(tally.n_valid_citations, tally.n_citations),
        },
        "span_correctness": span_scores(tally.spans),
        "source_change": {
            "n_claims": len(gold_flags),
            "n_checked": len(latest),
            **source_change_scores(gold_flags, predicted_flags),
        },
        "cross_tenant_links": cross_tenant_links,
        "reverify": {"jobs": reverify_ran},
        "models": models.ids,
        "calibrator": {
            "id": calibrator.artifact_id,
            "premise_mode": calibrator.premise_mode,
            "evidence_k": calibrator.evidence_k,
        },
        "env": env,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_seconds": time.perf_counter() - clock,
        "sample": sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    args.sample_out.write_text(render_sample(sample), encoding="utf-8", newline="\n")
    return result


# --- command line ------------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run_temporal")
    parser.add_argument(
        "--database-url",
        help="default: $HALLUCISCOPE_BENCH_DATABASE_URL, else " + DEFAULT_DATABASE_URL,
    )
    parser.add_argument("--families", type=_positive_int, default=40)
    parser.add_argument("--embedder", choices=["real", "fake"], default="real")
    parser.add_argument("--nli", choices=["real", "fake"], default="real")
    parser.add_argument(
        "--artifact",
        type=Path,
        help="the calibrator artifact, used unchanged; default: the configured calibrator path",
    )
    parser.add_argument("--out", type=Path, default=Path("evaluation/results/temporal.json"))
    parser.add_argument("--sample-out", type=Path, default=Path("docs/results/temporal-sample.md"))
    args = parser.parse_args(argv)
    if args.database_url is None:
        args.database_url = os.environ.get("HALLUCISCOPE_BENCH_DATABASE_URL", DEFAULT_DATABASE_URL)
    if args.artifact is None:
        args.artifact = Path(Settings().calibrator_path)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark; exit 1 when a tenant's evidence points into another (files written)."""
    args = parse_args(argv)
    result = asyncio.run(run(args))
    print(f"wrote {args.out} and {args.sample_out}")
    if result["cross_tenant_links"] != 0:
        print(
            f"FAILED: cross_tenant_links = {result['cross_tenant_links']}, must be 0",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
