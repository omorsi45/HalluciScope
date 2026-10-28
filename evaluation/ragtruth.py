"""RAGTruth: loading, source rendering, span-aligned sentence units, and source-level splits.

Leakage control: the role of a source (fit, select, risk, test) is a pure function of its id and
split, so every response of a source lands in one role. A manifest freezes the chosen source ids.

    python -m evaluation.ragtruth manifest --sizes fit=N,select=N,risk=N,test=all --out PATH
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection

from backend.db.repos import documents
from backend.ingest.parsers.text import parse_text
from backend.ingest.segment import segment_sentences
from backend.ingest.service import create_version
from backend.services import Services
from evaluation import data

CONFLICT = frozenset({"Evident Conflict", "Subtle Conflict"})
BASELESS = frozenset({"Evident Baseless Info", "Subtle Baseless Info"})

# The RAGTruth commit that data.REGISTRY pins.
COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
SEED = 20261026
ROLES = ("fit", "select", "risk", "test")

Role = Literal["fit", "select", "risk", "test"]


@dataclass(frozen=True)
class RTSpan:
    start: int
    end: int
    text: str
    label_type: str


@dataclass(frozen=True)
class RTSource:
    source_id: str
    task_type: str
    source_name: str
    text: str
    question: str | None


@dataclass(frozen=True)
class RTResponse:
    response_id: str
    source_id: str
    model: str
    split: str
    response: str
    labels: tuple[RTSpan, ...]


@dataclass(frozen=True)
class RTUnit:
    unit_id: str  # f"{response_id}:{index of the sentence in the response}"
    response_id: str
    source_id: str
    task_type: str
    role: str
    start: int
    end: int
    text: str
    gold: str  # "supported", "contradicted", or "insufficient"


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_ragtruth(root: Path) -> tuple[dict[str, RTSource], list[RTResponse]]:
    """Read ``source_info.jsonl`` and ``response.jsonl`` in ``root``, as fetched by ``data``.

    Every response is kept whatever its quality, and every label is kept, including the
    ``implicit_true`` and ``due_to_null`` ones, which RAGTruth counts as hallucinations.
    """
    sources: dict[str, RTSource] = {}
    for row in _read_jsonl(root / "source_info.jsonl"):
        text, question = render_source(row["task_type"], row["source_info"])
        source_id = str(row["source_id"])
        sources[source_id] = RTSource(source_id, row["task_type"], row["source"], text, question)
    responses = [
        RTResponse(
            str(row["id"]),
            str(row["source_id"]),
            row["model"],
            row["split"],
            row["response"],
            tuple(
                RTSpan(label["start"], label["end"], label["text"], label["label_type"])
                for label in row["labels"]
            ),
        )
        for row in _read_jsonl(root / "response.jsonl")
    ]
    return sources, responses


def render_source(task_type: str, source_info: Any) -> tuple[str, str | None]:
    """The source text and the question (QA only) of a RAGTruth source."""
    if task_type == "Summary":
        return source_info, None
    if task_type == "QA":
        return source_info["passages"], source_info["question"]
    if task_type == "Data2txt":
        return "\n".join(_flatten(source_info, "")), None
    raise ValueError(f"unknown RAGTruth task type: {task_type!r}")


def _flatten(value: Any, prefix: str) -> Iterator[str]:
    """One ``key: value`` line per scalar, in insertion order; None is skipped."""
    if value is None:
        return
    if isinstance(value, dict):
        children: Iterable[tuple[Any, Any]] = value.items()
    elif isinstance(value, list):
        children = enumerate(value)
    else:
        yield f"{prefix}: {_scalar(value)}"
        return
    for key, child in children:
        yield from _flatten(child, f"{prefix}.{key}" if prefix else str(key))


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def split_role(source_id: str, split: str) -> Role:
    """The R1 role of a source: ``test`` for the test split, else a bucket of its sha256."""
    if split == "test":
        return "test"
    bucket = int(hashlib.sha256(source_id.encode()).hexdigest(), 16) % 100
    if bucket < 70:
        return "fit"
    return "select" if bucket < 85 else "risk"


def unitize(
    response: RTResponse, source: RTSource, role: str
) -> tuple[list[RTUnit], dict[str, int]]:
    """Sentence units of a response, each with the gold verdict its overlapping spans imply.

    A span labels a unit when the overlap holds a non-whitespace character. A conflict span
    makes the unit ``contradicted``, else a baseless span makes it ``insufficient``, else it is
    ``supported``. Span offsets are used even when ``response[start:end]`` differs from the span
    text; both that and spans that overlap no unit are counted.
    """
    text = response.response
    sentences = segment_sentences(text)
    label_types: list[set[str]] = [set() for _ in sentences]
    n_mismatch = n_without_unit = 0
    for span in response.labels:
        if text[span.start : span.end] != span.text:
            n_mismatch += 1
        touched = False
        for sentence, types in zip(sentences, label_types, strict=True):
            low, high = max(span.start, sentence.start), min(span.end, sentence.end)
            if low < high and not text[low:high].isspace():
                types.add(span.label_type)
                touched = True
        if not touched:
            n_without_unit += 1
    units = [
        RTUnit(
            f"{response.response_id}:{index}",
            response.response_id,
            response.source_id,
            source.task_type,
            role,
            sentence.start,
            sentence.end,
            text[sentence.start : sentence.end],
            _gold(types),
        )
        for index, (sentence, types) in enumerate(zip(sentences, label_types, strict=True))
    ]
    stats = {
        "n_spans": len(response.labels),
        "n_span_text_mismatch": n_mismatch,
        "n_spans_without_unit": n_without_unit,
    }
    return units, stats


def _gold(label_types: set[str]) -> str:
    if label_types & CONFLICT:
        return "contradicted"
    if label_types & BASELESS:
        return "insufficient"
    return "supported"


def allocate(counts: Mapping[str, int], size: int) -> dict[str, int]:
    """Split ``size`` over the keys of ``counts`` in proportion to them (largest remainder).

    The result sums to ``size`` and never exceeds a count, given ``0 <= size <= sum(counts)``.
    Ties between equal remainders go to the smaller key.
    """
    total = sum(counts.values())
    allocation = {key: size * count // total for key, count in counts.items()}
    by_remainder = sorted(counts, key=lambda key: (-(size * counts[key] % total), key))
    for key in by_remainder[: size - sum(allocation.values())]:
        allocation[key] += 1
    return allocation


def build_manifest(
    sources: Mapping[str, RTSource],
    responses: Sequence[RTResponse],
    *,
    sizes: Mapping[str, int | None],
    seed: int = SEED,
    data_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Choose the source ids of each role: sampled without replacement, stratified by task type.

    ``sizes[role]`` is how many sources the role gets (``None`` keeps all of them). Each role's
    quota is split over task types in proportion to their share of the role's sources
    (largest remainder), so a size is met exactly. The sample of one (role, task type) depends
    only on the seed and its own population, never on the other roles' sizes or on input order.
    """
    if set(sizes) != set(ROLES):
        raise ValueError(f"sizes must name exactly the roles {ROLES}")
    role_of: dict[str, str] = {}
    for response in responses:
        role = split_role(response.source_id, response.split)
        if role_of.setdefault(response.source_id, role) != role:
            raise ValueError(f"source {response.source_id} has responses in more than one split")
    pool: dict[str, dict[str, list[str]]] = {role: {} for role in ROLES}
    for source_id in sorted(role_of):
        pool[role_of[source_id]].setdefault(sources[source_id].task_type, []).append(source_id)
    n_responses = Counter(response.source_id for response in responses)

    roles: dict[str, list[str]] = {}
    counts: dict[str, Any] = {}
    for role in ROLES:
        by_task = pool[role]
        total = sum(len(ids) for ids in by_task.values())
        size = sizes[role]
        wanted = total if size is None else size
        if wanted < 0:
            raise ValueError(f"{role}: size must not be negative")
        if wanted > total:
            raise ValueError(f"{role}: size {wanted} exceeds the population of {total} sources")
        quota = allocate({task: len(ids) for task, ids in by_task.items()}, wanted)
        chosen: list[str] = []
        for task in sorted(by_task):
            chosen += random.Random(f"{seed}:{role}:{task}").sample(by_task[task], quota[task])
        roles[role] = sorted(chosen)
        counts[role] = {
            "sources": dict(quota),
            "responses": sum(n_responses[source_id] for source_id in chosen),
        }
    return {
        "dataset": "ragtruth",
        "commit": COMMIT,
        "seed": seed,
        "sizes": {role: sizes[role] for role in ROLES},
        "roles": roles,
        "counts": counts,
        "data_sha256": dict(data_sha256),
    }


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """The sha256 of the canonical JSON of the roles, so it names the chosen source ids only."""
    canonical = json.dumps(manifest["roles"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _ensure_collection(conn: AsyncConnection, tenant_id: UUID, name: str) -> UUID:
    cur = await conn.execute(
        "select id, kind from collections where tenant_id = %s and name = %s", (tenant_id, name)
    )
    row = await cur.fetchone()
    if row is None:
        cur = await conn.execute(
            "insert into collections (tenant_id, name, kind) "
            "values (%s, %s, 'benchmark') returning id",
            (tenant_id, name),
        )
        row = await cur.fetchone()
    elif row["kind"] != "benchmark":
        raise ValueError(f"collection {name!r} exists and is not a benchmark")
    return row["id"]


async def ingest_ragtruth(
    services: Services,
    sources: Mapping[str, RTSource],
    source_ids: Iterable[str],
    *,
    tenant_id: UUID,
) -> dict[str, UUID]:
    """Ingest each source as the one document of its own benchmark collection.

    The collection is ``rt-<source_id>`` and the document ``rt-<source_id>.txt``, built through
    ``parse_text`` and ``create_version`` like an upload. Retrieval over one source can then be
    scoped to one collection. Existing collections and documents are reused, so a rerun adds
    nothing. Returns each source id's collection id.
    """
    effective_from = datetime.now(UTC)
    collection_ids: dict[str, UUID] = {}
    for source_id in dict.fromkeys(source_ids):
        raw = sources[source_id].text.encode()
        parsed = parse_text(raw)
        if not parsed.text:
            raise ValueError(f"source {source_id} has no text")
        name = f"rt-{source_id}"
        filename = f"{name}.txt"
        async with services.db.tenant_tx(tenant_id) as conn:
            collection_id = await _ensure_collection(conn, tenant_id, name)
            cur = await conn.execute(
                "select 1 from documents "
                "where collection_id = %s and filename = %s and deleted_at is null limit 1",
                (collection_id, filename),
            )
            if await cur.fetchone() is None:
                document_id = await documents.insert_document(
                    conn,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    title=name,
                    filename=filename,
                    mime_type="text/plain",
                    created_by=None,
                )
                # A benchmark document has no uploaded original, so no blob row is stored.
                await create_version(
                    conn,
                    services.embedder,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    parsed=parsed,
                    blob_sha256=hashlib.sha256(raw).digest(),
                    effective_from=effective_from,
                    created_by=None,
                    settings=services.settings,
                )
        collection_ids[source_id] = collection_id
    return collection_ids


def parse_sizes(text: str) -> dict[str, int | None]:
    """Parse ``fit=N,select=N,risk=N,test=N`` where N is a count or ``all`` (None)."""
    sizes: dict[str, int | None] = {}
    for part in text.split(","):
        role, _, value = part.partition("=")
        if role not in ROLES or role in sizes or not re.fullmatch(r"[0-9]+|all", value):
            break
        sizes[role] = None if value == "all" else int(value)
    if set(sizes) != set(ROLES):
        raise ValueError("sizes must be fit=N,select=N,risk=N,test=N, each N a count or 'all'")
    return sizes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.ragtruth")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest_parser = commands.add_parser("manifest", help="write the source-level split manifest")
    manifest_parser.add_argument("--sizes", type=parse_sizes, required=True)
    manifest_parser.add_argument("--out", type=Path, required=True)
    manifest_parser.add_argument("--data-root", type=Path, default=data.DATA_DIR)
    args = parser.parse_args(argv)

    # fetch verifies the pinned sha256 (and downloads only when no verified copy exists).
    paths = {
        name: data.fetch(name, args.data_root) for name in ("ragtruth_response", "ragtruth_source")
    }
    sources, responses = load_ragtruth(args.data_root)
    manifest = build_manifest(
        sources,
        responses,
        sizes=args.sizes,
        data_sha256={path.name: data.REGISTRY[name].sha256 for name, path in paths.items()},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{args.out} manifest_hash={manifest_hash(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
