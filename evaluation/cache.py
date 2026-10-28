"""A resumable cache of the evidence signals of verified units.

Extraction runs for hours, so every record is appended and flushed as soon as it is computed and
a crashed run restarts where it stopped. The cache of one configuration lives in a folder named by
``config_key``: ``signals.jsonl`` holds one record per unit, ``config.json`` the inputs the key
was made from. A record counts only once its line is complete (ends with a newline); the
unterminated tail a crash leaves behind is ignored on load and cut off before the next append.

One process writes a cache at a time.
"""

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.verify.types import EvidenceSignal, NLIResult

DEFAULT_ROOT = Path("evaluation/results/cache")


def config_dict(
    models: Mapping[str, str],
    *,
    premise_mode: str,
    depth: int,
    candidates: int,
    manifest_hash: str,
) -> dict[str, Any]:
    """The inputs a signal depends on: the models, the evidence settings, and the sources."""
    return {
        "models": dict(models),
        "premise_mode": premise_mode,
        "depth": depth,
        "candidates": candidates,
        "manifest_hash": manifest_hash,
    }


def config_key(
    models: Mapping[str, str],
    *,
    premise_mode: str,
    depth: int,
    candidates: int,
    manifest_hash: str,
) -> str:
    """The first 16 hex characters of the sha256 of the canonical JSON of the inputs."""
    config = config_dict(
        models,
        premise_mode=premise_mode,
        depth=depth,
        candidates=candidates,
        manifest_hash=manifest_hash,
    )
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8, LF) so that readers see the old file or the whole new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".tmp")
    scratch.write_text(text, encoding="utf-8", newline="\n")
    os.replace(scratch, path)


def _encode(unit_id: str, signals: Sequence[EvidenceSignal]) -> bytes:
    record = {
        "unit_id": unit_id,
        "signals": [
            {
                "passage_id": signal.passage_id,
                "text": signal.text,
                "nli": {
                    "entail": float(signal.nli.entail),
                    "contra": float(signal.nli.contra),
                    "neutral": float(signal.nli.neutral),
                },
                "rerank": None if signal.rerank is None else float(signal.rerank),
                "dense_cos": float(signal.dense_cos),
            }
            for signal in signals
        ],
    }
    return (json.dumps(record, separators=(",", ":")) + "\n").encode()


def _decode(record: Mapping[str, Any]) -> list[EvidenceSignal]:
    return [
        EvidenceSignal(
            row["passage_id"],
            row["text"],
            NLIResult(row["nli"]["entail"], row["nli"]["contra"], row["nli"]["neutral"]),
            row["rerank"],
            row["dense_cos"],
        )
        for row in record["signals"]
    ]


class SignalCache:
    """The signals of each unit (``unit_id`` to a list of ``EvidenceSignal``) of one config."""

    def __init__(self, root: Path, key: str, *, config: Mapping[str, Any] | None = None) -> None:
        """Open the cache ``root/key``; nothing is created until something is written.

        ``config`` is stored as ``config.json`` when the folder has none yet.
        """
        self.directory = root / key
        self._path = self.directory / "signals.jsonl"
        self._offsets: dict[str, int] = {}  # unit id -> byte offset of its line
        self._end = 0  # bytes up to the end of the last complete line
        self._load()
        config_path = self.directory / "config.json"
        if config is not None and not config_path.exists():
            write_atomic(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")

    def __contains__(self, unit_id: object) -> bool:
        return unit_id in self._offsets

    def get(self, unit_id: str) -> list[EvidenceSignal] | None:
        """The signals of a unit, or None when it is not cached."""
        offset = self._offsets.get(unit_id)
        if offset is None:
            return None
        with self._path.open("rb") as handle:
            handle.seek(offset)
            return _decode(json.loads(handle.readline()))

    def put_many(self, items: Iterable[tuple[str, Sequence[EvidenceSignal]]]) -> None:
        """Append the units not cached yet, then flush them to disk before returning."""
        batch = list(items)
        if not batch:
            return
        self._sync()
        lines: dict[str, bytes] = {}
        for unit_id, signals in batch:
            if unit_id not in self._offsets and unit_id not in lines:
                lines[unit_id] = _encode(unit_id, signals)
        if not lines:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab") as handle:
            handle.write(b"".join(lines.values()))
            handle.flush()
            os.fsync(handle.fileno())
        for unit_id, line in lines.items():
            self._offsets[unit_id] = self._end
            self._end += len(line)

    def load_all(self) -> dict[str, list[EvidenceSignal]]:
        """Every cached unit's signals. This reads the whole file."""
        return {unit_id: _decode(record) for unit_id, _, _, record in self._records()}

    def _records(self) -> Iterator[tuple[str, int, int, dict[str, Any]]]:
        """(unit id, byte offset, byte length, record) of each complete line.

        It stops before an unterminated line. A complete line that is not a record means the file
        is damaged: that raises, because skipping it would silently drop a unit.
        """
        if not self._path.exists():
            return
        offset = 0
        with self._path.open("rb") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.endswith(b"\n"):
                    return
                try:
                    record = json.loads(line)
                    unit_id = record["unit_id"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    raise ValueError(f"{self._path.name} line {number} is not a record") from None
                yield unit_id, offset, len(line), record
                offset += len(line)

    def _load(self) -> None:
        self._offsets = {}
        self._end = 0
        for unit_id, offset, length, _ in self._records():
            self._offsets[unit_id] = offset
            self._end = offset + length

    def _sync(self) -> None:
        """Pick up what changed on disk since the last load, and cut off an unterminated tail."""
        size = self._path.stat().st_size if self._path.exists() else 0
        if size != self._end:
            self._load()
        if size > self._end:
            os.truncate(self._path, self._end)
