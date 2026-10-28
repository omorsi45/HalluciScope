"""Pinned public benchmark datasets: streamed download, sha256 verification, safe extraction.

Every dataset in ``REGISTRY`` is pinned by sha256, so a benchmark can only run on the exact
bytes that were reviewed. Downloads land in ``evaluation/data/`` (gitignored).

    python -m evaluation.data fetch scifact nfcorpus
    python -m evaluation.data hash scifact   # prints the sha256, used once to pin a new file
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Literal
from urllib.parse import urlparse

import httpx

DATA_DIR = Path("evaluation/data")

_CHUNK_SIZE = 1024 * 1024
_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


@dataclass(frozen=True)
class DatasetFile:
    name: str
    url: str
    sha256: str
    kind: Literal["zip", "file"]


_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
# Pinned to a commit (2026-10-26), so a moved branch cannot change the bytes.
_RAGTRUTH_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
    "c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset"
)

REGISTRY: dict[str, DatasetFile] = {
    "scifact": DatasetFile(
        "scifact",
        f"{_BEIR_URL}/scifact.zip",
        "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "zip",
    ),
    "nfcorpus": DatasetFile(
        "nfcorpus",
        f"{_BEIR_URL}/nfcorpus.zip",
        "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "zip",
    ),
    "ragtruth_response": DatasetFile(
        "ragtruth_response",
        f"{_RAGTRUTH_URL}/response.jsonl",
        "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
        "file",
    ),
    "ragtruth_source": DatasetFile(
        "ragtruth_source",
        f"{_RAGTRUTH_URL}/source_info.jsonl",
        "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
        "file",
    ),
}


class DatasetIntegrityError(RuntimeError):
    """A download or archive is not what the registry pins."""


def _local_path(entry: DatasetFile, data_dir: Path) -> Path:
    """Where the downloaded file lives: the URL's basename inside the data directory."""
    return data_dir / Path(urlparse(entry.url).path).name


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _download(client: httpx.Client, entry: DatasetFile, data_dir: Path) -> tuple[Path, str]:
    """Stream the URL into a temporary file in ``data_dir``; return it with its sha256.

    The temporary file shares the directory (and so the filesystem) with the final path, which
    makes the later rename atomic. It is removed when the download fails.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=data_dir, prefix=f".{_local_path(entry, data_dir).name}.", suffix=".part"
    )
    temp = Path(temp_name)
    digest = hashlib.sha256()
    try:
        with (
            os.fdopen(handle, "wb") as out,
            client.stream("GET", entry.url, follow_redirects=True) as response,
        ):
            response.raise_for_status()
            for chunk in response.iter_bytes(_CHUNK_SIZE):
                digest.update(chunk)
                out.write(chunk)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp, digest.hexdigest()


def _check_member_names(entry: DatasetFile, archive: zipfile.ZipFile) -> None:
    """Reject absolute paths, drive letters, and any ``..`` component (zip-slip)."""
    for member in archive.namelist():
        parts = member.replace("\\", "/").split("/")
        if member.startswith(("/", "\\")) or PureWindowsPath(member).drive or ".." in parts:
            raise DatasetIntegrityError(f"unsafe path in archive for {entry.name}: {member!r}")


def _extract(entry: DatasetFile, archive_path: Path, target: Path) -> None:
    """Extract into a scratch directory next to ``target``, then rename it into place."""
    scratch = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _check_member_names(entry, archive)
            archive.extractall(scratch)
        if target.exists():
            shutil.rmtree(target)
        scratch.replace(target)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def fetch(name: str, data_dir: Path = DATA_DIR, *, client: httpx.Client | None = None) -> Path:
    """Download dataset ``name`` unless a verified copy exists; return its directory or file.

    A zip is extracted into ``data_dir/<name>/`` and that directory is returned. A plain file is
    returned as ``data_dir/<basename of the URL>``. A download whose sha256 differs from the
    registry is deleted and raises ``DatasetIntegrityError``.
    """
    entry = REGISTRY[name]
    path = _local_path(entry, data_dir)
    extracted = data_dir / name

    downloaded = False
    if not (path.is_file() and _file_sha256(path) == entry.sha256):
        owns_client = client is None
        http = client if client is not None else httpx.Client(timeout=_TIMEOUT)
        try:
            temp, actual = _download(http, entry, data_dir)
        finally:
            if owns_client:
                http.close()
        if actual != entry.sha256:
            temp.unlink()
            raise DatasetIntegrityError(
                f"sha256 mismatch for {name}: expected {entry.sha256}, got {actual}"
            )
        temp.replace(path)
        downloaded = True

    if entry.kind == "file":
        return path
    if downloaded or not extracted.is_dir():
        _extract(entry, path, extracted)
    return extracted


def remote_sha256(
    name: str, data_dir: Path = DATA_DIR, *, client: httpx.Client | None = None
) -> str:
    """Download dataset ``name`` without verification and return its sha256 (nothing is kept)."""
    entry = REGISTRY[name]
    owns_client = client is None
    http = client if client is not None else httpx.Client(timeout=_TIMEOUT)
    try:
        temp, digest = _download(http, entry, data_dir)
    finally:
        if owns_client:
            http.close()
    temp.unlink()
    return digest


def main(argv: Sequence[str] | None = None, *, client: httpx.Client | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.data")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="download and verify datasets")
    fetch_parser.add_argument("names", nargs="+", choices=sorted(REGISTRY))
    hash_parser = commands.add_parser("hash", help="print the sha256 of a remote dataset")
    hash_parser.add_argument("name", choices=sorted(REGISTRY))
    args = parser.parse_args(argv)

    if args.command == "hash":
        print(remote_sha256(args.name, DATA_DIR, client=client))
        return 0
    for name in args.names:
        try:
            print(f"{name}: {fetch(name, DATA_DIR, client=client)}")
        except DatasetIntegrityError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
