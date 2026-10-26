"""Forward-only SQL migration runner with checksum verification."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_FILENAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
_LOCK_KEY = 727274

_CREATE_TABLE = """
create table if not exists schema_migrations (
    version int primary key,
    name text not null,
    checksum text not null,
    applied_at timestamptz not null default now()
)
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    checksum: str


class MigrationError(RuntimeError):
    pass


class MigrationChecksumError(MigrationError):
    pass


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    found: dict[int, Migration] = {}
    for path in directory.iterdir():
        match = _FILENAME.match(path.name)
        if match is None:
            continue
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"duplicate migration version {version:04d}")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        found[version] = Migration(version, match.group(2), path, checksum)
    return [found[v] for v in sorted(found)]


async def _applied(conn: psycopg.AsyncConnection) -> dict[int, str]:
    cur = await conn.execute("select version, checksum from schema_migrations")
    return {version: checksum for version, checksum in await cur.fetchall()}


def _verify(migrations: list[Migration], applied: dict[int, str]) -> None:
    for migration in migrations:
        recorded = applied.get(migration.version)
        if recorded is not None and recorded != migration.checksum:
            raise MigrationChecksumError(
                f"migration {migration.version:04d}_{migration.name} changed after being applied"
            )


async def migrate(url: str, directory: Path = MIGRATIONS_DIR) -> list[int]:
    migrations = discover(directory)
    newly_applied: list[int] = []
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute(_CREATE_TABLE)
        _verify(migrations, await _applied(conn))
        for migration in migrations:
            async with conn.transaction():
                await conn.execute("select pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
                # Re-read under the lock: another runner may have applied it meanwhile.
                applied = await _applied(conn)
                _verify([migration], applied)
                if migration.version in applied:
                    continue
                await conn.execute(migration.path.read_text(encoding="utf-8"))
                await conn.execute(
                    "insert into schema_migrations (version, name, checksum) values (%s, %s, %s)",
                    (migration.version, migration.name, migration.checksum),
                )
                newly_applied.append(migration.version)
    return newly_applied


async def pending(url: str, directory: Path = MIGRATIONS_DIR) -> list[int]:
    migrations = discover(directory)
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute(_CREATE_TABLE)
        applied = await _applied(conn)
    return [m.version for m in migrations if m.version not in applied]
