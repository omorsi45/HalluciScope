"""Async Postgres pool with tenant-scoped transactions."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


class Database:
    def __init__(self, url: str, *, min_size: int = 1, max_size: int = 10) -> None:
        self._pool = AsyncConnectionPool(
            url,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )

    async def open(self) -> None:
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self._pool.close()

    @asynccontextmanager
    async def _tx(self, tenant: str) -> AsyncIterator[AsyncConnection]:
        async with self._pool.connection() as conn, conn.transaction():
            # is_local=true: the setting dies with the transaction, so it cannot leak
            # to the next user of this pooled connection.
            await conn.execute("select set_config('app.tenant_id', %s, true)", (tenant,))
            yield conn

    def tenant_tx(self, tenant_id: UUID) -> AbstractAsyncContextManager[AsyncConnection]:
        if not isinstance(tenant_id, UUID):
            # None or a str would otherwise become "None" or arbitrary text in the setting.
            raise TypeError("tenant_id must be a UUID")
        return self._tx(str(tenant_id))

    def system_tx(self) -> AbstractAsyncContextManager[AsyncConnection]:
        return self._tx("")
