import uuid

import psycopg
import pytest

from backend.db.pool import Database
from tests.pg_fixtures import ADMIN_URL


@pytest.mark.parametrize("tenant_id", [None, str(uuid.uuid4()), uuid.uuid4().int])
async def test_tenant_tx_rejects_non_uuid_tenant(tenant_id):
    database = Database("postgresql://eg_app@localhost:1/none")

    with pytest.raises(TypeError):
        database.tenant_tx(tenant_id)


@pytest.mark.pg
async def test_connection_killed_between_transactions_is_replaced(pg_urls):
    database = Database(pg_urls["app"], min_size=1, max_size=1)
    await database.open()
    try:
        async with database.system_tx() as conn:
            cur = await conn.execute("select pg_backend_pid() as pid")
            old_pid = (await cur.fetchone())["pid"]
        async with await psycopg.AsyncConnection.connect(ADMIN_URL, autocommit=True) as admin:
            await admin.execute("select pg_terminate_backend(%s)", (old_pid,))

        async with database.system_tx() as conn:
            cur = await conn.execute("select pg_backend_pid() as pid")
            new_pid = (await cur.fetchone())["pid"]
    finally:
        await database.close()

    assert new_pid != old_pid
