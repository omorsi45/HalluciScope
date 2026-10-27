import uuid

import pytest

from backend.db.pool import Database


@pytest.mark.parametrize("tenant_id", [None, str(uuid.uuid4()), uuid.uuid4().int])
async def test_tenant_tx_rejects_non_uuid_tenant(tenant_id):
    database = Database("postgresql://eg_app@localhost:1/none")

    with pytest.raises(TypeError):
        database.tenant_tx(tenant_id)
