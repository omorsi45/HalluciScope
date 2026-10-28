
import os

import pytest

# Re-export the pg fixtures so every test module sees them; __all__ limits the names.
from tests.pg_fixtures import *  # noqa: F403


@pytest.fixture(autouse=True)
def _isolate_from_halluciscope_environment(monkeypatch):
    """Settings read HALLUCISCOPE_*; a developer's shell must not change test outcomes."""
    for name in list(os.environ):
        if name.startswith("HALLUCISCOPE_") and name != "HALLUCISCOPE_TEST_ADMIN_URL":
            monkeypatch.delenv(name)
