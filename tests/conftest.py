
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


@pytest.fixture
def sample_document_text():
    return (
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany. "
        "He developed the theory of special relativity in 1905. "
        "Einstein received the Nobel Prize in Physics in 1921 for his "
        "explanation of the photoelectric effect. He moved to the United "
        "States in 1933 and worked at the Institute for Advanced Study "
        "in Princeton, New Jersey, until his death in 1955."
    )


@pytest.fixture
def sample_claims():
    return [
        "Albert Einstein was born on March 14, 1879.",
        "Einstein was born in Ulm, Germany.",
        "He developed the theory of special relativity in 1905.",
        "Einstein received the Nobel Prize in Physics in 1921.",
        "The Nobel Prize was for his explanation of the photoelectric effect.",
        "He moved to the United States in 1933.",
    ]


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test.db"
