import pytest
from pathlib import Path


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
