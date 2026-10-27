import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.config import Settings
from tests.fakes import HashEmbedder, KeywordReranker

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def test_hash_embedder_normalized_and_deterministic():
    texts = ["refund policy days", "quantum physics"]
    vectors = HashEmbedder().embed_passages(texts)

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(vectors, HashEmbedder().embed_passages(texts))


def test_hash_embedder_same_vectors_across_processes():
    # Python's str hash changes with PYTHONHASHSEED; the fake must not depend on it.
    code = (
        "import json; from tests.fakes import HashEmbedder; "
        "print(json.dumps(HashEmbedder().embed_passages(['refund policy days'])[0].tolist()))"
    )

    def run(seed: str) -> list[float]:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return json.loads(result.stdout)

    assert run("1") == run("2")


def test_hash_embedder_similarity_tracks_overlap():
    embedder = HashEmbedder()
    query, related, unrelated = embedder.embed_queries(
        ["refund policy days", "refund policy", "quantum physics"]
    )

    assert _cosine(query, related) > _cosine(query, unrelated)


def test_hash_embedder_queries_and_passages_are_embedded_identically():
    texts = ["refund policy days", "quantum physics"]
    embedder = HashEmbedder()

    assert np.array_equal(embedder.embed_queries(texts), embedder.embed_passages(texts))


def test_hash_embedder_ignores_case_and_punctuation():
    embedder = HashEmbedder()
    plain, noisy = embedder.embed_passages(["refund policy", "Refund, POLICY!"])

    assert np.array_equal(plain, noisy)


def test_hash_embedder_empty_text_and_empty_input():
    embedder = HashEmbedder()
    first, second = embedder.embed_passages(["", "   "])

    assert np.array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0, atol=1e-6)
    assert embedder.embed_passages([]).shape == (0, 384)
    assert embedder.embed_queries([]).shape == (0, 384)


def test_hash_embedder_dim_is_configurable():
    vectors = HashEmbedder(dim=64).embed_passages(["refund policy"])

    assert vectors.shape == (1, 64)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-6)


def test_keyword_reranker_counts():
    passages = ["Refund policy applies to all orders", "nothing relevant here", "REFUND"]

    scores = KeywordReranker().score("refund Refund policy", passages)

    # Distinct lowercase query tokens are {refund, policy}; repeats in the query count once.
    assert scores == [2.0, 0.0, 1.0]
    assert all(isinstance(score, float) for score in scores)


@pytest.mark.parametrize("setting", ["cpu", "cuda", "mps"])
def test_resolve_device_explicit(setting):
    from backend.models.loader import resolve_device

    assert resolve_device(setting) == setting


def test_importing_loader_does_not_import_ml_libraries():
    # Unit tests and the API app import the loader; only calling a loader function may load torch.
    code = (
        "import json, sys; import backend.models.loader; "
        "print(json.dumps([m for m in ('torch', 'sentence_transformers', 'transformers') "
        "if m in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    assert json.loads(result.stdout) == []


def test_settings_model_defaults():
    settings = Settings()

    assert settings.embedder_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dim == 384
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert settings.model_device == "auto"
    assert settings.max_upload_bytes == 20_971_520
    assert settings.parse_timeout_s == 60.0
    assert settings.embed_batch_size == 32
    # The legacy pipeline keeps its own embedding model.
    assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.models
def test_real_embedder_dim_and_prefix():
    from backend.models.loader import BGE_QUERY_PREFIX, load_embedder

    settings = Settings(model_device="cpu")
    embedder = load_embedder(settings)
    text = "how long do refunds take"

    passage = embedder.embed_passages([text])
    query = embedder.embed_queries([text])
    prefixed_passage = embedder.embed_passages([BGE_QUERY_PREFIX + text])

    assert embedder.model_id == "BAAI/bge-small-en-v1.5"
    assert embedder.dim == settings.embedding_dim == 384
    assert passage.shape == query.shape == (1, 384)
    assert passage.dtype == np.float32
    assert np.isclose(np.linalg.norm(passage[0]), 1.0, atol=1e-5)
    # The prefix changes the query vector ...
    assert not np.allclose(passage, query, atol=1e-3)
    # ... and it is exactly the documented prefix, not some other string.
    assert np.allclose(query, prefixed_passage, atol=1e-4)
    assert embedder.embed_passages([]).shape == (0, 384)
    assert embedder.embed_queries([]).shape == (0, 384)


@pytest.mark.models
def test_real_reranker_orders():
    from backend.models.loader import load_reranker

    reranker = load_reranker(Settings(model_device="cpu"))
    query = "how long do refunds take"
    refund = "Refunds are processed within 5 to 7 business days after we receive the item."
    unrelated = "The Eiffel Tower is in Paris and was completed in 1889."

    scores = reranker.score(query, [refund, unrelated])
    reversed_scores = reranker.score(query, [unrelated, refund])

    assert reranker.model_id == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
    assert scores[0] > scores[1]
    # Scores come back in input order.
    assert np.allclose(reversed_scores, scores[::-1], atol=1e-4)
    assert reranker.score(query, []) == []
