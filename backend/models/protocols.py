"""Structural types for the embedding and reranking models.

Services depend on these protocols, so tests can swap in deterministic fakes and the real
adapters in `backend.models.loader` stay out of the import path of unit tests.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim) with L2-normalized rows.

        Empty input returns shape (0, dim).
        """
        ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Same contract as `embed_passages`, with any query-side prefix applied."""
        ...


class Reranker(Protocol):
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one raw logit per passage, in input order."""
        ...
