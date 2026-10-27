"""Process-wide dependencies shared by the API app and the worker."""

from dataclasses import dataclass

from backend.config import Settings
from backend.db.pool import Database
from backend.models.protocols import Embedder, Reranker


@dataclass
class Services:
    settings: Settings
    db: Database
    embedder: Embedder
    reranker: Reranker | None = None
