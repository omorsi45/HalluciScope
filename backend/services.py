"""Process-wide dependencies shared by the API app and the worker."""

from dataclasses import dataclass

from backend.calibration.artifact import Calibrator
from backend.config import Settings
from backend.db.pool import Database
from backend.llm.client import LLMClient
from backend.models.protocols import Embedder, Reranker
from backend.retrieval.service import PostgresRetriever
from backend.verify.types import NLIScorer


@dataclass
class Services:
    settings: Settings
    db: Database
    embedder: Embedder
    reranker: Reranker | None = None
    # The worker alone holds these; the API process leaves them unset.
    llm: LLMClient | None = None
    nli: NLIScorer | None = None
    calibrator: Calibrator | None = None

    def retriever(self) -> PostgresRetriever:
        """A retriever over this process's database, embedder and reranker."""
        return PostgresRetriever(self.db, self.embedder, self.reranker)
