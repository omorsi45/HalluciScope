"""Process-wide dependencies shared by the API app, the worker and the CLI."""

import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.calibration.artifact import Calibrator, CalibratorError, CalibratorMismatch
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


class CalibratorRejected(Exception):
    """The calibrator artifact exists but cannot be used.

    ``field`` names the offending field only: the artifact's own message can quote content.
    """

    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field


def load_calibrator(path: str) -> Calibrator | None:
    """The calibrator artifact at `path`, or None when there is no file.

    Without an artifact the process still runs; analyses then fail as `calibrator_missing`. An
    artifact it cannot read or validate raises `CalibratorRejected`: serving with unusable
    weights is worse.
    """
    if not Path(path).is_file():
        return None
    try:
        return Calibrator.load(path)
    except CalibratorError as exc:
        # The message starts with the offending field, then a description that may quote content.
        raise CalibratorRejected(str(exc).partition(":")[0]) from None
    except json.JSONDecodeError:
        raise CalibratorRejected("json") from None


async def build_services(settings: Settings, db: Database, http: httpx.AsyncClient) -> Services:
    """Load the models and the calibrator and wire the services of a worker or a CLI run.

    This loads real models (and downloads them on first use). The LLM client uses `http`, which
    the caller opens and closes. Raises `CalibratorRejected` for an artifact that is unusable or
    was trained with other models than the ones loaded here.
    """
    # Resolved at call time: the model loaders pull in torch, and tests replace them and the
    # Ollama client.
    from backend.llm.client import OllamaClient
    from backend.models.loader import load_embedder, load_nli, load_reranker

    # The cheap file check comes before the model loads.
    calibrator = load_calibrator(settings.calibrator_path)
    embedder = load_embedder(settings)
    reranker = load_reranker(settings)
    nli = load_nli(settings)
    if calibrator is not None:
        try:
            calibrator.check_parity(
                nli_model=nli.model_id,
                embedder_model=embedder.model_id,
                reranker_model=reranker.model_id,
            )
        except CalibratorMismatch as exc:
            raise CalibratorRejected(exc.field) from None
    llm = OllamaClient(
        http, settings.ollama_base_url, settings.llm_model, timeout_s=settings.llm_timeout_s
    )
    return Services(settings, db, embedder, reranker, llm=llm, nli=nli, calibrator=calibrator)
