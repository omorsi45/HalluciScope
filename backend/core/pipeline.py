import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import httpx

from backend.config import Settings
from backend.core.chunker import Chunker, DocumentIndex
from backend.core.generator import generate_answer
from backend.core.decomposer import decompose_claims
from backend.core.verifiers.nli import NLIVerifier
from backend.core.verifiers.similarity import SimilarityVerifier
from backend.core.verifiers.consistency import ConsistencyVerifier
from backend.core.ensemble import EnsembleScorer, ScoredClaim


@dataclass
class AnalysisResult:
    question: str
    answer: str
    scored_claims: list[ScoredClaim]
    retrieved_chunks: list[str]
    overall_score: float


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        chunker: Chunker,
        nli_verifier: NLIVerifier,
        similarity_verifier: SimilarityVerifier,
        consistency_verifier: ConsistencyVerifier,
        ensemble: EnsembleScorer,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.chunker = chunker
        self.nli_verifier = nli_verifier
        self.similarity_verifier = similarity_verifier
        self.consistency_verifier = consistency_verifier
        self.ensemble = ensemble
        self.http_client = http_client
        self._index_cache: OrderedDict[str, DocumentIndex] = OrderedDict()
        self._cache_maxsize: int = settings.index_cache_maxsize

    def _get_doc_index(self, document_text: str) -> DocumentIndex:
        """Return a cached DocumentIndex or build and cache a new one (LRU)."""
        doc_hash = hashlib.sha256(document_text.encode()).hexdigest()
        if doc_hash in self._index_cache:
            self._index_cache.move_to_end(doc_hash)
            return self._index_cache[doc_hash]
        doc_index = self.chunker.build_index(document_text)
        self._index_cache[doc_hash] = doc_index
        if len(self._index_cache) > self._cache_maxsize:
            self._index_cache.popitem(last=False)
        return doc_index

    async def analyze(
        self,
        document_text: str,
        question: str,
    ) -> AnalysisResult:
        # 1. Get (possibly cached) document index and retrieve relevant chunks
        doc_index = self._get_doc_index(document_text)
        chunks = self.chunker.retrieve(question, doc_index)

        # 2. Generate answer
        answer = await generate_answer(
            question=question,
            context_chunks=chunks,
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            client=self.http_client,
        )

        # 3. Decompose into claims
        claims = await decompose_claims(
            answer=answer,
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            client=self.http_client,
        )

        if not claims:
            return AnalysisResult(
                question=question,
                answer=answer,
                scored_claims=[],
                retrieved_chunks=chunks,
                overall_score=0.0,
            )

        # 4. Run verifiers in parallel
        nli_scores, similarity_scores, consistency_scores = await asyncio.gather(
            self.nli_verifier.verify(claims, chunks),
            self.similarity_verifier.verify(claims, chunks),
            self.consistency_verifier.verify(claims, chunks, question=question),
        )

        # 5. Ensemble scoring
        scored_claims = self.ensemble.score(
            nli_scores=nli_scores,
            consistency_scores=consistency_scores,
            similarity_scores=similarity_scores,
        )

        # 6. Overall score = mean hallucination probability
        overall = sum(c.hallucination_score for c in scored_claims) / len(scored_claims)

        return AnalysisResult(
            question=question,
            answer=answer,
            scored_claims=scored_claims,
            retrieved_chunks=chunks,
            overall_score=overall,
        )
