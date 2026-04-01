import asyncio
from dataclasses import dataclass

from backend.config import Settings
from backend.core.chunker import Chunker
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
    ):
        self.settings = settings
        self.chunker = chunker
        self.nli_verifier = nli_verifier
        self.similarity_verifier = similarity_verifier
        self.consistency_verifier = consistency_verifier
        self.ensemble = ensemble

    async def analyze(
        self,
        document_text: str,
        question: str,
    ) -> AnalysisResult:
        # 1. Index document and retrieve relevant chunks
        self.chunker.index_document(document_text)
        chunks = self.chunker.retrieve(question)

        # 2. Generate answer
        answer = await generate_answer(
            question=question,
            context_chunks=chunks,
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
        )

        # 3. Decompose into claims
        claims = await decompose_claims(
            answer=answer,
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
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
            self.consistency_verifier.verify(claims, chunks),
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
