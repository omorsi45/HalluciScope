import asyncio
import numpy as np

from backend.core.generator import generate_answer
from backend.core.decomposer import decompose_claims
from backend.core.verifiers.base import BaseVerifier, ClaimScore


class ConsistencyVerifier(BaseVerifier):
    """Verify claims by self-consistency sampling.

    Re-asks the question N times, decomposes each answer into claims,
    checks if each original claim appears across samples.

    "Appears" = cosine similarity > threshold between claim embeddings.
    hallucination_score = 1 - (appearances / N)
    """

    def __init__(
        self,
        embedding_model,
        base_url: str,
        model: str,
        n_samples: int = 5,
        temperature: float = 0.7,
        similarity_threshold: float = 0.85,
    ):
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold

    async def _sample_once(self, context_chunks: list[str], question: str) -> list[str]:
        """Generate one sample answer and decompose it into claims."""
        answer = await generate_answer(
            question=question,
            context_chunks=context_chunks,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
        )
        return await decompose_claims(
            answer=answer,
            base_url=self.base_url,
            model=self.model,
        )

    def _claim_appears_in_sample(
        self,
        claim_embedding: np.ndarray,
        sample_embeddings: np.ndarray,
    ) -> bool:
        """Check if a claim appears in a sample based on cosine similarity."""
        if len(sample_embeddings) == 0:
            return False
        similarities = claim_embedding @ sample_embeddings.T
        return bool(np.max(similarities) >= self.similarity_threshold)

    async def verify(
        self,
        claims: list[str],
        context_chunks: list[str],
    ) -> list[ClaimScore]:
        question_context = "\n".join(context_chunks)

        # Sample N answers concurrently
        tasks = [
            self._sample_once(context_chunks, question_context)
            for _ in range(self.n_samples)
        ]
        sample_claim_lists = await asyncio.gather(*tasks)

        # Embed original claims
        claim_embeddings = self.embedding_model.encode(
            claims, normalize_embeddings=True
        )
        claim_embeds = np.array(claim_embeddings, dtype=np.float32)

        # Embed each sample's claims
        sample_embeddings_list = []
        for sample_claims in sample_claim_lists:
            if sample_claims:
                embeds = self.embedding_model.encode(
                    sample_claims, normalize_embeddings=True
                )
                sample_embeddings_list.append(np.array(embeds, dtype=np.float32))
            else:
                sample_embeddings_list.append(np.array([]).reshape(0, claim_embeds.shape[1]))

        # Count appearances for each claim
        results = []
        for i, claim in enumerate(claims):
            appearances = 0
            for sample_embeds in sample_embeddings_list:
                if self._claim_appears_in_sample(claim_embeds[i], sample_embeds):
                    appearances += 1

            hallucination_score = 1.0 - (appearances / self.n_samples)

            results.append(ClaimScore(
                claim=claim,
                hallucination_score=hallucination_score,
                details={
                    "verifier": "consistency",
                    "appearances": appearances,
                    "n_samples": self.n_samples,
                },
            ))
        return results
