import numpy as np

from backend.core.verifiers.base import BaseVerifier, ClaimScore


class SimilarityVerifier(BaseVerifier):
    """Verify claims by semantic similarity against source chunks.

    hallucination_score = 1 - max_cosine_similarity
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    async def verify(
        self,
        claims: list[str],
        context_chunks: list[str],
        question: str = "",
    ) -> list[ClaimScore]:
        chunk_embeddings = self.embedding_model.encode(
            context_chunks, normalize_embeddings=True
        )
        claim_embeddings = self.embedding_model.encode(
            claims, normalize_embeddings=True
        )

        chunk_embeds = np.array(chunk_embeddings, dtype=np.float32)
        claim_embeds = np.array(claim_embeddings, dtype=np.float32)

        # Cosine similarity matrix: (n_claims, n_chunks)
        sim_matrix = claim_embeds @ chunk_embeds.T

        results = []
        for i, claim in enumerate(claims):
            max_idx = int(np.argmax(sim_matrix[i]))
            max_sim = float(sim_matrix[i, max_idx])
            max_sim = max(0.0, min(1.0, max_sim))
            hallucination_score = 1.0 - max_sim

            results.append(ClaimScore(
                claim=claim,
                hallucination_score=hallucination_score,
                details={
                    "verifier": "similarity",
                    "max_similarity": max_sim,
                    "matched_chunk": context_chunks[max_idx],
                },
            ))
        return results
