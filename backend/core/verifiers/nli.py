import numpy as np
import torch
from torch.nn.functional import softmax

from backend.core.verifiers.base import BaseVerifier, ClaimScore


class NLIVerifier(BaseVerifier):
    """Verify claims against source chunks using NLI (DeBERTa).

    DeBERTa label order: [contradiction, neutral, entailment]
    Score mapping: hallucination = contradiction*1.0 + neutral*0.5 + entailment*0.0
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def _compute_hallucination_score(self, probs: np.ndarray) -> float:
        """Map NLI probabilities to hallucination score.

        Args:
            probs: Array of [contradiction, neutral, entailment] probabilities.
        """
        weights = np.array([1.0, 0.5, 0.0])
        return float(np.dot(probs, weights))

    async def verify(
        self,
        claims: list[str],
        context_chunks: list[str],
    ) -> list[ClaimScore]:
        results = []
        for claim in claims:
            best_score = 1.0
            best_probs = np.array([1.0, 0.0, 0.0])
            best_chunk = ""

            for chunk in context_chunks:
                inputs = self.tokenizer(
                    chunk, claim, return_tensors="pt", truncation=True, max_length=512
                )
                with torch.no_grad():
                    output = self.model(**inputs)
                probs = softmax(output.logits, dim=-1).numpy()[0]
                score = self._compute_hallucination_score(probs)

                if score < best_score:
                    best_score = score
                    best_probs = probs
                    best_chunk = chunk

            results.append(ClaimScore(
                claim=claim,
                hallucination_score=best_score,
                details={
                    "verifier": "nli",
                    "contradiction": float(best_probs[0]),
                    "neutral": float(best_probs[1]),
                    "entailment": float(best_probs[2]),
                    "matched_chunk": best_chunk,
                },
            ))
        return results
