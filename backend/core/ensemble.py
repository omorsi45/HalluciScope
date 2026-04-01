from dataclasses import dataclass
from enum import Enum

from backend.core.verifiers.base import ClaimScore


class ConfidenceTier(str, Enum):
    SUPPORTED = "supported"       # < 0.2
    UNCERTAIN = "uncertain"       # 0.2 - 0.6
    HALLUCINATED = "hallucinated" # > 0.6


@dataclass
class ScoredClaim:
    """Final scored claim with ensemble result."""
    claim: str
    hallucination_score: float
    tier: ConfidenceTier
    verifier_details: dict[str, dict]


class EnsembleScorer:
    def __init__(
        self,
        nli_weight: float = 0.5,
        consistency_weight: float = 0.3,
        similarity_weight: float = 0.2,
    ):
        self.nli_weight = nli_weight
        self.consistency_weight = consistency_weight
        self.similarity_weight = similarity_weight

    def _get_tier(self, score: float) -> ConfidenceTier:
        if score < 0.2:
            return ConfidenceTier.SUPPORTED
        elif score > 0.6:
            return ConfidenceTier.HALLUCINATED
        return ConfidenceTier.UNCERTAIN

    def score(
        self,
        nli_scores: list[ClaimScore],
        consistency_scores: list[ClaimScore],
        similarity_scores: list[ClaimScore],
    ) -> list[ScoredClaim]:
        results = []
        for nli, cons, sim in zip(nli_scores, consistency_scores, similarity_scores):
            combined = (
                self.nli_weight * nli.hallucination_score
                + self.consistency_weight * cons.hallucination_score
                + self.similarity_weight * sim.hallucination_score
            )
            results.append(ScoredClaim(
                claim=nli.claim,
                hallucination_score=combined,
                tier=self._get_tier(combined),
                verifier_details={
                    "nli": nli.details,
                    "consistency": cons.details,
                    "similarity": sim.details,
                },
            ))
        return results
