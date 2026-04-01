from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ClaimScore:
    """Score for a single claim from a single verifier."""
    claim: str
    hallucination_score: float  # 0.0 = supported, 1.0 = hallucinated
    details: dict  # Verifier-specific details for explainability


class BaseVerifier(ABC):
    """Interface for all verifiers."""

    @abstractmethod
    async def verify(
        self,
        claims: list[str],
        context_chunks: list[str],
    ) -> list[ClaimScore]:
        """Score each claim. Returns one ClaimScore per claim, in order."""
        ...
