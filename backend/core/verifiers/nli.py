import asyncio
import numpy as np
import torch
from torch.nn.functional import softmax

from backend.core.verifiers.base import BaseVerifier, ClaimScore


class NLIVerifier(BaseVerifier):
    """Verify claims against source chunks using NLI (DeBERTa).

    Label order is read from ``model.config.id2label`` (for
    cross-encoder/nli-deberta-v3-large it is contradiction, entailment,
    neutral), never assumed.
    Score mapping: hallucination = contradiction*1.0 + neutral*0.5 + entailment*0.0

    All claim×chunk pairs are tokenized as a single batch and run through
    the model in one forward pass, then offloaded via run_in_executor so
    the async event loop is not blocked.
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        names = {str(name).lower(): int(idx) for idx, name in model.config.id2label.items()}
        required = ("contradiction", "entailment", "neutral")
        if not all(name in names for name in required):
            raise ValueError("NLI model labels must include contradiction, entailment, neutral")
        self.label_index: dict[str, int] = {name: names[name] for name in required}

    def _compute_hallucination_score(
        self, contradiction: float, neutral: float, entailment: float
    ) -> float:
        """Map NLI probabilities to hallucination score."""
        return float(contradiction + 0.5 * neutral)

    async def verify(
        self,
        claims: list[str],
        context_chunks: list[str],
        question: str = "",
    ) -> list[ClaimScore]:
        if not claims or not context_chunks:
            return []
        # Build flat list: claim i with chunk j -> index i*n_chunks + j
        n_chunks = len(context_chunks)
        pairs = [(chunk, claim) for claim in claims for chunk in context_chunks]

        def _batch_infer() -> np.ndarray:
            inputs = self.tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding=True,
                truncation="only_first",
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**inputs)
            return softmax(output.logits, dim=-1).cpu().numpy()  # shape: [N*K, 3]

        loop = asyncio.get_running_loop()
        all_probs = await loop.run_in_executor(None, _batch_infer)

        results = []
        for i, claim in enumerate(claims):
            claim_probs = all_probs[i * n_chunks:(i + 1) * n_chunks]  # [K, 3]
            c_i = self.label_index["contradiction"]
            n_i = self.label_index["neutral"]
            e_i = self.label_index["entailment"]
            scores = [
                self._compute_hallucination_score(
                    contradiction=float(p[c_i]),
                    neutral=float(p[n_i]),
                    entailment=float(p[e_i]),
                )
                for p in claim_probs
            ]
            best_idx = int(np.argmin(scores))
            best_score = scores[best_idx]
            best_probs = claim_probs[best_idx]

            results.append(ClaimScore(
                claim=claim,
                hallucination_score=best_score,
                details={
                    "verifier": "nli",
                    "contradiction": float(best_probs[c_i]),
                    "neutral": float(best_probs[n_i]),
                    "entailment": float(best_probs[e_i]),
                    "matched_chunk": context_chunks[best_idx],
                },
            ))
        return results
