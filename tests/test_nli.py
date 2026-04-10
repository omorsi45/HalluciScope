import pytest
import numpy as np
from unittest.mock import MagicMock
from backend.core.verifiers.nli import NLIVerifier


@pytest.mark.asyncio
async def test_nli_score_mapping():
    """Test that NLI scores map correctly to hallucination probability."""
    verifier = NLIVerifier.__new__(NLIVerifier)

    # Scores: [contradiction, neutral, entailment] (DeBERTa order)
    # Full entailment -> hallucination_score = 0.0
    score = verifier._compute_hallucination_score(np.array([0.0, 0.0, 1.0]))
    assert abs(score - 0.0) < 1e-6

    # Full contradiction -> hallucination_score = 1.0
    score = verifier._compute_hallucination_score(np.array([1.0, 0.0, 0.0]))
    assert abs(score - 1.0) < 1e-6

    # Full neutral -> hallucination_score = 0.5
    score = verifier._compute_hallucination_score(np.array([0.0, 1.0, 0.0]))
    assert abs(score - 0.5) < 1e-6

    # Mixed: 0.6 entailment, 0.3 neutral, 0.1 contradiction
    # hallucination = 0.1*1.0 + 0.3*0.5 + 0.6*0.0 = 0.25
    score = verifier._compute_hallucination_score(np.array([0.1, 0.3, 0.6]))
    assert abs(score - 0.25) < 1e-6


@pytest.mark.asyncio
async def test_nli_verify_returns_correct_count():
    """Verifier calls tokenizer once with all pairs and returns one score per claim."""
    import torch

    tokenizer = MagicMock()
    model = MagicMock()

    def mock_tokenize(premises, hypotheses, **kwargs):
        n = len(premises)
        return {
            "input_ids": torch.zeros(n, 10, dtype=torch.long),
            "attention_mask": torch.ones(n, 10, dtype=torch.long),
        }
    tokenizer.side_effect = mock_tokenize

    def mock_forward(**kwargs):
        n = kwargs["input_ids"].shape[0]
        out = MagicMock()
        logits = torch.zeros(n, 3)
        logits[:, 2] = 5.0  # strong entailment for all pairs
        out.logits = logits
        return out
    model.side_effect = mock_forward

    verifier = NLIVerifier(tokenizer=tokenizer, model=model)
    results = await verifier.verify(
        claims=["Claim A.", "Claim B."],
        context_chunks=["Some context."],
    )

    assert len(results) == 2
    for r in results:
        assert r.hallucination_score < 0.1

    # Tokenizer called exactly once with both pairs
    assert tokenizer.call_count == 1
    # Model called exactly once
    assert model.call_count == 1
