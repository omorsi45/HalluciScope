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
    """Test that verify returns one score per claim."""
    import torch

    tokenizer = MagicMock()
    model = MagicMock()

    tokenizer.side_effect = lambda *args, **kwargs: {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }

    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[0.0, 0.0, 5.0]])  # Strong entailment
    model.return_value = mock_output

    verifier = NLIVerifier(tokenizer=tokenizer, model=model)
    claims = ["Claim A.", "Claim B."]
    chunks = ["Some context."]

    results = await verifier.verify(claims, chunks)
    assert len(results) == 2
    for r in results:
        assert r.hallucination_score < 0.1
