import pytest
from unittest.mock import MagicMock
from backend.core.verifiers.nli import NLIVerifier

# Label order of cross-encoder/nli-deberta-v3-large
DEBERTA_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def _fake_nli(strong_index: int, id2label=None):
    """Build a fake tokenizer and model whose logits put 6.0 at strong_index."""
    import torch

    tokenizer = MagicMock()
    model = MagicMock()
    model.config.id2label = DEBERTA_ID2LABEL if id2label is None else id2label

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
        logits[:, strong_index] = 6.0
        out.logits = logits
        return out
    model.side_effect = mock_forward
    return tokenizer, model


@pytest.mark.asyncio
async def test_nli_score_mapping():
    """Test that NLI scores map correctly to hallucination probability."""
    verifier = NLIVerifier.__new__(NLIVerifier)

    # Full entailment -> hallucination_score = 0.0
    score = verifier._compute_hallucination_score(
        contradiction=0.0, neutral=0.0, entailment=1.0
    )
    assert abs(score - 0.0) < 1e-6

    # Full contradiction -> hallucination_score = 1.0
    score = verifier._compute_hallucination_score(
        contradiction=1.0, neutral=0.0, entailment=0.0
    )
    assert abs(score - 1.0) < 1e-6

    # Full neutral -> hallucination_score = 0.5
    score = verifier._compute_hallucination_score(
        contradiction=0.0, neutral=1.0, entailment=0.0
    )
    assert abs(score - 0.5) < 1e-6

    # Mixed: 0.6 entailment, 0.3 neutral, 0.1 contradiction
    # hallucination = 0.1*1.0 + 0.3*0.5 + 0.6*0.0 = 0.25
    score = verifier._compute_hallucination_score(
        contradiction=0.1, neutral=0.3, entailment=0.6
    )
    assert abs(score - 0.25) < 1e-6


@pytest.mark.asyncio
async def test_nli_verify_returns_correct_count():
    """Verifier calls tokenizer once with all pairs and returns one score per claim."""
    tokenizer, model = _fake_nli(strong_index=1)  # strong entailment for all pairs

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


@pytest.mark.asyncio
async def test_entailment_index_from_config():
    tokenizer, model = _fake_nli(strong_index=1)
    verifier = NLIVerifier(tokenizer=tokenizer, model=model)

    results = await verifier.verify(claims=["Claim."], context_chunks=["Context."])

    assert results[0].hallucination_score < 0.05
    assert results[0].details["entailment"] > 0.95


@pytest.mark.asyncio
async def test_neutral_scores_half():
    tokenizer, model = _fake_nli(strong_index=2)
    verifier = NLIVerifier(tokenizer=tokenizer, model=model)

    results = await verifier.verify(claims=["Claim."], context_chunks=["Context."])

    assert abs(results[0].hallucination_score - 0.5) < 0.05
    assert results[0].details["neutral"] > 0.95


def test_missing_labels_raise():
    tokenizer, model = _fake_nli(
        strong_index=0, id2label={0: "a", 1: "b", 2: "c"}
    )

    with pytest.raises(ValueError):
        NLIVerifier(tokenizer=tokenizer, model=model)


@pytest.mark.asyncio
async def test_truncates_premise_only():
    tokenizer, model = _fake_nli(strong_index=1)
    verifier = NLIVerifier(tokenizer=tokenizer, model=model)

    await verifier.verify(claims=["Claim."], context_chunks=["Context."])

    assert tokenizer.call_count == 1
    assert tokenizer.call_args.kwargs["truncation"] == "only_first"
