import pytest
from backend.core.verifiers.base import ClaimScore
from backend.core.ensemble import EnsembleScorer, ScoredClaim, ConfidenceTier


def _make_claim_scores(claim: str, nli: float, consistency: float, similarity: float):
    return (
        ClaimScore(claim=claim, hallucination_score=nli, details={"verifier": "nli"}),
        ClaimScore(claim=claim, hallucination_score=consistency, details={"verifier": "consistency"}),
        ClaimScore(claim=claim, hallucination_score=similarity, details={"verifier": "similarity"}),
    )


def test_ensemble_default_weights():
    scorer = EnsembleScorer()
    nli, cons, sim = _make_claim_scores("Test claim.", 0.0, 0.0, 0.0)
    results = scorer.score(nli_scores=[nli], consistency_scores=[cons], similarity_scores=[sim])
    assert len(results) == 1
    assert results[0].hallucination_score == 0.0
    assert results[0].tier == ConfidenceTier.SUPPORTED


def test_ensemble_fully_hallucinated():
    scorer = EnsembleScorer()
    nli, cons, sim = _make_claim_scores("Fake claim.", 1.0, 1.0, 1.0)
    results = scorer.score(nli_scores=[nli], consistency_scores=[cons], similarity_scores=[sim])
    assert results[0].hallucination_score == 1.0
    assert results[0].tier == ConfidenceTier.HALLUCINATED


def test_ensemble_weighted_calculation():
    scorer = EnsembleScorer(nli_weight=0.5, consistency_weight=0.3, similarity_weight=0.2)
    nli, cons, sim = _make_claim_scores("Claim.", 0.2, 0.4, 0.6)
    results = scorer.score(nli_scores=[nli], consistency_scores=[cons], similarity_scores=[sim])
    assert abs(results[0].hallucination_score - 0.34) < 1e-6
    assert results[0].tier == ConfidenceTier.UNCERTAIN


def test_ensemble_tier_boundaries():
    scorer = EnsembleScorer()
    # Exactly 0.2 -> UNCERTAIN
    nli, cons, sim = _make_claim_scores("Edge.", 0.2, 0.2, 0.2)
    results = scorer.score(nli_scores=[nli], consistency_scores=[cons], similarity_scores=[sim])
    assert results[0].tier == ConfidenceTier.UNCERTAIN

    # Exactly 0.6 -> UNCERTAIN
    nli, cons, sim = _make_claim_scores("Edge2.", 0.6, 0.6, 0.6)
    results = scorer.score(nli_scores=[nli], consistency_scores=[cons], similarity_scores=[sim])
    assert results[0].tier == ConfidenceTier.UNCERTAIN


def test_ensemble_multiple_claims():
    scorer = EnsembleScorer()
    claims_data = [
        _make_claim_scores("Good.", 0.0, 0.0, 0.1),
        _make_claim_scores("Bad.", 0.9, 0.8, 0.7),
    ]
    results = scorer.score(
        nli_scores=[c[0] for c in claims_data],
        consistency_scores=[c[1] for c in claims_data],
        similarity_scores=[c[2] for c in claims_data],
    )
    assert len(results) == 2
    assert results[0].tier == ConfidenceTier.SUPPORTED
    assert results[1].tier == ConfidenceTier.HALLUCINATED


def test_ensemble_preserves_details():
    scorer = EnsembleScorer()
    nli = ClaimScore(claim="C.", hallucination_score=0.1, details={"verifier": "nli", "entailment": 0.9})
    cons = ClaimScore(claim="C.", hallucination_score=0.2, details={"verifier": "consistency", "appearances": 4})
    sim = ClaimScore(claim="C.", hallucination_score=0.3, details={"verifier": "similarity", "max_similarity": 0.7})
    results = scorer.score(nli_scores=[nli], consistency_scores=[cons], similarity_scores=[sim])
    assert results[0].verifier_details["nli"]["entailment"] == 0.9
    assert results[0].verifier_details["consistency"]["appearances"] == 4
    assert results[0].verifier_details["similarity"]["max_similarity"] == 0.7
