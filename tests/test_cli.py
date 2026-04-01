import pytest
from unittest.mock import AsyncMock, patch
from typer.testing import CliRunner
from backend.cli.main import app
from backend.core.pipeline import AnalysisResult
from backend.core.ensemble import ScoredClaim, ConfidenceTier


runner = CliRunner()


def test_check_command(tmp_path):
    doc_file = tmp_path / "test.txt"
    doc_file.write_text("Einstein was born in 1879 in Ulm, Germany.")

    mock_result = AnalysisResult(
        question="When was Einstein born?",
        answer="Einstein was born in 1879.",
        scored_claims=[
            ScoredClaim(
                claim="Einstein was born in 1879.",
                hallucination_score=0.1,
                tier=ConfidenceTier.SUPPORTED,
                verifier_details={
                    "nli": {"verifier": "nli", "entailment": 0.9, "neutral": 0.05, "contradiction": 0.05, "matched_chunk": ""},
                    "consistency": {"verifier": "consistency", "appearances": 5, "n_samples": 5},
                    "similarity": {"verifier": "similarity", "max_similarity": 0.92, "matched_chunk": ""},
                },
            )
        ],
        retrieved_chunks=["Einstein was born in 1879 in Ulm, Germany."],
        overall_score=0.1,
    )

    with patch("backend.cli.main._build_pipeline") as mock_build:
        mock_pipeline = AsyncMock()
        mock_pipeline.analyze = AsyncMock(return_value=mock_result)
        mock_build.return_value = mock_pipeline

        result = runner.invoke(app, [
            "check",
            "--doc", str(doc_file),
            "--question", "When was Einstein born?",
        ])

        assert result.exit_code == 0
        assert "Einstein was born in 1879" in result.stdout


def test_check_verbose(tmp_path):
    doc_file = tmp_path / "test.txt"
    doc_file.write_text("Einstein was born in 1879.")

    mock_result = AnalysisResult(
        question="Q?",
        answer="Answer.",
        scored_claims=[
            ScoredClaim(
                claim="A claim.",
                hallucination_score=0.3,
                tier=ConfidenceTier.UNCERTAIN,
                verifier_details={
                    "nli": {"verifier": "nli", "entailment": 0.6, "neutral": 0.3, "contradiction": 0.1, "matched_chunk": ""},
                    "consistency": {"verifier": "consistency", "appearances": 3, "n_samples": 5},
                    "similarity": {"verifier": "similarity", "max_similarity": 0.75, "matched_chunk": ""},
                },
            )
        ],
        retrieved_chunks=["chunk"],
        overall_score=0.3,
    )

    with patch("backend.cli.main._build_pipeline") as mock_build:
        mock_pipeline = AsyncMock()
        mock_pipeline.analyze = AsyncMock(return_value=mock_result)
        mock_build.return_value = mock_pipeline

        result = runner.invoke(app, [
            "check",
            "--doc", str(doc_file),
            "--question", "Q?",
            "--verbose",
        ])

        assert result.exit_code == 0
