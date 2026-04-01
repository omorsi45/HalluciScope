import pytest
from backend.db.repository import Repository


@pytest.mark.asyncio
async def test_init_creates_tables(tmp_db_path):
    repo = Repository(str(tmp_db_path))
    await repo.init()
    await repo.close()


@pytest.mark.asyncio
async def test_save_and_get_analysis(tmp_db_path):
    repo = Repository(str(tmp_db_path))
    await repo.init()

    analysis_id = await repo.save_analysis(
        question="When was Einstein born?",
        document_text="Einstein was born in 1879.",
        answer="Einstein was born in 1879.",
        overall_score=0.1,
        claims=[
            {
                "claim": "Einstein was born in 1879.",
                "hallucination_score": 0.1,
                "tier": "supported",
                "verifier_details": {"nli": {"entailment": 0.9}},
            }
        ],
    )
    assert analysis_id == 1

    analysis = await repo.get_analysis(analysis_id)
    assert analysis["question"] == "When was Einstein born?"
    assert analysis["overall_score"] == 0.1
    assert len(analysis["claims"]) == 1

    await repo.close()


@pytest.mark.asyncio
async def test_list_analyses(tmp_db_path):
    repo = Repository(str(tmp_db_path))
    await repo.init()

    await repo.save_analysis("Q1?", "Doc1", "A1", 0.1, [])
    await repo.save_analysis("Q2?", "Doc2", "A2", 0.5, [])

    analyses = await repo.list_analyses(limit=10)
    assert len(analyses) == 2
    assert analyses[0]["question"] == "Q2?"

    await repo.close()


@pytest.mark.asyncio
async def test_save_and_list_feedback(tmp_db_path):
    repo = Repository(str(tmp_db_path))
    await repo.init()

    analysis_id = await repo.save_analysis("Q?", "Doc", "A", 0.5, [
        {"claim": "Claim.", "hallucination_score": 0.5, "tier": "uncertain", "verifier_details": {}},
    ])

    await repo.save_feedback(
        analysis_id=analysis_id,
        claim_index=0,
        is_correct=False,
        note="This claim is actually true.",
    )

    feedbacks = await repo.list_feedback(analysis_id)
    assert len(feedbacks) == 1
    assert feedbacks[0]["is_correct"] is False
    assert feedbacks[0]["note"] == "This claim is actually true."

    await repo.close()
