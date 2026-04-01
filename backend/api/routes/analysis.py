from fastapi import APIRouter, Request

from backend.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisListItem,
    ClaimResponse,
)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, request: Request):
    pipeline = request.app.state.pipeline
    repo = request.app.state.repo

    result = await pipeline.analyze(
        document_text=req.document_text,
        question=req.question,
    )

    claims_data = [
        {
            "claim": sc.claim,
            "hallucination_score": sc.hallucination_score,
            "tier": sc.tier.value,
            "verifier_details": sc.verifier_details,
        }
        for sc in result.scored_claims
    ]

    analysis_id = await repo.save_analysis(
        question=result.question,
        document_text=req.document_text,
        answer=result.answer,
        overall_score=result.overall_score,
        claims=claims_data,
    )

    return AnalyzeResponse(
        id=analysis_id,
        question=result.question,
        answer=result.answer,
        scored_claims=[ClaimResponse(**c) for c in claims_data],
        retrieved_chunks=result.retrieved_chunks,
        overall_score=result.overall_score,
    )


@router.get("/analyses", response_model=list[AnalysisListItem])
async def list_analyses(request: Request, limit: int = 50):
    repo = request.app.state.repo
    analyses = await repo.list_analyses(limit=limit)
    return [AnalysisListItem(**a) for a in analyses]
