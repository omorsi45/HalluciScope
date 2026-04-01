from fastapi import APIRouter, Request

from backend.api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest, request: Request):
    repo = request.app.state.repo
    feedback_id = await repo.save_feedback(
        analysis_id=req.analysis_id,
        claim_index=req.claim_index,
        is_correct=req.is_correct,
        note=req.note,
    )
    return FeedbackResponse(id=feedback_id)
