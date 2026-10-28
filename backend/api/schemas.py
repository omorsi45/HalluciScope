"""Request and response shapes of the deprecated /api routes (see ``routes/legacy.py``)."""

from pydantic import BaseModel, Field

from backend.api.schemas_v1 import AskQuestion, DocumentText, FeedbackNote


class AnalyzeRequest(BaseModel):
    document_text: DocumentText
    question: AskQuestion


class ClaimResponse(BaseModel):
    claim: str
    hallucination_score: float
    tier: str
    verifier_details: dict


class AnalyzeResponse(BaseModel):
    id: str
    question: str
    answer: str
    scored_claims: list[ClaimResponse]
    retrieved_chunks: list[str]
    overall_score: float


class FeedbackRequest(BaseModel):
    analysis_id: str
    claim_index: int = Field(ge=0)
    is_correct: bool
    note: FeedbackNote | None = None


class FeedbackResponse(BaseModel):
    id: str


class AnalysisListItem(BaseModel):
    id: str
    question: str
    answer: str
    overall_score: float
    created_at: str | None = None
