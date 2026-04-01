from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    document_text: str
    question: str


class ClaimResponse(BaseModel):
    claim: str
    hallucination_score: float
    tier: str
    verifier_details: dict


class AnalyzeResponse(BaseModel):
    id: int | None = None
    question: str
    answer: str
    scored_claims: list[ClaimResponse]
    retrieved_chunks: list[str]
    overall_score: float


class FeedbackRequest(BaseModel):
    analysis_id: int
    claim_index: int
    is_correct: bool
    note: str | None = None


class FeedbackResponse(BaseModel):
    id: int


class AnalysisListItem(BaseModel):
    id: int
    question: str
    answer: str
    overall_score: float
    created_at: str | None = None
