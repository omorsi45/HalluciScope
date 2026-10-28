"""Request and response models for /api/v1."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from backend.retrieval.types import SearchMode

_SURROGATES = range(0xD800, 0xE000)


def _reject_surrogates(value: str) -> str:
    """Lone surrogates cannot be encoded as UTF-8 and would crash hashing or storage."""
    if any(ord(char) in _SURROGATES for char in value):
        raise ValueError("must not contain lone surrogate characters")
    return value


def _reject_nul(value: str) -> str:
    """Postgres text cannot hold NUL, so it would crash the insert instead of failing validation."""
    if "\x00" in value:
        raise ValueError("must not contain NUL characters")
    return value


# Every request string ends with the surrogate check so a JSON "\ud800" is a 422, not a 500.
# The constraints come first: stacking them on an already validated str skips the strip.
_NO_SURROGATES = AfterValidator(_reject_surrogates)
_NO_NUL = AfterValidator(_reject_nul)
Email = Annotated[str, StringConstraints(max_length=320), _NO_SURROGATES]
Password = Annotated[str, StringConstraints(max_length=256), _NO_SURROGATES]
DisplayName = Annotated[str, StringConstraints(max_length=100), _NO_SURROGATES]
Name = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    _NO_SURROGATES,
]
Description = Annotated[str, StringConstraints(max_length=1000), _NO_SURROGATES]
Title = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
    _NO_SURROGATES,
    _NO_NUL,
]
SearchQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
    _NO_SURROGATES,
    _NO_NUL,
]
AskQuestion = SearchQuery
# A verify question is optional context: it may be empty, and the route stores that as none.
VerifyQuestion = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=2000),
    _NO_SURROGATES,
    _NO_NUL,
]
# A feedback note may be empty; NUL is rejected like in the other free text a row stores.
FeedbackNote = Annotated[
    str,
    StringConstraints(max_length=1000),
    _NO_SURROGATES,
    _NO_NUL,
]
# The answer is stored exactly as sent: claim offsets point into it.
Answer = Annotated[
    str,
    StringConstraints(min_length=1, max_length=20000),
    _NO_SURROGATES,
    _NO_NUL,
]


class RegisterIn(BaseModel):
    email: Email
    password: Password
    display_name: DisplayName


class LoginIn(BaseModel):
    email: Email
    password: Password


class UserOut(BaseModel):
    id: UUID
    email: str
    display_name: str


class TenantOut(BaseModel):
    id: UUID
    name: str
    role: str


class SessionOut(BaseModel):
    user: UserOut
    tenant: TenantOut
    csrf_token: str


class MeOut(BaseModel):
    user: UserOut
    tenant: TenantOut
    tenants: list[TenantOut]


class SwitchTenantIn(BaseModel):
    tenant_id: UUID


class ApiKeyCreateIn(BaseModel):
    name: Name


class ApiKeyOut(BaseModel):
    id: UUID
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None


class ApiKeyCreatedOut(ApiKeyOut):
    token: str


class CollectionIn(BaseModel):
    name: Name
    description: Description = ""


class CollectionPatch(BaseModel):
    name: Name | None = None
    description: Description | None = None


class CollectionOut(BaseModel):
    id: UUID
    name: str
    description: str
    kind: str
    created_at: datetime
    permission: str


class GrantIn(BaseModel):
    user_id: UUID
    permission: Literal["read", "write", "admin"]


class GrantsIn(BaseModel):
    grants: list[GrantIn] = Field(max_length=500)


class GrantOut(BaseModel):
    user_id: UUID
    email: str
    display_name: str
    permission: str


class MemberOut(BaseModel):
    user_id: UUID
    email: str
    display_name: str
    role: str
    created_at: datetime


class MemberAddIn(BaseModel):
    email: Email
    role: Literal["owner", "admin", "member"]


class AuditOut(BaseModel):
    id: int
    action: str
    user_id: UUID | None
    target_type: str | None
    target_id: str | None
    at: datetime
    details: dict | None


class UploadAcceptedOut(BaseModel):
    document_id: UUID
    upload_id: UUID
    status: str


class UploadOut(BaseModel):
    id: UUID
    document_id: UUID
    status: str
    version_id: UUID | None
    version_number: int | None
    error_code: str | None
    stats: dict
    created_at: datetime
    updated_at: datetime


class DocumentOut(BaseModel):
    id: UUID
    collection_id: UUID
    title: str
    filename: str
    mime_type: str
    current_version_number: int | None
    version_count: int
    created_at: datetime


class VersionSummaryOut(BaseModel):
    id: UUID
    version_number: int
    effective_from: datetime
    created_at: datetime
    char_count: int
    n_passages: int
    parser: str
    is_current: bool


class DocumentDetailOut(DocumentOut):
    versions: list[VersionSummaryOut]


class VersionOut(VersionSummaryOut):
    page_map: list[list[int]]
    sections: list
    metadata: dict
    parser_version: str


class VersionTextOut(BaseModel):
    start: int
    end: int
    char_count: int
    text: str


class PassageBoundaryOut(BaseModel):
    id: UUID
    ordinal: int
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None
    section: str | None


class PassageOut(BaseModel):
    """A passage as stored in its own version, with its surroundings and its standing today.

    Mirrors ``backend.provenance.resolve.ResolvedPassage``. The context fields are up to 300
    characters of that version's text on each side.
    """

    passage_id: UUID
    version_id: UUID
    version_number: int
    effective_from: datetime
    document_id: UUID
    document_title: str
    text: str
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None
    context_before: str
    context_after: str
    is_current_version: bool
    current_equivalent_passage_id: UUID | None  # same text in the current version, if any
    superseded_by_version: int | None  # the current version number when it is not this one
    collection_id: UUID
    ordinal: int
    section: str | None


class DiffHunkOut(BaseModel):
    op: Literal["equal", "insert", "delete", "replace"]
    from_start: int
    from_end: int
    to_start: int
    to_end: int
    from_text: str
    to_text: str


class DiffOut(BaseModel):
    from_version: int
    to_version: int
    hunks: list[DiffHunkOut]
    stats: dict[str, int]  # hunk counts per op


class SearchIn(BaseModel):
    query: SearchQuery
    collection_ids: list[UUID] | None = None
    as_of: AwareDatetime | None = None
    k: int = Field(default=10, ge=1, le=50)
    mode: SearchMode = "hybrid"
    rerank: bool = False


class PassageHitOut(BaseModel):
    """One ranked passage; mirrors ``backend.retrieval.types.PassageHit``."""

    model_config = ConfigDict(from_attributes=True)

    passage_id: UUID
    document_id: UUID
    version_id: UUID
    version_number: int
    effective_from: datetime
    document_title: str
    text: str
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None
    section: str | None
    scores: dict[str, float]  # keys among 'bm25', 'dense', 'fts', 'rrf', 'rerank'
    rank: int


class SearchOut(BaseModel):
    hits: list[PassageHitOut]
    mode: str
    reranked: bool
    took_ms: int


class AnalysisOptionsIn(BaseModel):
    extraction: Literal["llm_atomic", "sentence"] = "llm_atomic"
    consistency_samples: int = Field(default=0, ge=0, le=5)


class AskIn(BaseModel):
    question: AskQuestion
    collection_ids: list[UUID] | None = None
    as_of: AwareDatetime | None = None
    options: AnalysisOptionsIn = Field(default_factory=AnalysisOptionsIn)


class VerifyIn(BaseModel):
    answer: Answer
    question: VerifyQuestion | None = None
    collection_ids: list[UUID] | None = None
    as_of: AwareDatetime | None = None
    options: AnalysisOptionsIn = Field(default_factory=AnalysisOptionsIn)

    @field_validator("options")
    @classmethod
    def _no_self_consistency(cls, options: AnalysisOptionsIn) -> AnalysisOptionsIn:
        # Self-consistency resamples the answer generation, which a verify request does not have.
        if options.consistency_samples != 0:
            raise ValueError("consistency_samples must be 0 when verifying a given answer")
        return options


class AnalysisAcceptedOut(BaseModel):
    analysis_id: UUID
    status: str


class ReverifyAcceptedOut(BaseModel):
    status: Literal["queued"]


class AnalysisSummaryOut(BaseModel):
    id: UUID
    kind: str
    question_preview: str | None  # the first 200 characters of the question
    status: str
    error_code: str | None
    live: bool  # true when the analysis reads the current documents (no as_of)
    as_of: datetime | None
    created_at: datetime
    completed_at: datetime | None
    summary: dict
    changed_since: bool  # some re-verification of one of its claims changed the verdict


class NLIOut(BaseModel):
    entail: float | None
    contra: float | None
    neutral: float | None


class EvidenceOut(BaseModel):
    """One passage cited for or against a claim.

    When the viewer can no longer read the passage's collection, `redacted` is true and the
    source material (text, span text, document, version number, dates, pages) is null. The ids,
    relation, rank, and NLI numbers stay.
    """

    passage_id: UUID
    version_id: UUID
    document_id: UUID | None
    document_title: str | None
    version_number: int | None
    effective_from: datetime | None
    page_start: int | None
    page_end: int | None
    relation: str
    rank: int
    span_start: int
    span_end: int
    nli: NLIOut
    rerank_score: float | None
    text: str | None
    span_text: str | None  # the version text between span_start and span_end
    passage_is_current: bool | None
    redacted: bool


class ClaimCheckOut(BaseModel):
    id: UUID
    reason: str
    verdict: str
    probs: dict[str, float]
    changed: bool
    checked_version_ids: list[UUID]
    created_at: datetime


class ClaimOut(BaseModel):
    id: UUID
    ordinal: int
    text: str
    answer_start: int
    answer_end: int
    extraction_method: str
    verdict: str
    probs: dict[str, float]
    features: dict[str, float]
    diagnostics: dict
    conflict: bool  # at least one supporting and one contradicting evidence item
    evidence: list[EvidenceOut]
    latest_check: ClaimCheckOut | None


class CitationOut(BaseModel):
    """A `[P#]` marker of the answer; redacted like evidence when its collection is unreadable."""

    number: int
    marker_start: int
    marker_end: int
    sentence_start: int
    sentence_end: int
    status: str
    passage_id: UUID | None
    version_id: UUID | None
    document_title: str | None
    version_number: int | None
    redacted: bool


class FeedbackIn(BaseModel):
    label: Literal["correct", "incorrect"]
    note: FeedbackNote | None = None


class FeedbackOut(BaseModel):
    id: UUID


class GraphNodeOut(BaseModel):
    """One node of the evidence graph; `label` and `data` are withheld for a redacted passage."""

    id: str  # claim:<uuid>, sentence:<start>:<end>, passage:<uuid>, version:<uuid>, document:<uuid>
    type: Literal["claim", "sentence", "passage", "version", "document"]
    label: str | None
    data: dict


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    type: Literal[
        "supports", "contradicts", "related", "cites", "part_of", "version_of", "superseded_by"
    ]


class GraphOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


class AnalysisOut(BaseModel):
    id: UUID
    kind: str
    question: str | None
    answer: str | None
    status: str
    error_code: str | None
    progress: dict
    live: bool
    as_of: datetime | None
    known_at: datetime
    collection_ids: list[UUID]
    created_at: datetime
    completed_at: datetime | None
    config: dict
    timings: dict
    summary: dict
    claims: list[ClaimOut]
    citations: list[CitationOut]
