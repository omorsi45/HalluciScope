"""Optional self-consistency diagnostic for the answers of ask analyses.

The answer prompt is asked again at temperature 0.7; for each claim of the real answer the
diagnostic counts the resampled answers that restate it. It is shown beside the verdict and
never feeds it: verdicts come only from the NLI features and the calibrator, and the diagnostic
cannot exist for an answer another model wrote (verify analyses, RAGTruth).

A resample counts a claim when its best-matching claim has an embedding cosine of at least
`threshold`. A resample whose LLM call fails is zero claims and is counted in `failed_samples`;
the failure never reaches the analysis.
"""

import asyncio
from collections.abc import Sequence

import numpy as np

from backend.generate.answer import build_answer_messages
from backend.llm.client import LLMClient, LLMFailure
from backend.models.protocols import Embedder
from backend.retrieval.types import PassageHit
from backend.verify.claims import sentence_claims
from backend.verify.types import Claim

SAMPLE_TEMPERATURE = 0.7
DEFAULT_THRESHOLD = 0.85


async def consistency_diagnostic(
    llm: LLMClient,
    embedder: Embedder,
    question: str,
    context: Sequence[PassageHit],
    claims: Sequence[Claim],
    *,
    n_samples: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """One `{"appearances", "n_samples", "failed_samples"}` per claim, in claim order.

    `n_samples` answers are drawn at once from the prompt the analysis answered with. Without
    claims there is nothing to compare, so no sample is drawn.
    """
    if not claims:
        return []

    messages = build_answer_messages(question, context)
    samples = await asyncio.gather(*(_sample_claim_texts(llm, messages) for _ in range(n_samples)))
    failed = sum(1 for sample in samples if sample is None)
    sample_texts = [sample or [] for sample in samples]

    texts = list(
        dict.fromkeys([claim.text for claim in claims] + [t for s in sample_texts for t in s])
    )
    vectors = await asyncio.to_thread(embedder.embed_passages, texts)
    row = {text: index for index, text in enumerate(texts)}

    def appears(claim: Claim, sample: Sequence[str]) -> bool:
        if not sample:
            return False
        # Rows are L2-normalized, so the dot product is the cosine.
        best = max(float(np.dot(vectors[row[claim.text]], vectors[row[t]])) for t in sample)
        return best >= threshold

    return [
        {
            "appearances": sum(1 for sample in sample_texts if appears(claim, sample)),
            "n_samples": n_samples,
            "failed_samples": failed,
        }
        for claim in claims
    ]


async def _sample_claim_texts(llm: LLMClient, messages: list[dict[str, str]]) -> list[str] | None:
    """The claim texts of one resampled answer; None when the LLM call failed."""
    try:
        answer = await llm.chat(messages, temperature=SAMPLE_TEMPERATURE)
    except LLMFailure:
        return None
    return [claim.text for claim in sentence_claims(answer)]
