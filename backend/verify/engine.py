"""The claim verification engine: evidence signals, calibrated verdicts, evidence spans.

A verdict comes only from NLI-derived features and the calibrator. No language model is
reachable from this module: the inputs are a retriever, a reranker, an NLI scorer, an embedder
and the calibrator artifact. Evidence spans are explanation only and never feed the verdict.
"""

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal

import numpy as np

from backend.calibration.artifact import Calibrator
from backend.models.protocols import Embedder, Reranker
from backend.obs.metrics import STAGE_SECONDS
from backend.retrieval.types import PassageHit, Retriever, SearchScope
from backend.verify.evidence import gather_evidence, relation_for, window_spans
from backend.verify.features import FEATURE_NAMES, compute_features, order_signals
from backend.verify.types import (
    VERDICT_ORDER,
    Claim,
    ClaimVerdict,
    EvidenceItem,
    EvidenceSignal,
    NLIResult,
    NLIScorer,
)

Collected = list[tuple[list[PassageHit], list[EvidenceSignal]]]


async def collect_signals(
    retriever: Retriever,
    reranker: Reranker,
    nli: NLIScorer,
    embedder: Embedder,
    claims: Sequence[Claim],
    scope: SearchScope,
    pinned: Mapping[int, Sequence[PassageHit]] = {},
    *,
    depth: int = 8,
    candidates: int = 20,
    premise_mode: Literal["passage", "window"] = "passage",
) -> Collected:
    """The evidence hits and scored signals of each claim, in claim order.

    Every NLI pair of every claim goes through one scorer call. With `premise_mode="window"`
    each hit is scored per sentence window and keeps one result: the window with the highest
    entail when that is at least the highest contra among its windows, else the window with
    the highest contra. Signals follow the hit order and carry the passage text.
    """
    if not claims:
        return []
    hit_lists = await gather_evidence(
        retriever, reranker, claims, scope, pinned=pinned, depth=depth, candidates=candidates
    )

    pairs: list[tuple[str, str]] = []
    pair_ranges: list[list[range]] = []
    for claim, hits in zip(claims, hit_lists, strict=True):
        ranges = []
        for hit in hits:
            if premise_mode == "window":
                premises = [text for _, _, text in window_spans(hit)]
            else:
                premises = [hit.text]
            ranges.append(range(len(pairs), len(pairs) + len(premises)))
            pairs.extend((premise, claim.text) for premise in premises)
        pair_ranges.append(ranges)

    results = await _score(nli, pairs)
    cosines = await _dense_cosines(embedder, claims, hit_lists)

    collected: Collected = []
    for hits, ranges, claim_cosines in zip(hit_lists, pair_ranges, cosines, strict=True):
        signals = []
        for hit, pair_range, dense_cos in zip(hits, ranges, claim_cosines, strict=True):
            signals.append(
                EvidenceSignal(
                    passage_id=str(hit.passage_id),
                    text=hit.text,
                    nli=_hit_result([results[i] for i in pair_range]),
                    rerank=hit.scores.get("rerank"),
                    dense_cos=dense_cos,
                )
            )
        collected.append((list(hits), signals))
    return collected


class VerificationEngine:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        nli: NLIScorer,
        embedder: Embedder,
        calibrator: Calibrator,
        *,
        depth: int = 8,
    ) -> None:
        # Features from other models mean something else to these weights: refuse to start.
        calibrator.check_parity(
            nli_model=nli.model_id,
            embedder_model=embedder.model_id,
            reranker_model=reranker.model_id,
        )
        self._retriever = retriever
        self._reranker = reranker
        self._nli = nli
        self._embedder = embedder
        self._calibrator = calibrator
        self._depth = depth

    async def collect(
        self,
        claims: Sequence[Claim],
        scope: SearchScope,
        pinned: Mapping[int, Sequence[PassageHit]] = {},
    ) -> Collected:
        """The evidence hits and signals of each claim, as the calibrator was trained on them."""
        return await collect_signals(
            self._retriever,
            self._reranker,
            self._nli,
            self._embedder,
            claims,
            scope,
            pinned,
            depth=self._depth,
            candidates=self._calibrator.candidates,
            premise_mode=self._calibrator.premise_mode,
        )

    async def verify(
        self,
        claims: Sequence[Claim],
        scope: SearchScope,
        pinned: Mapping[int, Sequence[PassageHit]] = {},
    ) -> list[ClaimVerdict]:
        """One calibrated verdict per claim, in claim order.

        A claim with no evidence gets the no-signal feature vector, which the calibrator turns
        into its usual verdict (insufficient), and an empty evidence tuple.
        """
        collected = await self.collect(claims, scope, pinned)
        if not collected:
            return []
        evidence_k = self._calibrator.evidence_k

        started = time.perf_counter()
        matrix = np.stack(
            [
                compute_features(claim.text, signals, evidence_k)
                for claim, (_, signals) in zip(claims, collected, strict=True)
            ]
        )
        probs = self._calibrator.predict_proba(matrix)
        verdicts = self._calibrator.decide(probs)
        STAGE_SECONDS.labels(stage="calibrate").observe(time.perf_counter() - started)

        evidence = [
            _evidence_items(hits, order_signals(signals)[:evidence_k])
            for hits, signals in collected
        ]
        evidence = await _localize(self._nli, claims, evidence)

        return [
            ClaimVerdict(
                claim=claim,
                verdict=verdict,
                probs={v.value: float(p) for v, p in zip(VERDICT_ORDER, row, strict=True)},
                features={name: float(x) for name, x in zip(FEATURE_NAMES, vector, strict=True)},
                evidence=tuple(items),
                signals=tuple(signals),
                diagnostics={
                    "n_candidates": len(hits),
                    "premise_mode": self._calibrator.premise_mode,
                },
            )
            for claim, (hits, signals), row, vector, verdict, items in zip(
                claims, collected, probs, matrix, verdicts, evidence, strict=True
            )
        ]


async def _score(nli: NLIScorer, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
    """One scorer call off the event loop; no pairs means no call."""
    if not pairs:
        return []
    started = time.perf_counter()
    try:
        return await asyncio.to_thread(nli.score, pairs)
    finally:
        STAGE_SECONDS.labels(stage="nli").observe(time.perf_counter() - started)


async def _dense_cosines(
    embedder: Embedder, claims: Sequence[Claim], hit_lists: Sequence[Sequence[PassageHit]]
) -> list[list[float]]:
    """Cosine of each claim with each of its hits; every distinct text is embedded once."""
    texts = list(
        dict.fromkeys(
            [claim.text for claim, hits in zip(claims, hit_lists, strict=True) if hits]
            + [hit.text for hits in hit_lists for hit in hits]
        )
    )
    if not texts:
        return [[] for _ in claims]
    vectors = await asyncio.to_thread(embedder.embed_passages, texts)
    row = {text: i for i, text in enumerate(texts)}
    # Rows are L2-normalized, so the dot product is the cosine.
    return [
        [float(np.dot(vectors[row[claim.text]], vectors[row[hit.text]])) for hit in hits]
        for claim, hits in zip(claims, hit_lists, strict=True)
    ]


def _hit_result(results: Sequence[NLIResult]) -> NLIResult:
    """The result of one hit: its only result, or the window result under the window rule."""
    best_entail = max(results, key=lambda r: r.entail)
    best_contra = max(results, key=lambda r: r.contra)
    return best_entail if best_entail.entail >= best_contra.contra else best_contra


def _evidence_items(
    hits: Sequence[PassageHit], ordered: Sequence[EvidenceSignal]
) -> list[EvidenceItem]:
    """An item per ordered signal, spanning its whole passage until localization narrows it."""
    by_id = {str(hit.passage_id): hit for hit in hits}
    return [
        EvidenceItem(
            hit=by_id[signal.passage_id],
            nli=signal.nli,
            relation=relation_for(signal.nli),
            span_start=by_id[signal.passage_id].start_char,
            span_end=by_id[signal.passage_id].end_char,
            dense_cos=signal.dense_cos,
            rerank=signal.rerank,
        )
        for signal in ordered
    ]


async def _localize(
    nli: NLIScorer, claims: Sequence[Claim], evidence: list[list[EvidenceItem]]
) -> list[list[EvidenceItem]]:
    """Narrow the span of each claim's best entail and best contra item to its best window.

    All windows of all targets are scored in one extra NLI call. The best window has the
    highest probability of the target class; ties keep the earliest window in `window_spans`
    order, so the single sentence wins over a pair that scores the same.
    """
    targets: list[tuple[int, int, Literal["entail", "contra"]]] = []
    pairs: list[tuple[str, str]] = []
    windows: list[list[tuple[int, int, str]]] = []
    for claim_index, (claim, items) in enumerate(zip(claims, evidence, strict=True)):
        for item_index, target in _targets(items).items():
            spans = window_spans(items[item_index].hit)
            targets.append((claim_index, item_index, target))
            windows.append(spans)
            pairs.extend((text, claim.text) for _, _, text in spans)

    results = await _score(nli, pairs)

    localized = [list(items) for items in evidence]
    offset = 0
    for (claim_index, item_index, target), spans in zip(targets, windows, strict=True):
        scored = results[offset : offset + len(spans)]
        offset += len(spans)
        best = max(range(len(spans)), key=lambda i: getattr(scored[i], target))
        start, end, _ = spans[best]
        localized[claim_index][item_index] = replace(
            localized[claim_index][item_index], span_start=start, span_end=end
        )
    return localized


def _targets(items: Sequence[EvidenceItem]) -> dict[int, Literal["entail", "contra"]]:
    """Item index to localize and the class to localize for.

    The item with the highest entail is always a target. The item with the highest contra is
    one when contra is its largest probability. If that is the same item, contra wins.
    """
    if not items:
        return {}
    targets: dict[int, Literal["entail", "contra"]] = {}
    targets[max(range(len(items)), key=lambda i: items[i].nli.entail)] = "entail"
    contra_index = max(range(len(items)), key=lambda i: items[i].nli.contra)
    if items[contra_index].relation == "contradicts":
        targets[contra_index] = "contra"
    return targets
