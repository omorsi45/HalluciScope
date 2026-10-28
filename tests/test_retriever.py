"""PostgresRetriever: hybrid fusion, rerank, access control, and bitemporal scope in every mode."""

import threading
import uuid
from datetime import UTC, datetime

import pytest

from backend.db.repos import documents
from backend.obs.metrics import STAGE_SECONDS
from backend.retrieval import service as service_module
from backend.retrieval.bm25 import bm25_term_score
from backend.retrieval.service import PostgresRetriever
from backend.retrieval.types import RerankUnavailable, SearchScope
from tests.fakes import HashEmbedder, KeywordReranker
from tests.pg_fixtures import seed_docs
from tests.test_bm25 import D_JAN, D_JUL, D_JUN, add_version, known_after, make_scope

pg = pytest.mark.pg

MODES = ("bm25", "dense", "fts", "hybrid")


class SpyEmbedder:
    """HashEmbedder that logs each query embedding and the thread it ran on."""

    model_id = "spy-hash-384"
    dim = 384

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.threads: list[int] = []
        self._inner = HashEmbedder()

    def embed_passages(self, texts):
        return self._inner.embed_passages(texts)

    def embed_queries(self, texts):
        self.events.append("embed")
        self.threads.append(threading.get_ident())
        return self._inner.embed_queries(texts)


class RecordingDb:
    """Database wrapper that logs each transaction it opens."""

    def __init__(self, db, events: list[str]) -> None:
        self._db = db
        self._events = events

    def tenant_tx(self, tenant_id):
        self._events.append("tx")
        return self._db.tenant_tx(tenant_id)


class MarkerReranker:
    """Scores 1.0 for passages containing the marker word, else 0.0; records what it was given."""

    model_id = "marker"

    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.seen: list[list[str]] = []

    def score(self, query, passages):
        self.seen.append(list(passages))
        return [1.0 if self.marker in passage.lower() else 0.0 for passage in passages]


KEYWORD_RERANKER = KeywordReranker()


def make_retriever(db, reranker=KEYWORD_RERANKER, **kwargs) -> PostgresRetriever:
    return PostgresRetriever(db, HashEmbedder(), reranker, **kwargs)


async def all_modes(retriever, query, scope, **kwargs):
    """Yield (mode, rerank, hits) for every mode, with and without rerank."""
    for mode in MODES:
        for rerank in (False, True):
            hits = await retriever.search(query, scope, mode=mode, rerank=rerank, **kwargs)
            yield mode, rerank, hits


async def passage_ids_in(db, tenant_id, document_ids) -> set[uuid.UUID]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(
            "select id from passages where document_id = any(%s)", (list(document_ids),)
        )
        return {row["id"] for row in await cur.fetchall()}


async def all_passage_ids(db, tenant_id) -> set[uuid.UUID]:
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select id from passages")
        return {row["id"] for row in await cur.fetchall()}


@pg
async def test_hybrid_combines_lexical_and_dense(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    # "what is the" are stopwords: the query's only lexeme is "harbor". The lexical passage
    # shares that lexeme but is diluted by filler, so it is far from the query vector; the two
    # stopword passages share query tokens and are the nearest vectors but match no lexeme.
    lexical_only = (
        "Harbor lantern orchard granite meadow falcon velvet ember quartz willow saddle timber."
    )
    dense_best = "What is the?"
    dense_next = "What is."
    unrelated = ["Weather report for tuesday afternoon.", "Quarterly numbers were revised."]
    await seed_docs(
        db, services, tenant_id, collection_id, [lexical_only, dense_best, dense_next, *unrelated]
    )
    scope = SearchScope(tenant_id, (collection_id,))
    retriever = make_retriever(db)
    query = "what is the harbor"

    bm25 = await retriever.search(query, scope, k=2, mode="bm25", rerank=False)
    dense = await retriever.search(query, scope, k=2, mode="dense", rerank=False)
    hybrid = await retriever.search(query, scope, k=2, candidates=2, mode="hybrid", rerank=False)

    # Neither component alone returns both passages ...
    assert [hit.text for hit in bm25] == [lexical_only]
    assert [hit.text for hit in dense] == [dense_best, dense_next]
    # ... and the fusion of their top 2 lists puts the best of each at the top.
    assert {hit.text for hit in hybrid} == {lexical_only, dense_best}
    assert [hit.rank for hit in hybrid] == [1, 2]
    by_text = {hit.text: hit for hit in hybrid}
    assert set(by_text[lexical_only].scores) == {"bm25", "rrf"}
    assert set(by_text[dense_best].scores) == {"dense", "rrf"}
    # Each is first in exactly one list: 1 / (60 + 1).
    assert by_text[lexical_only].scores["rrf"] == pytest.approx(1 / 61)
    assert by_text[dense_best].scores["rrf"] == pytest.approx(1 / 61)

    short_k = make_retriever(db, rrf_k=10)
    fused = await short_k.search(query, scope, k=2, candidates=2, mode="hybrid", rerank=False)
    assert [hit.scores["rrf"] for hit in fused] == [pytest.approx(1 / 11)] * 2


@pg
async def test_scores_carry_each_component(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    texts = [
        "Harbor copper.",
        "Harbor copper lantern.",
        "Copper orchard granite meadow.",
        "Weather report for tuesday afternoon.",
    ]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))
    retriever = make_retriever(db)
    query = "harbor copper"

    async def run(mode, *, rerank=False):
        return await retriever.search(query, scope, mode=mode, rerank=rerank)

    bm25, dense, fts = await run("bm25"), await run("dense"), await run("fts")
    hybrid, reranked = await run("hybrid"), await run("hybrid", rerank=True)

    assert bm25
    assert all(set(hit.scores) == {"bm25"} for hit in bm25)
    assert len(dense) == 4
    assert all(set(hit.scores) == {"dense"} for hit in dense)
    assert fts
    assert all(set(hit.scores) == {"fts"} for hit in fts)
    assert all({"rrf"} < set(hit.scores) <= {"bm25", "dense", "rrf"} for hit in hybrid)
    assert len(reranked) == 4
    assert all({"rrf", "rerank"} < set(hit.scores) for hit in reranked)
    # The component scores are the ones the single-component modes report.
    bm25_by_id = {hit.passage_id: hit.scores["bm25"] for hit in bm25}
    dense_by_id = {hit.passage_id: hit.scores["dense"] for hit in dense}
    assert len(hybrid) == 4
    assert len(bm25_by_id) == 3
    for hit in hybrid:
        assert hit.scores["dense"] == pytest.approx(dense_by_id[hit.passage_id])
        if hit.passage_id in bm25_by_id:
            assert hit.scores["bm25"] == pytest.approx(bm25_by_id[hit.passage_id])
        else:
            assert "bm25" not in hit.scores
    # rrf is the sum of 1 / (60 + rank) over the rankings that hold the passage.
    bm25_ranks = {hit.passage_id: hit.rank for hit in bm25}
    dense_ranks = {hit.passage_id: hit.rank for hit in dense}
    for hit in hybrid:
        expected = sum(
            1 / (60 + ranks[hit.passage_id])
            for ranks in (bm25_ranks, dense_ranks)
            if hit.passage_id in ranks
        )
        assert hit.scores["rrf"] == pytest.approx(expected)
    rrf_scores = [hit.scores["rrf"] for hit in hybrid]
    assert rrf_scores == sorted(rrf_scores, reverse=True)
    for hits in (bm25, dense, fts, hybrid, reranked):
        assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


@pg
async def test_rerank_reorders(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    texts = [
        "Harbor copper.",
        "Harbor copper lantern.",
        "Harbor copper lantern zebra granite.",
        "Weather report for tuesday afternoon.",
        "Quarterly numbers were revised downward.",
    ]
    await seed_docs(db, services, tenant_id, collection_id, texts)
    scope = SearchScope(tenant_id, (collection_id,))
    reranker = MarkerReranker("zebra")
    retriever = make_retriever(db, reranker)
    query = "harbor copper"

    plain = await retriever.search(query, scope, k=3, mode="hybrid", rerank=False)
    top_one = await retriever.search(query, scope, k=1, candidates=5, mode="hybrid", rerank=True)
    top_three = await retriever.search(query, scope, k=3, candidates=5, mode="hybrid", rerank=True)

    # Without rerank the zebra passage is last of the top 3 and the reranker is never asked.
    assert [hit.text for hit in plain] == texts[:3]
    assert len(reranker.seen) == 2
    # The reranker scores all 5 candidates, not only the top k, and the cut to k comes after.
    assert [len(seen) for seen in reranker.seen] == [5, 5]
    assert [hit.text for hit in top_one] == [texts[2]]
    assert top_one[0].rank == 1
    assert top_one[0].scores["rerank"] == 1.0
    assert {"bm25", "dense", "rrf"} < set(top_one[0].scores)
    # Equal logits keep the fused order.
    assert [hit.text for hit in top_three] == [texts[2], texts[0], texts[1]]
    assert [hit.rank for hit in top_three] == [1, 2, 3]


@pg
async def test_rerank_without_reranker_raises(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["Harbor copper lantern."])
    events: list[str] = []
    retriever = PostgresRetriever(RecordingDb(db, events), SpyEmbedder(events), None)
    scope = SearchScope(tenant_id, (collection_id,))

    for mode in MODES:
        with pytest.raises(RerankUnavailable):
            await retriever.search("harbor", scope, mode=mode, rerank=True)
    with pytest.raises(RerankUnavailable):
        await retriever.search("harbor", SearchScope(tenant_id, ()), rerank=True)

    # Raised before any work: nothing was embedded and no transaction was opened.
    assert events == []
    # Without rerank the same retriever works.
    hits = await retriever.search("harbor", scope, rerank=False)
    assert len(hits) == 1


@pg
async def test_scope_enforced_in_every_mode(db, services):
    tenant_id, (x, y) = await make_scope(db, collections=2)
    live_texts = [
        "Harbor copper lantern shipment arrived.",
        "Orchard copper granite shipment delayed.",
        "Meadow falcon velvet notice.",
    ]
    live_docs = await seed_docs(db, services, tenant_id, x, live_texts)
    retriever = make_retriever(db)
    scope_x = SearchScope(tenant_id, (x,))
    query = "harbor copper shipment"
    before = {
        hit.passage_id: hit.scores["bm25"]
        for hit in await retriever.search(query, scope_x, k=50, mode="bm25", rerank=False)
    }
    assert len(before) == 2

    # Collection Y holds the query terms in bulk; one document of X is deleted.
    other_docs = await seed_docs(
        db,
        services,
        tenant_id,
        y,
        [f"Harbor copper shipment manifest {n} for the dock." for n in range(6)],
    )
    (doomed,) = await seed_docs(
        db, services, tenant_id, x, ["Harbor copper shipment from a file that was deleted."]
    )
    known = await known_after(db, tenant_id)
    async with db.tenant_tx(tenant_id) as conn:
        assert await documents.soft_delete_document(conn, doomed) is True
    forbidden = await passage_ids_in(db, tenant_id, [*other_docs, doomed])
    readable = await passage_ids_in(db, tenant_id, live_docs)
    assert len(forbidden) == 7
    assert len(readable) == 3

    async for mode, rerank, hits in all_modes(retriever, query, scope_x, k=50):
        label = (mode, rerank)
        ids = {hit.passage_id for hit in hits}
        assert ids, label
        assert ids <= readable, label
        if mode == "dense":
            # Dense returns every visible passage, so a leak could not hide.
            assert ids == readable, label

    # BM25 statistics come from the readable set only: Y and the deleted file change nothing.
    after = {
        hit.passage_id: hit.scores["bm25"]
        for hit in await retriever.search(query, scope_x, k=50, mode="bm25", rerank=False)
    }
    assert after == pytest.approx(before, abs=1e-12, rel=0)

    # Controls: the same corpus leaks when Y is readable, and a scope that predates the delete
    # sees the deleted file, so the checks above can detect a leak.
    both = SearchScope(tenant_id, (x, y))
    predates_delete = SearchScope(tenant_id, (x,), known_at=known)
    for mode in MODES:
        for control, leaked in ((both, set(other_docs)), (predates_delete, {doomed})):
            hits = await retriever.search(query, control, k=50, mode=mode, rerank=False)
            leaked_ids = await passage_ids_in(db, tenant_id, leaked)
            assert {hit.passage_id for hit in hits} & leaked_ids, mode


@pg
async def test_as_of_in_every_mode(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    (doc1,) = await seed_docs(
        db, services, tenant_id, collection_id, ["Harbor copper lantern."], effective_from=D_JAN
    )
    # A second document whose first version takes effect only in July.
    await seed_docs(
        db, services, tenant_id, collection_id, ["Harbor copper granite."], effective_from=D_JUL
    )
    await add_version(db, services, tenant_id, doc1, "Harbor copper orchard.", D_JUN)
    retriever = make_retriever(db)
    query = "harbor copper"

    def scope(**kwargs) -> SearchScope:
        return SearchScope(tenant_id, (collection_id,), **kwargs)

    async def texts_in_every_mode(search_scope: SearchScope) -> dict[tuple[str, bool], list[str]]:
        return {
            (mode, rerank): sorted(hit.text for hit in hits)
            async for mode, rerank, hits in all_modes(retriever, query, search_scope)
        }

    def same_everywhere(found: dict, expected: list[str]) -> None:
        assert found == {(mode, rerank): expected for mode in MODES for rerank in (False, True)}

    # Before the first version took effect nothing is visible.
    same_everywhere(await texts_in_every_mode(scope(as_of=datetime(2025, 12, 1, tzinfo=UTC))), [])
    # Between the versions the older one answers, in every mode, and July's document is absent.
    between = scope(as_of=datetime(2026, 3, 1, tzinfo=UTC))
    same_everywhere(await texts_in_every_mode(between), ["Harbor copper lantern."])
    async for _, _, hits in all_modes(retriever, query, between):
        assert [hit.version_number for hit in hits] == [1]
        assert [hit.effective_from for hit in hits] == [D_JAN]

    known = await known_after(db, tenant_id)
    await add_version(db, services, tenant_id, doc1, "Harbor copper marble.", D_JUL)
    august = datetime(2026, 8, 1, tzinfo=UTC)
    # A version uploaded after known_at, even backdated to July, changes nothing for it.
    same_everywhere(
        await texts_in_every_mode(scope(as_of=august, known_at=known)),
        ["Harbor copper granite.", "Harbor copper orchard."],
    )
    same_everywhere(
        await texts_in_every_mode(scope(as_of=august)),
        ["Harbor copper granite.", "Harbor copper marble."],
    )


def unicode_text(first_word: str, tag: str) -> str:
    """Three paragraphs of about 600 characters (so three passages) with combining accents,
    CJK, and an emoji, each paragraph different."""
    paragraphs = []
    for n in range(3):
        sentence = (
            f"{first_word} lantern {tag}{n} re\u0301sume\u0301 \u65e5\u672c\u8a9e \U0001f642 "
            "caf\u00e9 orchard granite meadow falcon velvet ember."
        )
        paragraphs.append(" ".join([sentence] * 8))
    return "\n\n".join(paragraphs)


@pg
async def test_hit_fields_resolve_exactly(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    doc1, doc2 = await seed_docs(
        db,
        services,
        tenant_id,
        collection_id,
        [unicode_text("harbor", "old"), unicode_text("harbor", "other")],
        effective_from=D_JAN,
    )
    await add_version(db, services, tenant_id, doc1, unicode_text("harbor", "new"), D_JUN)
    expected = {doc1: (2, "Document 1", D_JUN), doc2: (1, "Document 2", D_JAN)}
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute("select id, text from document_versions")
        version_text = {row["id"]: row["text"] for row in await cur.fetchall()}
        cur = await conn.execute(
            "select p.id from passages p where p.is_current and p.document_id = any(%s)",
            ([doc1, doc2],),
        )
        current = {row["id"] for row in await cur.fetchall()}
    assert len(current) == 6
    retriever = make_retriever(db)
    scope = SearchScope(tenant_id, (collection_id,))

    seen_offset = seen_non_ascii = False
    async for mode, rerank, hits in all_modes(retriever, "harbor lantern", scope, k=20):
        label = (mode, rerank)
        assert hits, label
        assert {hit.passage_id for hit in hits} <= current, label
        if mode == "dense":
            assert {hit.passage_id for hit in hits} == current, label
        for hit in hits:
            assert hit.text == version_text[hit.version_id][hit.start_char : hit.end_char], label
            assert (hit.version_number, hit.document_title, hit.effective_from) == expected[
                hit.document_id
            ], label
            seen_offset = seen_offset or hit.start_char > 0
            seen_non_ascii = seen_non_ascii or not hit.text.isascii()
    # The corpus really exercises non-zero offsets and non-ASCII characters.
    assert seen_offset
    assert seen_non_ascii


@pg
async def test_cross_tenant_identical_text_isolated(db, services):
    texts = [
        "Harbor copper lantern shipment.",
        "Orchard copper granite shipment.",
        "Meadow falcon.",
    ]
    tenant_a, (collection_a,) = await make_scope(db)
    tenant_b, (collection_b,) = await make_scope(db)
    await seed_docs(db, services, tenant_a, collection_a, texts)
    await seed_docs(db, services, tenant_b, collection_b, texts)
    ids_a = await all_passage_ids(db, tenant_a)
    ids_b = await all_passage_ids(db, tenant_b)
    assert len(ids_a) == len(ids_b) == 3
    assert not ids_a & ids_b
    retriever = make_retriever(db)
    future = datetime(2100, 1, 1, tzinfo=UTC)

    for scope, own in (
        (SearchScope(tenant_a, (collection_a,)), ids_a),
        (SearchScope(tenant_a, (collection_a,), as_of=future), ids_a),
        (SearchScope(tenant_b, (collection_b,)), ids_b),
        (SearchScope(tenant_b, (collection_b,), as_of=future), ids_b),
    ):
        async for mode, rerank, hits in all_modes(retriever, "harbor copper shipment", scope):
            assert hits, (mode, rerank)
            assert {hit.passage_id for hit in hits} <= own, (mode, rerank)

    # Naming the other tenant's collection finds nothing: row security hides it.
    for scope in (
        SearchScope(tenant_a, (collection_b,)),
        SearchScope(tenant_a, (collection_b,), as_of=future),
    ):
        async for mode, rerank, hits in all_modes(retriever, "harbor copper shipment", scope):
            assert hits == [], (mode, rerank)


@pg
async def test_bm25_parameters_reach_the_scorer(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(
        db,
        services,
        tenant_id,
        collection_id,
        ["Lantern harbor.", "Harbor copper granite meadow.", "Orchard falcon."],
    )
    scope = SearchScope(tenant_id, (collection_id,))

    default = await make_retriever(db).search("lantern", scope, mode="bm25", rerank=False)
    tuned = await make_retriever(db, k1=0.5, b=0.2).search(
        "lantern", scope, mode="bm25", rerank=False
    )

    # One of 3 passages holds "lantern" once; passage lengths are 2, 4 and 2 (avgdl 8/3).
    assert default[0].scores["bm25"] == pytest.approx(bm25_term_score(1, 1, 3, 2, 8 / 3, 0.9, 0.4))
    assert tuned[0].scores["bm25"] == pytest.approx(bm25_term_score(1, 1, 3, 2, 8 / 3, 0.5, 0.2))
    assert default[0].scores["bm25"] != pytest.approx(tuned[0].scores["bm25"])


@pg
async def test_query_embedded_off_loop_before_transaction_and_only_for_dense_modes(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["Harbor copper lantern."])
    events: list[str] = []
    embedder = SpyEmbedder(events)
    retriever = PostgresRetriever(RecordingDb(db, events), embedder, KeywordReranker())
    scope = SearchScope(tenant_id, (collection_id,))

    for mode, embeds in (("hybrid", True), ("dense", True), ("bm25", False), ("fts", False)):
        events.clear()
        hits = await retriever.search("harbor", scope, mode=mode, rerank=True)
        assert len(hits) == 1, mode
        # One embedding (dense modes only), then exactly one transaction.
        assert events == (["embed", "tx"] if embeds else ["tx"]), mode

    assert embedder.threads
    assert threading.get_ident() not in embedder.threads


@pg
async def test_component_limits_exact_flag_and_lexical_before_dense(db, services, monkeypatch):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["Harbor copper lantern."])
    calls: list[tuple] = []

    def spy(name):
        real = getattr(service_module, f"{name}_search")

        async def wrapper(conn, query, scope, **kwargs):
            calls.append((name, kwargs["limit"], kwargs.get("exact")))
            return await real(conn, query, scope, **kwargs)

        return wrapper

    monkeypatch.setattr(service_module, "bm25_search", spy("bm25"))
    monkeypatch.setattr(service_module, "fts_search", spy("fts"))
    real_dense = service_module.dense_search

    async def dense_wrapper(conn, vector, scope, **kwargs):
        calls.append(("dense", kwargs["limit"], kwargs["exact"]))
        return await real_dense(conn, vector, scope, **kwargs)

    monkeypatch.setattr(service_module, "dense_search", dense_wrapper)
    retriever = make_retriever(db)
    scope = SearchScope(tenant_id, (collection_id,))

    async def run(mode, *, rerank, exact=False):
        calls.clear()
        await retriever.search(
            "harbor", scope, k=3, candidates=7, mode=mode, rerank=rerank, exact=exact
        )
        return list(calls)

    # Hybrid always fetches `candidates` per component, lexical before dense: dense's SET LOCAL
    # settings must not shape the lexical statement's plan.
    assert await run("hybrid", rerank=False) == [("bm25", 7, None), ("dense", 7, False)]
    assert await run("hybrid", rerank=False, exact=True) == [("bm25", 7, None), ("dense", 7, True)]
    # A single component fetches k, or `candidates` when a rerank follows.
    assert await run("bm25", rerank=False) == [("bm25", 3, None)]
    assert await run("bm25", rerank=True) == [("bm25", 7, None)]
    assert await run("fts", rerank=False) == [("fts", 3, None)]
    assert await run("dense", rerank=False) == [("dense", 3, False)]
    assert await run("dense", rerank=True, exact=True) == [("dense", 7, True)]
    # Candidates below k never shrink the fetch below k.
    calls.clear()
    await retriever.search("harbor", scope, k=5, candidates=2, mode="hybrid", rerank=False)
    assert calls == [("bm25", 5, None), ("dense", 5, False)]


@pg
async def test_validation(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["Harbor copper lantern."])
    events: list[str] = []
    retriever = PostgresRetriever(RecordingDb(db, events), SpyEmbedder(events), KeywordReranker())
    scope = SearchScope(tenant_id, (collection_id,))

    bad_arguments = [
        {"k": 0},
        {"k": -1},
        {"k": 301},
        {"candidates": 0},
        {"candidates": 1001},
        {"mode": "x"},
    ]
    for scope_under_test in (scope, SearchScope(tenant_id, ())):
        for bad in bad_arguments:
            with pytest.raises(ValueError):
                await retriever.search("harbor", scope_under_test, **bad)
    assert events == []

    for good in ({"k": 1}, {"k": 300}, {"candidates": 1}, {"candidates": 1000}):
        hits = await retriever.search("harbor", scope, rerank=False, **good)
        assert len(hits) == 1, good


@pg
async def test_empty_scope_returns_empty(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["Harbor copper lantern."])
    events: list[str] = []
    reranker = MarkerReranker("harbor")
    retriever = PostgresRetriever(RecordingDb(db, events), SpyEmbedder(events), reranker)

    for scope in (
        SearchScope(tenant_id, ()),
        SearchScope(tenant_id, (), as_of=D_JUN),
        SearchScope(tenant_id, (), known_at=D_JUN),
    ):
        async for mode, rerank, hits in all_modes(retriever, "harbor", scope):
            assert hits == [], (mode, rerank)

    # No embedding, no transaction, no rerank: there is nothing to search.
    assert events == []
    assert reranker.seen == []


@pg
async def test_retrieve_stage_is_timed(db, services):
    tenant_id, (collection_id,) = await make_scope(db)
    await seed_docs(db, services, tenant_id, collection_id, ["Harbor copper lantern."])
    retriever = make_retriever(db)
    scope = SearchScope(tenant_id, (collection_id,))
    stage = STAGE_SECONDS.labels(stage="retrieve")

    before = stage.total
    await retriever.search("harbor", scope, rerank=False)

    assert stage.total == before + 1
