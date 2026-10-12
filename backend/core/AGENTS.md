# Core pipeline instructions

Applies to `backend/core/` and descendants. Root and `backend/AGENTS.md` still apply.

## Invariants
- `pipeline.py` coordinates per-document retrieval, Ollama answer generation, atomic-claim decomposition, three verifier calls, and ensemble scoring. Keep that ordering and pass the original question to the consistency verifier.
- `chunker.py` constructs independent `DocumentIndex` objects; `Pipeline` caches them by document hash with bounded LRU behavior. Never mix chunks or embeddings from different documents or requests.
- Verifiers are invoked concurrently with `asyncio.gather`. Preserve the association and order between claims and all verifier outputs.
- `ensemble.py` combines hallucination scores, where 0 means supported and 1 means hallucinated. Current default weights are NLI 0.5, consistency 0.3, similarity 0.2; current tiers are supported for scores below 0.2, hallucinated above 0.6, and uncertain otherwise. Treat changes to these definitions as behavior changes requiring targeted tests.
- Empty decomposed-claim results return no scored claims and an overall score of 0.0.

## Checks
- Mock external generation, tokenizers, and embedding models in unit tests; do not rely on an Ollama server or downloaded model weights for routine checks.
- Relevant tests include `tests/test_pipeline.py`, `tests/test_chunker.py`, `tests/test_generator.py`, `tests/test_decomposer.py`, and `tests/test_ensemble.py`.
