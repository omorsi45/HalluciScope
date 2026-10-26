# Verifier instructions

Applies to `backend/core/verifiers/`. Follow the root, backend, and core instructions too.

## Contract
- Implement `BaseVerifier.verify(claims, context_chunks, question="")` asynchronously and return `ClaimScore` entries for the input claims in the same order on valid nonempty inputs. Handle empty inputs explicitly.
- Each `hallucination_score` is on the 0-to-1 scale: 0 is evidence of support; 1 is evidence of hallucination. Keep verifier `details` useful for explaining a score without changing their meaning without updating consumers.
- NLI reads label order from the model's `config.id2label` (never assumed; deberta-v3-large is `contradiction, entailment, neutral`), raises `ValueError` if a label is missing, and maps probabilities to `contradiction + 0.5 * neutral`. It currently uses the lowest-scoring source chunk per claim.
- Self-consistency resamples answers for the actual question, decomposes sampled answers, compares normalized claim embeddings, and scores `1 - appearances / n_samples`. Retain its question propagation and threshold behavior.
- Semantic similarity compares normalized embeddings against source chunks and scores `1 - max_similarity` after bounding the similarity to `[0, 1]`.
- Preserve batch processing and nonblocking handling of model inference where used; avoid unnecessary per-claim model initialization.

## Checks
Run the corresponding tests after verifier edits:
```bash
python -m pytest tests/test_nli.py tests/test_consistency.py tests/test_similarity.py tests/test_ensemble.py -q
```
Use fakes or mocks for network access and model inference.
