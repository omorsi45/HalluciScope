# Retrieval benchmark

## scifact (test split)

- Queries: 300, documents: 5183, passages: 14741, documents skipped as empty: 0
- Run: 2026-10-28T02:58:18+00:00 to 2026-10-28T03:10:23+00:00 (ingest 286.5 s, total 725.7 s)
- Commit: 05a6b403e187f3401f34bad4235b79a6f8cff433 (working tree: clean)
- CPU: AMD Ryzen 9 7900X 12-Core Processor, RAM: 15.2 GB
- Python: 3.14.9 (main, Oct 21 2026, 17:48:02) [Clang 23.1.3 ], torch: 2.14.1+cpu, PostgreSQL: 17.11 (Debian 17.11-1.pgdg12+2), pgvector: 0.8.7
- Models: embedder BAAI/bge-small-en-v1.5, reranker cross-encoder/ms-marco-MiniLM-L6-v2
- Parameters: BM25 k1 = 0.9, b = 0.4; RRF k = 60; candidates = 300; rerank depth = 50 documents; HNSW ef_search = 300; passage target = 800 characters

| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fts | 0.4398 | 0.6037 | 0.7911 | 0.3914 | 68.0 | 169.1 |
| bm25 | 0.6686 | 0.7850 | 0.9129 | 0.6363 | 26.2 | 36.2 |
| dense | 0.7105 | 0.8307 | 0.9600 | 0.6802 | 21.8 | 24.3 |
| dense_exact | 0.7104 | 0.8307 | 0.9600 | 0.6801 | 63.6 | 66.5 |
| hybrid | 0.7230 | 0.8489 | 0.9650 | 0.6887 | 43.1 | 52.6 |
| hybrid_rerank | 0.7006 | 0.8319 | 0.9650 | 0.6669 | 1124.8 | 1747.7 |

Checks:

- acl_leaks: 0 (must be 0)
- bm25_parity_max_abs_diff: 7.105e-15 (must be < 1e-9)

Methodology: BM25 parameters were fixed before evaluation. Passages are sentence-aligned and every system collapses its passage ranking to documents with MaxP (a document ranks by its best passage). There was no tuning on test qrels.

## nfcorpus (test split)

- Queries: 323, documents: 3633, passages: 10964, documents skipped as empty: 0
- Run: 2026-10-28T03:10:38+00:00 to 2026-10-28T03:21:43+00:00 (ingest 211.4 s, total 665.7 s)
- Commit: 05a6b403e187f3401f34bad4235b79a6f8cff433 (working tree: clean)
- CPU: AMD Ryzen 9 7900X 12-Core Processor, RAM: 15.2 GB
- Python: 3.14.9 (main, Oct 21 2026, 17:48:02) [Clang 23.1.3 ], torch: 2.14.1+cpu, PostgreSQL: 17.11 (Debian 17.11-1.pgdg12+2), pgvector: 0.8.7
- Models: embedder BAAI/bge-small-en-v1.5, reranker cross-encoder/ms-marco-MiniLM-L6-v2
- Parameters: BM25 k1 = 0.9, b = 0.4; RRF k = 60; candidates = 300; rerank depth = 50 documents; HNSW ef_search = 300; passage target = 800 characters

| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fts | 0.2534 | 0.1304 | 0.2254 | 0.4125 | 25.1 | 41.9 |
| bm25 | 0.3211 | 0.1578 | 0.2462 | 0.5174 | 26.3 | 47.2 |
| dense | 0.3488 | 0.1629 | 0.3081 | 0.5455 | 23.3 | 68.1 |
| dense_exact | 0.3500 | 0.1645 | 0.3093 | 0.5455 | 61.0 | 85.8 |
| hybrid | 0.3596 | 0.1757 | 0.3143 | 0.5629 | 45.9 | 59.5 |
| hybrid_rerank | 0.3504 | 0.1663 | 0.3143 | 0.5580 | 1115.8 | 1664.9 |

Checks:

- acl_leaks: 0 (must be 0)
- bm25_parity_max_abs_diff: 3.553e-15 (must be < 1e-9)

Methodology: BM25 parameters were fixed before evaluation. Passages are sentence-aligned and every system collapses its passage ranking to documents with MaxP (a document ranks by its best passage). There was no tuning on test qrels.
