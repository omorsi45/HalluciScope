# HalluciScope

Multi-signal hallucination detection for RAG question-answering.

Given a source document and a question, HalluciScope generates an answer using a local LLM, decomposes it into atomic claims, and verifies each claim through three independent signals:

- **NLI Verification** (DeBERTa) — checks if source documents entail or contradict each claim
- **Self-Consistency Sampling** — re-asks the question N times and flags claims that vary across samples
- **Semantic Similarity** — measures embedding distance between claims and source material

An ensemble scorer combines these signals into a per-claim hallucination probability.

## Architecture

```
Document + Question → RAG Generator (Ollama) → Claim Decomposer
    → [NLI | Self-Consistency | Similarity] (parallel)
    → Ensemble Scorer → Per-claim hallucination probabilities
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.1 8B via Ollama |
| NLI | cross-encoder/nli-deberta-v3-large |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Backend | FastAPI + Python 3.11 |
| Frontend | React + TypeScript + Tailwind |
| Vector Store | FAISS |
| Database | SQLite |

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.ai) with `llama3.1:8b` pulled

### Backend

```bash
cd halluciscope
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,eval]"
ollama pull llama3.1:8b
```

### Frontend

```bash
cd frontend
npm install
npm run build
```

### Run

```bash
# Start backend
uvicorn backend.api.startup:create_configured_app --factory --reload

# Start frontend (dev)
cd frontend && npm run dev
```

## CLI Usage

```bash
# Analyze a document
halluciscope check --doc paper.pdf --question "What were the findings?" --verbose

# Run benchmark
halluciscope evaluate --dataset halueval --output evaluation/results/

# Run ablation study
halluciscope evaluate --ablation --dataset halueval
```

## Evaluation

### Ensemble Weights

Default: NLI=0.5, Self-consistency=0.3, Similarity=0.2

### Ablation Results

| Configuration | Precision | Recall | F1 | Delta |
|---|---|---|---|---|
| NLI only | - | - | - | - |
| Self-consistency only | - | - | - | - |
| Semantic similarity only | - | - | - | - |
| NLI + Self-consistency | - | - | - | - |
| NLI + Similarity | - | - | - | - |
| Consistency + Similarity | - | - | - | - |
| **Full ensemble** | - | - | - | - |

*(Fill after running `halluciscope evaluate --ablation --dataset halueval`)*

## Limitations

- Single-document RAG only (no multi-doc)
- English only
- Static ensemble weights (no online learning)
- Self-consistency adds latency (5x LLM calls)
- Semantic similarity can be fooled by paraphrased misinformation
