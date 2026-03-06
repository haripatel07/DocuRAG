# DocuRAG

**Production-Grade Retrieval-Augmented Generation for Technical Documentation**

DocuRAG lets you ask natural-language questions against your own documentation
and receive precise, grounded answers with source citations — powered by
LangChain, FAISS, and FastAPI.

---

## Architecture

```
User Query
    │
    ▼
FastAPI (POST /ask)
    │
    ▼
Query Processing
  ├─ Guardrails / validation
  ├─ Query rewriting (LLM)
  └─ Query classification
    │
    ▼
FAISS Semantic Search (top-k)
    │
    ▼
FlashRank Cross-Encoder Reranking
    │
    ▼
LLM Generation (GPT-4o-mini)
    │
    ▼
Grounded Answer + Source Citations
```

---

## Quick Start

### 1. Clone & configure

```bash
git clone <repo-url> docurag
cd docurag
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (required for LLM)
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Ingest documents

Place PDFs or Markdown files in `data/raw/`, then run:

```bash
python -m app.ingestion --source data/raw
```

Or ingest a web URL:

```bash
python -m app.ingestion --url https://fastapi.tiangolo.com/tutorial/
```

### 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at **http://localhost:8000/docs**

---

## API Reference

### `POST /ask`

Ask a question against the indexed documentation.

**Request:**
```json
{
  "query": "How does dependency injection work in FastAPI?",
  "top_k": 5,
  "metadata_filter": {"source_document": "fastapi_guide.pdf"},
  "stream": false
}
```

**Response:**
```json
{
  "answer": "Dependency injection in FastAPI ...",
  "sources": ["fastapi_guide.pdf (page 5)", "fastapi_guide.pdf (page 12)"],
  "query_type": "conceptual",
  "rewritten_query": "Explain the dependency injection mechanism in FastAPI",
  "latency_ms": 842.3
}
```

Set `"stream": true` to receive a Server-Sent Events (SSE) token stream.

### `POST /ingest`

Upload PDF/Markdown files via multipart form to index them immediately.

### `POST /ingest/url?url=<URL>`

Ingest a web page by URL.

### `GET /health`

Liveness probe — returns `{"status": "ok"}`.

---

## Docker Deployment

```bash
# Build and run
docker compose up --build

# Or with plain Docker
docker build -t docurag .
docker run -p 8000:8000 --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/vectorstore:/app/vectorstore \
  docurag
```

---

## Evaluation (RAGAS)

Add benchmark questions to `evaluation/benchmark.json`, then:

```bash
python evaluation/ragas_eval.py --dataset evaluation/benchmark.json
```

Reports **context precision**, **answer relevance**, and **faithfulness**.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All settings live in `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required for LLM) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `EMBEDDING_PROVIDER` | `huggingface` | `openai` or `huggingface` |
| `HF_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace model |
| `CHUNK_SIZE` | `800` | Token chunk size |
| `CHUNK_OVERLAP` | `150` | Chunk overlap |
| `RETRIEVAL_TOP_K` | `5` | Candidates fetched from FAISS |
| `RERANK_TOP_N` | `3` | Final chunks after reranking |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |

---

## Project Structure

```
docurag/
├── app/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings
│   ├── embeddings.py      # Embedding factory (OpenAI / HuggingFace)
│   ├── ingestion.py       # Load → clean → chunk → embed → store
│   ├── retriever.py       # FAISS search + FlashRank reranking
│   ├── generator.py       # Prompt builder + LLM generation
│   ├── rag_pipeline.py    # Full pipeline orchestrator + query processing
│   └── main.py            # FastAPI application
├── data/
│   └── raw/               # Place source documents here
├── vectorstore/           # FAISS index (auto-created)
├── evaluation/
│   ├── ragas_eval.py      # RAGAS evaluation script
│   └── benchmark.json     # Sample benchmark queries
├── tests/
│   └── test_docurag.py    # Unit + integration tests
├── logs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Notes

The system is intentionally minimal but production-aware. The query layer rewrites
vague inputs before retrieval and classifies them so the response style can adapt.
FlashRank adds a cross-encoder reranking pass after FAISS, which meaningfully
improves answer quality on longer documents without the latency cost of a full
re-embedding.

LangSmith tracing is opt-in — set `LANGCHAIN_TRACING_V2=true` in `.env` to start
recording chain traces. The RAGAS evaluation script is designed to run against a
curated benchmark; results below 0.7 on faithfulness usually indicate the chunking
strategy needs tuning for that document type.
