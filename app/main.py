"""FastAPI application — exposes /ask, /ingest, and /health endpoints."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag_pipeline import get_pipeline, validate_query, RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Configure LangSmith tracing if enabled
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info(f"LangSmith tracing enabled: project='{settings.langchain_project}'")

    logger.info("DocuRAG starting — pre-warming RAG pipeline …")
    try:
        get_pipeline()
        logger.info("RAG pipeline ready.")
    except Exception as exc:
        logger.warning(
            f"Pipeline warm-up failed ({exc}). "
            "The first request will trigger lazy initialisation."
        )

    yield
    logger.info("DocuRAG shutting down.")


settings = get_settings()

app = FastAPI(
    title="DocuRAG",
    description=(
        "Production-grade Retrieval-Augmented Generation API "
        "for technical documentation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask the documentation.",
        examples=["How does dependency injection work in FastAPI?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of retrieved chunks.",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filter, e.g. {\"source_document\": \"guide.pdf\"}.",
    )
    stream: bool = Field(
        default=False,
        description="If true, returns a streaming SSE response.",
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    query_type: str
    rewritten_query: str
    latency_ms: float


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness probe — always returns 200 if the service is running."""
    return HealthResponse(status="ok", version=app.version)


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest) -> AskResponse | StreamingResponse:
    """
    Ask a question against the indexed documentation.

    - For JSON responses leave ``stream=false`` (default).
    - For token-by-token streaming set ``stream=true``; the response will be
      a ``text/event-stream`` Server-Sent-Events stream.
    """
    try:
        validate_query(request.query)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    pipeline: RAGPipeline = get_pipeline()

    # --- Streaming branch ---------------------------------------------------
    if request.stream:
        async def _event_stream() -> AsyncIterator[str]:
            try:
                async for token in pipeline.astream(
                    request.query,
                    metadata_filter=request.metadata_filter,
                    top_k=request.top_k,
                ):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(f"Streaming error: {exc}")
                yield f"data: [ERROR] {exc}\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # --- Standard JSON branch -----------------------------------------------
    start = time.perf_counter()
    try:
        result = pipeline.run(
            request.query,
            metadata_filter=request.metadata_filter,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vector store not found. "
                "Please ingest documents first via POST /ingest."
            ),
        )
    except Exception as exc:
        logger.exception("Unhandled error in /ask")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(f"/ask | latency={latency_ms:.1f}ms | sources={result.sources}")

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        query_type=result.query_type,
        rewritten_query=result.rewritten_query,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_files(
    files: list[UploadFile] = File(...),
) -> IngestResponse:
    """
    Upload and ingest one or more PDF / Markdown documents into the vector
    store.  The endpoint streams files to the ``data/raw`` directory and then
    runs the ingestion pipeline.
    """
    from app.ingestion import run_ingestion_pipeline, add_documents_to_vectorstore, load_directory, chunk_documents

    settings = get_settings()
    saved_paths: list[Path] = []

    for upload in files:
        dest = settings.data_dir / upload.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = await upload.read()
        dest.write_bytes(content)
        saved_paths.append(dest)
        logger.info(f"Saved uploaded file: {dest}")

    try:
        docs = load_directory(settings.data_dir)
        chunks = chunk_documents(docs)
        from app.ingestion import build_vectorstore
        build_vectorstore(chunks)
        return IngestResponse(
            status="success",
            chunks_indexed=len(chunks),
            message=f"Ingested {len(saved_paths)} file(s) → {len(chunks)} chunks indexed.",
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url(
    url: Annotated[str, Query(description="Web URL to ingest")],
) -> IngestResponse:
    """Ingest a web page by URL into the vector store."""
    from app.ingestion import load_urls, chunk_documents, add_documents_to_vectorstore

    try:
        docs = load_urls([url])
        chunks = chunk_documents(docs)
        add_documents_to_vectorstore(docs)
        return IngestResponse(
            status="success",
            chunks_indexed=len(chunks),
            message=f"Ingested URL '{url}' → {len(chunks)} chunks indexed.",
        )
    except Exception as exc:
        logger.exception("URL ingestion failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
