"""FAISS-backed semantic retriever with optional metadata filtering and FlashRank reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from app.config import get_settings
from app.ingestion import load_vectorstore


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its relevance score."""

    content: str
    metadata: dict[str, Any]
    score: float = 0.0

    @property
    def source(self) -> str:
        doc = self.metadata.get("source_document", "unknown")
        page = self.metadata.get("page_number", "?")
        return f"{doc} (page {page})"

    @classmethod
    def from_document(cls, doc: Document, score: float = 0.0) -> "RetrievedChunk":
        return cls(content=doc.page_content, metadata=doc.metadata, score=score)


class FlashRankReranker:
    """Lightweight cross-encoder reranker using FlashRank."""

    def __init__(self, model: str | None = None, top_n: int | None = None):
        settings = get_settings()
        self._top_n = top_n or settings.rerank_top_n
        try:
            from flashrank import Ranker, RerankRequest

            self._ranker = Ranker(model_name=model or settings.reranker_model)
            self._RerankRequest = RerankRequest
            self._enabled = True
            logger.info(f"FlashRank reranker loaded: {settings.reranker_model}")
        except Exception as exc:
            logger.warning(f"FlashRank unavailable ({exc}); skipping reranking.")
            self._enabled = False

    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not self._enabled or not chunks:
            return chunks[: self._top_n]

        passages = [
            {"id": i, "text": c.content, "meta": c.metadata}
            for i, c in enumerate(chunks)
        ]
        req = self._RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(req)

        reranked: list[RetrievedChunk] = []
        for r in results[: self._top_n]:
            original = chunks[r["id"]]
            reranked.append(
                RetrievedChunk(
                    content=original.content,
                    metadata=original.metadata,
                    score=float(r.get("score", original.score)),
                )
            )
        return reranked


class DocumentRetriever:
    """Semantic retriever backed by FAISS with optional metadata filtering and reranking."""

    def __init__(
        self,
        vectorstore: FAISS | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
        persist_path: Path | None = None,
    ):
        settings = get_settings()
        self._top_k = top_k or settings.retrieval_top_k
        self._vectorstore = vectorstore or load_vectorstore(persist_path)
        self._reranker = FlashRankReranker(top_n=rerank_top_n or settings.rerank_top_n)

    # public

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top reranked chunks for *query*, optionally filtered by metadata."""
        k = top_k or self._top_k
        raw_results = self._similarity_search(query, k, metadata_filter)
        logger.debug(f"FAISS returned {len(raw_results)} candidate chunks.")

        chunks = [RetrievedChunk.from_document(doc, score) for doc, score in raw_results]
        reranked = self._reranker.rerank(query, chunks)
        logger.debug(f"After reranking: {len(reranked)} chunks selected.")
        return reranked

    # internal

    def _similarity_search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[tuple[Document, float]]:
        try:
            if metadata_filter:
                return self._vectorstore.similarity_search_with_score(
                    query, k=k, filter=metadata_filter
                )
            return self._vectorstore.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            raise


def build_retriever(
    vectorstore: FAISS | None = None,
    top_k: int | None = None,
) -> DocumentRetriever:
    """Factory that wires together a ready-to-use *DocumentRetriever*."""
    return DocumentRetriever(vectorstore=vectorstore, top_k=top_k)
