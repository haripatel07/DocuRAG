"""RAG pipeline orchestrator — query validation, rewriting, classification, retrieval, and generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import get_settings
from app.generator import RAGGenerator
from app.retriever import DocumentRetriever, RetrievedChunk, build_retriever


_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query optimisation assistant. "
            "Rewrite the user's query to maximise retrieval precision "
            "in a technical documentation search. "
            "Return ONLY the rewritten query — no explanations.",
        ),
        ("human", "Original query: {query}"),
    ]
)

_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the following query into exactly one category: "
            "conceptual | implementation | troubleshooting. "
            "Return only the category word.",
        ),
        ("human", "Query: {query}"),
    ]
)

BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(drop table|delete from|insert into)\b", re.I),
    re.compile(r"<script", re.I),
    re.compile(r"ignore previous instructions", re.I),
]


def validate_query(query: str) -> str:
    """Raise ValueError if the query is empty or contains a blocked pattern."""
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if len(query) > 2000:
        raise ValueError("Query exceeds maximum length of 2000 characters.")
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(query):
            raise ValueError(f"Query contains a disallowed pattern: {pattern.pattern}")
    return query


class QueryProcessor:
    """Rewrites and classifies user queries using a lightweight LLM call."""

    def __init__(self):
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            openai_api_key=settings.openai_api_key,
        )
        self._rewrite_chain = _REWRITE_PROMPT | llm | StrOutputParser()
        self._classify_chain = _CLASSIFY_PROMPT | llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        """Return an expanded, retrieval-optimised version of *query*."""
        try:
            rewritten = self._rewrite_chain.invoke({"query": query}).strip()
            logger.debug(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as exc:
            logger.warning(f"Query rewriting failed ({exc}); using original.")
            return query

    def classify(self, query: str) -> str:
        """Return the query type: conceptual | implementation | troubleshooting."""
        try:
            category = self._classify_chain.invoke({"query": query}).strip().lower()
            if category not in {"conceptual", "implementation", "troubleshooting"}:
                category = "conceptual"
            logger.debug(f"Query classified as: {category}")
            return category
        except Exception as exc:
            logger.warning(f"Query classification failed ({exc}).")
            return "conceptual"


@dataclass
class PipelineResponse:
    """Structured response returned by the RAG pipeline."""

    answer: str
    sources: list[str] = field(default_factory=list)
    query_type: str = "conceptual"
    rewritten_query: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query_type": self.query_type,
            "rewritten_query": self.rewritten_query,
        }


class RAGPipeline:
    """Full RAG pipeline wiring query validation, rewriting, retrieval, and generation."""

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        generator: RAGGenerator | None = None,
        enable_query_rewriting: bool = True,
    ):
        self._retriever = retriever or build_retriever()
        self._generator = generator or RAGGenerator()
        self._query_processor = QueryProcessor()
        self._enable_rewriting = enable_query_rewriting

    # sync

    def run(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> PipelineResponse:
        """Execute the full RAG pipeline and return a *PipelineResponse*."""
        # 1. Guardrails
        query = validate_query(query)

        # 2. Query processing
        query_type = self._query_processor.classify(query)
        rewritten = (
            self._query_processor.rewrite(query)
            if self._enable_rewriting
            else query
        )

        # 3. Retrieval
        chunks = self._retriever.retrieve(
            rewritten, top_k=top_k, metadata_filter=metadata_filter
        )
        logger.info(
            f"Retrieved {len(chunks)} chunks for query: '{rewritten[:80]}…'"
        )

        # 4. Generation
        result = self._generator.generate(rewritten, chunks)

        return PipelineResponse(
            answer=result["answer"],
            sources=result["sources"],
            query_type=query_type,
            rewritten_query=rewritten,
            retrieved_chunks=chunks,
        )

    # streaming

    def stream(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> Iterator[str]:
        """Stream answer tokens; raises on validation failure."""
        query = validate_query(query)
        rewritten = (
            self._query_processor.rewrite(query)
            if self._enable_rewriting
            else query
        )
        chunks = self._retriever.retrieve(
            rewritten, top_k=top_k, metadata_filter=metadata_filter
        )
        yield from self._generator.stream(rewritten, chunks)

    async def astream(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming pipeline."""
        query = validate_query(query)
        rewritten = (
            self._query_processor.rewrite(query)
            if self._enable_rewriting
            else query
        )
        chunks = self._retriever.retrieve(
            rewritten, top_k=top_k, metadata_filter=metadata_filter
        )
        async for token in self._generator.astream(rewritten, chunks):
            yield token


_pipeline_instance: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return a module-level singleton *RAGPipeline* (lazy initialisation)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
