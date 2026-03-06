"""LLM generation layer — builds prompts from retrieved context and streams or returns grounded answers."""

from __future__ import annotations

from typing import AsyncIterator, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.retriever import RetrievedChunk


SYSTEM_PROMPT = (
    "You are DocuRAG — a precise technical documentation assistant. "
    "Answer questions ONLY using the provided context. "
    "If the answer cannot be found in the context, respond exactly with: "
    "'Not found in documents.' "
    "Always cite your sources at the end using the format: "
    "Sources: <document_name> (page <page_number>)."
)

HUMAN_PROMPT_TEMPLATE = """\
Context:
{context}

Question:
{question}

Instructions:
- Answer only using the information from the context above.
- Be concise but complete.
- If the context does not contain the answer, say "Not found in documents."
- After your answer, list the sources you used, one per line, in this exact format:
  Sources:
  - <document_name> (page <page_number>)
"""


def build_llm(streaming: bool = False) -> ChatOpenAI:
    """Instantiate the configured LLM."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        openai_api_key=settings.openai_api_key,
        streaming=streaming,
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Convert retrieved chunks into a numbered context block."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk.metadata.get("source_document", "unknown")
        page = chunk.metadata.get("page_number", "?")
        section = chunk.metadata.get("section", "")
        header = f"[{i}] {src} | page {page}"
        if section:
            header += f" | section: {section}"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def extract_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Return a de-duplicated list of source strings from retrieved chunks."""
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        src = chunk.metadata.get("source_document", "unknown")
        page = chunk.metadata.get("page_number", "?")
        label = f"{src} (page {page})"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


class RAGGenerator:
    """Generates grounded answers using an LLM and retrieved context."""

    def __init__(self, streaming: bool = False):
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT_TEMPLATE),
            ]
        )
        self._llm = build_llm(streaming=streaming)
        self._chain = self._prompt | self._llm | StrOutputParser()

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> dict[str, str | list[str]]:
        """Generate a grounded answer for *query* given retrieved *chunks*."""
        if not chunks:
            logger.warning("No context chunks provided — returning fallback answer.")
            return {
                "answer": "Not found in documents.",
                "sources": [],
            }

        context = format_context(chunks)
        sources = extract_sources(chunks)
        logger.debug(f"Invoking LLM with {len(chunks)} context chunks.")

        answer = self._chain.invoke({"context": context, "question": query})
        return {"answer": answer, "sources": sources}

    def stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> Iterator[str]:
        """Yield answer tokens incrementally (streaming mode)."""
        if not chunks:
            yield "Not found in documents."
            return

        context = format_context(chunks)
        for token in self._chain.stream({"context": context, "question": query}):
            yield token

    async def astream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> AsyncIterator[str]:
        """Async token streaming."""
        if not chunks:
            yield "Not found in documents."
            return

        context = format_context(chunks)
        async for token in self._chain.astream({"context": context, "question": query}):
            yield token
