"""Embedding factory — returns and caches the configured embedding backend."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from loguru import logger

from app.config import get_settings


@lru_cache(maxsize=1)
def build_embeddings() -> Embeddings:
    """Return a cached embedding model based on *EMBEDDING_PROVIDER*."""
    settings = get_settings()

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info(f"Using OpenAI embeddings: {settings.openai_embedding_model}")
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    # Default: HuggingFace / sentence-transformers (free, local)
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(f"Using HuggingFace embeddings: {settings.hf_embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
