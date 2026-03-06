"""Pydantic-settings configuration for DocuRAG. All values can be overridden via .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide configuration resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="Chat model name")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1)

    # embeddings
    embedding_provider: Literal["openai", "huggingface"] = Field(
        default="huggingface",
        description="Embedding backend: 'openai' or 'huggingface'",
    )
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    hf_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace sentence-transformer model",
    )

    # chunking
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    # retrieval
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    rerank_top_n: int = Field(default=3, ge=1)
    reranker_model: str = Field(default="ms-marco-MiniLM-L-12-v2")

    # paths
    data_dir: Path = Field(default=BASE_DIR / "data" / "raw")
    vectorstore_dir: Path = Field(default=BASE_DIR / "vectorstore")
    log_dir: Path = Field(default=BASE_DIR / "logs")

    # langsmith
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="DocuRAG")

    # api
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1)
    cors_origins: list[str] = Field(default=["*"])

    @field_validator("data_dir", "vectorstore_dir", "log_dir", mode="before")
    @classmethod
    def _ensure_path(cls, v: str | Path) -> Path:
        return Path(v)

    def create_directories(self) -> None:
        for d in (self.data_dir, self.vectorstore_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.create_directories()
    return settings
