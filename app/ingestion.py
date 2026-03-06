"""Document ingestion pipeline: load → clean → chunk → embed → FAISS.

CLI usage:
    python -m app.ingestion --source data/raw
    python -m app.ingestion --url https://fastapi.tiangolo.com/tutorial/
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

from app.config import get_settings
from app.embeddings import build_embeddings


def _clean_text(text: str) -> str:
    """Normalise and lightly clean raw extracted text."""
    # Unicode normalisation (e.g. ligatures → ascii)
    text = unicodedata.normalize("NFKC", text)
    # Collapse excessive whitespace / blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Strip zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", text)
    return text.strip()


def _document_id(source: str, page: int | str) -> str:
    """Stable content-addressable ID for deduplication."""
    raw = f"{source}::{page}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_pdfs(directory: Path) -> list[Document]:
    """Recursively load all PDF files from *directory*."""
    docs: list[Document] = []
    pdf_files = list(directory.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF file(s) in {directory}")
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata.setdefault("source_document", pdf_path.name)
            page.metadata.setdefault("page_number", page.metadata.get("page", 0))
            page.metadata.setdefault("section", "")
            page.page_content = _clean_text(page.page_content)
        docs.extend([p for p in pages if p.page_content])
    return docs


def load_markdown(directory: Path) -> list[Document]:
    """Recursively load all Markdown files from *directory*."""
    md_files = list(directory.rglob("*.md")) + list(directory.rglob("*.mdx"))
    logger.info(f"Found {len(md_files)} Markdown file(s) in {directory}")
    docs: list[Document] = []
    for md_path in tqdm(md_files, desc="Loading Markdown"):
        loader = UnstructuredMarkdownLoader(str(md_path), mode="elements")
        elements = loader.load()
        for el in elements:
            el.metadata.setdefault("source_document", md_path.name)
            el.metadata.setdefault("page_number", el.metadata.get("page_number", 1))
            el.metadata.setdefault("section", el.metadata.get("category", ""))
            el.page_content = _clean_text(el.page_content)
        docs.extend([e for e in elements if e.page_content])
    return docs


def load_urls(urls: list[str]) -> list[Document]:
    """Fetch and load documents from a list of web URLs."""
    logger.info(f"Loading {len(urls)} URL(s)")
    loader = WebBaseLoader(urls)
    raw_docs = loader.load()
    docs: list[Document] = []
    for doc in raw_docs:
        doc.metadata.setdefault("source_document", doc.metadata.get("source", "web"))
        doc.metadata.setdefault("page_number", 1)
        doc.metadata.setdefault("section", doc.metadata.get("title", ""))
        doc.page_content = _clean_text(doc.page_content)
        if doc.page_content:
            docs.append(doc)
    return docs


def load_directory(directory: Path) -> list[Document]:
    """Load all supported document types from *directory*."""
    docs: list[Document] = []
    docs.extend(load_pdfs(directory))
    docs.extend(load_markdown(directory))
    logger.info(f"Total raw documents loaded: {len(docs)}")
    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping text chunks."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    # Attach stable IDs to metadata
    for chunk in chunks:
        src = chunk.metadata.get("source_document", "unknown")
        page = chunk.metadata.get("page_number", 0)
        chunk.metadata["chunk_id"] = _document_id(src, page)
    logger.info(f"Total chunks after splitting: {len(chunks)}")
    return chunks


def build_vectorstore(
    chunks: list[Document],
    persist_path: Path | None = None,
) -> FAISS:
    """Embed *chunks* and save a FAISS index to *persist_path*."""
    settings = get_settings()
    persist_path = persist_path or settings.vectorstore_dir

    embeddings = build_embeddings()
    logger.info("Generating embeddings and building FAISS index …")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(persist_path))
    logger.info(f"FAISS index saved to {persist_path}")
    return vectorstore


def load_vectorstore(persist_path: Path | None = None) -> FAISS:
    """Load an existing FAISS index from disk."""
    settings = get_settings()
    persist_path = persist_path or settings.vectorstore_dir
    embeddings = build_embeddings()
    vectorstore = FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info(f"FAISS index loaded from {persist_path}")
    return vectorstore


def add_documents_to_vectorstore(
    new_docs: list[Document],
    persist_path: Path | None = None,
) -> FAISS:
    """Incrementally add new documents to an existing FAISS index."""
    settings = get_settings()
    persist_path = persist_path or settings.vectorstore_dir

    chunks = chunk_documents(new_docs)

    index_path = persist_path / "index.faiss"
    if index_path.exists():
        vectorstore = load_vectorstore(persist_path)
        vectorstore.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to existing index.")
    else:
        vectorstore = FAISS.from_documents(chunks, build_embeddings())
        logger.info(f"Created new index with {len(chunks)} chunks.")

    vectorstore.save_local(str(persist_path))
    return vectorstore


def run_ingestion_pipeline(
    source_dir: Path | None = None,
    urls: list[str] | None = None,
    persist_path: Path | None = None,
) -> FAISS:
    """Run the full ingestion pipeline: load, chunk, embed, and save to FAISS."""
    settings = get_settings()
    source_dir = source_dir or settings.data_dir

    all_docs: list[Document] = []
    if source_dir.exists():
        all_docs.extend(load_directory(source_dir))
    if urls:
        all_docs.extend(load_urls(urls))

    if not all_docs:
        raise ValueError(
            "No documents found. Provide a non-empty source directory or URLs."
        )

    chunks = chunk_documents(all_docs)
    vectorstore = build_vectorstore(chunks, persist_path)
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocuRAG ingestion pipeline")
    parser.add_argument("--source", type=Path, help="Directory with raw documents")
    parser.add_argument("--url", action="append", dest="urls", help="URL(s) to ingest")
    parser.add_argument("--persist", type=Path, help="Path to save the FAISS index")
    args = parser.parse_args()

    run_ingestion_pipeline(
        source_dir=args.source,
        urls=args.urls,
        persist_path=args.persist,
    )
