from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

import fitz

from app.config import settings


def sanitize_text(value: str) -> str:
    # PostgreSQL text/json fields reject embedded NUL bytes.
    return (value or "").replace("\x00", " ")


def compute_file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_local_file(file_path: str) -> Dict[str, Any]:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    suffix = path.suffix.lower()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parser = "text"
    chunks: List[Dict[str, Any]]
    page_count = 0

    if suffix == ".pdf":
        parser = "pymupdf"
        chunks, page_count = _parse_pdf(path)
    elif suffix in {".md", ".markdown", ".txt"}:
        chunks = _chunk_text(path.read_text(encoding="utf-8", errors="ignore"), path.name)
    else:
        raise ValueError(f"Unsupported file type: {suffix or path.name}")

    return {
        "filename": path.name,
        "file_path": str(path),
        "file_hash": compute_file_hash(path),
        "media_type": media_type,
        "parser": parser,
        "page_count": page_count,
        "chunks": chunks,
    }


def parse_file_bytes(filename: str, data: bytes) -> Dict[str, Any]:
    """Parse PDF or text from uploaded bytes (no local filesystem path on the server)."""
    name = Path(filename).name or "upload"
    suffix = Path(name).suffix.lower()
    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    parser = "text"
    chunks: List[Dict[str, Any]]
    page_count = 0

    if suffix == ".pdf":
        parser = "pymupdf"
        document = fitz.open(stream=data, filetype="pdf")
        try:
            chunks = []
            next_chunk_index = 0
            for page_index, page in enumerate(document):
                text = sanitize_text(page.get_text("text")).strip()
                if not text:
                    continue
                page_chunks = _chunk_text(
                    text,
                    name,
                    page_index=page_index + 1,
                    start_chunk_index=next_chunk_index,
                )
                chunks.extend(page_chunks)
                next_chunk_index += len(page_chunks)
            page_count = document.page_count
        finally:
            document.close()
    elif suffix in {".md", ".markdown", ".txt"}:
        text = data.decode("utf-8", errors="ignore")
        chunks = _chunk_text(text, name)
    else:
        raise ValueError(f"Unsupported file type: {suffix or name}")

    return {
        "filename": name,
        "file_path": f"upload:{name}",
        "file_hash": compute_bytes_hash(data),
        "media_type": media_type,
        "parser": parser,
        "page_count": page_count,
        "chunks": chunks,
    }


def _parse_pdf(path: Path) -> tuple[List[Dict[str, Any]], int]:
    document = fitz.open(path)
    chunks: List[Dict[str, Any]] = []
    try:
        next_chunk_index = 0
        for page_index, page in enumerate(document):
            text = sanitize_text(page.get_text("text")).strip()
            if not text:
                continue
            page_chunks = _chunk_text(
                text,
                path.name,
                page_index=page_index + 1,
                start_chunk_index=next_chunk_index,
            )
            chunks.extend(page_chunks)
            next_chunk_index += len(page_chunks)
        return chunks, document.page_count
    finally:
        document.close()


def _chunk_text(
    text: str,
    source_name: str,
    page_index: int | None = None,
    start_chunk_index: int = 0,
) -> List[Dict[str, Any]]:
    normalized = " ".join(line.strip() for line in sanitize_text(text).splitlines() if line.strip())
    if not normalized:
        return []
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            metadata = {
                "chunk_index": start_chunk_index + chunk_index,
                "source_name": source_name,
            }
            if page_index is not None:
                metadata["page"] = page_index
            chunks.append({"text": chunk_text, "metadata": metadata})
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
        chunk_index += 1
    return chunks
