import os
from pathlib import Path


def fetch_url(url: str) -> str:
    import trafilatura
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Failed to fetch URL: {url}")
    text = trafilatura.extract(downloaded)
    if not text:
        raise ValueError(f"Failed to extract text from URL: {url}")
    return text


def fetch_pdf(path: str) -> str:
    import fitz  # pymupdf
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts)


def fetch_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def fetch_source(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return fetch_url(url)
    lower = url.lower()
    if lower.endswith(".pdf"):
        return fetch_pdf(url)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return fetch_text(url)
    ext = Path(url).suffix or "(none)"
    raise ValueError(
        f"Unsupported source type: extension {ext}. "
        "Supported: http/https URLs, .pdf, .txt, .md"
    )
