"""Document fetchers.

Security:
  - fetch_url: SSRF guard (private-IP blocklist, scheme allowlist, 30s timeout, size cap).
  - fetch_text / fetch_pdf: optional path-jail via documents_root (SCRIBBLESLM_DOCUMENTS_ROOT).
  - fetch_source: size cap applied after extraction (raw bytes → extracted text).
"""
from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from pathlib import Path


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

_PRIVATE_NETS = [
    ipaddress.ip_network(n) for n in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
        "169.254.0.0/16",   # link-local / AWS metadata
        "100.64.0.0/10",    # CGNAT
        "::1/128", "fc00::/7", "fe80::/10",
    )
]


def _is_private(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in _PRIVATE_NETS)
    except ValueError:
        pass
    try:
        ip = socket.gethostbyname(host)
        return _is_private(ip)
    except OSError:
        return False


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme '{parsed.scheme}' not allowed — use http or https")
    host = parsed.hostname or ""
    if not host:
        raise ValueError("URL has no host")
    if _is_private(host):
        raise ValueError(f"Requests to private/internal hosts are not allowed (host: {host})")


# ---------------------------------------------------------------------------
# Path jail
# ---------------------------------------------------------------------------

def _check_local_path(path: str, documents_root: Path | None) -> Path:
    p = Path(path).resolve()
    if documents_root is not None:
        root = documents_root.resolve()
        if not str(p).startswith(str(root) + "/") and p != root:
            raise ValueError(
                f"Path '{p}' is outside the allowed documents root '{root}'. "
                "Set SCRIBBLESLM_DOCUMENTS_ROOT to the directory you want to allow."
            )
    return p


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_url(url: str, max_bytes: int = 50 * 1024 * 1024, timeout: int = 30) -> str:
    _check_url(url)
    import trafilatura
    downloaded = trafilatura.fetch_url(url, no_ssl=False, config=None)
    if not downloaded:
        raise ValueError(f"Failed to fetch URL: {url}")
    if max_bytes and len(downloaded.encode()) > max_bytes:
        raise ValueError(
            f"Document at {url} exceeds the {max_bytes // (1024*1024)} MB size limit "
            "(set SCRIBBLESLM_MAX_DOCUMENT_BYTES to override)"
        )
    text = trafilatura.extract(downloaded)
    if not text:
        raise ValueError(f"Failed to extract text from URL: {url}")
    return text


def fetch_pdf(path: str, documents_root: Path | None = None,
              max_bytes: int = 50 * 1024 * 1024) -> str:
    p = _check_local_path(path, documents_root)
    if max_bytes and p.stat().st_size > max_bytes:
        raise ValueError(
            f"PDF '{p}' exceeds the {max_bytes // (1024*1024)} MB size limit "
            "(set SCRIBBLESLM_MAX_DOCUMENT_BYTES to override)"
        )
    import fitz  # pymupdf
    doc = fitz.open(str(p))
    parts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(parts)


def fetch_text(path: str, documents_root: Path | None = None,
               max_bytes: int = 50 * 1024 * 1024) -> str:
    p = _check_local_path(path, documents_root)
    if max_bytes and p.stat().st_size > max_bytes:
        raise ValueError(
            f"File '{p}' exceeds the {max_bytes // (1024*1024)} MB size limit "
            "(set SCRIBBLESLM_MAX_DOCUMENT_BYTES to override)"
        )
    return p.read_text(encoding="utf-8")


def fetch_source(url: str, documents_root: Path | None = None,
                 max_bytes: int = 50 * 1024 * 1024) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return fetch_url(url, max_bytes=max_bytes)
    lower = url.lower()
    if lower.endswith(".pdf"):
        return fetch_pdf(url, documents_root=documents_root, max_bytes=max_bytes)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return fetch_text(url, documents_root=documents_root, max_bytes=max_bytes)
    ext = Path(url).suffix or "(none)"
    raise ValueError(
        f"Unsupported source type: extension {ext}. "
        "Supported: http/https URLs, .pdf, .txt, .md"
    )
