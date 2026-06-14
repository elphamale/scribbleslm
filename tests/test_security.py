"""Security and correctness fixes: SEC-1 SSRF, SEC-2 path traversal, SEC-3 size cap,
BUG-1/2/3 source_refresh, BUG-4 window guard on fallback model."""
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scribbleslm.sources import _check_url, fetch_source, fetch_text


# ── SEC-1: SSRF guard ──────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:5984/_all_dbs",
    "http://localhost/admin",
    "http://10.0.0.1/internal",
    "http://192.168.1.1/router",
    "http://172.16.0.1/meta",
    "http://169.254.169.254/latest/meta-data",   # AWS metadata
    "ftp://example.com/file",
    "file:///etc/passwd",
])
def test_ssrf_blocked(url):
    with pytest.raises(ValueError, match="not allowed|scheme"):
        _check_url(url)


def test_public_url_allowed():
    _check_url("https://example.com/page")  # must not raise


# ── SEC-2: path traversal / path jail ─────────────────────────────────────

def test_path_traversal_blocked_with_root():
    root = Path(tempfile.mkdtemp())
    with pytest.raises(ValueError, match="outside the allowed"):
        fetch_text("/etc/passwd", documents_root=root)


def test_path_within_root_allowed():
    root = Path(tempfile.mkdtemp())
    f = root / "doc.txt"
    f.write_text("hello")
    assert fetch_text(str(f), documents_root=root) == "hello"


def test_no_root_allows_any_path():
    f = Path(tempfile.mktemp(suffix=".txt"))
    f.write_text("ok")
    assert fetch_text(str(f), documents_root=None) == "ok"
    f.unlink()


# ── SEC-3: document size cap ──────────────────────────────────────────────

def test_local_file_size_cap():
    f = Path(tempfile.mktemp(suffix=".txt"))
    f.write_bytes(b"x" * 200)
    with pytest.raises(ValueError, match="size limit"):
        fetch_text(str(f), max_bytes=100)
    f.unlink()


# ── BUG-4: window guard applies to all models (not only contextual) ────────

async def test_window_guard_applies_to_fallback_model(monkeypatch):
    from scribbleslm.config import get_settings, reset_settings_cache
    from scribbleslm.embeddings.base import EmbedRequest
    from scribbleslm.embeddings.voyage import CONTEXT_WINDOW_TOKENS, VoyageBackend
    from scribbleslm.embeddings.dispatcher import Dispatcher

    monkeypatch.setenv("VOYAGE_API_KEY", "test")
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-4-lite")
    reset_settings_cache()
    settings = get_settings()
    backend = VoyageBackend(settings, Dispatcher())
    assert not backend._contextual   # confirm it's the non-contextual path

    big_chunk = " ".join(["word"] * (CONTEXT_WINDOW_TOKENS + 100))
    req = EmbedRequest(documents=[[big_chunk]], private=False)
    with pytest.raises(ValueError, match="ceiling"):
        await backend.embed_batch(req)
    reset_settings_cache()


# ── BUG-1/2/3: source_refresh import + hash + enrichment ─────────────────

async def test_source_refresh_callable_without_import_error():
    """BUG-1: source_refresh must not raise ImportError (chunk_text → build_chunks)."""
    from scribbleslm import server as S
    import tempfile, os
    os.environ["SCRIBBLESLM_DB_PATH"] = tempfile.mktemp(suffix=".db")
    # Call with non-existent ids — we want to confirm it returns {"error": ...}
    # not crash with ImportError
    result = await S.source_refresh(notebook_id=999, source_id=999)
    assert "error" in result and "ImportError" not in result["error"]
    try:
        os.remove(os.environ["SCRIBBLESLM_DB_PATH"])
    except OSError:
        pass
