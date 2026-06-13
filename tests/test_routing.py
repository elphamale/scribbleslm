"""Sensitivity routing + guards."""
import pytest

from scribbleslm.config import get_settings, reset_settings_cache
from scribbleslm.embeddings.base import EmbeddingBackend, EmbedRequest, PrivacyViolation
from scribbleslm.routing import Router


class FakeBackend(EmbeddingBackend):
    def __init__(self, backend_id, private_ok):
        self.backend_id = backend_id
        self._private_ok = private_ok

    @property
    def dimension(self):
        return 4

    async def embed_batch(self, req: EmbedRequest):
        if not self._private_ok and req.private:
            raise PrivacyViolation("private content reached a remote backend")
        return [[[0.0] * 4 for _ in doc] for doc in req.documents]

    async def embed_query(self, text):
        return [0.0] * 4


def _router(monkeypatch, default_private=False):
    monkeypatch.setenv("DEFAULT_PRIVATE", "1" if default_private else "0")
    reset_settings_cache()
    voyage = FakeBackend("voyage-context-3", private_ok=False)
    private = FakeBackend("bge-m3-gguf-q5_k_m", private_ok=True)
    return Router(get_settings(), voyage, private), voyage, private


async def test_public_routes_to_voyage(monkeypatch):
    r, _, _ = _router(monkeypatch)
    _, bid, eff = await r.embed_source(1, False, [["a"]])
    assert bid == "voyage-context-3" and eff is False


async def test_private_routes_to_local(monkeypatch):
    r, _, _ = _router(monkeypatch)
    _, bid, eff = await r.embed_source(2, True, [["a"]])
    assert bid == "bge-m3-gguf-q5_k_m" and eff is True


async def test_default_private_env_controls_unset_flag(monkeypatch):
    r, _, _ = _router(monkeypatch, default_private=True)
    _, bid, eff = await r.embed_source(3, None, [["a"]])
    assert eff is True and bid.startswith("bge-m3")


def test_query_leak_guard_excludes_remote_for_sensitive(monkeypatch):
    r, _, _ = _router(monkeypatch)
    present = ["voyage-context-3", "bge-m3-gguf-q5_k_m"]
    searched, excluded = r.query_plan(present, query_private=True)
    assert excluded == ["voyage-context-3"]
    assert searched == ["bge-m3-gguf-q5_k_m"]
    # non-sensitive query searches everything
    searched2, excluded2 = r.query_plan(present, query_private=False)
    assert excluded2 == [] and set(searched2) == set(present)


async def test_embed_query_guard_refuses_remote_for_sensitive(monkeypatch):
    r, _, _ = _router(monkeypatch)
    with pytest.raises(RuntimeError):
        await r.embed_query("voyage-context-3", "secret query", query_private=True)
    # local backend is allowed for a sensitive query
    assert await r.embed_query("bge-m3-gguf-q5_k_m", "secret query", query_private=True) == [0.0] * 4
