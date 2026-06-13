"""R2: voyage-context-3 embedding is document-atomic — one call per document,
a document's chunks never split across calls. Plus the egress guard."""
import pytest

from scribbleslm.config import get_settings, reset_settings_cache
from scribbleslm.embeddings.base import EmbedRequest, PrivacyViolation
from scribbleslm.embeddings.dispatcher import Dispatcher
from scribbleslm.embeddings.voyage import VoyageBackend


class _FakeCtxResult:
    def __init__(self, embs):
        self.results = [type("R", (), {"embeddings": embs})()]


class FakeVoyageClient:
    def __init__(self):
        self.calls = []

    def contextualized_embed(self, inputs, model, input_type):
        self.calls.append(("ctx", inputs, input_type))
        return _FakeCtxResult([[0.1, 0.2, 0.3, 0.4] for _ in inputs[0]])

    def embed(self, texts, model, input_type):
        self.calls.append(("emb", texts, input_type))
        return type("E", (), {"embeddings": [[0.0] * 4 for _ in texts]})()


def _backend(monkeypatch, model="voyage-context-3"):
    monkeypatch.setenv("VOYAGE_MODEL", model)
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    reset_settings_cache()
    be = VoyageBackend(get_settings(), Dispatcher(wait_initial=0.001))
    fake = FakeVoyageClient()
    be._client = fake  # inject, bypassing real client
    return be, fake


async def test_document_atomic_one_call_per_doc(monkeypatch):
    be, fake = _backend(monkeypatch)
    docs = [["a", "b"], ["c", "d", "e"]]
    out = await be.embed_batch(EmbedRequest(documents=docs, private=False))
    ctx_calls = [c for c in fake.calls if c[0] == "ctx"]
    # exactly one call per document, each carrying that doc's chunks as ONE inner list
    assert len(ctx_calls) == 2
    assert ctx_calls[0][1] == [["a", "b"]]
    assert ctx_calls[1][1] == [["c", "d", "e"]]
    # shape preserved: per-doc, per-chunk vectors
    assert len(out) == 2 and len(out[0]) == 2 and len(out[1]) == 3


async def test_egress_guard_blocks_private(monkeypatch):
    be, fake = _backend(monkeypatch)
    with pytest.raises(PrivacyViolation):
        await be.embed_batch(EmbedRequest(documents=[["x"]], private=True, source_id=9))
    assert fake.calls == []  # nothing sent to the client


async def test_oversize_document_raises(monkeypatch):
    be, fake = _backend(monkeypatch)
    big = "слово " * 20000  # well over the 30K-token context-3 window
    with pytest.raises(ValueError):
        await be.embed_batch(EmbedRequest(documents=[[big]], private=False))


async def test_query_embedding(monkeypatch):
    be, fake = _backend(monkeypatch)
    vec = await be.embed_query("що таке вбивство")
    assert len(vec) == 4
    assert any(c[2] == "query" for c in fake.calls)
