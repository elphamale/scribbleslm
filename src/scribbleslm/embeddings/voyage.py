"""Voyage public-path backend: voyage-context-3 (contextual) + voyage-3.5 fallback.

Hard egress guard (R-spec): embed_batch raises PrivacyViolation if ever handed a
private-sourced request — defense in depth behind the router.
R2: each document is embedded in exactly one call (document-atomic); for the
contextual model a document must fit the 32K-token window.
"""
from __future__ import annotations

import asyncio

import tiktoken

from ..config import Settings
from .base import Document, EmbeddingBackend, EmbedRequest, PrivacyViolation, Vector
from .dispatcher import Dispatcher

CONTEXT_WINDOW_TOKENS = 30_000  # margin under voyage-context-3's 32K hard limit


class VoyageBackend(EmbeddingBackend):
    def __init__(self, settings: Settings, dispatcher: Dispatcher):
        self._settings = settings
        self._dispatcher = dispatcher
        self.backend_id = settings.voyage_model
        self._contextual = settings.voyage_model.startswith("voyage-context")
        self._dim: int | None = None
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._client = None  # lazy

    @property
    def dimension(self) -> int | None:
        return self._dim

    def _get_client(self):
        if self._client is None:
            if not self._settings.voyage_api_key:
                raise RuntimeError(
                    "VOYAGE_API_KEY is not set (env or ~/.scribbleslm/voyage_key). "
                    "Required for the public embedding path."
                )
            import voyageai
            self._client = voyageai.Client(api_key=self._settings.voyage_api_key)
        return self._client

    def _doc_tokens(self, doc: Document) -> int:
        return sum(len(self._enc.encode(c)) for c in doc)

    async def embed_batch(self, req: EmbedRequest) -> list[list[Vector]]:
        if req.private:
            raise PrivacyViolation(
                f"private-sourced content (source_id={req.source_id}) routed to "
                f"Voyage backend '{self.backend_id}' — egress guard tripped"
            )
        out: list[list[Vector]] = []
        for doc in req.documents:
            tok = self._doc_tokens(doc)
            if self._contextual and tok > CONTEXT_WINDOW_TOKENS:
                raise ValueError(
                    f"document of {tok} tokens exceeds the {CONTEXT_WINDOW_TOKENS} "
                    f"context-3 window — window it via the chunker before embedding"
                )
            vecs = await self._dispatcher.run(lambda d=doc: self._embed_doc(d), tokens=tok)
            if self._dim is None and vecs:
                self._dim = len(vecs[0])
            out.append(vecs)
        return out

    async def _embed_doc(self, doc: Document) -> list[Vector]:
        client = self._get_client()
        if self._contextual:
            r = await asyncio.to_thread(
                client.contextualized_embed, inputs=[doc],
                model=self.backend_id, input_type="document",
            )
            return [list(e) for e in r.results[0].embeddings]
        r = await asyncio.to_thread(
            client.embed, doc, model=self.backend_id, input_type="document",
        )
        return [list(e) for e in r.embeddings]

    async def embed_query(self, text: str) -> Vector:
        client = self._get_client()
        if self._contextual:
            r = await asyncio.to_thread(
                client.contextualized_embed, inputs=[[text]],
                model=self.backend_id, input_type="query",
            )
            vec = list(r.results[0].embeddings[0])
        else:
            r = await asyncio.to_thread(
                client.embed, [text], model=self.backend_id, input_type="query",
            )
            vec = list(r.embeddings[0])
        if self._dim is None:
            self._dim = len(vec)
        return vec
