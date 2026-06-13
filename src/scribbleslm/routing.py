"""Sensitivity routing + hard guards.

- Per-document routing by `private` flag (default from settings.default_private).
- Egress guard: enforced inside VoyageBackend.embed_batch (PrivacyViolation);
  the router never routes private content to a remote backend in the first place.
- Query-leak guard: a sensitive query is never embedded by a remote backend —
  only local (private) spaces + FTS5 are searched, and excluded remote spaces
  are reported.
- Every routing decision is logged for audit.
"""
from __future__ import annotations

import logging

from .config import Settings
from .embeddings.base import Document, EmbeddingBackend, EmbedRequest, Vector

log = logging.getLogger("scribbleslm.routing")


class Router:
    def __init__(self, settings: Settings, voyage_backend: EmbeddingBackend,
                 private_backend: EmbeddingBackend):
        self.settings = settings
        self.voyage = voyage_backend
        self.private = private_backend

    def effective_private(self, private: bool | None) -> bool:
        return self.settings.default_private if private is None else private

    def is_remote(self, backend_id: str) -> bool:
        return backend_id == self.voyage.backend_id or backend_id.startswith("voyage")

    # ---- ingestion routing ----------------------------------------------
    async def embed_source(
        self, source_id: int | None, private: bool | None, documents: list[Document]
    ) -> tuple[list[list[Vector]], str, bool]:
        """Route a source's documents to the correct backend, audit, embed.
        Returns (per-doc vectors, backend_id, effective_private)."""
        eff = self.effective_private(private)
        backend = self.private if eff else self.voyage
        log.info("route ingest source_id=%s backend=%s private=%s docs=%d",
                 source_id, backend.backend_id, eff, len(documents))
        vecs = await backend.embed_batch(
            EmbedRequest(documents=documents, private=eff, source_id=source_id)
        )
        return vecs, backend.backend_id, eff

    # ---- query routing (leak guard) -------------------------------------
    def query_plan(self, present_backend_ids: list[str], query_private: bool | None
                   ) -> tuple[list[str], list[str]]:
        """Given the backend_ids present in a notebook and the query's sensitivity,
        return (searched_backend_ids, excluded_remote_backend_ids)."""
        eff = self.effective_private(query_private)
        if eff:
            searched = [b for b in present_backend_ids if not self.is_remote(b)]
            excluded = [b for b in present_backend_ids if self.is_remote(b)]
            if excluded:
                log.info("query-leak guard: sensitive query excludes remote spaces %s", excluded)
            return searched, excluded
        return list(present_backend_ids), []

    async def embed_query(self, backend_id: str, text: str, query_private: bool | None
                          ) -> Vector:
        """Embed a query for a specific present backend_id, enforcing the leak guard."""
        if self.effective_private(query_private) and self.is_remote(backend_id):
            raise RuntimeError(
                f"query-leak guard: refusing to embed a sensitive query via remote "
                f"backend '{backend_id}'"
            )
        backend = self.voyage if self.is_remote(backend_id) else self.private
        return await backend.embed_query(text)
