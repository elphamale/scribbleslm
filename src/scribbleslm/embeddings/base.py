"""Embedding backend interface.

R2 (hard contract): the unit of embedding is a *document* — an ordered list of
chunks. `embed_batch` embeds each document atomically; a single document's chunks
are NEVER split across separate backend calls (splitting silently degrades
voyage-context-3's contextual quality with no error). For contextual models a
document must fit the model's context window (the chunker is responsible for
window-aligned segmentation; see Milestone B parallel-partitioning backlog item).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

Vector = list[float]
Document = list[str]  # ordered chunks that share one contextualization unit


@dataclass
class EmbedRequest:
    """Carries privacy intent alongside the documents so backends can enforce
    the hard egress guard at their own boundary (defense in depth)."""
    documents: list[Document]
    private: bool
    source_id: int | None = None


class PrivacyViolation(RuntimeError):
    """Raised if private-sourced content is ever routed to a remote backend."""


class EmbeddingBackend(ABC):
    backend_id: str  # identity string pinned per chunk, e.g. "voyage-context-3"

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Vector dim; may be None until the first successful embed."""

    @abstractmethod
    async def embed_batch(self, req: EmbedRequest) -> list[list[Vector]]:
        """Embed each document atomically. Returns per-document, per-chunk vectors
        in the same order as req.documents / each Document."""

    @abstractmethod
    async def embed_query(self, text: str) -> Vector:
        """Embed a single query string in this backend's vector space."""
