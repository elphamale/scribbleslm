"""Private-path backend: local bge-m3 GGUF via llama-cpp-python.

In-process, no daemon: the model loads lazily on first private embed and is
released with the server session (call .unload()). Content never leaves the host.

Sparse note: llama.cpp exposes only DENSE pooled embeddings — bge-m3's SPLADE-style
sparse weights are not available through this path (verified at build). The lexical
channel is therefore FTS5 for every notebook; do not claim a native sparse channel.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ..config import Settings
from .base import EmbeddingBackend, EmbedRequest, Vector


class BgeM3GgufBackend(EmbeddingBackend):
    def __init__(self, settings: Settings):
        self._path: Path = settings.private_model_path
        self._threads = settings.private_threads
        self._n_ctx = 8192
        self.backend_id = "bge-m3-gguf-" + self._quant_suffix(self._path)
        self._llama = None
        self._dim: int | None = None

    @staticmethod
    def _quant_suffix(path: Path) -> str:
        # bge-m3-Q5_K_M.gguf -> q5_k_m
        stem = path.stem
        return stem.split("-")[-1].lower() if "-" in stem else stem.lower()

    @property
    def dimension(self) -> int | None:
        return self._dim

    def _load(self):
        if self._llama is None:
            if not self._path.exists():
                raise RuntimeError(
                    f"private embedding model not found at {self._path}. "
                    f"Download a bge-m3 GGUF (e.g. gpustack/bge-m3-GGUF) or set "
                    f"PRIVATE_EMBEDDING_MODEL_PATH."
                )
            from llama_cpp import Llama
            self._llama = Llama(
                model_path=str(self._path),
                embedding=True,
                n_ctx=self._n_ctx,
                n_threads=self._threads,
                verbose=False,
            )
        return self._llama

    def unload(self) -> None:
        """Release the model (footprint returns to ~0 between sessions)."""
        self._llama = None

    def _embed_texts(self, texts: list[str]) -> list[Vector]:
        m = self._load()
        res = m.create_embedding(texts)
        return [list(d["embedding"]) for d in res["data"]]

    async def embed_batch(self, req: EmbedRequest) -> list[list[Vector]]:
        # local path: private content is expected here; no egress guard needed.
        out: list[list[Vector]] = []
        for doc in req.documents:
            vecs = await asyncio.to_thread(self._embed_texts, doc)
            if self._dim is None and vecs:
                self._dim = len(vecs[0])
            out.append(vecs)
        return out

    async def embed_query(self, text: str) -> Vector:
        vecs = await asyncio.to_thread(self._embed_texts, [text])
        if self._dim is None and vecs:
            self._dim = len(vecs[0])
        return vecs[0]
