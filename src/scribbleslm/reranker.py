"""Sensitivity-routed reranker (Milestone C, DEFAULT OFF).

- PUBLIC query/candidates -> Voyage rerank-2.5 (API).
- PRIVATE -> local bge-reranker-v2-m3 run in a SELF-EXITING SUBPROCESS
  (scribbleslm.reranker_worker), spawned on first use, exits after idle. The main
  process imports nothing heavy here — only `subprocess` — preserving zero idle RSS.

Egress guard: the Voyage path raises if handed any private-sourced candidate.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Settings
from .embeddings.base import PrivacyViolation

_PKG_PARENT = str(Path(__file__).resolve().parent.parent)  # dir containing `scribbleslm`


class LocalReranker:
    """Manages the self-exiting subprocess worker. Never imports the model itself."""

    def __init__(self, settings: Settings):
        self.model = settings.reranker_model
        self.idle = settings.reranker_idle_exit
        self._proc: subprocess.Popen | None = None

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:  # never spawned, or self-exited
            env = os.environ.copy()
            env["RERANKER_MODEL"] = self.model
            env["RERANKER_IDLE_EXIT"] = str(self.idle)
            env["PYTHONPATH"] = _PKG_PARENT + os.pathsep + env.get("PYTHONPATH", "")
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "scribbleslm.reranker_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        return self._proc

    def score(self, query: str, docs: list[str]) -> list[float]:
        for _ in range(2):  # respawn once if the worker had self-exited between bursts
            proc = self._ensure_proc()
            try:
                proc.stdin.write(json.dumps({"query": query, "documents": docs}) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise BrokenPipeError("reranker worker closed")
                resp = json.loads(line)
                if "error" in resp:
                    raise RuntimeError(resp["error"])
                return resp["scores"]
            except (BrokenPipeError, OSError, ValueError):
                self._proc = None
        raise RuntimeError("local reranker subprocess failed")

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None


class VoyageReranker:
    def __init__(self, settings: Settings):
        self._settings = settings
        self.model = settings.voyage_rerank_model
        self._client = None

    def score(self, query: str, docs: list[str]) -> list[float]:
        if self._client is None:
            import voyageai
            self._client = voyageai.Client(api_key=self._settings.voyage_api_key)
        res = self._client.rerank(query, docs, model=self.model, top_k=len(docs))
        scores = [0.0] * len(docs)
        for r in res.results:
            scores[r.index] = r.relevance_score
        return scores


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.reranker_enabled
        self.local = LocalReranker(settings)
        self.voyage = VoyageReranker(settings)

    def route(self, any_private: bool, query_private: bool) -> str:
        return "local" if (any_private or query_private) else "voyage"

    async def rerank(self, query: str, items: list[tuple[str, bool]],
                     query_private: bool) -> list[float]:
        """items = [(text, source_is_private)]. Returns a score per item (higher better)."""
        docs = [t for t, _ in items]
        any_private = any(p for _, p in items)
        if self.route(any_private, query_private) == "local":
            return await asyncio.to_thread(self.local.score, query, docs)
        if any_private:  # defense-in-depth; route() should have prevented this
            raise PrivacyViolation("private candidate routed to Voyage reranker")
        return await asyncio.to_thread(self.voyage.score, query, docs)
