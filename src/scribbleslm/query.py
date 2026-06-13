"""Query pipeline (Milestone C): hybrid retrieval — per-backend dense KNN + FTS5
lexical, fused with Reciprocal Rank Fusion. RRF is rank-based, so it is valid across
the (incomparable) dense vector spaces AND the lexical channel.

- `mode`: hybrid (dense + FTS5) | dense | lexical.
- Query-leak guard: a sensitive query never DENSE-embeds via the remote backend
  (Voyage). FTS5 is local (no egress), so it runs for sensitive queries too.
- Per-stage latency is timed and returned (C-1): query-embed / KNN / FTS5 / RRF / total.

FTS5 note (C-2): unicode61 has no Ukrainian stemming, so we OR the query terms as
PREFIX matches (term*) to recover inflected forms — measured, not assumed; see the C-2
report. The Ukrainian inflection limit is real and surfaced, not hidden behind RRF.
"""
from __future__ import annotations

import re
import sqlite3
import time

from . import db
from .routing import Router

RRF_K = 60


def _fts_terms(query: str, prefix: bool = True) -> str | None:
    words = re.findall(r"\w+", query.lower())
    if not words:
        return None
    return " OR ".join(f"{w}*" if prefix else w for w in words)


def fts_search(conn: sqlite3.Connection, notebook_id: int, query: str, k: int,
               prefix: bool = True) -> list[int]:
    terms = _fts_terms(query, prefix)
    if not terms:
        return []
    try:
        rows = conn.execute(
            "SELECT f.chunk_id FROM fts_chunks f JOIN chunks c ON c.id = f.chunk_id "
            "WHERE f.fts_chunks MATCH ? AND c.notebook_id = ? ORDER BY rank LIMIT ?",
            (terms, notebook_id, k),
        ).fetchall()
        return [r["chunk_id"] for r in rows]
    except sqlite3.OperationalError:
        return []


def _rrf(ranked_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    score: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)


RERANK_POOL = 30


def _result(row, rrf_score: float) -> dict:
    return {
        "chunk_id": row["id"], "source_id": row["source_id"], "source_url": row["url"],
        "display_name": row["display_name"], "original_text": row["original_text"],
        "contextualized_text": row["contextualized_text"], "backend_id": row["backend_id"],
        "enrichment_status": row["enrichment_status"], "rrf_score": round(rrf_score, 6),
    }


async def notebook_query(conn: sqlite3.Connection, router: Router, notebook_id: int,
                         query: str, top_k: int = 10, private: bool | None = None,
                         mode: str = "hybrid", reranker=None) -> dict:
    t0 = time.perf_counter()
    lat = {"query_embed_ms": 0.0, "knn_ms": 0.0, "fts_ms": 0.0, "rrf_ms": 0.0, "rerank_ms": 0.0}

    present = db.present_backend_ids(conn, notebook_id)
    if not present:
        return {"results": [], "notebook_id": notebook_id, "query": query, "note": "notebook is empty"}
    searched, excluded = router.query_plan(present, private)

    ranked_lists: list[list[int]] = []

    if mode in ("hybrid", "dense"):
        for backend_id in searched:
            te = time.perf_counter()
            qv = await router.embed_query(backend_id, query, private)
            lat["query_embed_ms"] += (time.perf_counter() - te) * 1000
            tk = time.perf_counter()
            hits = db.knn(conn, backend_id, qv, top_k)
            lat["knn_ms"] += (time.perf_counter() - tk) * 1000
            ranked_lists.append([cid for cid, _ in hits])

    if mode in ("hybrid", "lexical"):
        tf = time.perf_counter()
        lex = fts_search(conn, notebook_id, query, top_k * 2)
        lat["fts_ms"] += (time.perf_counter() - tf) * 1000
        if lex:
            ranked_lists.append(lex)

    tr = time.perf_counter()
    fused = _rrf(ranked_lists)
    lat["rrf_ms"] = (time.perf_counter() - tr) * 1000

    use_rerank = bool(reranker and getattr(reranker, "enabled", False) and fused)
    pool = fused[: max(RERANK_POOL, top_k)] if use_rerank else fused[:top_k]

    rows = []
    for cid, rrf_score in pool:
        row = conn.execute(
            "SELECT c.id, c.source_id, c.original_text, c.contextualized_text, c.backend_id, "
            "c.enrichment_status, s.url, s.display_name, s.private FROM chunks c JOIN sources s "
            "ON s.id = c.source_id WHERE c.id = ?", (cid,)).fetchone()
        if row:
            rows.append((row, rrf_score))

    reranked = False
    if use_rerank and rows:
        items = [(r["contextualized_text"] or r["original_text"], bool(r["private"])) for r, _ in rows]
        trk = time.perf_counter()
        scores = await reranker.rerank(query, items, router.effective_private(private))
        lat["rerank_ms"] = (time.perf_counter() - trk) * 1000
        order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:top_k]
        rows = [rows[i] for i in order]
        reranked = True
    else:
        rows = rows[:top_k]

    results = [_result(r, sc) for r, sc in rows]
    lat["total_ms"] = (time.perf_counter() - t0) * 1000
    lat = {k: round(v, 2) for k, v in lat.items()}
    return {"results": results, "notebook_id": notebook_id, "query": query, "mode": mode,
            "searched_spaces": searched, "excluded_spaces": excluded, "reranked": reranked,
            "latency_ms": lat,
            "note": "hybrid dense+FTS5 (RRF k=60)" + (" + reranker" if reranked else "")}
