"""Query pipeline (Milestone B): dense retrieval with the query-leak guard and
Reciprocal Rank Fusion across (incomparable) per-backend vector spaces.

FTS5 lexical channel and the sensitivity-routed reranker are Milestone C — this
returns dense results only. RRF is rank-based, so it is valid across spaces whose
raw distances are not comparable; that is exactly why we fuse on rank, not score.
"""
from __future__ import annotations

import sqlite3

from . import db
from .routing import Router

RRF_K = 60


def _rrf(ranked_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    score: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)


async def notebook_query(conn: sqlite3.Connection, router: Router, notebook_id: int,
                         query: str, top_k: int = 10, private: bool | None = None) -> dict:
    present = db.present_backend_ids(conn, notebook_id)
    if not present:
        return {"results": [], "notebook_id": notebook_id, "query": query,
                "note": "notebook is empty"}

    searched, excluded = router.query_plan(present, private)
    if not searched:
        return {"results": [], "notebook_id": notebook_id, "query": query,
                "excluded_spaces": excluded,
                "note": "sensitive query: all present spaces are remote and were excluded"}

    ranked_lists: list[list[int]] = []
    for backend_id in searched:
        qv = await router.embed_query(backend_id, query, private)
        hits = db.knn(conn, backend_id, qv, top_k)
        ranked_lists.append([cid for cid, _ in hits])

    fused = _rrf(ranked_lists)[:top_k]
    results = []
    for cid, rrf_score in fused:
        row = conn.execute(
            "SELECT c.id, c.source_id, c.original_text, c.contextualized_text, "
            "c.backend_id, c.enrichment_status, s.url, s.display_name "
            "FROM chunks c JOIN sources s ON s.id = c.source_id WHERE c.id = ?", (cid,)
        ).fetchone()
        if not row:
            continue
        results.append({
            "chunk_id": row["id"],
            "source_id": row["source_id"],
            "source_url": row["url"],
            "display_name": row["display_name"],
            "original_text": row["original_text"],
            "contextualized_text": row["contextualized_text"],
            "backend_id": row["backend_id"],
            "enrichment_status": row["enrichment_status"],
            "rrf_score": round(rrf_score, 6),
        })
    return {"results": results, "notebook_id": notebook_id, "query": query,
            "searched_spaces": searched, "excluded_spaces": excluded,
            "note": "dense-only; FTS5 lexical + reranker arrive in Milestone C"}
