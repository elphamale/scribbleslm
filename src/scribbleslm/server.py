"""ScribblesLM v2 MCP server (stdio).

Sensitivity-routed embeddings on a SQLite + sqlite-vec + FTS5 foundation:
public docs -> voyage-context-3 (contextual, fast), private docs -> local
bge-m3-GGUF. Ingest-now / enrich-later. All tools return {"error": ...} on
failure rather than raising to the MCP layer.
"""
from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from . import costs, db
from .config import get_settings
from .embeddings.bge_gguf import BgeM3GgufBackend
from .embeddings.dispatcher import Dispatcher
from .embeddings.voyage import VoyageBackend
from .ingest import enrich_source, ingest_source, run_ingest
from .query import notebook_query as _notebook_query
from .reranker import Reranker
from .routing import Router

mcp = FastMCP("ScribblesLM")

_conn = None
_settings = None
_router: Router | None = None
_reranker: Reranker | None = None
_bg_tasks: set = set()  # retain background ingest tasks so they aren't GC'd mid-flight


def _ensure():
    """Lazy init: clean startup even with missing keys; backends validate on use."""
    global _conn, _settings, _router, _reranker
    if _conn is None:
        _settings = get_settings()
        _conn = db.connect(_settings.db_path)
        db.init_schema(_conn)
        disp = Dispatcher(concurrency=_settings.voyage_concurrency,
                          tpm_ceiling=_settings.voyage_tpm_ceiling)
        _router = Router(_settings, VoyageBackend(_settings, disp), BgeM3GgufBackend(_settings))
        _reranker = Reranker(_settings)
    return _conn, _router, _settings


# ---------------------------------------------------------------------------
# Notebook management
# ---------------------------------------------------------------------------

@mcp.tool()
async def notebook_create(name: str) -> dict:
    """Create a notebook."""
    try:
        conn, _, _ = _ensure()
        row = conn.execute("INSERT INTO notebooks(name) VALUES(?) RETURNING id, created_at",
                           (name,)).fetchone()
        conn.commit()
        return {"id": row["id"], "name": name, "created_at": row["created_at"]}
    except Exception as e:
        if "unique" in str(e).lower():
            return {"error": f"notebook '{name}' already exists"}
        return {"error": str(e)}


@mcp.tool()
async def notebook_list() -> list[dict]:
    """List notebooks with source counts."""
    try:
        conn, _, _ = _ensure()
        rows = conn.execute(
            "SELECT n.id, n.name, n.created_at, COUNT(s.id) source_count "
            "FROM notebooks n LEFT JOIN sources s ON s.notebook_id = n.id "
            "GROUP BY n.id ORDER BY n.created_at").fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def notebook_delete(notebook_id: int) -> dict:
    """Delete a notebook and all its sources, chunks, and vectors."""
    try:
        conn, _, _ = _ensure()
        row = conn.execute("SELECT name FROM notebooks WHERE id=?", (notebook_id,)).fetchone()
        if not row:
            return {"error": f"notebook {notebook_id} not found"}
        db.purge_vectors_for_notebook(conn, notebook_id)
        conn.execute("DELETE FROM notebooks WHERE id=?", (notebook_id,))
        conn.commit()
        return {"deleted": True, "name": row["name"]}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------

@mcp.tool()
async def source_add(notebook_id: int, url: str, display_name: str | None = None,
                     private: bool | None = None, enrich: bool = False) -> dict:
    """Fetch and ingest a source. Returns after embed+store (queryable immediately);
    no DeepSeek runs inline. Documents default to public (private=false), embedded via
    a remote API for speed. Set private=true to force fully-local embedding for any
    document that must not leave this host. Note: on a rented/cloud host the document
    already resides on infrastructure the operator may not fully control; private=true
    limits further transmission but is not a substitute for not ingesting truly
    sensitive material onto an untrusted host. enrich=true runs contextualization in
    the background (private sources only; public sources are already contextual)."""
    try:
        conn, router, settings = _ensure()
        if not conn.execute("SELECT 1 FROM notebooks WHERE id=?", (notebook_id,)).fetchone():
            return {"error": f"notebook {notebook_id} not found"}
        eff = router.effective_private(private)
        src = conn.execute(
            "INSERT INTO sources(notebook_id, url, display_name, private, ingest_state) "
            "VALUES(?,?,?,?, 'pending') RETURNING id",
            (notebook_id, url, display_name, int(eff))).fetchone()["id"]
        conn.commit()
        # Background the embed (+enrich): return source_id immediately so the agent isn't
        # blocked for the duration of a large ingest; chunks become queryable as they
        # stream in. Poll source_status(source_id) for progress.
        task = asyncio.create_task(
            run_ingest(conn, router, settings, notebook_id, src, url, private, enrich))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
        return {"source_id": src, "ingest_state": "ingesting", "private": eff,
                "note": "ingesting in background — poll source_status(source_id); "
                        "chunks are queryable as they embed"}
    except Exception as e:
        if "unique" in str(e).lower():
            return {"error": f"source '{url}' already exists in this notebook"}
        return {"error": str(e)}


@mcp.tool()
async def source_list(notebook_id: int) -> list[dict]:
    """List sources with chunk counts, enrichment progress, and per-model cost."""
    try:
        conn, _, _ = _ensure()
        srcs = conn.execute(
            "SELECT id, url, display_name, private, backend_id, ingested_at "
            "FROM sources WHERE notebook_id=? ORDER BY ingested_at", (notebook_id,)).fetchall()
        out = []
        for s in srcs:
            status_rows = conn.execute(
                "SELECT enrichment_status, COUNT(*) n FROM chunks WHERE source_id=? "
                "GROUP BY enrichment_status", (s["id"],)).fetchall()
            status = {r["enrichment_status"]: r["n"] for r in status_rows}
            out.append({
                "source_id": s["id"], "url": s["url"], "display_name": s["display_name"],
                "private": bool(s["private"]), "backend_id": s["backend_id"],
                "ingested_at": s["ingested_at"],
                "chunk_count": sum(status.values()),
                "enrichment": status,
                "cost": costs.source_cost_breakdown(conn, s["id"]),
            })
        return out
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def source_refresh(notebook_id: int, source_id: int,
                         enrich: bool = False) -> dict:
    """Re-fetch a source; re-ingest in background only if content hash changed.
    Pass enrich=true to re-enrich private sources after re-ingest (matches source_add)."""
    try:
        import hashlib
        from .sources import fetch_source
        conn, router, settings = _ensure()
        s = conn.execute("SELECT url, content_hash, private FROM sources WHERE id=? AND notebook_id=?",
                         (source_id, notebook_id)).fetchone()
        if not s:
            return {"error": f"source {source_id} not found in notebook {notebook_id}"}
        new_text = await asyncio.to_thread(
            fetch_source, s["url"], settings.documents_root, settings.max_document_bytes)
        new_hash = hashlib.sha256(new_text.encode()).hexdigest()
        if new_hash == s["content_hash"]:
            return {"refreshed": False, "reason": "content unchanged"}
        old = conn.execute("SELECT COUNT(*) n FROM chunks WHERE source_id=?", (source_id,)).fetchone()["n"]
        db.purge_vectors_for_source(conn, source_id)
        conn.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
        conn.execute("UPDATE sources SET ingest_state='pending', ingest_error=NULL, "
                     "chunks_planned=0 WHERE id=?", (source_id,))
        conn.commit()
        task = asyncio.create_task(
            run_ingest(conn, router, settings, notebook_id, source_id, s["url"],
                       bool(s["private"]), enrich))
        _bg_tasks.add(task); task.add_done_callback(_bg_tasks.discard)
        return {"refreshed": True, "old_chunk_count": old, "source_id": source_id,
                "ingest_state": "ingesting",
                "note": "re-ingesting in background — poll source_status(source_id)"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def source_delete(notebook_id: int, source_id: int) -> dict:
    """Delete a source and its chunks/vectors."""
    try:
        conn, _, _ = _ensure()
        if not conn.execute("SELECT 1 FROM sources WHERE id=? AND notebook_id=?",
                            (source_id, notebook_id)).fetchone():
            return {"error": f"source {source_id} not found in notebook {notebook_id}"}
        db.purge_vectors_for_source(conn, source_id)
        conn.execute("DELETE FROM sources WHERE id=?", (source_id,))
        conn.commit()
        return {"deleted": True}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def source_enrich(source_id: int, force: bool = False) -> dict:
    """Run DeepSeek contextualization over a PRIVATE source's pending chunks, re-embed
    locally, and mark them enriched. Public (context-3) sources are already contextual."""
    try:
        conn, router, settings = _ensure()
        return await enrich_source(conn, router, settings, source_id, force=force)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Status (pollable — stdio is request/response, so poll; there is no push/stream)
# ---------------------------------------------------------------------------

@mcp.tool()
async def source_status(source_id: int) -> dict:
    """Ingestion/enrichment status for a source, derived from current DB state. Poll
    this to track progress (no streaming — stdio is request/response). Returns chunk
    counts (total/embedded/pending/enriched/failed), the enrichment_status rollup,
    `queryable` (true once any chunk is stored — partial coverage is searchable, and
    failed-enrichment chunks stay queryable on their plain embeddings), backend, and a
    terse human-readable `summary`. Retrieval itself is sub-second, so there is no
    retrieval progress to report."""
    try:
        conn, _, _ = _ensure()
        st = db.source_status(conn, source_id)
        return st if st else {"error": f"source {source_id} not found"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def notebook_status(notebook_id: int) -> dict:
    """Aggregate ingestion/enrichment status across all sources in a notebook — 'is my
    corpus ready'. Same rollup as source_status, plus source_count."""
    try:
        conn, _, _ = _ensure()
        st = db.notebook_status(conn, notebook_id)
        return st if st else {"error": f"notebook {notebook_id} not found"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@mcp.tool()
async def notebook_query(notebook_id: int, query: str, top_k: int = 10,
                         private: bool | None = None, mode: str = "hybrid") -> dict:
    """Hybrid search within a notebook: dense KNN + FTS5 lexical, fused with RRF.
    Returns raw chunks for the calling agent to synthesize. mode = hybrid | dense |
    lexical. Set private=true for a sensitive query so it is never dense-embedded via
    the remote backend (FTS5 is local and still runs); excluded remote spaces are
    flagged. Per-stage latency is returned under 'latency_ms'."""
    try:
        conn, router, _ = _ensure()
        return await _notebook_query(conn, router, notebook_id, query, top_k, private, mode, _reranker)
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
