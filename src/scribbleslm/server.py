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
from .ingest import enrich_source, ingest_source
from .query import notebook_query as _notebook_query
from .routing import Router

mcp = FastMCP("ScribblesLM")

_conn = None
_settings = None
_router: Router | None = None


def _ensure():
    """Lazy init: clean startup even with missing keys; backends validate on use."""
    global _conn, _settings, _router
    if _conn is None:
        _settings = get_settings()
        _conn = db.connect(_settings.db_path)
        db.init_schema(_conn)
        disp = Dispatcher(concurrency=_settings.voyage_concurrency,
                          tpm_ceiling=_settings.voyage_tpm_ceiling)
        _router = Router(_settings, VoyageBackend(_settings, disp), BgeM3GgufBackend(_settings))
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
            "INSERT INTO sources(notebook_id, url, display_name, private) VALUES(?,?,?,?) "
            "RETURNING id", (notebook_id, url, display_name, int(eff))).fetchone()["id"]
        conn.commit()
        try:
            summary = await ingest_source(conn, router, settings, notebook_id, src, url, private)
        except Exception:
            # roll back the orphan source row so a failed add leaves no phantom source
            db.purge_vectors_for_source(conn, src)
            conn.execute("DELETE FROM sources WHERE id=?", (src,))
            conn.commit()
            raise
        out = {"source_id": src, **summary}
        if enrich and summary["private"]:
            asyncio.create_task(enrich_source(conn, router, settings, src))
            out["enrich"] = "started (background)"
        elif enrich:
            out["enrich"] = "skipped (public source is natively contextual)"
        return out
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
async def source_refresh(notebook_id: int, source_id: int) -> dict:
    """Re-fetch a source; re-ingest only if its content hash changed."""
    try:
        conn, router, settings = _ensure()
        s = conn.execute("SELECT url, content_hash, private FROM sources WHERE id=? AND notebook_id=?",
                         (source_id, notebook_id)).fetchone()
        if not s:
            return {"error": f"source {source_id} not found in notebook {notebook_id}"}
        from .ingest import chunk_text
        from .sources import fetch_source
        import hashlib
        new_hash = hashlib.sha256(fetch_source(s["url"]).encode()).hexdigest()
        if new_hash == s["content_hash"]:
            return {"refreshed": False, "reason": "content unchanged"}
        old = conn.execute("SELECT COUNT(*) n FROM chunks WHERE source_id=?", (source_id,)).fetchone()["n"]
        db.purge_vectors_for_source(conn, source_id)
        conn.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
        conn.commit()
        summary = await ingest_source(conn, router, settings, notebook_id, source_id,
                                      s["url"], bool(s["private"]))
        return {"refreshed": True, "old_chunk_count": old, "new_chunk_count": summary["chunk_count"]}
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
# Query
# ---------------------------------------------------------------------------

@mcp.tool()
async def notebook_query(notebook_id: int, query: str, top_k: int = 10,
                         private: bool | None = None) -> dict:
    """Semantic search within a notebook (dense + RRF across backends). Returns raw
    chunks for the calling agent to synthesize. Set private=true for a sensitive query
    so it is never sent to the remote backend (searches only local spaces + flags any
    excluded remote spaces)."""
    try:
        conn, router, _ = _ensure()
        return await _notebook_query(conn, router, notebook_id, query, top_k, private)
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
