import asyncio
import hashlib
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .db import get_conn, get_db_url, init_schema
from .ingest import ingest_source
from .query import query_notebook as _query_notebook
from .sources import fetch_source

mcp = FastMCP("ScribblesLM")

_schema_ready = False


async def _ensure_schema() -> None:
    global _schema_ready
    if not _schema_ready:
        get_db_url()  # raises if unset
        await init_schema()
        _schema_ready = True


# ---------------------------------------------------------------------------
# Notebook management
# ---------------------------------------------------------------------------

@mcp.tool()
async def notebook_create(name: str) -> dict:
    """Create a new notebook."""
    try:
        await _ensure_schema()
        conn = await get_conn()
        row = await conn.execute(
            "INSERT INTO notebooks(name) VALUES(%s) "
            "RETURNING id, name, created_at",
            (name,),
        )
        result = await row.fetchone()
        await conn.commit()
        return {
            "id": str(result["id"]),
            "name": result["name"],
            "created_at": result["created_at"].isoformat(),
        }
    except Exception as e:
        msg = str(e)
        if "unique" in msg.lower() or "duplicate" in msg.lower():
            return {"error": f"Notebook '{name}' already exists"}
        return {"error": msg}


@mcp.tool()
async def notebook_list() -> list[dict]:
    """List all notebooks with source counts."""
    try:
        await _ensure_schema()
        conn = await get_conn()
        rows = await conn.execute(
            """
            SELECT n.id, n.name, n.created_at,
                   COUNT(s.id) AS source_count
            FROM notebooks n
            LEFT JOIN sources s ON s.notebook_id = n.id
            GROUP BY n.id, n.name, n.created_at
            ORDER BY n.created_at
            """
        )
        results = await rows.fetchall()
        return [
            {
                "id": str(r["id"]),
                "name": r["name"],
                "created_at": r["created_at"].isoformat(),
                "source_count": r["source_count"],
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def notebook_delete(notebook_id: str) -> dict:
    """Delete a notebook and all its sources and chunks."""
    try:
        await _ensure_schema()
        conn = await get_conn()
        row = await conn.execute(
            "DELETE FROM notebooks WHERE id = %s RETURNING name",
            (notebook_id,),
        )
        result = await row.fetchone()
        await conn.commit()
        if not result:
            return {"error": f"Notebook {notebook_id} not found"}
        return {"deleted": True, "name": result["name"]}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------

@mcp.tool()
async def source_add(
    notebook_id: str,
    url: str,
    display_name: Optional[str] = None,
) -> dict:
    """Fetch and ingest a source into a notebook."""
    try:
        await _ensure_schema()
        conn = await get_conn()

        # Insert source record first (to get source_id)
        row = await conn.execute(
            """
            INSERT INTO sources(notebook_id, url, display_name)
            VALUES(%s, %s, %s)
            RETURNING id
            """,
            (notebook_id, url, display_name),
        )
        result = await row.fetchone()
        await conn.commit()
        source_id = str(result["id"])

        chunk_count, content_hash = await ingest_source(notebook_id, source_id, url)

        await conn.execute(
            "UPDATE sources SET content_hash = %s WHERE id = %s",
            (content_hash, source_id),
        )
        await conn.commit()

        return {
            "source_id": source_id,
            "chunk_count": chunk_count,
            "content_hash": content_hash,
        }
    except Exception as e:
        msg = str(e)
        if "unique" in msg.lower() or "duplicate" in msg.lower():
            return {"error": f"Source '{url}' already exists in this notebook"}
        return {"error": msg}


@mcp.tool()
async def source_list(notebook_id: str) -> list[dict]:
    """List sources in a notebook with chunk counts."""
    try:
        await _ensure_schema()
        conn = await get_conn()
        rows = await conn.execute(
            """
            SELECT s.id, s.url, s.display_name, s.ingested_at,
                   COUNT(c.id) AS chunk_count
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id
            WHERE s.notebook_id = %s
            GROUP BY s.id, s.url, s.display_name, s.ingested_at
            ORDER BY s.ingested_at
            """,
            (notebook_id,),
        )
        results = await rows.fetchall()
        return [
            {
                "source_id": str(r["id"]),
                "url": r["url"],
                "display_name": r["display_name"],
                "chunk_count": r["chunk_count"],
                "ingested_at": r["ingested_at"].isoformat(),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def source_refresh(notebook_id: str, source_id: str) -> dict:
    """Re-fetch a source and re-ingest if content changed."""
    try:
        await _ensure_schema()
        conn = await get_conn()

        row = await conn.execute(
            "SELECT url, content_hash FROM sources WHERE id = %s AND notebook_id = %s",
            (source_id, notebook_id),
        )
        result = await row.fetchone()
        if not result:
            return {"error": f"Source {source_id} not found in notebook {notebook_id}"}

        url = result["url"]
        old_hash = result["content_hash"]

        doc_text = fetch_source(url)
        new_hash = hashlib.sha256(doc_text.encode()).hexdigest()

        if new_hash == old_hash:
            return {"refreshed": False, "reason": "content unchanged"}

        old_count_row = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE source_id = %s",
            (source_id,),
        )
        old_count = (await old_count_row.fetchone())["cnt"]

        await conn.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
        await conn.commit()

        new_count, _ = await ingest_source(notebook_id, source_id, url)

        await conn.execute(
            "UPDATE sources SET content_hash = %s, ingested_at = now() WHERE id = %s",
            (new_hash, source_id),
        )
        await conn.commit()

        return {
            "refreshed": True,
            "old_chunk_count": old_count,
            "new_chunk_count": new_count,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def source_delete(notebook_id: str, source_id: str) -> dict:
    """Delete a source and its chunks."""
    try:
        await _ensure_schema()
        conn = await get_conn()
        row = await conn.execute(
            "DELETE FROM sources WHERE id = %s AND notebook_id = %s RETURNING id",
            (source_id, notebook_id),
        )
        result = await row.fetchone()
        await conn.commit()
        if not result:
            return {"error": f"Source {source_id} not found in notebook {notebook_id}"}
        return {"deleted": True}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@mcp.tool()
async def notebook_query(
    notebook_id: str,
    query: str,
    top_k: int = 10,
) -> dict:
    """Semantic search within a notebook. Returns raw chunks for agent synthesis."""
    try:
        await _ensure_schema()
        results = await _query_notebook(notebook_id, query, top_k)
        return {
            "results": results,
            "notebook_id": notebook_id,
            "query": query,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
