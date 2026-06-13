"""Ingest-now / enrich-later pipeline (Milestone B).

source_add path: fetch -> hash -> chunk -> window -> stream(embed + store) and
return. No DeepSeek inline. Chunks are queryable as each window lands.

- PUBLIC sources -> voyage-context-3: embeddings are natively contextual, so
  chunks are marked 'enriched' immediately (no DeepSeek pass).
- PRIVATE sources -> bge-m3-GGUF: embedded on raw text, marked 'pending';
  enrich_source() later runs DeepSeek contextualization (private path only),
  re-embeds, and marks 'enriched' (or 'failed').
"""
from __future__ import annotations

import hashlib
import sqlite3

import tiktoken

from . import costs, db
from .config import Settings
from .routing import Router
from .sources import fetch_source

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 100
WINDOW_CHUNKS = 50           # context-3 atomic window (~25.6K tok < 30K limit)
CONTEXT_NEIGHBORS = 4        # +/- chunks used as DeepSeek context window (private)

_enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> tuple[list[str], list[int], list[int]]:
    toks = _enc.encode(text)
    chunks, starts, ends = [], [], []
    s = 0
    while s < len(toks):
        e = min(s + CHUNK_TOKENS, len(toks))
        chunks.append(_enc.decode(toks[s:e]))
        starts.append(s)
        ends.append(e)
        if e == len(toks):
            break
        s += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks, starts, ends


def _windows(n: int, size: int = WINDOW_CHUNKS):
    for i in range(0, n, size):
        yield i, min(i + size, n)


async def ingest_source(conn: sqlite3.Connection, router: Router, settings: Settings,
                        notebook_id: int, source_id: int, url: str, private: bool) -> dict:
    """Stream embed+store. Returns summary; commits per window (progressive)."""
    doc_text = fetch_source(url)
    content_hash = hashlib.sha256(doc_text.encode()).hexdigest()
    chunks, starts, ends = chunk_text(doc_text)

    backend_id: str | None = None
    eff_private = router.effective_private(private)
    status = "pending" if eff_private else "enriched"  # public = context-3 native
    total = 0

    for a, b in _windows(len(chunks)):
        window = chunks[a:b]
        vecs, backend_id, _ = await router.embed_source(source_id, private, [window])
        win_vecs = vecs[0]
        db.ensure_vec_table(conn, backend_id, len(win_vecs[0]))
        win_tokens = 0
        for j, (text, vec) in enumerate(zip(window, win_vecs)):
            idx = a + j
            cid = conn.execute(
                "INSERT INTO chunks(source_id, notebook_id, backend_id, chunk_index, "
                "token_start, token_end, original_text, enrichment_status) "
                "VALUES(?,?,?,?,?,?,?,?) RETURNING id",
                (source_id, notebook_id, backend_id, idx, starts[idx], ends[idx], text, status),
            ).fetchone()["id"]
            db.insert_vector(conn, backend_id, cid, vec)
            db.insert_fts(conn, cid, text)
            win_tokens += ends[idx] - starts[idx]
        conn.commit()  # stream: this window is now queryable
        costs.record_cost(conn, source_id, backend_id, api_calls=1, input_tokens=win_tokens)
        total += len(window)

    conn.execute("UPDATE sources SET content_hash=?, backend_id=?, private=? WHERE id=?",
                 (content_hash, backend_id, int(eff_private), source_id))
    conn.commit()
    return {"chunk_count": total, "content_hash": content_hash,
            "backend_id": backend_id, "status": status, "private": eff_private}


def _context_client(settings: Settings):
    if not (settings.context_llm_base_url and settings.context_llm_api_key
            and settings.context_llm_model):
        raise RuntimeError(
            "CONTEXT_LLM_BASE_URL/API_KEY/MODEL must be set to enrich private sources."
        )
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=settings.context_llm_base_url,
                       api_key=settings.context_llm_api_key)


async def _gen_context(client, model: str, window: str, chunk: str) -> tuple[str, int, int]:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                f"<context>\n{window}\n</context>\n\n<chunk>\n{chunk}\n</chunk>\n\n"
                "Give a short succinct context to situate this chunk within the broader "
                "document for search retrieval purposes. Answer only with the context, "
                "nothing else."
            )},
        ],
        max_tokens=200,
    )
    u = resp.usage
    return (resp.choices[0].message.content.strip(),
            getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))


async def enrich_source(conn: sqlite3.Connection, router: Router, settings: Settings,
                        source_id: int, force: bool = False) -> dict:
    """DeepSeek contextualization over a PRIVATE source's pending chunks, re-embed
    locally, mark enriched/failed. Public (context-3) sources need no enrichment."""
    rows = conn.execute(
        "SELECT id, chunk_index, original_text, enrichment_status, backend_id "
        "FROM chunks WHERE source_id=? ORDER BY chunk_index", (source_id,)
    ).fetchall()
    if not rows:
        return {"error": f"source {source_id} has no chunks"}
    backend_id = rows[0]["backend_id"]
    if not backend_id.startswith("bge"):
        return {"enriched": 0, "reason": "public source is natively contextual (context-3)"}

    texts = [r["original_text"] for r in rows]
    targets = [r for r in rows if force or r["enrichment_status"] in ("pending", "failed")]
    if not targets:
        return {"enriched": 0, "reason": "nothing pending"}

    client = _context_client(settings)
    enriched = failed = calls = in_tok = out_tok = 0
    for r in targets:
        i = r["chunk_index"]
        window = "\n".join(texts[max(0, i - CONTEXT_NEIGHBORS): i + CONTEXT_NEIGHBORS + 1])
        try:
            ctx, ui, uo = await _gen_context(client, settings.context_llm_model, window, r["original_text"])
            calls += 1; in_tok += ui; out_tok += uo
            ctext = f"{ctx}\n\n{r['original_text']}"
            vec = await router.private.embed_query(ctext)
            db.replace_vector(conn, backend_id, r["id"], vec)
            conn.execute(
                "UPDATE chunks SET contextualized_text=?, enrichment_status='enriched' WHERE id=?",
                (ctext, r["id"]))
            conn.commit(); enriched += 1
        except Exception:
            conn.execute("UPDATE chunks SET enrichment_status='failed' WHERE id=?", (r["id"],))
            conn.commit(); failed += 1

    if calls:
        costs.record_cost(conn, source_id, settings.context_llm_model,
                          api_calls=calls, input_tokens=in_tok, output_tokens=out_tok)
    return {"enriched": enriched, "failed": failed, "processed": len(targets)}
