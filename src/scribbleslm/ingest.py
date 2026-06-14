"""Ingest-now / enrich-later pipeline (Milestone B + D induction).

source_add: fetch -> hash -> INDUCE chunking profile -> build article-aligned chunks
-> stream(embed + store). No DeepSeek inline. Queryable as each window lands.

Chunking (D): the induction ladder derives a profile and segments the document into
article-aligned sections; each section is token-split to <=512 tokens (the storage
guarantee — a 224K flat region becomes ~440 capped chunks, not one blob), inheriting
the section breadcrumb. Such chunks are marked 'structural' and EXCLUDED from LLM
enrichment (the breadcrumb is their context). Non-structural chunks (preamble, or a
token-fallback document) follow the public/private path:
  - PUBLIC  -> voyage-context-3 (natively contextual) -> 'enriched'
  - PRIVATE -> bge-m3-GGUF on raw text -> 'pending' (enrich_source adds DeepSeek later)
"""
from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from . import costs, db
from .config import Settings
from .induction.cache import ProfileCache
from .induction.ladder import induce
from .routing import Router
from .sources import fetch_source

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 100
WINDOW_CHUNKS = 50           # context-3 atomic embedding window (~25.6K tok < 32K)
CONTEXT_NEIGHBORS = 4        # +/- chunks as DeepSeek context window (private enrich)
_SHIPPED_PROFILES = Path(__file__).resolve().parent / "profiles"

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    breadcrumb: str | None
    structural: bool
    pos_start: int   # char offset of the parent unit (informational)
    pos_end: int


def _split_tokens(text: str, pos: int) -> list[Chunk]:
    """Token-split a span to <=512-token pieces. This is the storage-size guarantee
    enforced on EVERY segment, including oversized flat regions."""
    toks = _enc.encode(text)
    out, s = [], 0
    while s < len(toks):
        e = min(s + CHUNK_TOKENS, len(toks))
        out.append(_enc.decode(toks[s:e]))
        if e == len(toks):
            break
        s += CHUNK_TOKENS - OVERLAP_TOKENS
    return out


def build_chunks(doc_text: str, induction) -> list[Chunk]:
    """Article-aligned chunks from an induction result. Every chunk is <=512 tokens."""
    segs = induction.segments
    chunks: list[Chunk] = []
    if not segs:  # rung-6 token fallback: whole doc, non-structural
        for piece in _split_tokens(doc_text, 0):
            chunks.append(Chunk(piece, None, False, 0, len(doc_text)))
        return chunks

    first = segs[0].start
    preamble = doc_text[:first]
    if preamble.strip():  # content before the first heading -> non-structural
        for piece in _split_tokens(preamble, 0):
            chunks.append(Chunk(piece, None, False, 0, first))

    for s in segs:  # each section -> <=512-tok structural chunks, inherit breadcrumb
        for piece in _split_tokens(doc_text[s.start:s.end], s.start):
            chunks.append(Chunk(piece, s.breadcrumb, True, s.start, s.end))
    return chunks


def _embed_text(c: Chunk) -> str:
    """Structural chunks embed breadcrumb-prefixed text (the breadcrumb is context)."""
    return f"{c.breadcrumb}\n\n{c.text}" if (c.structural and c.breadcrumb) else c.text


def _profile_cache(settings: Settings) -> ProfileCache:
    return ProfileCache(settings.profile_cache_dir, shipped_dir=_SHIPPED_PROFILES)


def _windows(n: int, size: int = WINDOW_CHUNKS):
    for i in range(0, n, size):
        yield i, min(i + size, n)


def _prepare(doc_text: str, settings: Settings):
    """Blocking CPU work (induction + chunking) — run off the event loop via to_thread."""
    induction = induce(doc_text, cache=_profile_cache(settings))
    return induction, build_chunks(doc_text, induction)


async def run_ingest(conn: sqlite3.Connection, router: Router, settings: Settings,
                     notebook_id: int, source_id: int, url: str, private: bool,
                     enrich: bool) -> dict:
    """Background ingest lifecycle: ingesting -> ready -> (optional) enrich. On error,
    mark 'failed' but KEEP partial chunks (they stay queryable); never roll back."""
    try:
        conn.execute("UPDATE sources SET ingest_state='ingesting' WHERE id=?", (source_id,))
        conn.commit()
        summary = await ingest_source(conn, router, settings, notebook_id, source_id, url, private)
        conn.execute("UPDATE sources SET ingest_state='ready' WHERE id=?", (source_id,))
        conn.commit()
        if enrich and summary.get("private"):
            await enrich_source(conn, router, settings, source_id)
        return summary
    except Exception as e:
        conn.execute("UPDATE sources SET ingest_state='failed', ingest_error=? WHERE id=?",
                     (str(e)[:500], source_id))
        conn.commit()
        return {"error": str(e), "source_id": source_id}


async def ingest_source(conn: sqlite3.Connection, router: Router, settings: Settings,
                        notebook_id: int, source_id: int, url: str, private: bool) -> dict:
    doc_text = await asyncio.to_thread(
        fetch_source, url,
        settings.documents_root, settings.max_document_bytes)     # blocking I/O off-loop
    content_hash = hashlib.sha256(doc_text.encode()).hexdigest()
    induction, chunks = await asyncio.to_thread(_prepare, doc_text, settings)  # CPU off-loop
    conn.execute("UPDATE sources SET chunks_planned=? WHERE id=?", (len(chunks), source_id))
    conn.commit()  # denominator available to source_status before the embed loop starts

    eff_private = router.effective_private(private)
    backend_id: str | None = None
    structural = 0

    for a, b in _windows(len(chunks)):
        window = chunks[a:b]
        embed_texts = [_embed_text(c) for c in window]
        vecs, backend_id, _ = await router.embed_source(source_id, private, [embed_texts])
        win_vecs = vecs[0]
        db.ensure_vec_table(conn, backend_id, len(win_vecs[0]))
        win_tokens = 0
        for j, (c, vec, etext) in enumerate(zip(window, win_vecs, embed_texts)):
            if c.structural:
                status, ctext = "structural", etext
            elif eff_private:
                status, ctext = "pending", None
            else:
                status, ctext = "enriched", None     # context-3 native
            if c.structural:
                structural += 1
            cid = conn.execute(
                "INSERT INTO chunks(source_id, notebook_id, backend_id, chunk_index, "
                "token_start, token_end, original_text, contextualized_text, "
                "enrichment_status) VALUES(?,?,?,?,?,?,?,?,?) RETURNING id",
                (source_id, notebook_id, backend_id, a + j, c.pos_start, c.pos_end,
                 c.text, ctext, status),
            ).fetchone()["id"]
            db.insert_vector(conn, backend_id, cid, vec)
            db.insert_fts(conn, cid, c.text)
            win_tokens += len(_enc.encode(etext))
        conn.commit()
        costs.record_cost(conn, source_id, backend_id, api_calls=1, input_tokens=win_tokens)

    conn.execute("UPDATE sources SET content_hash=?, backend_id=?, private=? WHERE id=?",
                 (content_hash, backend_id, int(eff_private), source_id))
    conn.commit()
    return {"chunk_count": len(chunks), "content_hash": content_hash,
            "backend_id": backend_id, "private": eff_private,
            "rung": induction.rung, "structural_chunks": structural,
            "profile_levels": [l.name for l in induction.profile.levels] if induction.profile else []}


# ---------------------------------------------------------------------------
# enrich (private path only): DeepSeek contextualization over PENDING chunks.
# Structural chunks (status='structural') are excluded automatically.
# ---------------------------------------------------------------------------

def _context_client(settings: Settings):
    if not (settings.context_llm_base_url and settings.context_llm_api_key
            and settings.context_llm_model):
        raise RuntimeError(
            "CONTEXT_LLM_BASE_URL/API_KEY/MODEL must be set to enrich private sources.")
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
                "nothing else.")},
        ],
        max_tokens=200,
    )
    u = resp.usage
    return (resp.choices[0].message.content.strip(),
            getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))


async def enrich_source(conn: sqlite3.Connection, router: Router, settings: Settings,
                        source_id: int, force: bool = False) -> dict:
    rows = conn.execute(
        "SELECT id, chunk_index, original_text, enrichment_status, backend_id "
        "FROM chunks WHERE source_id=? ORDER BY chunk_index", (source_id,)).fetchall()
    if not rows:
        return {"error": f"source {source_id} has no chunks"}
    backend_id = rows[0]["backend_id"]
    if not backend_id.startswith("bge"):
        return {"enriched": 0, "reason": "public source is natively contextual (context-3)"}

    texts = [r["original_text"] for r in rows]
    # structural chunks already carry a breadcrumb context -> excluded; only pending/failed
    targets = [r for r in rows if force or r["enrichment_status"] in ("pending", "failed")]
    if not targets:
        return {"enriched": 0, "reason": "nothing pending (structural chunks are excluded)"}

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
            conn.execute("UPDATE chunks SET contextualized_text=?, enrichment_status='enriched' WHERE id=?",
                         (ctext, r["id"]))
            conn.commit(); enriched += 1
        except Exception:
            conn.execute("UPDATE chunks SET enrichment_status='failed' WHERE id=?", (r["id"],))
            conn.commit(); failed += 1

    if calls:
        costs.record_cost(conn, source_id, settings.context_llm_model,
                          api_calls=calls, input_tokens=in_tok, output_tokens=out_tok)
    return {"enriched": enriched, "failed": failed, "processed": len(targets)}
