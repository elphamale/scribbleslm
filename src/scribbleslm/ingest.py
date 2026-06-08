import asyncio
import hashlib
import os
import random

import httpx
import tiktoken
from openai import AsyncOpenAI

from .db import get_conn, config_get, config_set
from .sources import fetch_source


CHUNK_TOKENS = 512
OVERLAP_TOKENS = 100
WINDOW_TOKENS = 2000   # tokens of surrounding context sent to DeepSeek
CONCURRENCY = 8        # DeepSeek concurrent calls
EMBED_CONCURRENCY = 4  # Ollama concurrent calls


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _make_context_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.environ["CONTEXT_LLM_BASE_URL"],
        api_key=os.environ["CONTEXT_LLM_API_KEY"],
    )


def _get_enc():
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> tuple[list[str], list[int], list[int]]:
    """Returns (chunks, token_starts, token_ends)."""
    enc = _get_enc()
    tokens = enc.encode(text)
    chunks, starts, ends = [], [], []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        starts.append(start)
        ends.append(end)
        if end == len(tokens):
            break
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks, starts, ends


def extract_window(doc_tokens: list[int], chunk_start: int, chunk_end: int) -> str:
    enc = _get_enc()
    win_start = max(0, chunk_start - WINDOW_TOKENS)
    win_end = min(len(doc_tokens), chunk_end + WINDOW_TOKENS)
    return enc.decode(doc_tokens[win_start:win_end])


_embed_sem = asyncio.Semaphore(EMBED_CONCURRENCY)


async def embed(text: str) -> list[float]:
    model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3")
    async with _embed_sem:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{_ollama_base_url()}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]


async def validate_embedding_dim(actual_dim: int) -> None:
    stored = await config_get("embedding_dim")
    if stored is None:
        await config_set("embedding_dim", str(actual_dim))
    elif int(stored) != actual_dim:
        raise ValueError(
            f"Embedding dimension mismatch: stored {stored}, got {actual_dim}. "
            "Drop and recreate the database to switch models."
        )


async def _call_context_llm_with_backoff(
    client: AsyncOpenAI,
    model: str,
    window: str,
    chunk_text_str: str,
    max_retries: int = 6,
) -> str:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": (
                            f"<context>\n{window}\n</context>\n\n"
                            f"<chunk>\n{chunk_text_str}\n</chunk>\n\n"
                            "Give a short succinct context to situate this chunk within the "
                            "broader document for search retrieval purposes. Answer only with "
                            "the context, nothing else."
                        ),
                    },
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                if attempt == max_retries - 1:
                    raise
                jitter = random.uniform(0, delay * 0.3)
                await asyncio.sleep(delay + jitter)
                delay = min(delay * 2, 60.0)
            else:
                raise
    raise RuntimeError("Max retries exceeded on context LLM call")


async def _contextualize_chunk(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    doc_tokens: list[int],
    chunk_text_str: str,
    chunk_start: int,
    chunk_end: int,
) -> str:
    async with sem:
        window = extract_window(doc_tokens, chunk_start, chunk_end)
        ctx = await _call_context_llm_with_backoff(client, model, window, chunk_text_str)
        return f"{ctx}\n\n{chunk_text_str}"


async def ingest_source(
    notebook_id: str,
    source_id: str,
    url: str,
) -> tuple[int, str]:
    """Returns (chunk_count, content_hash)."""
    doc_text = fetch_source(url)
    content_hash = hashlib.sha256(doc_text.encode()).hexdigest()

    enc = _get_enc()
    doc_tokens = enc.encode(doc_text)

    chunks, starts, ends = chunk_text(doc_text)

    context_client = _make_context_client()
    context_model = os.environ["CONTEXT_LLM_MODEL"]

    sem = asyncio.Semaphore(CONCURRENCY)
    contextualized_chunks = await asyncio.gather(
        *[
            _contextualize_chunk(sem, context_client, context_model, doc_tokens, chunk, start, end)
            for chunk, start, end in zip(chunks, starts, ends)
        ]
    )

    embeddings = await asyncio.gather(*[embed(ctx) for ctx in contextualized_chunks])
    if embeddings:
        await validate_embedding_dim(len(embeddings[0]))

    conn = await get_conn()
    async with conn.transaction():
        for i, (chunk, ctx_chunk, embedding) in enumerate(
            zip(chunks, contextualized_chunks, embeddings)
        ):
            await conn.execute(
                """
                INSERT INTO chunks
                  (source_id, notebook_id, chunk_index, original_text,
                   contextualized_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (source_id, notebook_id, i, chunk, ctx_chunk, embedding),
            )

    return len(chunks), content_hash
