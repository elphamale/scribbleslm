import asyncio
import hashlib
import os
from typing import Optional

import tiktoken
from openai import AsyncOpenAI

from .db import get_conn, config_get, config_set
from .sources import fetch_source


CHUNK_TOKENS = 512
OVERLAP_TOKENS = 100
CONTEXT_DELAY = 1.5  # seconds between LLM calls


def _make_context_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.environ["CONTEXT_LLM_BASE_URL"],
        api_key=os.environ["CONTEXT_LLM_API_KEY"],
    )


def _make_embedding_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.environ["EMBEDDING_API_KEY"],
    )


def chunk_text(text: str) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks


async def generate_context(
    client: AsyncOpenAI,
    model: str,
    document_text: str,
    chunk_text_str: str,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{document_text}\n</document>\n\n",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"<chunk>\n{chunk_text_str}\n</chunk>\n\n"
                            "Please give a short succinct context to situate this chunk "
                            "within the overall document for the purposes of improving "
                            "search retrieval of the chunk. Answer only with the succinct "
                            "context and nothing else."
                        ),
                    },
                ],
            },
        ],
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


async def embed_texts(client: AsyncOpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


async def validate_embedding_dim(actual_dim: int) -> None:
    stored = await config_get("embedding_dim")
    if stored is None:
        await config_set("embedding_dim", str(actual_dim))
    elif int(stored) != actual_dim:
        raise ValueError(
            f"Embedding dimension mismatch: stored {stored}, got {actual_dim}. "
            "Drop and recreate the database to switch models."
        )


async def ingest_source(
    notebook_id: str,
    source_id: str,
    url: str,
) -> tuple[int, str]:
    """
    Returns (chunk_count, content_hash).
    Fetches, chunks, contextualizes, embeds, and stores.
    """
    doc_text = fetch_source(url)
    content_hash = hashlib.sha256(doc_text.encode()).hexdigest()

    chunks = chunk_text(doc_text)

    context_client = _make_context_client()
    context_model = os.environ["CONTEXT_LLM_MODEL"]
    embedding_client = _make_embedding_client()
    embedding_model = os.environ["EMBEDDING_MODEL"]

    contextualized_chunks = []
    for i, chunk in enumerate(chunks):
        ctx = await generate_context(context_client, context_model, doc_text, chunk)
        contextualized = f"{ctx}\n\n{chunk}"
        contextualized_chunks.append(contextualized)
        if i < len(chunks) - 1:
            await asyncio.sleep(CONTEXT_DELAY)

    embeddings = await embed_texts(embedding_client, embedding_model, contextualized_chunks)
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
