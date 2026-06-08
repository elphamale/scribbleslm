import os
from openai import AsyncOpenAI

from .db import get_conn
from .ingest import embed_texts, validate_embedding_dim


async def query_notebook(notebook_id: str, query: str, top_k: int = 10) -> list[dict]:
    client = AsyncOpenAI(
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.environ["EMBEDDING_API_KEY"],
    )
    model = os.environ["EMBEDDING_MODEL"]

    embeddings = await embed_texts(client, model, [query])
    query_embedding = embeddings[0]
    await validate_embedding_dim(len(query_embedding))

    conn = await get_conn()
    rows = await conn.execute(
        """
        SELECT
            c.id AS chunk_id,
            c.source_id,
            s.url AS source_url,
            s.display_name,
            c.original_text,
            c.contextualized_text,
            1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        WHERE c.notebook_id = %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, notebook_id, query_embedding, top_k),
    )
    results = await rows.fetchall()
    return [
        {
            "chunk_id": str(r["chunk_id"]),
            "source_id": str(r["source_id"]),
            "source_url": r["source_url"],
            "display_name": r["display_name"],
            "original_text": r["original_text"],
            "contextualized_text": r["contextualized_text"],
            "similarity": float(r["similarity"]),
        }
        for r in results
    ]
