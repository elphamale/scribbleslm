import os
import asyncio
from typing import Optional
import psycopg
from psycopg.rows import dict_row


_pool: Optional[psycopg.AsyncConnection] = None


def get_db_url() -> str:
    url = os.environ.get("SCRIBBLESLM_DB_URL")
    if not url:
        raise RuntimeError(
            "SCRIBBLESLM_DB_URL is not set. "
            "Set it to postgresql://user:pass@host:port/dbname"
        )
    return url


async def get_conn() -> psycopg.AsyncConnection:
    global _pool
    if _pool is None or _pool.closed:
        _pool = await psycopg.AsyncConnection.connect(
            get_db_url(), row_factory=dict_row, autocommit=False
        )
    return _pool


async def init_schema() -> None:
    conn = await get_conn()
    async with conn.transaction():
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS notebooks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                notebook_id UUID REFERENCES notebooks(id) ON DELETE CASCADE,
                url TEXT NOT NULL,
                display_name TEXT,
                content_hash TEXT,
                ingested_at TIMESTAMPTZ DEFAULT now(),
                UNIQUE(notebook_id, url)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_id UUID REFERENCES sources(id) ON DELETE CASCADE,
                notebook_id UUID REFERENCES notebooks(id) ON DELETE CASCADE,
                chunk_index INTEGER,
                original_text TEXT,
                contextualized_text TEXT,
                embedding vector(1024),
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_notebook_id_idx
            ON chunks(notebook_id)
        """)
        # ivfflat index requires rows to exist; create only if chunks exist
        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'chunks_embedding_idx'
                ) THEN
                    BEGIN
                        CREATE INDEX chunks_embedding_idx
                        ON chunks USING ivfflat (embedding vector_cosine_ops);
                    EXCEPTION WHEN OTHERS THEN
                        NULL;
                    END;
                END IF;
            END$$
        """)


async def config_get(key: str) -> Optional[str]:
    conn = await get_conn()
    row = await conn.execute(
        "SELECT value FROM config WHERE key = %s", (key,)
    )
    result = await row.fetchone()
    return result["value"] if result else None


async def config_set(key: str, value: str) -> None:
    conn = await get_conn()
    await conn.execute(
        "INSERT INTO config(key, value) VALUES(%s, %s) "
        "ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )
    await conn.commit()
