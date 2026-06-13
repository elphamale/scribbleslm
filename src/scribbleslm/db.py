"""SQLite + sqlite-vec + FTS5 storage layer.

Design notes:
- INTEGER PRIMARY KEY on `chunks` doubles as the `vec0` rowid (no separate map).
- `chunks.backend_id` pins the vector space; vectors live in one `vec0` table
  PER backend_id (dims/spaces differ and must never be mixed in a KNN).
- FTS5 is shared (lexical is space-independent), indexing `original_text`.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import sqlite_vec


SCHEMA = """
CREATE TABLE IF NOT EXISTS notebooks (
    id          INTEGER PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sources (
    id            INTEGER PRIMARY KEY,
    notebook_id   INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    url           TEXT NOT NULL,
    display_name  TEXT,
    content_hash  TEXT,
    private       INTEGER NOT NULL DEFAULT 0,   -- 0 public / 1 private
    backend_id    TEXT,                         -- backend used to embed this source
    ingested_at   TEXT NOT NULL DEFAULT (datetime('now')),
    -- per-source cost log
    api_calls         INTEGER NOT NULL DEFAULT 0,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cache_hit_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(notebook_id, url)
);

CREATE TABLE IF NOT EXISTS chunks (
    id                   INTEGER PRIMARY KEY,   -- == vec0 rowid
    source_id            INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    notebook_id          INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    backend_id           TEXT NOT NULL,         -- pins the vector space
    chunk_index          INTEGER NOT NULL,
    token_start          INTEGER,
    token_end            INTEGER,
    original_text        TEXT NOT NULL,
    contextualized_text  TEXT,
    enrichment_status    TEXT NOT NULL DEFAULT 'pending',  -- structural|pending|enriched|failed
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS chunks_notebook_idx ON chunks(notebook_id);
CREATE INDEX IF NOT EXISTS chunks_source_idx   ON chunks(source_id);
CREATE INDEX IF NOT EXISTS chunks_backend_idx  ON chunks(notebook_id, backend_id);

CREATE TABLE IF NOT EXISTS config (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cost_log (
    id                INTEGER PRIMARY KEY,
    source_id         INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    model             TEXT NOT NULL,             -- model identity, not just totals
    api_calls         INTEGER NOT NULL DEFAULT 0,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cache_hit_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL NOT NULL DEFAULT 0.0,
    recorded_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS cost_log_source_idx ON cost_log(source_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    original_text,
    chunk_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# config k/v
# ---------------------------------------------------------------------------

def config_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def config_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO config(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# per-backend vec0 tables
# ---------------------------------------------------------------------------

def vec_table_name(backend_id: str) -> str:
    return "vec_" + re.sub(r"[^a-zA-Z0-9]", "_", backend_id)


def ensure_vec_table(conn: sqlite3.Connection, backend_id: str, dim: int) -> str:
    """Create (once) the vec0 table for a backend_id and pin its dimension.

    Raises on dimension mismatch — switching a backend's model requires
    re-embedding (source_refresh --re-embed).
    """
    dim_key = f"dim::{backend_id}"
    stored = config_get(conn, dim_key)
    if stored is not None and int(stored) != dim:
        raise ValueError(
            f"Embedding dimension mismatch for backend '{backend_id}': "
            f"stored {stored}, got {dim}. Re-embed (source_refresh --re-embed) "
            f"to change the model for this backend."
        )
    table = vec_table_name(backend_id)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(embedding float[{dim}])"
    )
    if stored is None:
        config_set(conn, dim_key, str(dim))
    conn.commit()
    return table


# ---------------------------------------------------------------------------
# vectors + retrieval helpers
# ---------------------------------------------------------------------------

def insert_vector(conn: sqlite3.Connection, backend_id: str, rowid: int, vec: list[float]) -> None:
    conn.execute(
        f"INSERT INTO {vec_table_name(backend_id)}(rowid, embedding) VALUES(?, ?)",
        (rowid, sqlite_vec.serialize_float32(vec)),
    )


def replace_vector(conn: sqlite3.Connection, backend_id: str, rowid: int, vec: list[float]) -> None:
    table = vec_table_name(backend_id)
    conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
    conn.execute(f"INSERT INTO {table}(rowid, embedding) VALUES(?, ?)",
                 (rowid, sqlite_vec.serialize_float32(vec)))


def insert_fts(conn: sqlite3.Connection, rowid: int, text: str) -> None:
    conn.execute("INSERT INTO fts_chunks(original_text, chunk_id) VALUES(?, ?)", (text, rowid))


def knn(conn: sqlite3.Connection, backend_id: str, query_vec: list[float], k: int
        ) -> list[tuple[int, float]]:
    table = vec_table_name(backend_id)
    rows = conn.execute(
        f"SELECT rowid, distance FROM {table} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (sqlite_vec.serialize_float32(query_vec), k),
    ).fetchall()
    return [(r["rowid"], r["distance"]) for r in rows]


def present_backend_ids(conn: sqlite3.Connection, notebook_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT backend_id FROM chunks WHERE notebook_id = ?", (notebook_id,)
    ).fetchall()
    return [r["backend_id"] for r in rows]


# ---------------------------------------------------------------------------
# cascade-safe deletion of virtual-table rows (vec0/FTS5 don't honor FK CASCADE)
# ---------------------------------------------------------------------------

def _chunk_rows(conn: sqlite3.Connection, where: str, param) -> list[tuple[int, str]]:
    rows = conn.execute(
        f"SELECT id, backend_id FROM chunks WHERE {where}", (param,)
    ).fetchall()
    return [(r["id"], r["backend_id"]) for r in rows]


def purge_vectors_for_source(conn: sqlite3.Connection, source_id: int) -> None:
    _purge(conn, _chunk_rows(conn, "source_id = ?", source_id))


def purge_vectors_for_notebook(conn: sqlite3.Connection, notebook_id: int) -> None:
    _purge(conn, _chunk_rows(conn, "notebook_id = ?", notebook_id))


def _purge(conn: sqlite3.Connection, chunk_rows: list[tuple[int, str]]) -> None:
    from collections import defaultdict
    by_backend: dict[str, list[int]] = defaultdict(list)
    for cid, bid in chunk_rows:
        by_backend[bid].append(cid)
    for bid, ids in by_backend.items():
        table = vec_table_name(bid)
        conn.executemany(f"DELETE FROM {table} WHERE rowid = ?", [(i,) for i in ids])
    conn.executemany("DELETE FROM fts_chunks WHERE chunk_id = ?",
                     [(cid,) for cid, _ in chunk_rows])

