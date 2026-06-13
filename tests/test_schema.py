import os
import tempfile

import pytest

from scribbleslm import db


def _make():
    p = tempfile.mktemp(suffix=".db")
    c = db.connect(p)
    db.init_schema(c)
    return c, p


def test_one_vec_table_per_backend():
    c, p = _make()
    try:
        t1 = db.ensure_vec_table(c, "voyage-context-3", 1024)
        t2 = db.ensure_vec_table(c, "bge-m3-gguf-q5_k_m", 1024)
        assert t1 == "vec_voyage_context_3"
        assert t2 == "vec_bge_m3_gguf_q5_k_m"
        names = {r["name"] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert t1 in names and t2 in names
    finally:
        os.remove(p)


def test_dimension_mismatch_guard():
    c, p = _make()
    try:
        db.ensure_vec_table(c, "b", 8)
        with pytest.raises(ValueError):
            db.ensure_vec_table(c, "b", 16)
    finally:
        os.remove(p)


def test_chunk_requires_backend_id():
    c, p = _make()
    try:
        nb = c.execute("INSERT INTO notebooks(name) VALUES('n') RETURNING id").fetchone()["id"]
        src = c.execute(
            "INSERT INTO sources(notebook_id,url) VALUES(?,?) RETURNING id", (nb, "u")
        ).fetchone()["id"]
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            c.execute(
                "INSERT INTO chunks(source_id,notebook_id,chunk_index,original_text) "
                "VALUES(?,?,?,?)", (src, nb, 0, "x"),  # backend_id NULL -> NOT NULL violation
            )
    finally:
        os.remove(p)


def test_fts_roundtrip():
    c, p = _make()
    try:
        c.execute("INSERT INTO fts_chunks(original_text, chunk_id) VALUES(?,?)",
                  ("кримінальне правопорушення це діяння", 7))
        c.commit()
        rows = c.execute(
            "SELECT chunk_id FROM fts_chunks WHERE fts_chunks MATCH 'правопорушення'"
        ).fetchall()
        assert [r["chunk_id"] for r in rows] == [7]
    finally:
        os.remove(p)
