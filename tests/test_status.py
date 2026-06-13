"""source_status / notebook_status — rollup correctness + observable background-embed progress."""
import asyncio
import os
import tempfile

from scribbleslm import db
from scribbleslm.config import get_settings, reset_settings_cache
from scribbleslm.ingest import ingest_source, run_ingest


def _db():
    c = db.connect(tempfile.mktemp(suffix=".db"))
    db.init_schema(c)
    return c


def _source(c, statuses, private=0, backend="voyage-context-3", ingest_state="ready"):
    nb = c.execute("INSERT INTO notebooks(name) VALUES(?) RETURNING id",
                   (f"n{os.urandom(3).hex()}",)).fetchone()["id"]
    src = c.execute("INSERT INTO sources(notebook_id,url,private,backend_id,ingest_state) "
                    "VALUES(?,?,?,?,?) RETURNING id", (nb, "u", private, backend, ingest_state)).fetchone()["id"]
    for i, s in enumerate(statuses):
        c.execute("INSERT INTO chunks(source_id,notebook_id,backend_id,chunk_index,original_text,"
                  "enrichment_status) VALUES(?,?,?,?,?,?)", (src, nb, backend, i, "t", s))
    c.commit()
    return nb, src


# ---- rollup correctness (post-embed sources) -----------------------------

def test_embedded_count_and_queryable():
    c = _db()
    _, src = _source(c, ["structural", "structural", "structural"])
    st = db.source_status(c, src)
    assert st["chunks_embedded"] == 3 and st["chunks_total"] == 3 and st["queryable"] is True


def test_fully_enriched():
    c = _db()
    _, src = _source(c, ["enriched", "enriched", "enriched"], private=1, backend="bge-m3-gguf-q5_k_m")
    st = db.source_status(c, src)
    assert st["chunks_enriched"] == 3 and st["chunks_pending"] == 0
    assert "enrichment complete" in st["summary"]


def test_failed_chunk_counts_and_stays_queryable():
    c = _db()
    _, src = _source(c, ["enriched", "enriched", "failed", "pending"], private=1)
    st = db.source_status(c, src)
    assert st["chunks_failed"] == 1 and st["chunks_pending"] == 1 and st["queryable"] is True


def test_notebook_rollup():
    c = _db()
    nb, _ = _source(c, ["enriched", "pending"])
    src2 = c.execute("INSERT INTO sources(notebook_id,url,private,backend_id,ingest_state) "
                     "VALUES(?,?,?,?,?) RETURNING id", (nb, "u2", 1, "bge", "ingesting")).fetchone()["id"]
    c.execute("INSERT INTO chunks(source_id,notebook_id,backend_id,chunk_index,original_text,"
              "enrichment_status) VALUES(?,?,?,?,?,?)", (src2, nb, "bge", 0, "t", "failed"))
    c.commit()
    st = db.notebook_status(c, nb)
    assert st["source_count"] == 2 and st["chunks_total"] >= 3 and st["chunks_failed"] == 1
    assert st["source_states"] == {"ready": 1, "ingesting": 1}


def test_missing():
    c = _db()
    assert db.source_status(c, 999) is None and db.notebook_status(c, 999) is None


# ---- background embed: observable mid-EMBED progress ----------------------

class _FakeRouter:
    """Minimal router for ingest: dim-8 zero vectors, optional per-window sleep/fail."""
    def __init__(self, dim=8, sleep=0.0, fail_after=None):
        self.dim, self.sleep, self.fail_after, self.calls = dim, sleep, fail_after, 0

    def effective_private(self, p):
        return bool(p)

    async def embed_source(self, source_id, private, documents):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("embed boom")
        if self.sleep:
            await asyncio.sleep(self.sleep)
        return ([[[0.1] * self.dim for _ in documents[0]]], "voyage-context-3", bool(private))


def _big_txt(lines=3000):
    p = tempfile.mktemp(suffix=".txt")
    with open(p, "w") as f:
        for i in range(lines):
            f.write(f"Це загальне положення законодавства України, рядок номер {i}.\n")
    return p


def _settings(monkeypatch):
    monkeypatch.setenv("SCRIBBLESLM_HOME", tempfile.mkdtemp())
    reset_settings_cache()
    return get_settings()


async def test_mid_embed_progress_observable(monkeypatch):
    c = _db()
    settings = _settings(monkeypatch)
    nb = c.execute("INSERT INTO notebooks(name) VALUES('n') RETURNING id").fetchone()["id"]
    src = c.execute("INSERT INTO sources(notebook_id,url,ingest_state) VALUES(?,?, 'pending') RETURNING id",
                    (nb, _big_txt())).fetchone()["id"]
    url = c.execute("SELECT url FROM sources WHERE id=?", (src,)).fetchone()["url"]
    router = _FakeRouter(sleep=0.08)
    task = asyncio.create_task(run_ingest(c, router, settings, nb, src, url, False, False))
    caught = False
    for _ in range(400):
        await asyncio.sleep(0.01)
        st = db.source_status(c, src)
        if st["chunks_planned"] > 0 and 0 < st["chunks_embedded"] < st["chunks_total"]:
            assert st["queryable"] is True and st["ingest_state"] == "ingesting"
            caught = True
            break
    await task
    assert caught, "never observed chunks_embedded < chunks_total mid-ingest"
    fin = db.source_status(c, src)
    assert fin["chunks_embedded"] == fin["chunks_total"] and fin["ingest_state"] == "ready"


async def test_mid_embed_failure_keeps_partial_queryable(monkeypatch):
    c = _db()
    settings = _settings(monkeypatch)
    nb = c.execute("INSERT INTO notebooks(name) VALUES('n') RETURNING id").fetchone()["id"]
    src = c.execute("INSERT INTO sources(notebook_id,url,ingest_state) VALUES(?,?, 'pending') RETURNING id",
                    (nb, _big_txt())).fetchone()["id"]
    url = c.execute("SELECT url FROM sources WHERE id=?", (src,)).fetchone()["url"]
    out = await run_ingest(c, _FakeRouter(fail_after=1), settings, nb, src, url, False, False)
    assert "error" in out
    st = db.source_status(c, src)
    assert st["ingest_state"] == "failed" and "boom" in (st["ingest_error"] or "")
    assert st["chunks_embedded"] > 0 and st["queryable"] is True   # partial chunks NOT rolled back
