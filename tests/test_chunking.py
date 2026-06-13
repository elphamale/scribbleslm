"""Induction -> chunk integration: the storage-size guarantee and structural status.

This is the item-1 enforcement point: a segment far over the gate's ceiling must
still yield only <=512-token chunks once it reaches build_chunks (the bound the
profile gate does NOT provide — it lives here, at ingest)."""
import tiktoken

from scribbleslm.induction.ladder import induce
from scribbleslm.ingest import CHUNK_TOKENS, build_chunks

_enc = tiktoken.get_encoding("cl100k_base")


def _doc_with_huge_article() -> str:
    parts = ["Преамбула документа перед першою статтею.\n"]
    for n in range(1, 7):  # >=5 статті so rung-3 mining fires
        body = "слово " * (6000 if n == 3 else 20)  # Стаття 3 is huge (flat)
        parts.append(f"Стаття {n}. Заголовок статті номер {n}\n{body}\n")
    return "".join(parts)


def test_storage_guarantee_every_chunk_capped():
    doc = _doc_with_huge_article()
    ind = induce(doc)  # no cache -> rung-3 line mining
    assert ind.profile is not None and any(l.name == "стаття" for l in ind.profile.levels)
    chunks = build_chunks(doc, ind)
    # the guarantee: NO chunk exceeds the token cap, even from the 6000-word article
    assert all(len(_enc.encode(c.text)) <= CHUNK_TOKENS for c in chunks)


def test_huge_article_becomes_many_structural_chunks():
    doc = _doc_with_huge_article()
    chunks = build_chunks(doc, induce(doc))
    art3 = [c for c in chunks if c.breadcrumb and "стаття 3" in c.breadcrumb.lower()]
    assert len(art3) > 5                      # one huge section -> many capped chunks
    assert all(c.structural for c in art3)    # all carry the section breadcrumb -> structural


def test_preamble_is_non_structural():
    doc = _doc_with_huge_article()
    chunks = build_chunks(doc, induce(doc))
    assert any((not c.structural) and c.breadcrumb is None for c in chunks)


def test_token_fallback_doc_is_non_structural():
    # no repeated headings -> no profile -> rung-6 token fallback
    doc = "Просто суцільний текст без жодної структури. " * 200
    ind = induce(doc)
    chunks = build_chunks(doc, ind)
    assert ind.profile is None
    assert chunks and all((not c.structural) and c.breadcrumb is None for c in chunks)
    assert all(len(_enc.encode(c.text)) <= CHUNK_TOKENS for c in chunks)
