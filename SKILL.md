---
name: scribbleslm
description: "Grounded retrieval over ingested structured-document corpora (legal codes, contracts, regulations, specs, manuals) with hierarchical breadcrumb citations, via the ScribblesLM MCP server. Not a web/general-knowledge tool."
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [rag, retrieval, citation, documents, legal, mcp]
    related_skills: []
---

# ScribblesLM — grounded corpus retrieval with citations

## When to use (and not)

- **Use** for grounded retrieval over corpora the user has ingested into ScribblesLM
  notebooks — any structured document family (legislation, contracts, regulations,
  standards, technical specs, manuals, internal policies). Answers come back as raw chunks
  **with breadcrumb citations** drawn from the document's own heading hierarchy
  (`Part › Chapter › Section`, `Article › §`, `Розділ › Стаття › Пункт`, etc.) — you cite
  and synthesize from these.
- **Do NOT use** for web search, general knowledge, or any document not ingested. ScribblesLM
  only knows its notebooks. If the needed corpus isn't ingested, ingest it first (or use a
  different tool). It is a *citator over a known corpus*, not a knowledge source.

## ID hygiene

Never guess `notebook_id` / `source_id`. If you don't already hold the id from this
conversation, call `notebook_list` (or `source_list`) first.

## Ingest, then poll (ingestion runs in the background)

- `source_add(notebook_id, url, private=…, enrich=…)` returns `source_id` **immediately** —
  embedding runs in the background. **Do not block** waiting for it.
- For anything but a tiny doc, poll `source_status(source_id)`: it reports
  `chunks_embedded / chunks_total`, `queryable`, the enrichment rollup, and a one-line
  `summary` you can relay verbatim.
- Tell the user it is **searchable as soon as `queryable` is true** (`chunks_embedded > 0`) —
  partial coverage is queryable; you do **not** need to wait for full enrichment to start
  answering. `notebook_status(notebook_id)` answers "is my whole corpus ready."

## The `private` flag — set deliberately, never guess

- **Default is public** (`private=false`): embedded via the remote Voyage API (fast). Use for
  published or non-confidential material.
- `private=true`: embedded **locally**, the document **never leaves the host**. Use **only**
  for material that must not be transmitted (privileged / confidential / client documents).
- **Trust-boundary caveat:** on a rented/cloud host the document already resides on
  infrastructure the operator may not fully control; `private=true` limits *further*
  transmission but is **not** a substitute for not ingesting truly sensitive material onto an
  untrusted host. Decide per-document from what the user tells you; never guess the flag.

## Query — hybrid by default, always

- `notebook_query(notebook_id, query, top_k=…, mode="hybrid")`: **always use `mode=hybrid`**
  (the default). It fuses dense semantic search with lexical search; dense carries word forms
  the lexical channel misses. This matters most for **inflected / morphologically-rich
  languages**, where the same term varies by grammatical case (measured: dense cosine
  0.86–0.92 across inflected forms of one term, vs near-zero lexical overlap).
- Use **`mode=lexical` ONLY for exact identifier lookups** — a clause/section number or code
  (`14.1.159`, `§ 230`, an article number). **Never** for concept / word search, where the
  lexical channel fragments across surface forms.
- **The breadcrumb is the citation.** Each result's breadcrumb (the heading hierarchy,
  carried in `contextualized_text`) plus `source_url` / `display_name` is what you cite.
  **Surface it; never strip it.** Results are raw chunks — synthesize a coherent, **cited**
  answer; never hand the user raw chunks as the answer.
- `notebook_query` is scoped to **one** notebook. Never query across notebooks — run separate
  queries and synthesize, citing each source.

## Honest limitations you must respect

- **Validated only on Ukrainian legal codes.** The whole pipeline was tested on that corpus;
  other document families, jurisdictions, and languages are **unverified**. The design is
  general and should extend, but do not overstate confidence on untested corpora — if
  results look thin or mis-cited on a new corpus, say so to the user rather than asserting.
- **Documents without keyword headings** — structure carried only by typography (font/bold)
  or by bare digit-led numbering (`1.`, `1.1.`), and PDFs with no embedded table of contents
  — ingest via token fallback and carry **no fine-grained breadcrumbs**. Their citations are
  coarser (source / top-section level). **Do NOT fabricate fine-grained clause numbers** for
  such sources — cite exactly what the breadcrumb says, and tell the user the citation is
  coarse if a precise clause is requested.
- **Pure-lexical inflected-language queries are degraded** (no stemming). Stay on hybrid; do
  not switch to `mode=lexical` for concept search.
- `enrichment_status: failed` chunks remain retrievable on plain embeddings — they are not
  lost; they are just not extra-contextualized.

## Reranker (optional, default OFF)

A precision reranker (remote for public, local for private) exists but is off by default
(adds latency and token cost). Hybrid + RRF is the standard path; reach for the reranker
only when rank-1 precision is critical and the user accepts the cost.
