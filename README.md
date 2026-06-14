# ScribblesLM

Notebook-scoped RAG over your own corpora, delivered as a **stdio MCP server**. Documents
(URLs, PDFs, plain text) are ingested into named persistent notebooks; queries return raw,
**breadcrumbed** chunks for the calling agent to cite and synthesize. The breadcrumb is the
document's own heading hierarchy — `Part › Chapter › Section › Clause`, `Article › §`,
`Розділ › Стаття › Пункт`, whatever the source uses — so answers carry a real citation, not
just a similarity score.

It works on any **structured document** whose sections are marked by consistent keyword
headings (legal codes, contracts, regulations, standards, technical specs, manuals…). The
chunking profile is **induced from each document**, not hard-coded.

Embeddings are **sensitivity-routed** (see below): public docs → Voyage (fast, contextual);
private docs → a local **bge-m3 GGUF** that never leaves the host. Storage is **SQLite +
sqlite-vec + FTS5** — no Docker, no daemon, near-zero idle footprint.

## Tested scope & maturity

> **ScribblesLM has been built and validated end-to-end on Ukrainian legal codes only**
> (Criminal, Tax, Civil, Family, and Labour Codes). The structure induction, the
> retrieval/inflection measurements, and the full ingest→query pipeline were all exercised
> on that corpus. **Everything else is untested** — other document families (contracts,
> specs, manuals, prose), other jurisdictions, other languages, other PDF layouts. The
> design is deliberately general (pattern-based heading induction; language-agnostic dense
> embeddings), so it *should* extend — but treat any non-Ukrainian-legal use as unverified
> and **check retrieval quality on your own corpus before relying on it.**

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager / runner)
- A **Voyage API key** from **[voyageai.com](https://www.voyageai.com)** — **use a paid
  account** (add a payment method). The free tier (3 RPM / 10K TPM) is far too rate-limited
  for bulk ingestion; adding a card unlocks usable limits (~2000 RPM / millions of TPM).
  Cost stays ~\$0 for typical corpora — the models used carry large free-token allotments
  (e.g. 200M tokens for `voyage-context-3`); paying is essentially a *rate* unlock, not a
  bill.
- *(private path only)* a **bge-m3 GGUF** model file — see install step 3
- *(optional)* an OpenAI-compatible LLM key for private-path contextualization (any
  provider — see "Context LLM" below)

## Install

```bash
# 1. Clone
git clone https://github.com/elphamale/scribbleslm
cd scribbleslm

# 2. Configure env (secrets live OUTSIDE the repo; .env is git-ignored)
mkdir -p ~/.scribbleslm
cp .env.example ~/.scribbleslm/.env
#    then edit ~/.scribbleslm/.env and set VOYAGE_API_KEY=<your key>

# 3. (private path only) download the local embedding model (~445 MB, ungated)
uvx --from huggingface_hub hf download gpustack/bge-m3-GGUF bge-m3-Q5_K_M.gguf \
    --local-dir ~/.scribbleslm/models
#    Skip if you only ingest PUBLIC sources. Adding a private source without this
#    model returns a clear error, not a crash. (See "Embedding backends" for why GGUF.)

# 4. Smoke test — should print one JSON line listing 11 tools
{ printf '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}\n{"jsonrpc":"2.0","method":"notifications/initialized"}\n{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}\n'; sleep 3; } | uv run scribbleslm
```

The first `uv run` resolves dependencies (including a prebuilt `llama-cpp-python` CPU wheel
via `pyproject`'s configured index) — it may take a minute. Subsequent runs are instant.

## MCP client config

Add to your agent's MCP config. **Secrets are not inlined** — the server reads
`~/.scribbleslm/.env` at startup:

```json
{
  "mcpServers": {
    "scribbleslm": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/scribbleslm", "run", "scribbleslm"]
    }
  }
}
```

## Sensitivity routing (public vs private)

Every source is embedded by one of two backends, chosen **per document** by the `private`
flag on `source_add`:

- `private=false` (**default**) → **Voyage** (remote API): fast, contextual. For published /
  non-confidential documents.
- `private=true` → **local bge-m3 GGUF**: the document's text **never leaves the host**. For
  confidential / privileged material.

The default is set by `DEFAULT_PRIVATE`. Queries are guarded the same way: a query you flag
sensitive is never sent to the remote backend — it searches only locally-embedded content
plus the lexical index. The two backends produce different vector spaces; results are merged
by rank fusion, never by comparing raw scores across spaces.

**Trust boundary:** on a host you don't fully control (e.g. a rented VPS), `private=true`
limits *transmission* but does not make the host trustworthy — the real boundary is whether
you ingest sensitive material onto that host at all. Set the flag deliberately.

## Ingestion: when is it ready?

`source_add` returns a `source_id` **immediately** and embeds in the **background**. Two
phases affect result quality — poll **`source_status(source_id)`** to see where a source is
(`chunks_embedded/chunks_total`, `queryable`, the `pending/enriched/failed` rollup, and a
one-line summary); **`notebook_status(notebook_id)`** aggregates across a whole notebook
("is my corpus ready").

1. **Coverage (during embedding).** Chunks become queryable as soon as the first batch
   embeds (`queryable=true`), but a query run mid-ingest only searches the chunks embedded
   *so far* — it can miss sections not yet embedded. **Coverage is complete when
   `chunks_embedded == chunks_total`.** So early queries are usable but may have lower recall
   than queries run after the embed finishes; the improvement plateaus at full coverage (it
   does not keep getting better indefinitely).
2. **Enrichment quality (private path only).** Private docs are first embedded on raw text
   (`pending`), then — if you run `source_enrich` / pass `enrich=true` *and* have a context
   LLM configured — re-embedded **with surrounding context** (`enriched`), which improves
   retrieval quality. **Public docs are already contextually embedded at ingest time**, so
   they have no separate enrichment step and don't improve further after coverage completes.

## Embedding backends & context LLM

- **Public embeddings (Voyage):** model is configurable via `VOYAGE_MODEL` (default
  `voyage-context-3`, chosen by benchmark). Any Voyage embedding model works.
- **Private embeddings (local):** ships a **bge-m3 GGUF** backend run in-process via
  `llama-cpp-python`. GGUF was chosen **for the build environment** — modest CPU, limited
  RAM, no GPU, and no compiler to build from source (a prebuilt CPU wheel is used). The
  embedding layer is a pluggable interface (`embed_batch` / `embed_query`); on a host with
  more RAM/CPU or a GPU you could add an alternative local backend (e.g. non-GGUF bge-m3 via
  sentence-transformers, or a larger model). **Only the GGUF backend is implemented and
  tested today** — alternatives are an extension point, not a config switch.
- **Context LLM (private-path enrichment):** any **OpenAI-compatible** chat API — DeepSeek
  is only the default example. Point `CONTEXT_LLM_BASE_URL` / `CONTEXT_LLM_API_KEY` /
  `CONTEXT_LLM_MODEL` at any provider (OpenAI, OpenRouter, a self-hosted vLLM or Ollama
  OpenAI endpoint, etc.). It is used **only** to contextualize *private* documents during
  `source_enrich`; public documents never use it.

## Environment (`~/.scribbleslm/.env`)

| Variable | Required | Notes |
|---|---|---|
| `VOYAGE_API_KEY` | **yes** | public embedding path; paid Voyage account (see Prerequisites) |
| `VOYAGE_MODEL` | no | default `voyage-context-3` |
| `PRIVATE_EMBEDDING_MODEL_PATH` | private path | default `~/.scribbleslm/models/bge-m3-Q5_K_M.gguf` |
| `CONTEXT_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | private enrich | any OpenAI-compatible API; **placeholder by default** |
| `DEFAULT_PRIVATE` | no | default `false` (public) |
| `RERANKER_ENABLED` | no | default `false` (reranker off) |

**Unrun without a context LLM key:** `CONTEXT_LLM_API_KEY` ships as a placeholder. Until you
set it, **private-path enrichment** and **LLM profile synthesis** do not run (the rest works;
private docs are embedded and queryable, just not extra-contextualized). The local reranker's
model-load path is exercised only when `RERANKER_ENABLED=true`.

## How chunking works (why the breadcrumbs)

Each document is run through an **induction ladder** that derives its structure:
format-native (markdown headings / PDF table-of-contents) → a cached profile → **line-shape
mining** (detects recurring `Keyword + Number/Roman` headings — `Article 12`, `Section 4`,
`Розділ II`, `§ 3` — with no model) → optional LLM synthesis → semantic segmentation →
plain token-splitting as the floor. The winning profile segments the document into
heading-aligned chunks, each carrying its breadcrumb; oversized sections are token-split but
inherit the breadcrumb. A small pre-warmed profile ships for one common document family;
others are induced automatically and cached by structural fingerprint.

## Tools (11)

`notebook_create` · `notebook_list` · `notebook_delete` · `source_add` · `source_list` ·
`source_refresh` · `source_delete` · `source_enrich` · `source_status` · `notebook_status`
· `notebook_query`

## Known limitations

- Without a context-LLM key: private-path enrichment + LLM profile synthesis unrun.
- Documents whose structure is carried only by **typography** (font size/bold) or by
  **bare digit-led numbering** (`1.`, `1.1.`), and PDFs with **no embedded table of
  contents**, fall back to plain token chunking — **no fine-grained breadcrumbs** (coarser
  citation). Heading-keyword–structured documents get full breadcrumbs.
- For **morphologically-rich / inflected languages**, the pure-lexical channel (FTS5, no
  stemming) misses inflected word forms; the default **hybrid** query mode covers this via
  dense embeddings.
