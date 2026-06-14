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

Embeddings are **sensitivity-routed**: public docs → Voyage (fast, contextual); private docs
→ a local **bge-m3 GGUF** that never leaves the host. Storage is **SQLite + sqlite-vec +
FTS5** — no Docker, no daemon, near-zero idle footprint.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager / runner)
- A **Voyage API key** — required for the public embedding path
- *(private path only)* a **bge-m3 GGUF** model file — see install step 3
- *(optional)* a DeepSeek / OpenAI-compatible key for private-path contextualization

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
#    model returns a clear error, not a crash.

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

## Environment (`~/.scribbleslm/.env`)

| Variable | Required | Notes |
|---|---|---|
| `VOYAGE_API_KEY` | **yes** | public embedding path (voyage-context-3) |
| `VOYAGE_MODEL` | no | default `voyage-context-3` |
| `PRIVATE_EMBEDDING_MODEL_PATH` | private path | default `~/.scribbleslm/models/bge-m3-Q5_K_M.gguf` |
| `CONTEXT_LLM_API_KEY` | private enrich | **placeholder by default** — see below |
| `DEFAULT_PRIVATE` | no | default `false` (public) |
| `RERANKER_ENABLED` | no | default `false` (reranker off) |

**Unrun without a DeepSeek key:** `CONTEXT_LLM_API_KEY` ships as a placeholder. Until you
set it, the **private-path DeepSeek enrichment** and **LLM profile synthesis** paths do not
run (the rest works; private docs are embedded and queryable, just not DeepSeek-
contextualized). The local reranker's model-load path is likewise exercised only when
`RERANKER_ENABLED=true`.

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

Ingestion is **backgrounded**: `source_add` returns `source_id` immediately and chunks
become queryable as they embed — poll `source_status(source_id)` for progress.

## Known limitations

- Without `CONTEXT_LLM_API_KEY`: private-path enrichment + LLM profile synthesis unrun.
- Documents whose structure is carried only by **typography** (font size/bold) or by
  **bare digit-led numbering** (`1.`, `1.1.`), and PDFs with **no embedded table of
  contents**, fall back to plain token chunking — **no fine-grained breadcrumbs** (coarser
  citation). Heading-keyword–structured documents get full breadcrumbs.
- For **morphologically-rich / inflected languages**, the pure-lexical channel (FTS5, no
  stemming) misses inflected word forms; the default **hybrid** mode covers this via dense
  embeddings.
