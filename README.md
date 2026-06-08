# ScribblesLM

ScribblesLM is a self-hosted, notebook-scoped RAG pipeline with contextual retrieval, delivered as an MCP server. Documents (URLs, PDFs, plain text) are ingested into named persistent notebooks; queries return semantically ranked raw chunks for the calling agent to synthesize. Transport is stdio, making it a drop-in tool for any MCP-compatible agent.

## Prerequisites

- Docker (for the pgvector container)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.com) (local embeddings)
- DeepSeek API key (context generation during ingestion)

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/elphamale/scribbleslm
cd scribbleslm

# 2. Install Ollama and pull the embedding model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull bge-m3

# 3. Configure
cp .env.example .env
# Edit .env and fill in your DeepSeek API key

# 4. Start the database
docker compose up -d

# 5. Run the server
uv run scribbleslm
```

Add to your agent's MCP config:

```json
{
  "mcpServers": {
    "scribbleslm": {
      "command": "uv",
      "args": ["--directory", "/path/to/scribbleslm", "run", "scribbleslm"],
      "env": {
        "SCRIBBLESLM_DB_URL": "postgresql://scribbleslm:scribbleslm@localhost:5433/scribbleslm",
        "CONTEXT_LLM_BASE_URL": "https://api.deepseek.com/v1",
        "CONTEXT_LLM_API_KEY": "your_deepseek_key",
        "CONTEXT_LLM_MODEL": "deepseek-v4-flash",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "bge-m3"
      }
    }
  }
}
```

## MCP Tools

**Notebook management**
- `notebook_create` — create a named notebook
- `notebook_list` — list all notebooks with source counts
- `notebook_delete` — delete a notebook and all its data

**Source management**
- `source_add` — fetch and ingest a URL, PDF, or text file into a notebook
- `source_list` — list sources in a notebook with chunk counts
- `source_refresh` — re-fetch a source and re-ingest only if content changed
- `source_delete` — delete a source and its chunks

**Query**
- `notebook_query` — semantic search within a notebook; returns raw ranked chunks

## Switching embedding models

Change `OLLAMA_EMBEDDING_MODEL` in your `.env` and pull the new model with `ollama pull <model>`. If the new model has a different vector dimension, you must drop and recreate the database:

```bash
docker compose down -v
docker compose up -d
```

Then re-ingest your sources. ScribblesLM will detect the new dimension on first use and store it.

To point at a remote Ollama instance, set `OLLAMA_BASE_URL` to its address.

## Switching context LLM

Any OpenAI-compatible API works. Set in `.env`:

```env
CONTEXT_LLM_BASE_URL=https://api.openai.com/v1
CONTEXT_LLM_API_KEY=your_openai_key
CONTEXT_LLM_MODEL=gpt-4o-mini
```

No restart of the database required.

## Notes on ingestion cost and speed

- One DeepSeek call is made per chunk during ingestion (contextual retrieval technique).
- Up to 8 context generation calls run concurrently; 429 responses are retried with exponential backoff.
- Embeddings are generated locally via Ollama — no rate limit, ~10–50ms per chunk.
- The Ukrainian Criminal Code (~1031 chunks) ingests in approximately 5–10 minutes depending on DeepSeek server load.
- `source_refresh` is a no-op if the content hash has not changed — safe to call frequently.
