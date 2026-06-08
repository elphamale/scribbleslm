# ScribblesLM — Hermes Agent Skill

## When to use ScribblesLM vs direct context window

Use ScribblesLM when:
- Corpus is larger than ~50 pages
- Knowledge must persist across sessions
- User is explicitly managing notebooks (named collections of sources)

Use context window when:
- Single short document, one-off query
- No persistence needed
- Document fits comfortably in the window

## Notebook lifecycle

```
notebook_create → source_add (blocks until ingestion complete) → notebook_query
```

Never query a notebook before at least one `source_add` has returned successfully.

## Source refresh trigger

Call `source_refresh` when:
- The user mentions a document was amended or updated
- `ingested_at` is older than the user's stated amendment date

`source_refresh` is a no-op if content is unchanged — safe to call proactively.

## Query behaviour

`notebook_query` returns raw ranked chunks. Always:
- Synthesize chunks into a coherent answer before responding
- Cite `source_url` and `display_name` for every claim drawn from results
- Never present raw chunk text verbatim as the final answer

## Scope isolation

`notebook_query` is always scoped to a single `notebook_id`. Never attempt cross-notebook queries. If the user's question spans multiple notebooks, run separate queries and merge the synthesis yourself.

## Notebook ID hygiene

Never guess or construct a notebook ID. If the target notebook ID is not already known from earlier in the conversation, call `notebook_list` first to resolve the name to an ID.

## Ingestion timing

`source_add` is blocking — it runs the full fetch → chunk → contextualize → embed pipeline before returning. For large documents (>50 pages) this may take several minutes. Inform the user before calling `source_add` on a large source so they are not surprised by the delay.
