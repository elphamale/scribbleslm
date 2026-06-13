# ScribblesLM v2 — Backlog

## Milestone B — deferred optimization (do NOT let it compete with profile induction / temporal layer)

### Parallel context-3 embedding via window-aligned partitioning
voyage-context-3 only attends within ~32K tokens (~55 chunks). A large document can
be partitioned at ~32K boundaries and those segments embedded **concurrently** —
losing only context the model wasn't using anyway. The concurrency seam already
exists (Dispatcher semaphore + token governor, R1); this is mostly a config flip
(`VOYAGE_CONCURRENCY` > 1) plus the partitioner that emits window-aligned documents.

- Diagnosis: context-3 ran ~4.8 chunks/sec flat across 400 and 5,460 chunks
  (linear → serialization-bound, ~6% of the 3M TPM budget). Concurrency is the win,
  not a rate problem.
- Expected: Tax Code full-embed ~19.5 min → ~2–3 min.
- NOT a Milestone A blocker: it's optimization on a BACKGROUND process; progressive
  insert already keeps time-to-first-queryable < 5 s.
- Honor R2: a single window (document) is still embedded atomically in one call;
  only distinct windows run in parallel.

## Roadmap (per locked spec)
- **B** — ingest-now/enrich-later; `source_enrich`; batched DeepSeek contextualization
  (PRIVATE path only — public uses context-3 natively); cost log; MCP tools rebuild;
  parallel context-3 (above).
- **C** — hybrid retrieval (sqlite-vec + FTS5 + RRF, k=60); `mode` param; reranker
  sensitivity-routed (rerank-2.5 public / bge-reranker-v2-m3 local private), default off.
- **D** — profile induction ladder (format-native → cached → line-shape mining →
  LLM synth → semantic → token fallback), validation gate, regional fallback.

## Assumptions to validate
- **voyage-4-lite (public fallback) is UNMEASURED on the Tax Code harness.** We benched
  voyage-3.5 (16/25) but never lite. Chosen as the fallback default on cost/free-tier
  grounds ($0.02/M + 200M free vs 3.5's $0.06/M, no free tier) for a rarely-firing
  fallback — this is an assumption, not a tested retrieval result. Bench it before
  relying on it as a primary.
- Auto-failover to the fallback model mid-source is NOT implemented (and is unsafe:
  switching models mid-document mixes incomparable vector spaces). The fallback is a
  config switch (set VOYAGE_MODEL), not runtime failover.

## Known carry-overs
- v1 modules `server.py`, `ingest.py`, `query.py` are Postgres-coupled and dormant;
  they are rewritten in Milestone B against the SQLite + routing foundation.
  `sources.py` (URL/PDF/text extraction) is reused as-is.
- Ukrainian FTS5 lexical is inflection-limited (no stemming; bge-m3 GGUF is dense-only).
  Mitigation (prefix/trigram tokenizer or query expansion) is a Milestone C task.
