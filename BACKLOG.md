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

## Milestone D — rungs 1/4/5 deferred (with evidence)
- **Rung 1 (format-native) — DEFERRED, refuted against the real corpus.** Tested the
  committed second shape: NBU постанова PDF (bank.gov.ua Resolution_25072025_80,
  /admin_uploads/law/25072025_80.pdf). Result: 22 pages, NO embedded TOC (get_toc=0),
  flat typography (847/849 lines at 14.0pt — font heuristic finds nothing), 0 line-start
  keyword headings, 88 digit-led `1.`/`1.1.` lines; розділ/пункт appear 35/39× but never
  as line-start headings. So get_toc, markdown, AND font-heuristic all fail on the actual
  shape → rung-6 token fallback. Building rung 1 would be dead code for this corpus.
- **Rung 4 (LLM profile synthesis) — DEFERRED.** Un-live-testable (no DeepSeek key on the
  box). Joins the same deferred bucket as private-path DeepSeek enrichment; both get tested
  together once a key lands. Do not accumulate a second unverified DeepSeek path.
- **Rung 5 (semantic segmentation) — DEFERRED (YAGNI).** No structureless source in the
  corpus (legislation is the least structureless text there is). Earns its place when
  commentary/transcripts/articles actually arrive.
- **Digit-led numbered-point detection — now corpus-justified** (the NBU PDF needs it), but
  tricky: розділ headings there aren't line-started, and naive `^\d+\.` matching risks
  false positives on in-body numbered lists. Scope to docs/regions where keyword mining
  returned nothing. Test fixture: the NBU PDF above. NBU regs meanwhile work via rung-6
  (embedded + context-3-contextualized + queryable, just no breadcrumbs) — functional for
  the few small regs in scope.

## Milestone D refinements (induction)
- **Digit-led numbered-point segmentation.** Line-mining only treats alpha-keyword+
  number lines as headings (Стаття/Розділ/Підрозділ), by design — detecting bare
  "1." / "1.1." lines would false-positive on every list. Consequence: ПКУ's flat
  transitional tail (numbered points, no headings) stays one ~134K-token region
  (down from a 585K blob after regional fallback), token-split at integration. If
  finer breadcrumbs there ever matter, add a guarded numbered-point detector scoped
  to regions that already failed keyword mining. After the working gate + regional
  fallback, ПКУ's tail segments correctly into `підрозділ`, but its largest
  підрозділ (~224K tokens of numbered points) stays one flat region, token-split at
  ingest under its correct breadcrumb.
- Profile cache key is the dominant signature (`стаття|<NUM>`); MEASURED to generalize
  across ККУ/ЦКУ/СКУ/КЗпП/ПКУ (all rung-2 reuse, 96–100% align except ПКУ's genuinely
  non-article 75%). Enriching to top-2 would separate structural families at the cost
  of less sharing — not needed today; revisit if a non-statute corpus mis-hits.

## Qdrant — deferred engine evaluation (NOT now)

Considered Qdrant as a replacement for sqlite-vec. Decision: stay on sqlite-vec for
v2/v3; re-evaluate Qdrant Edge (in-process build — NOT the docker server, which
violates the zero-daemon principle) only when a MEASURED trigger fires. Two distinct
triggers, two distinct rationales:

1. RETRIEVAL SPEED (Milestone C onward): only relevant if the query-stage latency log
   (add per-stage timing in C: query-embed ms / KNN ms / fusion ms / total) shows KNN
   search is a meaningful fraction of total. Expected NOT to fire — at ~30k vectors
   brute-force KNN is single-digit ms and the Voyage query-embedding round-trip
   (~100-300ms) dominates. Qdrant does not embed, so it cannot touch the dominant
   term. Revisit only if the log contradicts this.

2. v4 TEMPORAL (probable backbone): the stronger case. If v4's dominant query turns
   out to be PAYLOAD-FILTERED SEMANTIC SEARCH over a six-figure version store
   ("similar provisions, but only versions чинні on date X"), that is payload-filtered
   HNSW — Qdrant Edge's strongest feature and sqlite-vec's structural weakness
   (brute-force cannot index-prefilter). If instead v4's dominant query is "fetch the
   dated version of a KNOWN provision_key," that is a SQLite B-tree range lookup and
   Qdrant buys nothing. Decide on v4's measured query-pattern mix, not corpus size.
   Version-count growth is amendment-churn-bound (~15-25k chunks for a decade of Tax
   Code history), not corpus×time, so scale alone does not justify the switch.

Qdrant Edge bonus if ever adopted: native sparse vectors (which GGUF did NOT expose)
and DBSF fusion. Only matters if FTS5 lexical recall on Ukrainian proves insufficient
in C — itself an open measurement.

Do NOT pivot speculatively. Each trigger is a number to measure, not a plan to execute.

## Known carry-overs
- v1 modules `server.py`, `ingest.py`, `query.py` are Postgres-coupled and dormant;
  they are rewritten in Milestone B against the SQLite + routing foundation.
  `sources.py` (URL/PDF/text extraction) is reused as-is.
- Ukrainian FTS5 lexical is inflection-limited (no stemming; bge-m3 GGUF is dense-only).
  Mitigation (prefix/trigram tokenizer or query expansion) is a Milestone C task.
