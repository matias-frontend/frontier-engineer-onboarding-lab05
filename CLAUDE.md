# production-rag

RAG with production concerns: hybrid retrieval (ported from Lab 04), cited
generation, evaluation harness, rate limiting, semantic caching, injection
defenses, cost tracking. Hono API.

## Commands

```bash
npm run smoke:local   # 83 checks; no keys, no network, no quota — run this constantly
npm run smoke:e2e     # end-to-end (needs both keys)
npm run evaluate      # golden set + regression gate (expensive: ~36 chat calls)
npm run tune:cache    # cache threshold measurement (embeddings only, cheap)
npm run ab:chunking   # chunk size A/B (embeddings only)
npm run build         # typecheck
```

`smoke:local` is the workhorse. `evaluate` is the quota-expensive one — 12 cases
× (1 generation + 2 judges). Don't run it casually.

## Invariants

- **Rate limit before anything expensive.** The check runs before embedding and
  generation, so a rejected request costs nothing. Never reorder this.
- **A rejected request must not consume budget.** Otherwise a blocked client
  extends its own lockout by retrying. There's a test for it.
- **Suspicious queries are blocked, never silently rewritten.** Rewriting
  produces confusing answers and hides attacks from logs.
- **Retrieved documents are untrusted input.** Anything from the corpus goes
  through `neutralizeContext` before entering a prompt. In RAG the injection
  usually arrives through the corpus, not the query.
- **Never return a raw upstream error.** `errorResponse` logs specifics and
  returns a generic message; upstream errors can carry internal detail.
- **Retrieval and generation metrics stay separate.** Precision@K is exact
  arithmetic; judge scores are one model's opinion. Averaging them into one
  "accuracy" number would be dishonest.
- **Precision@K divides by k, not by results returned.** Under-returning must
  not score the same as returning k good results.
- **Citations outside the supplied source range are dropped**, so a hallucinated
  `[Source 9]` never surfaces as real.

## Gotchas

- `GEMINI_MODEL` defaults to `gemini-3.6-flash`. Never `gemini-2.0-flash` (shut
  down) or `gemini-2.5-flash` (404s for new keys).
- Quota is per-model per-day (~20 chat calls); embeddings bill separately and
  are far more generous. Scripts that only embed are cheap.
- Rate limiter and cache are **per-process memory** — they don't survive restart
  and don't coordinate across replicas. Single instance only without Redis.
- Pinecone is eventually consistent; e2e waits after upserting. Don't remove it.
- This project uses the `lab05` namespace in the index Lab 04 also uses. Scripts
  claim their own (`lab05-smoke`, `lab05-eval`, `lab05-ab-*`) and wipe on entry.
- Don't add transformers.js for local embeddings — measured and rejected in
  Lab 03 (400–640MB, unfixable CVEs via `sharp`).
