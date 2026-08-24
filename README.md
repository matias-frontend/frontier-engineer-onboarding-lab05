# production-rag

A RAG system with the production concerns attached: hybrid retrieval, cited
generation, an evaluation harness, rate limiting, semantic caching, prompt
injection defenses, and cost observability.

Frontier Engineer Onboarding, Day 5 / Lab 05. Builds directly on Lab 04's
retrieval. See [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) for departures from the
handout and — more usefully — for what this system does *not* guarantee.

## Pipeline

```
POST /query
   │
   ├─ rate limit ........... before any spend, so a rejected request costs nothing
   ├─ sanitize ............. block injection attempts (refuse, don't rewrite)
   ├─ embed + cache check ... semantic hit returns immediately
   ├─ self-RAG gate ........ optionally skip retrieval entirely
   ├─ hybrid retrieve ...... Pinecone vectors + BM25, fused
   ├─ generate ............. context fenced, rule restated after it
   ├─ validate output ...... redact keys/PII before it leaves
   └─ record ............... latency, tokens, cost, citations
```

## Requirements

- Node **20.x**
- `PINECONE_API_KEY` — free Starter plan at [app.pinecone.io](https://app.pinecone.io)
- `GOOGLE_API_KEY` — free at [aistudio.google.com](https://aistudio.google.com)

> Gemini's free tier allows roughly 20 chat requests/day **per model**, and
> embeddings draw on a **separate** quota. `npm run evaluate` runs 12 cases with
> 2 judge calls each, so it is the expensive script — budget accordingly, or
> switch `GEMINI_MODEL` for a fresh allowance.

## Setup

```bash
npm install
export PINECONE_API_KEY=... GOOGLE_API_KEY=...
```

## Verify

```bash
npm run smoke:local   # 83 checks — no keys, no network, no quota
npm run smoke:e2e     # end-to-end against real Pinecone + Gemini
```

`smoke:local` covers the parts where the logic actually lives: sliding-window
rate limiting, cache eviction and TTL, Precision@K / MRR / Recall@K arithmetic,
citation extraction, every injection defense, and cost maths. Run it while
iterating — it is instant and free.

## Run it

```bash
npm start
```

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/query` | `{ query, limit?, use_cache?, client_id? }` |
| POST | `/index` | `{ doc_id, title, text }` |
| POST | `/evaluate` | `{ dataset: [{query, relevant_doc_ids}], k? }` |
| GET | `/metrics` | RAG, cache, cost, rate-limiter, index stats |
| GET | `/health` | Liveness |

```bash
# index something
curl -X POST localhost:8000/index -H 'content-type: application/json' \
  -d '{"doc_id":"bm25","title":"BM25","text":"The k1 parameter controls term frequency saturation..."}'

# ask
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"What does k1 control?","client_id":"user1"}'

# ask again — cache_hit should be true, latency_ms 0
```

Responses carry `cited_sources` (which `[Source N]` markers the model actually
used, with out-of-range citations dropped), token counts, estimated cost, and
any `security_flags` raised by output validation.

## Evaluation

```bash
npm run evaluate -- --index    # index the corpus, then evaluate
npm run evaluate               # evaluate against what's already indexed
```

Runs 12 golden cases spanning exact-term, conceptual, multi-document, and one
**deliberately unanswerable** question — that last one is the faithfulness test:
a grounded system says the corpus does not cover it, a hallucinating one invents
a number.

Reports Precision@K, MRR, Recall@K (exact arithmetic against hand-labelled
ground truth) and LLM-judged faithfulness and relevance (one model grading
another — useful for regression detection, not as absolute quality). The two are
reported **separately** and deliberately not averaged together.

Writes a timestamped JSON report, diffs against the previous run, and exits
non-zero on a regression or floor breach — so it works as a cron job:

```bash
0 3 * * * cd /path/to/production-rag && npm run evaluate >> eval.log 2>&1
```

## Extension challenges

```bash
npm run tune:cache     # measure the cache threshold against labelled pairs
npm run ab:chunking    # 300 vs 800 char chunks on identical ground truth
SELF_RAG=true npm start  # gate retrieval on whether it's actually needed
```

`tune:cache` is the interesting one: it measures the **false** hit rate, not the
hit rate. Paraphrases that should share an answer are labelled separately from
confusables that must not ("cancel my order" vs "cancel my account"), and the
sweep reports whether any threshold separates them cleanly.

## Prompt injection defenses

Four layers, documented in full in [DEVIATIONS §5](docs/DEVIATIONS.md):
input sanitization that **blocks rather than rewrites**, neutralization of
instruction-shaped text inside *retrieved documents*, sandwich prompting that
restates the rule after untrusted content, and output validation that redacts
credentials and PII.

The honest framing: prompt injection has no complete fix. These layers raise
cost and catch common cases. The real containment is that this system has no
tools and no write access, so a successful injection can at worst produce a
wrong answer.

## Deploy

Render → **New → Blueprint** → point at the repo. Set `PINECONE_API_KEY` and
`GOOGLE_API_KEY`. No disk required — Pinecone holds the vectors and BM25 rebuilds
from Pinecone metadata on boot.

**Single instance only.** The rate limiter and cache are per-process memory;
behind a load balancer each replica would enforce its own limit. Redis is the
fix. See DEVIATIONS §7.

## Layout

```
src/
  rag.ts             system prompt, generation, citations, metrics
  evaluation.ts      Precision@K, MRR, Recall@K, LLM judges, orchestrator
  rate-limiter.ts    sliding window, per-client, request + token budgets
  semantic-cache.ts  cosine-similarity cache with TTL and batch eviction
  security.ts        injection defenses, sandwich prompt, output validation
  cost.ts            per-request token and cost accounting
  self-rag.ts        retrieval gating (extension)
  corpus.ts          demo corpus + 12-case golden evaluation set
  server.ts          Hono API
  # ported from Lab 04, unchanged:
  chunking.ts bm25.ts fusion.ts embeddings.ts pinecone.ts hydrate.ts
  search.ts retry.ts quota.ts rerank.ts
scripts/
  smoke-local.ts     83 offline checks
  smoke-e2e.ts       end-to-end
  evaluate.ts        scheduled evaluation + regression gate
  tune-cache.ts      cache threshold measurement
  ab-chunking.ts     chunk size A/B
```
