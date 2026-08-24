# Deviations from the lab handout

## 1. `@google/generative-ai` → `@google/genai`, and the model is different

The handout's SDK is deprecated (support ended 30 November 2025) and its model,
`gemini-2.0-flash`, is **shut down**. `gemini-2.5-flash` is not a substitute
either — it is closed to new API keys and 404s. This uses `@google/genai` v2 and
defaults to `gemini-3.6-flash`, overridable via `GEMINI_MODEL`.

## 2. Retrieval is Lab 04's, ported

The handout offers "reuse your Lab 04 retrieval or stub it". Stubbing would make
the end-to-end pipeline hollow and, worse, make Precision@K and MRR measure
nothing — you cannot evaluate retrieval you did not perform.

Lab 04's chunking, BM25, fusion, embeddings, and Pinecone modules are copied in
whole, along with the boot-time BM25 rehydration that fixes the in-memory
keyword index problem. They arrive with 87 passing tests' worth of prior
verification. This project runs in its own `lab05` Pinecone namespace so it
shares the single free-tier index without colliding with Lab 04's data.

## 3. Embeddings via Gemini at 384 dimensions

Same reasoning as Lab 04: the handout's Ollama path cannot run on a deployed
container, and transformers.js was measured and rejected (388–642 MB with
unfixable high/critical CVEs via an unused `sharp` dependency). Gemini's
embedding model accepts an explicit `outputDimensionality`, so 384 matches the
Pinecone index exactly.

## 4. Render instead of Railway

Railway's free tier no longer exists. `render.yaml` deploys the Dockerfile the
handout asks for.

## 5. Added from the in-class brief, absent from Lab 05's checklist

The in-class brief lists prompt injection defenses and cost tracking as
objectives and deliverables; Lab 05's own checklist omits both. Both are built:

**Prompt injection** (`src/security.ts`) is layered, because no single layer
works:

| Layer | What it does |
| --- | --- |
| `sanitizeQuery` | Strips control and zero-width characters, then **blocks** rather than silently rewrites when a query reads as an override attempt |
| `neutralizeContext` | Defangs instruction-shaped phrases and fence-escape attempts inside *retrieved* text |
| `sandwichPrompt` | Fences untrusted content and restates the rule *after* it, so the last thing read is the constraint |
| `validateOutput` | Redacts API keys, private keys, emails, and card numbers before a response leaves |

Two deliberate choices worth defending. First, a suspicious query is **refused,
not sanitised into something else** — silent rewriting produces confusing
answers and hides attacks from your logs. Second, the most important defense is
not in this file at all: the system has no tools and no write access, so a
successful injection can at worst produce a wrong answer. Prompt injection has
no complete fix, and treating a regex filter as one is worse than knowing you
have partial cover.

In RAG specifically, the realistic attack arrives through the **corpus**, not the
query — a planted document reaches the model looking exactly like legitimate
context. That is why context neutralisation exists alongside query sanitisation.

**Cost tracking** (`src/cost.ts`) records per-request token usage, estimates
spend, breaks it down by model, and reports what the cache saved. Every figure
is labelled an estimate and the prices are configuration, not constants —
a stale hardcoded price presented as authoritative spend is worse than no number.

## 6. Metrics endpoint goes beyond the handout

The handout asks for RAG metrics and cache size. `/metrics` also reports cost,
rate-limiter state, index sizes, blocked-injection and rate-limited counts.
Security events belong in metrics: an injection attempt you cannot see is one
you cannot respond to.

## 7. Honest limits

Stated so the system is not read as more production-ready than it is.

- **The rate limiter and semantic cache are per-process, in-memory state.**
  They do not survive a restart and do not coordinate across replicas. On a
  single free-tier instance that is correct behaviour; behind a load balancer,
  each replica would enforce its own limit and a client could multiply its
  budget by the replica count. Redis is the fix, and the handout's own
  prerequisites note it as optional.
- **LLM-judge scores are not ground truth.** They are one model's opinion of
  another's output. `runEvaluation` reports retrieval and generation metrics
  separately and says so in the payload, specifically so nobody averages them
  into a single "accuracy" number.
- **Self-RAG is off by default.** The gate is itself an LLM call, so skipping
  retrieval saves an embedding call and a vector query but adds a generation
  call. It wins on latency and context size, not necessarily on token spend.
  Its failure mode is biased toward retrieving, because a wrong "skip" answers
  with no grounding and no citations — the exact failure RAG exists to prevent.
- **The evaluation corpus is small** (12 documents, 12 golden cases). Enough to
  exercise every metric and catch regressions; not enough to conclude much about
  retrieval quality in general.

## 8. Extension challenges

All four, with measurement rather than assertion:

- **Self-RAG** — `src/self-rag.ts`, `SELF_RAG=true`.
- **Cache threshold tuning** — `npm run tune:cache` measures the threshold
  against *labelled* pairs: paraphrases that should hit and confusables that
  must not. The interesting number is the **false** hit rate, not the hit rate —
  a cache that answers "how do I cancel my order" with the cached answer to
  "how do I cancel my account" has served a confidently wrong answer, not saved
  money.
- **A/B chunking** — `npm run ab:chunking` indexes the corpus at 300 and 800
  characters into separate namespaces and scores both on identical ground truth.
  It measures retrieval only, and says so; whether answers improve needs the
  judges.
- **Scheduled evaluation** — `npm run evaluate` writes a timestamped report,
  diffs against the previous run, and **exits non-zero** on a regression or a
  floor breach, so it works as a cron job or CI gate rather than something read
  by eye.

## 9. Measured: the handout's cache threshold is wrong for this embedding model

The handout suggests a similarity threshold of 0.95. Measured against labelled
pairs (`npm run tune:cache`), that setting makes the cache almost useless.

Six paraphrase pairs that *should* share an answer, and six confusables that
*must not*, scored:

```
0.9566  SAME    "Why use overlapping chunks?"          / "What is the purpose of chunk overlap?"
0.9456  SAME    "How is cosine similarity calculated?" / "What is the formula for cosine similarity?"
0.9298  SAME    "When should retrieval be skipped?"    / "In what cases is retrieval unnecessary?"
0.9235  SAME    "How do I return an item?"             / "What is the returns process?"
0.9175  SAME    "How do I reset my password?"          / "What are the steps to change my password?"
0.9080  DIFFER  "What is precision at k?"              / "What is recall at k?"
0.8849  SAME    "What does the k1 parameter control?"  / "What is k1 used for?"
0.8654  DIFFER  "How do I cancel my order?"            / "How do I cancel my account?"
0.8561  DIFFER  "What does the k1 parameter control?"  / "What does the b parameter control?"
0.8463  DIFFER  "How do I reset my password?"          / "How do I reset my device?"
0.8383  DIFFER  "What is BM25?"                        / "What is TF-IDF?"
0.8340  DIFFER  "How do I return an item?"             / "How do I track my return shipment?"
```

Sweep:

| threshold | true hits | **false hits** | verdict |
| --- | --- | --- | --- |
| 0.85 | 6/6 | **3/6** | unsafe — serves wrong answers |
| 0.90 | 5/6 | **1/6** | unsafe |
| **0.92** | **4/6** | **0/6** | best safe setting |
| 0.95 | 1/6 | 0/6 | safe but nearly useless |
| 0.97+ | 0/6 | 0/6 | never fires |

**The default is now 0.92.** At 0.95 the cache caught one paraphrase in six —
paying for a full generation on almost every rewording.

### The finding that matters more than the number

**The populations overlap.** The highest-scoring pair of genuinely *different*
questions (0.9080) outranks the lowest-scoring pair of *equivalent* ones
(0.8849). No threshold separates them cleanly, so any setting trades false hits
against missed hits — this is a calibration, not a solution.

`"What is precision at k?"` vs `"What is recall at k?"` is the instructive case:
near-identical phrasing, genuinely different answers, and embedding similarity
cannot tell them apart. That class of query is where a semantic cache is
dangerous, and it is common in technical documentation.

Consequences, recorded honestly:

- 0.92 clears the worst measured false pair by only ~0.012. That margin is thin,
  and twelve pairs is a small sample — the boundary is fitted to this data.
- The TTL is the other half of the mitigation: a wrong cached answer expires
  rather than persisting.
- Domains where a wrong answer is costly (billing, medical, legal) should raise
  the threshold and accept the lower hit rate. The right setting is a function
  of blast radius, not of the embedding model alone.
- A production system with real traffic should verify borderline hits more
  cheaply than regenerating — a fast model asked "are these the same question?"
  costs far less than a full RAG turn. Not built here; noted as the obvious
  next step.
