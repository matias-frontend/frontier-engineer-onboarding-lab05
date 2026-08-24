/**
 * Production RAG API.
 *
 *   POST /query     rate limit -> sanitize -> cache -> retrieve -> generate
 *   POST /index     chunk, embed, index into Pinecone + BM25
 *   POST /evaluate  run the evaluation suite over a dataset
 *   GET  /metrics   RAG, cache, cost, and rate-limiter stats
 *   GET  /health    liveness
 */
import { serve } from '@hono/node-server';
import { Hono } from 'hono';

import { bm25 } from './bm25.js';
import { GEMINI_MODEL, HOST, PINECONE_NAMESPACE, PORT, REHYDRATE_ON_BOOT } from './config.js';
import { costTracker } from './cost.js';
import { embedQuery } from './embeddings.js';
import { runEvaluation, type EvalItem } from './evaluation.js';
import { rehydrateBm25 } from './hydrate.js';
import { ensureIndex, indexStats, setNamespace } from './pinecone.js';
import { isDailyQuotaExhausted, quotaGuidance } from './quota.js';
import { generateAnswer, metrics, type ContextChunk, type RAGResponse } from './rag.js';
import { rateLimiter } from './rate-limiter.js';
import { indexDocument, search } from './search.js';
import { semanticCache } from './semantic-cache.js';
import { sanitizeQuery } from './security.js';
import { SELF_RAG_ENABLED, shouldRetrieve } from './self-rag.js';

setNamespace(PINECONE_NAMESPACE);

const app = new Hono();

function errorResponse(error: unknown): { body: Record<string, unknown>; status: 429 | 500 } {
    const detail = error instanceof Error ? error.message : String(error);
    if (isDailyQuotaExhausted(error)) {
        return {
            body: { error: 'Gemini free-tier daily quota exhausted.', detail: quotaGuidance(error) },
            status: 429
        };
    }
    // Never surface a raw upstream error to the caller — it can carry internal
    // detail. Log the specifics, return something safe.
    console.error('[server]', detail);
    return { body: { error: 'The request could not be completed. Please try again.' }, status: 500 };
}

/** Retrieval adapter: hybrid search results -> RAG context chunks. */
async function retrieve(query: string, limit: number): Promise<ContextChunk[]> {
    const response = await search(query, { limit, method: 'weighted' });
    return response.results.map(result => ({
        id: result.id,
        text: result.text,
        metadata: result.metadata
    }));
}

app.get('/health', c => c.json({ status: 'ok' }));

app.get('/metrics', async c => {
    const stats = await indexStats().catch(() => null);
    return c.json({
        rag: metrics.snapshot(),
        cache: semanticCache.stats(),
        cost: costTracker.summary(),
        rate_limiter: rateLimiter.stats(),
        index: {
            bm25_docs: bm25.totalDocs,
            pinecone_vectors: stats?.vectors ?? null,
            namespace: PINECONE_NAMESPACE
        },
        config: { model: GEMINI_MODEL, self_rag: SELF_RAG_ENABLED }
    });
});

app.post('/query', async c => {
    const body = (await c.req.json().catch(() => null)) as Record<string, unknown> | null;
    if (!body) {
        return c.json({ error: 'Request body must be a JSON object.' }, 400);
    }

    const clientId = typeof body.client_id === 'string' && body.client_id.trim() ? body.client_id.trim() : 'default';

    // Rate limit FIRST: an abusive caller must not reach the embedding or
    // generation path, which is where the money is spent.
    const limit = rateLimiter.checkRateLimit(clientId);
    if (!limit.ok) {
        metrics.rateLimited += 1;
        c.header('Retry-After', String(limit.retryAfterSeconds ?? 60));
        return c.json({ error: limit.error, limit: limit.limit, retry_after_seconds: limit.retryAfterSeconds }, 429);
    }

    if (typeof body.query !== 'string') {
        return c.json({ error: 'Field "query" is required and must be a string.' }, 400);
    }

    const sanitized = sanitizeQuery(body.query);
    if (sanitized.blocked) {
        metrics.injectionBlocked += 1;
        console.warn(`[security] blocked query from ${clientId}: ${sanitized.flags.join(', ')}`);
        return c.json({ error: sanitized.reason, flags: sanitized.flags }, 400);
    }
    const query = sanitized.clean;

    const topK = body.limit === undefined ? 5 : Number(body.limit);
    if (!Number.isInteger(topK) || topK < 1 || topK > 20) {
        return c.json({ error: 'Field "limit" must be an integer between 1 and 20.' }, 400);
    }
    const useCache = body.use_cache !== false;

    try {
        const embedding = await embedQuery(query);

        if (useCache) {
            const hit = semanticCache.get(embedding);
            if (hit) {
                const cached = hit.response as RAGResponse;
                metrics.totalQueries += 1;
                metrics.cacheHits += 1;
                // Count what the cache saved, so the metrics endpoint can show
                // the feature paying for itself.
                costTracker.recordCacheSaving(cached.prompt_tokens, cached.output_tokens);
                return c.json({
                    ...cached,
                    cache_hit: true,
                    latency_ms: 0,
                    cache_similarity: Number(hit.similarity.toFixed(4)),
                    cached_query: hit.cachedQuery,
                    cache_age_seconds: hit.ageSeconds
                });
            }
        }

        const decision = await shouldRetrieve(query);
        const chunks = decision.retrieve ? await retrieve(query, topK) : [];
        if (!decision.retrieve) {
            metrics.retrievalSkipped += 1;
        }

        const answer = await generateAnswer(query, chunks);
        const result: RAGResponse = { ...answer, retrieval_skipped: !decision.retrieve };

        if (useCache) {
            semanticCache.put(query, embedding, result);
        }
        rateLimiter.recordTokens(clientId, answer.prompt_tokens + answer.output_tokens);

        return c.json({ ...result, retrieval_reason: decision.reason });
    } catch (error) {
        const { body: payload, status } = errorResponse(error);
        return c.json(payload, status);
    }
});

app.post('/index', async c => {
    const body = (await c.req.json().catch(() => null)) as Record<string, unknown> | null;
    if (!body) {
        return c.json({ error: 'Request body must be a JSON object.' }, 400);
    }
    const { doc_id: docId, title, text } = body;
    for (const [name, value] of [['doc_id', docId], ['title', title], ['text', text]] as const) {
        if (typeof value !== 'string' || !value.trim()) {
            return c.json({ error: `Field "${name}" is required and must be a non-empty string.` }, 400);
        }
    }

    try {
        return c.json(await indexDocument(String(docId).trim(), String(title).trim(), String(text)));
    } catch (error) {
        const { body: payload, status } = errorResponse(error);
        return c.json(payload, status);
    }
});

app.post('/evaluate', async c => {
    const body = (await c.req.json().catch(() => null)) as Record<string, unknown> | null;
    const dataset = body?.dataset;
    if (!Array.isArray(dataset) || dataset.length === 0) {
        return c.json({ error: 'Field "dataset" is required and must be a non-empty array.' }, 400);
    }
    for (const item of dataset) {
        if (typeof item?.query !== 'string' || !Array.isArray(item?.relevant_doc_ids)) {
            return c.json({ error: 'Each dataset item needs a "query" string and a "relevant_doc_ids" array.' }, 400);
        }
    }

    const k = body?.k === undefined ? 5 : Number(body.k);
    if (!Number.isInteger(k) || k < 1 || k > 20) {
        return c.json({ error: 'Field "k" must be an integer between 1 and 20.' }, 400);
    }

    try {
        const report = await runEvaluation(
            dataset as EvalItem[],
            query => retrieve(query, k),
            (query, chunks) => generateAnswer(query, chunks),
            k
        );
        return c.json(report);
    } catch (error) {
        const { body: payload, status } = errorResponse(error);
        return c.json(payload, status);
    }
});

async function bootstrap(): Promise<void> {
    await ensureIndex();
    if (REHYDRATE_ON_BOOT) {
        const result = await rehydrateBm25();
        console.log(`[production-rag] BM25 rehydrated: ${result.documents} chunks in ${result.durationMs}ms`);
    }
}

// Serve before bootstrap finishes so the health check can answer while the
// index warms up; otherwise the platform kills a slow-starting deploy.
serve({ fetch: app.fetch, port: PORT, hostname: HOST }, info => {
    console.log(
        `[production-rag] listening on http://${HOST}:${info.port} ` +
            `(model: ${GEMINI_MODEL}, namespace: ${PINECONE_NAMESPACE}, self-rag: ${SELF_RAG_ENABLED})`
    );
});

bootstrap().catch(error => {
    console.error('[production-rag] bootstrap failed:', error instanceof Error ? error.message : error);
});

export { app };
