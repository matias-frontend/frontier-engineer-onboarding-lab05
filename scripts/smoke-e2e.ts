/**
 * End-to-end verification against real Pinecone and real Gemini.
 *
 * Runs in its own Pinecone namespace, wiped on entry, so it is idempotent and
 * isolated from Lab 04's data in the same free-tier index.
 *
 * Costs a handful of chat calls plus embeddings. Embeddings draw on a separate
 * quota from chat, so the chat budget is the binding constraint.
 */
import { bm25 } from '../src/bm25.js';
import { CORPUS } from '../src/corpus.js';
import { costTracker } from '../src/cost.js';
import { embedQuery } from '../src/embeddings.js';
import { meanReciprocalRank, precisionAtK } from '../src/evaluation.js';
import { rehydrateBm25 } from '../src/hydrate.js';
import { clearNamespace, ensureIndex, setNamespace } from '../src/pinecone.js';
import { isDailyQuotaExhausted, quotaGuidance } from '../src/quota.js';
import { generateAnswer, metrics } from '../src/rag.js';
import { InMemoryRateLimiter } from '../src/rate-limiter.js';
import { indexDocument, search } from '../src/search.js';
import { SemanticCache } from '../src/semantic-cache.js';

for (const key of ['PINECONE_API_KEY', 'GOOGLE_API_KEY']) {
    if (!process.env[key]) {
        console.error(`\n  ${key} is not set — cannot run the end-to-end test.\n`);
        process.exit(1);
    }
}

let passed = 0;
let failed = 0;

function check(label: string, condition: boolean, detail?: unknown): void {
    if (condition) {
        passed += 1;
        console.log(`  PASS  ${label}`);
    } else {
        failed += 1;
        console.log(`  FAIL  ${label}${detail === undefined ? '' : `\n        ${JSON.stringify(detail).slice(0, 400)}`}`);
    }
}

async function guarded<T>(label: string, run: () => Promise<T>): Promise<T> {
    try {
        return await run();
    } catch (error) {
        if (isDailyQuotaExhausted(error)) {
            console.error(`\n[stopped during: ${label}]\n\n${quotaGuidance(error)}\n`);
            process.exit(2);
        }
        throw error;
    }
}

setNamespace('lab05-smoke');
console.log('\nnamespace: lab05-smoke\n');

console.log('setup');
await guarded('ensureIndex', () => ensureIndex());
await guarded('clear', () => clearNamespace().catch(() => undefined));
check('Pinecone ready', true);

console.log('\nindexing');
for (const doc of CORPUS.slice(0, 6)) {
    const result = await guarded(`index ${doc.doc_id}`, () => indexDocument(doc.doc_id, doc.title, doc.text));
    check(`indexed ${doc.doc_id}`, result.chunks_indexed > 0, result);
}
await new Promise(resolve => setTimeout(resolve, 8000));
check('BM25 populated on the write path', bm25.totalDocs > 0, bm25.totalDocs);

console.log('\nretrieval');
const hits = await guarded('search', () => search('What does the k1 parameter control?', { limit: 5 }));
check('returns results', hits.results.length > 0, hits.count);
const topDoc = String(hits.results[0]?.metadata.doc_id ?? '');
check('retrieves the right document for an exact term', topDoc === 'bm25', hits.results.map(r => r.metadata.doc_id));

// Ground-truth metrics against the retrieved ids.
const retrievedDocs = hits.results.map(r => String(r.metadata.doc_id));
check('precision@5 is computable and positive', precisionAtK(retrievedDocs, new Set(['bm25']), 5) > 0);
check('mrr rewards the top hit', meanReciprocalRank(retrievedDocs, new Set(['bm25'])) === 1);

console.log('\ngeneration with citations');
const chunks = hits.results.map(r => ({ id: r.id, text: r.text, metadata: r.metadata }));
const answer = await guarded('generate', () => generateAnswer('What does the k1 parameter control?', chunks));
console.log(`\n> ${answer.answer.slice(0, 400)}\n`);

check('produces an answer', answer.answer.length > 0);
check('cites at least one source', answer.cited_sources.length > 0, answer.cited_sources);
check('citations are within range', answer.cited_sources.every(n => n >= 1 && n <= chunks.length), answer.cited_sources);
check('mentions saturation (the actual answer)', /saturat/i.test(answer.answer), answer.answer.slice(0, 200));
check('reports token usage', answer.prompt_tokens > 0 && answer.output_tokens > 0, {
    prompt: answer.prompt_tokens,
    output: answer.output_tokens
});
check('estimates a cost', answer.estimated_cost_usd >= 0);
check('no security flags on a benign answer', answer.security_flags.length === 0, answer.security_flags);

console.log('\ngrounding: refuses what the corpus cannot answer');
// The central faithfulness property — an ungrounded question must produce a
// refusal, not an invention.
const ungrounded = await guarded('generate ungrounded', () =>
    generateAnswer('What is the current stock price of Alphabet?', chunks)
);
console.log(`\n> ${ungrounded.answer.slice(0, 300)}\n`);
check(
    'declines to answer from outside the context',
    /do(?:es)? not|cannot|no information|not (?:contain|cover|provide)|insufficient|unable/i.test(ungrounded.answer),
    ungrounded.answer.slice(0, 300)
);

console.log('\nsemantic cache against real embeddings');
{
    const cache = new SemanticCache<string>(0.95, 100);
    const a = await guarded('embed a', () => embedQuery('How do I return an item?'));
    const b = await guarded('embed b', () => embedQuery('What is the returns process?'));
    const c = await guarded('embed c', () => embedQuery('How do I reset my password?'));

    cache.put('How do I return an item?', a, 'returns answer');
    check('identical query hits', cache.get(a)?.response === 'returns answer');

    const paraphrase = cache.get(b);
    const unrelated = cache.get(c);
    const simParaphrase = SemanticCache.cosineSimilarity(a, b);
    const simUnrelated = SemanticCache.cosineSimilarity(a, c);

    console.log(`   paraphrase similarity: ${simParaphrase.toFixed(4)}`);
    console.log(`   unrelated similarity:  ${simUnrelated.toFixed(4)}`);

    // The property that matters is ordering, not whether either crosses 0.95 —
    // that depends on the threshold, which tune-cache.ts measures.
    check('paraphrase scores higher than unrelated', simParaphrase > simUnrelated, {
        paraphrase: simParaphrase,
        unrelated: simUnrelated
    });
    check('an unrelated query never hits', unrelated === null, unrelated);
    if (paraphrase) {
        console.log('   (paraphrase cleared the 0.95 threshold)');
    } else {
        console.log('   (paraphrase below 0.95 — conservative, see tune:cache)');
    }
}

console.log('\nrate limiting under real sequencing');
{
    const limiter = new InMemoryRateLimiter({ requestsPerMinute: 2, requestsPerHour: 100, tokensPerMinute: 999_999 });
    check('first allowed', limiter.checkRateLimit('e2e').ok);
    check('second allowed', limiter.checkRateLimit('e2e').ok);
    check('third blocked with 429 semantics', !limiter.checkRateLimit('e2e').ok);
}

console.log('\nrehydration');
{
    const before = bm25.totalDocs;
    bm25.clear();
    check('BM25 empty after clear', bm25.totalDocs === 0);
    await guarded('rehydrate', () => rehydrateBm25());
    check('rehydration restores every chunk', bm25.totalDocs === before, { before, after: bm25.totalDocs });
}

console.log('\nmetrics and cost');
{
    const snapshot = metrics.snapshot();
    check('queries counted', snapshot.total_queries >= 2, snapshot);
    check('latency recorded', snapshot.avg_latency_ms > 0, snapshot);
    const cost = costTracker.summary();
    check('cost tracked', cost.calls >= 2 && cost.total_tokens > 0, cost);
    console.log(`   ${cost.calls} calls, ${cost.total_tokens} tokens, ~$${cost.estimated_cost_usd}`);
}

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed === 0 ? 0 : 1);
