/**
 * Extension: A/B test two chunking strategies against the golden set.
 *
 * Indexes the same corpus twice into separate Pinecone namespaces, one with
 * small chunks and one with large, then scores retrieval on both with identical
 * queries and ground truth.
 *
 * Retrieval metrics only — no chat calls, so this is cheap on the constrained
 * quota. What it can measure is whether chunk size changes which documents come
 * back; whether the resulting answers are better is a separate question that
 * needs the judges in `npm run evaluate`.
 */
import { CORPUS, GOLDEN_SET } from '../src/corpus.js';
import { CHUNK_OVERLAP, CHUNK_SIZE } from '../src/config.js';
import { chunkDocument } from '../src/chunking.js';
import { meanReciprocalRank, precisionAtK, recallAtK } from '../src/evaluation.js';
import { rehydrateBm25 } from '../src/hydrate.js';
import { clearNamespace, ensureIndex, setNamespace } from '../src/pinecone.js';
import { isDailyQuotaExhausted, quotaGuidance } from '../src/quota.js';
import { indexDocument, search } from '../src/search.js';

for (const key of ['PINECONE_API_KEY', 'GOOGLE_API_KEY']) {
    if (!process.env[key]) {
        console.error(`\n  ${key} is not set.\n`);
        process.exit(1);
    }
}

async function guarded<T>(run: () => Promise<T>): Promise<T> {
    try {
        return await run();
    } catch (error) {
        if (isDailyQuotaExhausted(error)) {
            console.error(`\n${quotaGuidance(error)}\n`);
            process.exit(2);
        }
        throw error;
    }
}

interface Variant {
    name: string;
    namespace: string;
    chunkSize: number;
    overlap: number;
}

const VARIANTS: Variant[] = [
    { name: 'small', namespace: 'lab05-ab-small', chunkSize: 300, overlap: 60 },
    { name: 'large', namespace: 'lab05-ab-large', chunkSize: 800, overlap: 160 }
];

const K = 5;

interface VariantScore {
    variant: string;
    chunks: number;
    /** P@K over DISTINCT documents — the cross-strategy comparable figure. */
    precisionDistinct: number;
    /** How many distinct documents the top K actually contained. */
    avgDistinctDocs: number;
    precision: number;
    mrr: number;
    recall: number;
    perQuery: Array<{ query: string; precision: number; mrr: number; retrieved: string[] }>;
}

const results: VariantScore[] = [];

for (const variant of VARIANTS) {
    console.log(`\n=== variant "${variant.name}" (${variant.chunkSize} chars, ${variant.overlap} overlap) ===\n`);

    // indexDocument reads chunk size from config, so chunk here and index the
    // pieces as individual documents to control the split per variant.
    setNamespace(variant.namespace);
    await guarded(() => ensureIndex());
    await guarded(() => clearNamespace().catch(() => undefined));

    let chunkCount = 0;
    for (const doc of CORPUS) {
        const pieces = chunkDocument(doc.text, variant.chunkSize, variant.overlap);
        chunkCount += pieces.length;
        for (const [i, piece] of pieces.entries()) {
            // doc_id carries the parent so ground-truth labels still match.
            await guarded(() => indexDocument(doc.doc_id, `${doc.title} (part ${i + 1})`, piece.text));
        }
    }
    console.log(`   indexed ${chunkCount} chunks from ${CORPUS.length} documents`);

    await new Promise(resolve => setTimeout(resolve, 8000));
    await guarded(() => rehydrateBm25());

    const perQuery: VariantScore['perQuery'] = [];
    const perQueryDeduped: Array<{ precision: number; mrr: number }> = [];
    for (const testCase of GOLDEN_SET) {
        // Skip the deliberately unanswerable case: it has no relevant docs, so
        // retrieval metrics are undefined for it.
        if (testCase.relevant_doc_ids.length === 0) {
            continue;
        }
        const hits = await guarded(() => search(testCase.query, { limit: K, method: 'weighted' }));
        const retrieved = hits.results.map(r => String(r.metadata.doc_id ?? r.id));
        const relevant = new Set(testCase.relevant_doc_ids);
        perQuery.push({
            query: testCase.query,
            precision: precisionAtK(retrieved, relevant, K),
            mrr: meanReciprocalRank(retrieved, relevant),
            retrieved
        });

        // Chunk-level P@K is NOT comparable across chunking strategies: a
        // strategy that splits one document into three chunks can fill three of
        // the top five slots with that single document, inflating precision
        // without retrieving anything more relevant. Deduplicating to distinct
        // documents first is what makes the two variants measurable against
        // each other.
        const seen = new Set<string>();
        const distinct = retrieved.filter(id => (seen.has(id) ? false : (seen.add(id), true)));
        perQueryDeduped.push({
            precision: precisionAtK(distinct, relevant, K),
            mrr: meanReciprocalRank(distinct, relevant)
        });
    }

    const mean = (xs: number[]) => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
    const score: VariantScore = {
        variant: variant.name,
        chunks: chunkCount,
        precisionDistinct: mean(perQueryDeduped.map(q => q.precision)),
        avgDistinctDocs: mean(
            perQuery.map(q => new Set(q.retrieved.slice(0, K)).size)
        ),
        precision: mean(perQuery.map(q => q.precision)),
        mrr: mean(perQuery.map(q => q.mrr)),
        recall: mean(
            perQuery.map((q, i) =>
                recallAtK(q.retrieved, new Set(GOLDEN_SET.filter(g => g.relevant_doc_ids.length > 0)[i]!.relevant_doc_ids), K)
            )
        ),
        perQuery
    };
    results.push(score);

    console.log(
        `   P@${K}(chunk)=${score.precision.toFixed(4)}  P@${K}(distinct docs)=${score.precisionDistinct.toFixed(4)}  ` +
            `MRR=${score.mrr.toFixed(4)}  recall=${score.recall.toFixed(4)}  avg distinct docs in top ${K}: ${score.avgDistinctDocs.toFixed(1)}`
    );
}

console.log('\n\n=== comparison ===\n');
console.log('   variant   chunks   P@5-chunk   P@5-distinct   MRR      recall   distinct/5');
for (const score of results) {
    console.log(
        `   ${score.variant.padEnd(9)} ${String(score.chunks).padStart(5)}   ` +
            `${score.precision.toFixed(4)}      ${score.precisionDistinct.toFixed(4)}       ` +
            `${score.mrr.toFixed(4)}   ${score.recall.toFixed(4)}   ${score.avgDistinctDocs.toFixed(1)}`
    );
}

console.log('\n   Read P@5-distinct, not P@5-chunk. Chunk-level precision rewards a');
console.log('   strategy for splitting one document across several top-K slots, which');
console.log('   is not the same as retrieving more relevant material.');

const [small, large] = results;
if (small && large) {
    console.log('\n--- where they differ ---\n');
    let differing = 0;
    for (const [i, q] of small.perQuery.entries()) {
        const other = large.perQuery[i];
        if (!other) continue;
        if (q.retrieved[0] !== other.retrieved[0] || Math.abs(q.mrr - other.mrr) > 0.001) {
            differing += 1;
            console.log(`   "${q.query.slice(0, 56)}"`);
            console.log(`      small: ${q.retrieved.slice(0, 3).join(', ')}  (MRR ${q.mrr.toFixed(2)})`);
            console.log(`      large: ${other.retrieved.slice(0, 3).join(', ')}  (MRR ${other.mrr.toFixed(2)})`);
        }
    }
    if (differing === 0) {
        console.log('   No query changed its top result between strategies.');
        console.log('   On a corpus this small that is a plausible outcome, not a bug —');
        console.log('   chunk size matters most when documents are long enough to split');
        console.log('   into genuinely different passages.');
    } else {
        console.log(`\n   ${differing}/${small.perQuery.length} queries ranked differently.`);
    }

    const better = large.mrr > small.mrr ? 'large' : small.mrr > large.mrr ? 'small' : 'neither';
    console.log(`\n   Better MRR: ${better}`);
    console.log(`   Default config is ${CHUNK_SIZE} chars / ${CHUNK_OVERLAP} overlap.`);
    console.log('\n   Caveat: this measures RETRIEVAL only. Whether the answers are better');
    console.log('   needs the LLM judges — run npm run evaluate against each variant.\n');
}
