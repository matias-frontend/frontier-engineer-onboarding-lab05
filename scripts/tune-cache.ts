/**
 * Extension: measure the semantic cache similarity threshold.
 *
 * The threshold trades hit rate against correctness, and the interesting number
 * is not the hit rate — it is the FALSE hit rate. A cache that answers "how do
 * I cancel my order" with the cached answer to "how do I cancel my account" has
 * not saved you money, it has served a confidently wrong answer.
 *
 * So this measures both, using labelled pairs: paraphrases that SHOULD hit, and
 * near-miss confusables that MUST NOT.
 *
 * Embeddings only — no chat calls, so it barely touches the constrained quota.
 */
import { embedQuery } from '../src/embeddings.js';
import { isDailyQuotaExhausted, quotaGuidance } from '../src/quota.js';
import { SemanticCache } from '../src/semantic-cache.js';

if (!process.env.GOOGLE_API_KEY) {
    console.error('\n  GOOGLE_API_KEY is not set.\n');
    process.exit(1);
}

/** Pairs that mean the same thing — a good cache SHOULD serve these. */
const SHOULD_HIT: Array<[string, string]> = [
    ['How do I return an item?', 'What is the returns process?'],
    ['What does the k1 parameter control?', 'What is k1 used for?'],
    ['How is cosine similarity calculated?', 'What is the formula for cosine similarity?'],
    ['Why use overlapping chunks?', 'What is the purpose of chunk overlap?'],
    ['How do I reset my password?', 'What are the steps to change my password?'],
    ['When should retrieval be skipped?', 'In what cases is retrieval unnecessary?']
];

/** Superficially similar, semantically different — a cache MUST NOT serve these. */
const MUST_NOT_HIT: Array<[string, string]> = [
    ['How do I cancel my order?', 'How do I cancel my account?'],
    ['What does the k1 parameter control?', 'What does the b parameter control?'],
    ['How do I return an item?', 'How do I track my return shipment?'],
    ['What is precision at k?', 'What is recall at k?'],
    ['How do I reset my password?', 'How do I reset my device?'],
    ['What is BM25?', 'What is TF-IDF?']
];

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

console.log('\nembedding query pairs...\n');

const scored: Array<{ a: string; b: string; sim: number; shouldHit: boolean }> = [];

for (const [a, b] of SHOULD_HIT) {
    const [va, vb] = await Promise.all([guarded(() => embedQuery(a)), guarded(() => embedQuery(b))]);
    scored.push({ a, b, sim: SemanticCache.cosineSimilarity(va, vb), shouldHit: true });
}
for (const [a, b] of MUST_NOT_HIT) {
    const [va, vb] = await Promise.all([guarded(() => embedQuery(a)), guarded(() => embedQuery(b))]);
    scored.push({ a, b, sim: SemanticCache.cosineSimilarity(va, vb), shouldHit: false });
}

console.log('--- measured similarities ---\n');
for (const entry of [...scored].sort((x, y) => y.sim - x.sim)) {
    const tag = entry.shouldHit ? 'SAME  ' : 'DIFFER';
    console.log(`   ${entry.sim.toFixed(4)}  [${tag}]  "${entry.a.slice(0, 38)}" vs "${entry.b.slice(0, 38)}"`);
}

console.log('\n--- threshold sweep ---\n');
console.log('   thresh   true-hits   FALSE-hits   missed   note');

const thresholds = [0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99];
let safest: { threshold: number; trueHits: number } | null = null;

for (const threshold of thresholds) {
    const trueHits = scored.filter(e => e.shouldHit && e.sim >= threshold).length;
    const falseHits = scored.filter(e => !e.shouldHit && e.sim >= threshold).length;
    const missed = SHOULD_HIT.length - trueHits;

    let note = '';
    if (falseHits > 0) {
        note = 'UNSAFE — serves wrong answers';
    } else if (trueHits === 0) {
        note = 'useless — never hits';
    } else {
        note = 'safe';
        if (!safest || trueHits > safest.trueHits) {
            safest = { threshold, trueHits };
        }
    }

    console.log(
        `   ${threshold.toFixed(2)}     ${String(trueHits).padStart(2)}/${SHOULD_HIT.length}        ` +
            `${String(falseHits).padStart(2)}/${MUST_NOT_HIT.length}        ${String(missed).padStart(2)}      ${note}`
    );
}

const maxFalse = Math.max(...scored.filter(e => !e.shouldHit).map(e => e.sim));
const minTrue = Math.min(...scored.filter(e => e.shouldHit).map(e => e.sim));

console.log('\n--- interpretation ---\n');
console.log(`   highest similarity among DIFFERENT questions : ${maxFalse.toFixed(4)}`);
console.log(`   lowest  similarity among SAME      questions : ${minTrue.toFixed(4)}`);

if (minTrue > maxFalse) {
    const midpoint = (minTrue + maxFalse) / 2;
    console.log(`\n   The two populations separate cleanly. Any threshold between`);
    console.log(`   ${maxFalse.toFixed(4)} and ${minTrue.toFixed(4)} is safe; ${midpoint.toFixed(3)} is the midpoint.`);
} else {
    console.log(`\n   The populations OVERLAP: some different questions score higher than`);
    console.log(`   some equivalent ones. No threshold separates them perfectly — any`);
    console.log(`   setting trades false hits against missed hits. This is the honest`);
    console.log(`   result, and it argues for a conservative threshold plus a short TTL.`);
}

if (safest) {
    console.log(`\n   Highest threshold with zero false hits and maximum true hits: ${safest.threshold}`);
} else {
    console.log('\n   No threshold achieved hits without false hits on this set.');
}
console.log('');
