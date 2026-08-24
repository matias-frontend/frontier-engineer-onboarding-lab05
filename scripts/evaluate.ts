/**
 * Extension: scheduled evaluation with regression detection.
 *
 * Runs the golden set, writes a timestamped report, compares against the
 * previous run, and exits non-zero on a regression — which is what makes it
 * usable as a cron job or CI gate rather than something you read by eye.
 *
 *   npm run evaluate                 run and store
 *   npm run evaluate -- --index      index the corpus first
 *
 * Cron example (daily at 03:00):
 *   0 3 * * * cd /path/to/production-rag && npm run evaluate >> eval.log 2>&1
 */
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { CORPUS, GOLDEN_SET } from '../src/corpus.js';
import { runEvaluation, type EvalResult } from '../src/evaluation.js';
import { rehydrateBm25 } from '../src/hydrate.js';
import { clearNamespace, ensureIndex, setNamespace } from '../src/pinecone.js';
import { isDailyQuotaExhausted, quotaGuidance } from '../src/quota.js';
import { generateAnswer } from '../src/rag.js';
import { indexDocument, search } from '../src/search.js';

for (const key of ['PINECONE_API_KEY', 'GOOGLE_API_KEY']) {
    if (!process.env[key]) {
        console.error(`\n  ${key} is not set.\n`);
        process.exit(1);
    }
}

const RESULTS_DIR = process.env.EVAL_RESULTS_DIR ?? 'eval-results';
const NAMESPACE = process.env.EVAL_NAMESPACE ?? 'lab05-eval';

/** A drop larger than this against the previous run is treated as a regression. */
const REGRESSION_TOLERANCE = Number(process.env.EVAL_REGRESSION_TOLERANCE ?? 0.1);
/** Absolute floors — below these, the run fails regardless of history. */
const FLOORS = {
    precision_at_k: Number(process.env.EVAL_MIN_PRECISION ?? 0.15),
    mrr: Number(process.env.EVAL_MIN_MRR ?? 0.5),
    faithfulness: Number(process.env.EVAL_MIN_FAITHFULNESS ?? 0.7),
    relevance: Number(process.env.EVAL_MIN_RELEVANCE ?? 0.7)
};

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

function previousReport(): EvalResult | null {
    try {
        const files = readdirSync(RESULTS_DIR).filter(f => f.endsWith('.json')).sort();
        const last = files[files.length - 1];
        return last ? (JSON.parse(readFileSync(join(RESULTS_DIR, last), 'utf8')) as EvalResult) : null;
    } catch {
        return null;
    }
}

setNamespace(NAMESPACE);
await guarded(() => ensureIndex());

if (process.argv.includes('--index')) {
    console.log('indexing corpus...');
    await guarded(() => clearNamespace().catch(() => undefined));
    for (const doc of CORPUS) {
        await guarded(() => indexDocument(doc.doc_id, doc.title, doc.text));
    }
    await new Promise(resolve => setTimeout(resolve, 8000));
}

await guarded(() => rehydrateBm25());

console.log(`\nrunning evaluation over ${GOLDEN_SET.length} golden cases...\n`);

const K = Number(process.env.EVAL_K ?? 5);

const report = await guarded(() =>
    runEvaluation(
        // Chunk ids are `{doc_id}_{hash}`; the golden labels name doc_ids, so
        // compare on the metadata rather than the composite id.
        GOLDEN_SET.map(c => ({ query: c.query, relevant_doc_ids: c.relevant_doc_ids })),
        async query => {
            const results = await search(query, { limit: K, method: 'weighted' });
            return results.results.map(r => ({
                id: String(r.metadata.doc_id ?? r.id),
                text: r.text,
                metadata: r.metadata
            }));
        },
        (query, chunks) => generateAnswer(query, chunks),
        K
    )
);

console.log('--- per-query ---\n');
for (const [i, detail] of report.details.entries()) {
    const kind = GOLDEN_SET[i]?.kind ?? '?';
    console.log(`[${kind}] ${detail.query}`);
    console.log(
        `   P@${K}=${detail.precision_at_k.toFixed(2)}  MRR=${detail.mrr.toFixed(2)}  ` +
            `recall=${detail.recall_at_k.toFixed(2)}  faith=${detail.faithfulness.toFixed(2)}  rel=${detail.relevance.toFixed(2)}` +
            (detail.judge_failed ? '  [JUDGE FAILED]' : '')
    );
    console.log(`   retrieved: ${detail.retrieved_ids.slice(0, K).join(', ') || '(none)'}`);
    if (detail.faithfulness < 0.7) {
        console.log(`   faithfulness note: ${detail.faithfulness_explanation.slice(0, 140)}`);
    }
    console.log('');
}

console.log('--- summary ---');
console.log(`retrieval : P@${K}=${report.retrieval.precision_at_k}  MRR=${report.retrieval.mrr}  recall=${report.retrieval.recall_at_k}`);
console.log(`generation: faithfulness=${report.generation.faithfulness}  relevance=${report.generation.relevance}`);
console.log(`judge failures: ${report.generation.judge_failures}/${report.items}`);
console.log(`\n${report.note}\n`);

// Store before gating, so a failing run is still recorded for comparison.
mkdirSync(RESULTS_DIR, { recursive: true });
const stamp = report.ran_at.replace(/[:.]/g, '-');
const path = join(RESULTS_DIR, `eval-${stamp}.json`);
writeFileSync(path, JSON.stringify(report, null, 2));
console.log(`report written to ${path}`);

const problems: string[] = [];

for (const [metric, floor] of Object.entries(FLOORS)) {
    const value =
        metric in report.retrieval
            ? (report.retrieval as Record<string, number>)[metric]!
            : (report.generation as unknown as Record<string, number>)[metric]!;
    if (value < floor) {
        problems.push(`${metric} ${value} is below the floor of ${floor}`);
    }
}

const previous = previousReport();
if (previous) {
    const comparisons: Array<[string, number, number]> = [
        ['precision_at_k', previous.retrieval.precision_at_k, report.retrieval.precision_at_k],
        ['mrr', previous.retrieval.mrr, report.retrieval.mrr],
        ['faithfulness', previous.generation.faithfulness, report.generation.faithfulness],
        ['relevance', previous.generation.relevance, report.generation.relevance]
    ];
    console.log(`\n--- vs previous run (${previous.ran_at}) ---`);
    for (const [name, before, after] of comparisons) {
        const delta = after - before;
        const arrow = delta > 0.001 ? '↑' : delta < -0.001 ? '↓' : '=';
        console.log(`   ${name.padEnd(16)} ${before.toFixed(4)} -> ${after.toFixed(4)}  ${arrow} ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`);
        if (delta < -REGRESSION_TOLERANCE) {
            problems.push(`${name} dropped ${Math.abs(delta).toFixed(4)} (tolerance ${REGRESSION_TOLERANCE})`);
        }
    }
} else {
    console.log('\n(no previous run to compare against — this becomes the baseline)');
}

if (problems.length > 0) {
    console.error('\nALERT: evaluation problems detected');
    for (const problem of problems) {
        console.error(`  - ${problem}`);
    }
    console.error('');
    process.exit(1);
}

console.log('\nno regressions detected.\n');
