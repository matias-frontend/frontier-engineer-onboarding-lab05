/**
 * Evaluation: retrieval metrics (deterministic) and generation metrics
 * (LLM-as-judge).
 *
 * The two halves have very different trust levels, and it matters. Precision@K
 * and MRR are arithmetic over a ground-truth set — reproducible and exact. The
 * judge scores are one model's opinion of another's output: useful for spotting
 * regressions across runs, unreliable as an absolute measure of quality. The
 * report labels them separately so nobody averages them into a single number
 * and calls it "accuracy".
 */
import { GoogleGenAI } from '@google/genai';

import { GEMINI_MODEL, requireEnv } from './config.js';
import type { ContextChunk, RAGResponse } from './rag.js';
import { withRetry } from './retry.js';

// ------------------------------------------------------- retrieval metrics

/**
 * Precision@K — what fraction of the top K retrieved documents are relevant.
 *
 * Divides by K, not by the number retrieved: a retriever that returns 2 results
 * when asked for 5 has not earned the same score as one that returned 5 good
 * ones. Under-returning is a failure mode this must not reward.
 */
export function precisionAtK(retrievedIds: string[], relevantIds: Set<string>, k: number): number {
    if (k <= 0) {
        return 0;
    }
    const topK = retrievedIds.slice(0, k);
    if (topK.length === 0) {
        return 0;
    }
    const hits = topK.filter(id => relevantIds.has(id)).length;
    return hits / k;
}

/** Reciprocal rank of the first relevant hit: 1/rank, or 0 if none. */
export function meanReciprocalRank(retrievedIds: string[], relevantIds: Set<string>): number {
    for (let i = 0; i < retrievedIds.length; i += 1) {
        if (relevantIds.has(retrievedIds[i]!)) {
            return 1 / (i + 1);
        }
    }
    return 0;
}

/** Recall@K — of all known relevant documents, how many made the top K. */
export function recallAtK(retrievedIds: string[], relevantIds: Set<string>, k: number): number {
    if (relevantIds.size === 0) {
        return 0;
    }
    const topK = new Set(retrievedIds.slice(0, k));
    let found = 0;
    for (const id of relevantIds) {
        if (topK.has(id)) {
            found += 1;
        }
    }
    return found / relevantIds.size;
}

// -------------------------------------------------------- LLM-as-judge

let judge: GoogleGenAI | undefined;

function judgeClient(): GoogleGenAI {
    if (!judge) {
        judge = new GoogleGenAI({ apiKey: requireEnv('GOOGLE_API_KEY') });
    }
    return judge;
}

export interface JudgeScore {
    score: number;
    explanation: string;
    /** True when the judge call or its parsing failed — score is not meaningful. */
    failed?: boolean;
}

/** Pull a JSON object out of a response that may be fenced or prefixed. */
function parseJudgeJson(raw: string): JudgeScore {
    const text = raw.trim().replace(/^```(?:json)?/i, '').replace(/```$/, '').trim();
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');

    if (start === -1 || end === -1 || end <= start) {
        return { score: 0, explanation: `Unparseable judge response: ${text.slice(0, 200)}`, failed: true };
    }

    try {
        const parsed = JSON.parse(text.slice(start, end + 1)) as { score?: unknown; explanation?: unknown };
        const score = Number(parsed.score);
        if (!Number.isFinite(score)) {
            return { score: 0, explanation: 'Judge returned no numeric score.', failed: true };
        }
        return {
            // Clamp: a judge that returns 1.5 must not inflate the average.
            score: Math.min(1, Math.max(0, score)),
            explanation: String(parsed.explanation ?? '')
        };
    } catch {
        return { score: 0, explanation: `Judge JSON did not parse: ${text.slice(0, 200)}`, failed: true };
    }
}

async function runJudge(prompt: string): Promise<JudgeScore> {
    try {
        const response = await withRetry(() =>
            judgeClient().models.generateContent({
                model: GEMINI_MODEL,
                contents: prompt,
                // Judging must be as close to deterministic as the API allows,
                // or run-to-run score drift gets mistaken for a regression.
                config: { temperature: 0, maxOutputTokens: 400 }
            })
        );
        return parseJudgeJson(response.text ?? '');
    } catch (error) {
        return {
            score: 0,
            explanation: `Judge call failed: ${error instanceof Error ? error.message : String(error)}`,
            failed: true
        };
    }
}

/** Is every claim in the answer supported by the context? */
export function faithfulnessPrompt(query: string, answer: string, context: string): string {
    return [
        'You are grading whether an ANSWER is faithful to its CONTEXT.',
        '',
        'Scoring:',
        '  1.0 — every factual claim is supported by the context',
        '  0.5 — partially supported; some claims go beyond the context',
        '  0.0 — contradicts the context, or fabricates claims it does not contain',
        '',
        'Judge ONLY support by the context. A correct-sounding claim that the context',
        'does not establish is unfaithful. An answer that correctly says the context is',
        'insufficient is fully faithful and scores 1.0.',
        '',
        `CONTEXT:\n${context}`,
        '',
        `QUESTION:\n${query}`,
        '',
        `ANSWER:\n${answer}`,
        '',
        'Reply with ONLY a JSON object: {"score": <number>, "explanation": "<one sentence>"}'
    ].join('\n');
}

/** Does the answer address what was asked? */
export function relevancePrompt(query: string, answer: string): string {
    return [
        'You are grading whether an ANSWER addresses the QUESTION asked.',
        '',
        'Scoring:',
        '  1.0 — directly answers the question',
        '  0.5 — partially on topic, or answers a related but different question',
        '  0.0 — does not address the question',
        '',
        'Judge relevance only. Do not reward or penalise factual correctness here.',
        'An honest "the sources do not cover this" is relevant if it is responsive to',
        'the question asked.',
        '',
        `QUESTION:\n${query}`,
        '',
        `ANSWER:\n${answer}`,
        '',
        'Reply with ONLY a JSON object: {"score": <number>, "explanation": "<one sentence>"}'
    ].join('\n');
}

export function llmJudgeFaithfulness(query: string, answer: string, context: string): Promise<JudgeScore> {
    return runJudge(faithfulnessPrompt(query, answer, context));
}

export function llmJudgeRelevance(query: string, answer: string): Promise<JudgeScore> {
    return runJudge(relevancePrompt(query, answer));
}

// ------------------------------------------------------------ orchestrator

export interface EvalItem {
    query: string;
    relevant_doc_ids: string[];
}

export interface EvalDetail {
    query: string;
    retrieved_ids: string[];
    relevant_ids: string[];
    precision_at_k: number;
    mrr: number;
    recall_at_k: number;
    faithfulness: number;
    relevance: number;
    faithfulness_explanation: string;
    relevance_explanation: string;
    judge_failed: boolean;
    answer_preview: string;
    latency_ms: number;
}

export interface EvalResult {
    k: number;
    items: number;
    retrieval: { precision_at_k: number; mrr: number; recall_at_k: number };
    generation: { faithfulness: number; relevance: number; judge_failures: number };
    details: EvalDetail[];
    ran_at: string;
    note: string;
}

const mean = (values: number[]) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);

export async function runEvaluation(
    dataset: EvalItem[],
    retrievalFn: (query: string) => Promise<ContextChunk[]>,
    generationFn: (query: string, chunks: ContextChunk[]) => Promise<RAGResponse>,
    k = 5
): Promise<EvalResult> {
    const details: EvalDetail[] = [];

    for (const item of dataset) {
        const relevant = new Set(item.relevant_doc_ids);
        const chunks = await retrievalFn(item.query);
        const retrievedIds = chunks.map(chunk => chunk.id);

        const generated = await generationFn(item.query, chunks);
        const context = chunks.map((chunk, i) => `[Source ${i + 1}] ${chunk.text}`).join('\n\n');

        // Both judges concurrently — they are independent.
        const [faithfulness, relevance] = await Promise.all([
            llmJudgeFaithfulness(item.query, generated.answer, context),
            llmJudgeRelevance(item.query, generated.answer)
        ]);

        details.push({
            query: item.query,
            retrieved_ids: retrievedIds,
            relevant_ids: item.relevant_doc_ids,
            precision_at_k: precisionAtK(retrievedIds, relevant, k),
            mrr: meanReciprocalRank(retrievedIds, relevant),
            recall_at_k: recallAtK(retrievedIds, relevant, k),
            faithfulness: faithfulness.score,
            relevance: relevance.score,
            faithfulness_explanation: faithfulness.explanation,
            relevance_explanation: relevance.explanation,
            judge_failed: Boolean(faithfulness.failed || relevance.failed),
            answer_preview: generated.answer.slice(0, 200),
            latency_ms: generated.latency_ms
        });
    }

    return {
        k,
        items: details.length,
        retrieval: {
            precision_at_k: Number(mean(details.map(d => d.precision_at_k)).toFixed(4)),
            mrr: Number(mean(details.map(d => d.mrr)).toFixed(4)),
            recall_at_k: Number(mean(details.map(d => d.recall_at_k)).toFixed(4))
        },
        generation: {
            faithfulness: Number(mean(details.map(d => d.faithfulness)).toFixed(4)),
            relevance: Number(mean(details.map(d => d.relevance)).toFixed(4)),
            judge_failures: details.filter(d => d.judge_failed).length
        },
        details,
        ran_at: new Date().toISOString(),
        note:
            'Retrieval metrics are exact. Generation metrics are one model judging another — ' +
            'useful for tracking regressions between runs, not as absolute quality measures.'
    };
}
