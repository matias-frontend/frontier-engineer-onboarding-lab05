/**
 * Extension: Self-RAG retrieval gating.
 *
 * Ask the model whether a query actually needs the corpus before paying for
 * retrieval. "What is 2+2" does not; "what does our refund policy say" does.
 *
 * The economics are worth stating, because they are not obviously favourable:
 * the gate is itself an LLM call, so skipping retrieval saves an embedding call
 * and a vector query but adds a generation call. It wins on latency and on
 * context size, not necessarily on token spend. It is off by default for that
 * reason (SELF_RAG=true to enable).
 *
 * Failure is biased toward retrieving. A wrong "skip" answers from parametric
 * memory with no grounding and no citations — the exact failure RAG exists to
 * prevent — so anything ambiguous, errored, or unparseable retrieves.
 */
import { GoogleGenAI } from '@google/genai';

import { GEMINI_MODEL, requireEnv } from './config.js';
import { withRetry } from './retry.js';

export const SELF_RAG_ENABLED = process.env.SELF_RAG === 'true';

export interface RetrievalDecision {
    retrieve: boolean;
    reason: string;
    /** True when we defaulted to retrieving because the gate was unusable. */
    defaulted: boolean;
}

const GATE_PROMPT = `Decide whether answering this question requires looking up documents in a private knowledge base.

Answer NO only when the question is general knowledge, arithmetic, a definition, or pure small talk that any competent assistant answers correctly without sources.

Answer YES when the question refers to specific documents, systems, policies, projects, data, or anything organisation-specific — and whenever you are unsure.

Reply with exactly one word: YES or NO.

Question: `;

let client: GoogleGenAI | undefined;

export async function shouldRetrieve(query: string): Promise<RetrievalDecision> {
    if (!SELF_RAG_ENABLED) {
        return { retrieve: true, reason: 'Self-RAG disabled; always retrieving.', defaulted: false };
    }

    try {
        if (!client) {
            client = new GoogleGenAI({ apiKey: requireEnv('GOOGLE_API_KEY') });
        }
        const response = await withRetry(() =>
            client!.models.generateContent({
                model: GEMINI_MODEL,
                contents: GATE_PROMPT + query,
                config: { temperature: 0, maxOutputTokens: 5 }
            })
        );

        const verdict = (response.text ?? '').trim().toUpperCase();
        if (verdict.startsWith('NO')) {
            return { retrieve: false, reason: 'Model judged the question answerable without sources.', defaulted: false };
        }
        if (verdict.startsWith('YES')) {
            return { retrieve: true, reason: 'Model judged sources necessary.', defaulted: false };
        }
        return { retrieve: true, reason: `Unparseable gate response (${verdict.slice(0, 20)}); retrieving.`, defaulted: true };
    } catch (error) {
        return {
            retrieve: true,
            reason: `Gate failed (${error instanceof Error ? error.message : String(error)}); retrieving.`,
            defaulted: true
        };
    }
}
