/**
 * Extension challenge: Cohere Rerank over the fused candidate set.
 *
 * Optional by design. Without COHERE_API_KEY the feature is simply off and
 * `search` returns the fused ranking unchanged — no third signup required to
 * run this project. Called via Cohere's REST endpoint rather than their SDK to
 * avoid a dependency for one HTTP POST.
 *
 * Reranking is a different operation from fusion: fusion merges two cheap
 * retrievers by score or rank, while a cross-encoder reranker reads the query
 * and each document *together* and is far more accurate — but too slow to run
 * over a whole corpus. Hence retrieve-then-rerank over a small candidate set.
 */
import { COHERE_API_KEY, COHERE_RERANK_MODEL } from './config.js';
import type { FusedResult } from './fusion.js';

const COHERE_ENDPOINT = 'https://api.cohere.com/v2/rerank';

export function rerankEnabled(): boolean {
    return Boolean(COHERE_API_KEY);
}

interface CohereRerankResponse {
    results?: Array<{ index: number; relevance_score: number }>;
}

/**
 * Returns the reranked list, or null when reranking is unavailable or fails.
 *
 * Null rather than throwing: a reranker outage should degrade the ranking, not
 * fail the search. The caller reports whether reranking actually applied.
 */
export async function rerankWithCohere(query: string, candidates: FusedResult[]): Promise<FusedResult[] | null> {
    if (!COHERE_API_KEY || candidates.length === 0) {
        return null;
    }

    try {
        const response = await fetch(COHERE_ENDPOINT, {
            method: 'POST',
            headers: {
                'content-type': 'application/json',
                authorization: `Bearer ${COHERE_API_KEY}`
            },
            body: JSON.stringify({
                model: COHERE_RERANK_MODEL,
                query,
                documents: candidates.map(candidate => candidate.text),
                top_n: candidates.length
            }),
            signal: AbortSignal.timeout(15_000)
        });

        if (!response.ok) {
            console.warn(`[rerank] Cohere returned ${response.status}; keeping fused order.`);
            return null;
        }

        const body = (await response.json()) as CohereRerankResponse;
        const ordered = body.results
            ?.map(result => {
                const candidate = candidates[result.index];
                return candidate ? { ...candidate, rerank_score: result.relevance_score } : null;
            })
            .filter((entry): entry is FusedResult & { rerank_score: number } => entry !== null);

        return ordered?.length ? ordered : null;
    } catch (error) {
        console.warn('[rerank] failed, keeping fused order:', error instanceof Error ? error.message : error);
        return null;
    }
}
