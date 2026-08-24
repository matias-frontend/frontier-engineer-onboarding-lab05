/**
 * Score fusion: weighted min-max blending and Reciprocal Rank Fusion.
 *
 * Kept free of Pinecone, Gemini, and BM25 imports on purpose — fusion is pure
 * ranking maths, and keeping it pure is what lets it be tested exhaustively
 * with no API keys and no network.
 *
 * Weighted vs RRF: weighted needs the two score sets to be comparable after
 * normalisation, which is an assumption — cosine similarity and BM25 have
 * completely different distributions, and min-max on a handful of results is
 * easily skewed by one outlier. RRF ignores raw scores entirely and uses only
 * rank position, which is why it tends to be more robust across systems whose
 * score scales you don't control.
 */
import { RRF_K, BM25_WEIGHT, VECTOR_WEIGHT } from './config.js';

export interface Candidate {
    id: string;
    text: string;
    metadata: Record<string, unknown>;
}

export interface ScoredCandidate extends Candidate {
    score: number;
}

export interface FusedResult extends Candidate {
    hybrid_score: number;
    vector_score: number;
    bm25_score: number;
    /** 0-based positions in each source list; null when absent from it. */
    vector_rank: number | null;
    bm25_rank: number | null;
}

/**
 * Min-max normalise to [0,1].
 *
 * A single-item or all-equal list has no spread to normalise, so every member
 * maps to 1 — dropping them to 0 would silently discard a result that was, by
 * definition, the best its retriever found.
 */
export function minMaxNormalizer(values: number[]): (value: number) => number {
    if (values.length === 0) {
        return () => 0;
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    if (range === 0) {
        return () => 1;
    }
    return (value: number) => (value - min) / range;
}

function emptyResult(candidate: Candidate, vectorRank: number | null, bm25Rank: number | null): FusedResult {
    return {
        ...candidate,
        hybrid_score: 0,
        vector_score: 0,
        bm25_score: 0,
        vector_rank: vectorRank,
        bm25_rank: bm25Rank
    };
}

/**
 * Weighted fusion over the UNION of both candidate sets.
 *
 * Union, not intersection: a chunk that only semantic search found (a
 * paraphrase sharing no keywords) and one that only BM25 found (a rare exact
 * term the embedding blurred) are both exactly what hybrid search exists to
 * catch. Intersecting would discard precisely those cases.
 */
export function weightedFusion(
    vectorHits: ScoredCandidate[],
    bm25Hits: ScoredCandidate[],
    limit = 10,
    vectorWeight = VECTOR_WEIGHT,
    bm25Weight = BM25_WEIGHT
): FusedResult[] {
    const normVector = minMaxNormalizer(vectorHits.map(hit => hit.score));
    const normBm25 = minMaxNormalizer(bm25Hits.map(hit => hit.score));

    const merged = new Map<string, FusedResult>();

    vectorHits.forEach((hit, rank) => {
        const entry = emptyResult(hit, rank, null);
        entry.vector_score = hit.score;
        entry.hybrid_score = vectorWeight * normVector(hit.score);
        merged.set(hit.id, entry);
    });

    bm25Hits.forEach((hit, rank) => {
        const existing = merged.get(hit.id);
        const contribution = bm25Weight * normBm25(hit.score);
        if (existing) {
            existing.bm25_score = hit.score;
            existing.bm25_rank = rank;
            existing.hybrid_score += contribution;
        } else {
            const entry = emptyResult(hit, null, rank);
            entry.bm25_score = hit.score;
            entry.hybrid_score = contribution;
            merged.set(hit.id, entry);
        }
    });

    return [...merged.values()].sort((a, b) => b.hybrid_score - a.hybrid_score).slice(0, limit);
}

/**
 * Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank + 1).
 *
 * k damps how sharply early ranks dominate. Small k (20) makes rank 0 hugely
 * more valuable than rank 5; large k (100) flattens the curve so appearing in
 * both lists matters more than placing highly in one.
 */
export function rrfFusion(
    vectorHits: ScoredCandidate[],
    bm25Hits: ScoredCandidate[],
    limit = 10,
    k = RRF_K
): FusedResult[] {
    const merged = new Map<string, FusedResult>();

    const accumulate = (hits: ScoredCandidate[], which: 'vector' | 'bm25') => {
        hits.forEach((hit, rank) => {
            let entry = merged.get(hit.id);
            if (!entry) {
                entry = emptyResult(hit, null, null);
                merged.set(hit.id, entry);
            }
            if (which === 'vector') {
                entry.vector_score = hit.score;
                entry.vector_rank = rank;
            } else {
                entry.bm25_score = hit.score;
                entry.bm25_rank = rank;
            }
            entry.hybrid_score += 1 / (k + rank + 1);
        });
    };

    accumulate(vectorHits, 'vector');
    accumulate(bm25Hits, 'bm25');

    return [...merged.values()].sort((a, b) => b.hybrid_score - a.hybrid_score).slice(0, limit);
}
