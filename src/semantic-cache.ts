/**
 * Semantic response cache keyed by query meaning rather than exact text.
 *
 * "How do I return an item?" and "what's the returns process?" are the same
 * question; an exact-match cache misses both. Comparing query embeddings
 * catches them.
 *
 * The threshold is the whole safety story, and measurement beats intuition
 * here. Too low and semantically *distinct* questions collide — measured,
 * "how do I cancel my order" and "how do I cancel my account" sit at 0.8654,
 * and they have very different answers. Too high and the cache never fires.
 *
 * Worse, the two populations OVERLAP: "precision at k" vs "recall at k" scores
 * 0.9080 while the equivalent pair "what does k1 control" / "what is k1 used
 * for" scores only 0.8849. No threshold separates them perfectly, so this is a
 * calibrated trade-off rather than a solved problem. `scripts/tune-cache.ts`
 * measures it against labelled pairs; DEVIATIONS §9 records the numbers.
 */

export interface CacheEntry<T = unknown> {
    query: string;
    embedding: number[];
    response: T;
    createdAt: number;
    ttlSeconds: number;
}

export interface CacheHit<T = unknown> {
    response: T;
    similarity: number;
    cachedQuery: string;
    ageSeconds: number;
}

export interface CacheStats {
    size: number;
    hits: number;
    misses: number;
    hitRate: number;
    evictions: number;
    expirations: number;
}

/**
 * Measured, not guessed — see `npm run tune:cache` and docs/DEVIATIONS.md §9.
 *
 * 0.95 (the handout's suggestion) caught 1 of 6 paraphrase pairs: a cache that
 * essentially never fires. 0.92 caught 4 of 6 with zero false hits on the same
 * labelled set.
 *
 * The margin is thin and deliberately documented: the highest-scoring pair of
 * genuinely DIFFERENT questions measured 0.9080 ("precision at k" vs "recall at
 * k"), so 0.92 clears it by only ~0.012. The populations overlap, meaning no
 * threshold is perfectly safe. Raise this toward 0.95 if a wrong cached answer
 * would be costly in your domain; the short TTL is the other half of the
 * mitigation.
 */
export const DEFAULT_SIMILARITY_THRESHOLD = Number(process.env.CACHE_SIMILARITY_THRESHOLD ?? 0.92);

export class SemanticCache<T = unknown> {
    private entries: Array<CacheEntry<T>> = [];
    private hits = 0;
    private misses = 0;
    private evictions = 0;
    private expirations = 0;

    constructor(
        public similarityThreshold: number = DEFAULT_SIMILARITY_THRESHOLD,
        private maxEntries = Number(process.env.CACHE_MAX_ENTRIES ?? 1000)
    ) {}

    static cosineSimilarity(a: number[], b: number[]): number {
        if (a.length !== b.length || a.length === 0) {
            return 0;
        }
        let dot = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < a.length; i += 1) {
            const x = a[i]!;
            const y = b[i]!;
            dot += x * y;
            normA += x * x;
            normB += y * y;
        }
        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom === 0 ? 0 : dot / denom;
    }

    private evictExpired(now: number): void {
        const before = this.entries.length;
        this.entries = this.entries.filter(entry => (now - entry.createdAt) / 1000 < entry.ttlSeconds);
        this.expirations += before - this.entries.length;
    }

    get(queryEmbedding: number[], now = Date.now()): CacheHit<T> | null {
        this.evictExpired(now);

        let best: CacheEntry<T> | undefined;
        let bestScore = -1;

        for (const entry of this.entries) {
            const score = SemanticCache.cosineSimilarity(entry.embedding, queryEmbedding);
            if (score > bestScore) {
                bestScore = score;
                best = entry;
            }
        }

        if (best && bestScore >= this.similarityThreshold) {
            this.hits += 1;
            return {
                response: best.response,
                similarity: bestScore,
                cachedQuery: best.query,
                ageSeconds: Math.round((now - best.createdAt) / 1000)
            };
        }

        this.misses += 1;
        return null;
    }

    put(query: string, queryEmbedding: number[], response: T, ttlSeconds = 3600, now = Date.now()): void {
        this.evictExpired(now);

        if (this.entries.length >= this.maxEntries) {
            // Drop the oldest quarter rather than a single entry: evicting one
            // at a time means every subsequent write pays an eviction once the
            // cache is warm.
            const dropCount = Math.max(1, Math.floor(this.maxEntries * 0.25));
            this.entries.splice(0, dropCount);
            this.evictions += dropCount;
        }

        this.entries.push({ query, embedding: queryEmbedding, response, createdAt: now, ttlSeconds });
    }

    get size(): number {
        return this.entries.length;
    }

    stats(): CacheStats {
        const total = this.hits + this.misses;
        return {
            size: this.entries.length,
            hits: this.hits,
            misses: this.misses,
            hitRate: total === 0 ? 0 : this.hits / total,
            evictions: this.evictions,
            expirations: this.expirations
        };
    }

    clear(): void {
        this.entries = [];
        this.hits = 0;
        this.misses = 0;
        this.evictions = 0;
        this.expirations = 0;
    }
}

export const semanticCache = new SemanticCache();
