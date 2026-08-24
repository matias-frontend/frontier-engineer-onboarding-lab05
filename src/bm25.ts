/**
 * BM25 keyword index, implemented from scratch.
 *
 * BM25 scores a document for a query term as:
 *
 *   idf(t) * ( tf * (k1 + 1) ) / ( tf + k1 * (1 - b + b * |d| / avgdl) )
 *
 * where k1 controls term-frequency saturation (a term appearing 20 times isn't
 * 20x more relevant than once) and b controls length normalisation (a long
 * document shouldn't win purely by containing more words).
 */
import { BM25_B, BM25_K1 } from './config.js';

export interface Bm25Document {
    id: string;
    text: string;
    metadata: Record<string, unknown>;
}

export interface Bm25Hit extends Bm25Document {
    bm25_score: number;
}

/**
 * The handout's list (first two rows) plus interrogatives and common verbs.
 *
 * The extension is not cosmetic. With only the handout's list, the query
 * "how do I prepare noodles from scratch" matched a TF-IDF document — because
 * that document happens to contain "how" twice, and "how" carried real IDF
 * weight. Question words are exactly the tokens a natural-language query is
 * built from and exactly the ones that carry no topical signal, so leaving
 * them in makes BM25 actively noisy on conversational queries.
 *
 * See docs/DEVIATIONS.md.
 */
const STOP_WORDS = new Set([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to',
    'for', 'of', 'and', 'or', 'but', 'not', 'with', 'this', 'that', 'it',
    // interrogatives and pronouns — the scaffolding of a natural-language query
    'how', 'what', 'why', 'when', 'where', 'which', 'who', 'i', 'you', 'my', 'me',
    // high-frequency verbs and prepositions with no topical content
    'do', 'does', 'did', 'be', 'been', 'have', 'has', 'had', 'can', 'could',
    'would', 'should', 'will', 'from', 'by', 'as', 'so', 'if', 'then', 'than',
    'there', 'here', 'about', 'into', 'out', 'up', 'down', 'over', 'under'
]);

interface StoredDoc {
    id: string;
    text: string;
    metadata: Record<string, unknown>;
    tokens: string[];
    tf: Map<string, number>;
    length: number;
}

export class BM25Index {
    private documents = new Map<string, StoredDoc>();
    /** term -> number of documents containing it */
    private df = new Map<string, number>();
    /** term -> ids of documents containing it, so scoring doesn't scan everything */
    private postings = new Map<string, Set<string>>();
    private totalLength = 0;

    constructor(
        private readonly k1: number = BM25_K1,
        private readonly b: number = BM25_B
    ) {}

    static tokenize(text: string): string[] {
        return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter(token => !STOP_WORDS.has(token));
    }

    get totalDocs(): number {
        return this.documents.size;
    }

    get avgDocLength(): number {
        return this.documents.size === 0 ? 0 : this.totalLength / this.documents.size;
    }

    addDocument(id: string, text: string, metadata: Record<string, unknown> = {}): void {
        // Re-adding the same id must not double-count it in df or avgdl.
        if (this.documents.has(id)) {
            this.removeDocument(id);
        }

        const tokens = BM25Index.tokenize(text);
        const tf = new Map<string, number>();
        for (const token of tokens) {
            tf.set(token, (tf.get(token) ?? 0) + 1);
        }

        this.documents.set(id, { id, text, metadata, tokens, tf, length: tokens.length });
        this.totalLength += tokens.length;

        for (const term of tf.keys()) {
            this.df.set(term, (this.df.get(term) ?? 0) + 1);
            let posting = this.postings.get(term);
            if (!posting) {
                posting = new Set();
                this.postings.set(term, posting);
            }
            posting.add(id);
        }
    }

    removeDocument(id: string): boolean {
        const doc = this.documents.get(id);
        if (!doc) {
            return false;
        }
        for (const term of doc.tf.keys()) {
            const next = (this.df.get(term) ?? 1) - 1;
            if (next <= 0) {
                this.df.delete(term);
                this.postings.delete(term);
            } else {
                this.df.set(term, next);
                this.postings.get(term)?.delete(id);
            }
        }
        this.totalLength -= doc.length;
        this.documents.delete(id);
        return true;
    }

    clear(): void {
        this.documents.clear();
        this.df.clear();
        this.postings.clear();
        this.totalLength = 0;
    }

    search(query: string, limit = 10): Bm25Hit[] {
        const terms = BM25Index.tokenize(query);
        if (terms.length === 0 || this.documents.size === 0) {
            return [];
        }

        const N = this.documents.size;
        const avgdl = this.avgDocLength || 1;
        const scores = new Map<string, number>();

        // Count each distinct term once: repeating a word in the query
        // shouldn't multiply its weight.
        for (const term of new Set(terms)) {
            const df = this.df.get(term);
            if (!df) {
                continue;
            }
            const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);

            for (const docId of this.postings.get(term) ?? []) {
                const doc = this.documents.get(docId)!;
                const tf = doc.tf.get(term) ?? 0;
                const tfScore = (tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + (this.b * doc.length) / avgdl));
                scores.set(docId, (scores.get(docId) ?? 0) + idf * tfScore);
            }
        }

        return [...scores.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([id, score]) => {
                const doc = this.documents.get(id)!;
                return { id, text: doc.text, metadata: doc.metadata, bm25_score: score };
            });
    }
}

/** Process-wide index. Rebuilt from Pinecone on boot — see src/hydrate.ts. */
export const bm25 = new BM25Index();
