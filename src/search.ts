/**
 * Indexing and search orchestration: the layer that wires chunking,
 * embeddings, Pinecone, BM25, and fusion together.
 */
import { bm25 } from './bm25.js';
import { chunkDocument } from './chunking.js';
import { BM25_WEIGHT, CHUNK_OVERLAP, CHUNK_SIZE, RRF_K, VECTOR_WEIGHT } from './config.js';
import { embedBatch, embedQuery } from './embeddings.js';
import { rrfFusion, weightedFusion, type FusedResult, type ScoredCandidate } from './fusion.js';
import { queryVectors, upsertVectors, type ChunkMetadata, type UpsertRecord } from './pinecone.js';
import { rerankWithCohere, rerankEnabled } from './rerank.js';

export interface IndexResult {
    doc_id: string;
    chunks_indexed: number;
    total_chars: number;
}

export async function indexDocument(docId: string, title: string, text: string): Promise<IndexResult> {
    const chunks = chunkDocument(text, CHUNK_SIZE, CHUNK_OVERLAP);
    if (chunks.length === 0) {
        return { doc_id: docId, chunks_indexed: 0, total_chars: 0 };
    }

    const embeddings = await embedBatch(chunks.map(chunk => chunk.text));

    const records: UpsertRecord[] = chunks.map((chunk, i) => {
        const metadata: ChunkMetadata = {
            doc_id: docId,
            title,
            text: chunk.text,
            chunk_index: chunk.chunk_index
        };
        return { id: `${docId}_${chunk.id}`, values: embeddings[i]!, metadata };
    });

    await upsertVectors(records);

    // Keep BM25 in step with Pinecone on the same write path, so a fresh index
    // is immediately searchable by keyword without waiting for a rehydrate.
    for (const record of records) {
        bm25.addDocument(record.id, record.metadata.text, record.metadata);
    }

    return {
        doc_id: docId,
        chunks_indexed: records.length,
        total_chars: chunks.reduce((sum, chunk) => sum + chunk.text.length, 0)
    };
}

export type SearchMethod = 'weighted' | 'rrf';

export interface SearchOptions {
    limit?: number;
    vectorWeight?: number;
    bm25Weight?: number;
    method?: SearchMethod;
    k?: number;
    /** Extension: metadata filter, applied to both retrieval arms. */
    filter?: Record<string, unknown>;
    /** Extension: Cohere rerank over the fused candidates. */
    rerank?: boolean;
}

export interface SearchResponse {
    results: FusedResult[];
    count: number;
    method: SearchMethod;
    reranked: boolean;
    vector_candidates: number;
    bm25_candidates: number;
}

/**
 * Does a BM25 document satisfy a metadata filter?
 *
 * Pinecone applies filters server-side; BM25 is in-process, so the same
 * predicate has to be applied here or the two arms would disagree about what
 * is eligible — and the filter would appear to "work" while keyword hits
 * leaked through.
 */
function matchesFilter(metadata: Record<string, unknown>, filter: Record<string, unknown>): boolean {
    return Object.entries(filter).every(([key, expected]) => {
        const actual = metadata[key];
        if (expected !== null && typeof expected === 'object' && '$in' in (expected as object)) {
            const allowed = (expected as { $in: unknown[] }).$in;
            return Array.isArray(allowed) && allowed.includes(actual);
        }
        return actual === expected;
    });
}

export async function search(query: string, options: SearchOptions = {}): Promise<SearchResponse> {
    const {
        limit = 10,
        vectorWeight = VECTOR_WEIGHT,
        bm25Weight = BM25_WEIGHT,
        method = 'weighted',
        k = RRF_K,
        filter,
        rerank = false
    } = options;

    // Over-fetch from each arm so fusion can reorder across a wider pool than
    // it returns — otherwise the blend can only shuffle the final page.
    const candidatePool = limit * 3;

    const embedding = await embedQuery(query);

    const [vectorRaw, bm25Raw] = await Promise.all([
        queryVectors(embedding, candidatePool, filter),
        Promise.resolve(bm25.search(query, candidatePool * 2))
    ]);

    const vectorHits: ScoredCandidate[] = vectorRaw.map(hit => ({
        id: hit.id,
        text: hit.text,
        metadata: hit.metadata,
        score: hit.score
    }));

    const bm25Hits: ScoredCandidate[] = bm25Raw
        .filter(hit => !filter || matchesFilter(hit.metadata, filter))
        .slice(0, candidatePool)
        .map(hit => ({ id: hit.id, text: hit.text, metadata: hit.metadata, score: hit.bm25_score }));

    let results =
        method === 'rrf'
            ? rrfFusion(vectorHits, bm25Hits, limit, k)
            : weightedFusion(vectorHits, bm25Hits, limit, vectorWeight, bm25Weight);

    let reranked = false;
    if (rerank && rerankEnabled() && results.length > 1) {
        const ordered = await rerankWithCohere(query, results);
        if (ordered) {
            results = ordered;
            reranked = true;
        }
    }

    return {
        results,
        count: results.length,
        method,
        reranked,
        vector_candidates: vectorHits.length,
        bm25_candidates: bm25Hits.length
    };
}
