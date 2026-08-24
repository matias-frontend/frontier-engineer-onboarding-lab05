/**
 * Environment and tuning configuration.
 */

export const PINECONE_INDEX = process.env.PINECONE_INDEX ?? 'hybrid-search-lab';

/** Own namespace inside the shared free-tier index (Lab 04 holds the others). */
export const PINECONE_NAMESPACE = process.env.PINECONE_NAMESPACE ?? 'lab05';
export const PINECONE_CLOUD = process.env.PINECONE_CLOUD ?? 'aws';
export const PINECONE_REGION = process.env.PINECONE_REGION ?? 'us-east-1';

/**
 * 384 to match the lab's required index dimension (the width of
 * all-MiniLM-L6-v2). Gemini's embedding model accepts an explicit
 * outputDimensionality, so we hit the spec exactly without needing a local
 * model. Changing this requires recreating the Pinecone index — its dimension
 * is fixed at creation.
 */
export const EMBEDDING_DIM = Number(process.env.EMBEDDING_DIM ?? 384);
export const GEMINI_EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL ?? 'gemini-embedding-2';

/** Chat model, used only by the optional Cohere-free reranking fallback and diagnostics. */
export const GEMINI_MODEL = process.env.GEMINI_MODEL ?? 'gemini-3.6-flash';

/** Chunking defaults from the lab. */
export const CHUNK_SIZE = Number(process.env.CHUNK_SIZE ?? 500);
export const CHUNK_OVERLAP = Number(process.env.CHUNK_OVERLAP ?? 100);

/** Fusion defaults from the lab: 0.7 vector / 0.3 BM25, RRF k=60. */
export const VECTOR_WEIGHT = Number(process.env.VECTOR_WEIGHT ?? 0.7);
export const BM25_WEIGHT = Number(process.env.BM25_WEIGHT ?? 0.3);
export const RRF_K = Number(process.env.RRF_K ?? 60);

/** BM25 tuning constants. */
export const BM25_K1 = Number(process.env.BM25_K1 ?? 1.5);
export const BM25_B = Number(process.env.BM25_B ?? 0.75);

/** Optional: Cohere rerank. Absent key simply disables the feature. */
export const COHERE_API_KEY = process.env.COHERE_API_KEY;
export const COHERE_RERANK_MODEL = process.env.COHERE_RERANK_MODEL ?? 'rerank-v3.5';

/** Rebuild the BM25 index from Pinecone on startup. */
export const REHYDRATE_ON_BOOT = process.env.REHYDRATE_ON_BOOT !== 'false';

export const PORT = Number(process.env.PORT ?? 8000);
export const HOST = process.env.HOST ?? '0.0.0.0';

export function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(`Missing required environment variable ${name}. See .env.example.`);
    }
    return value;
}
