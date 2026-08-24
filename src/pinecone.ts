/**
 * Pinecone index lifecycle and vector operations.
 */
import { Pinecone, type Index, type RecordMetadata } from '@pinecone-database/pinecone';

import {
    EMBEDDING_DIM,
    PINECONE_CLOUD,
    PINECONE_INDEX,
    PINECONE_REGION,
    requireEnv
} from './config.js';

export interface ChunkMetadata extends RecordMetadata {
    doc_id: string;
    title: string;
    text: string;
    chunk_index: number;
}

let client: Pinecone | undefined;
let handle: Index<ChunkMetadata> | undefined;
let namespace = process.env.PINECONE_NAMESPACE ?? '';

function pinecone(): Pinecone {
    if (!client) {
        client = new Pinecone({ apiKey: requireEnv('PINECONE_API_KEY') });
    }
    return client;
}

/**
 * Scope all reads and writes to a Pinecone namespace.
 *
 * The free tier allows a single index, so scripts that index their own corpora
 * would otherwise contend for it — one script's documents changing another's
 * search results, and rehydration pulling in everything indiscriminately.
 * Namespaces partition the one index cheaply, which is also how you would
 * separate tenants in a real deployment.
 *
 * Call before any other operation; it resets the cached handle.
 */
export function setNamespace(name: string): void {
    namespace = name;
    handle = undefined;
}

export function currentNamespace(): string {
    return namespace;
}

/**
 * Create the index if absent, then wait until it reports ready.
 *
 * A freshly created serverless index is not immediately queryable, and upserts
 * against a not-ready index fail in confusing ways — so this polls rather than
 * sleeping a fixed guess.
 */
export async function ensureIndex(waitMs = 60_000): Promise<Index<ChunkMetadata>> {
    const pc = pinecone();
    const existing = await pc.listIndexes();
    const found = existing.indexes?.find(index => index.name === PINECONE_INDEX);

    if (!found) {
        console.log(`[pinecone] creating index "${PINECONE_INDEX}" (dim ${EMBEDDING_DIM}, cosine)...`);
        await pc.createIndex({
            name: PINECONE_INDEX,
            dimension: EMBEDDING_DIM,
            metric: 'cosine',
            spec: { serverless: { cloud: PINECONE_CLOUD, region: PINECONE_REGION } }
        });
    } else if (found.dimension !== EMBEDDING_DIM) {
        // An index's dimension is immutable, so this can only be fixed by
        // deleting and recreating it. Say so rather than failing at upsert.
        throw new Error(
            `Pinecone index "${PINECONE_INDEX}" has dimension ${found.dimension}, but EMBEDDING_DIM is ` +
                `${EMBEDDING_DIM}. Delete the index in the Pinecone console and let it be recreated, ` +
                'or set EMBEDDING_DIM to match.'
        );
    }

    const deadline = Date.now() + waitMs;
    for (;;) {
        const description = await pc.describeIndex(PINECONE_INDEX).catch(() => null);
        if (description?.status?.ready) {
            break;
        }
        if (Date.now() > deadline) {
            throw new Error(`Pinecone index "${PINECONE_INDEX}" was not ready within ${waitMs}ms.`);
        }
        await new Promise(resolve => setTimeout(resolve, 2000));
    }

    return index();
}

/** The index handle, scoped to the active namespace when one is set. */
export function index(): Index<ChunkMetadata> {
    if (!handle) {
        const base = pinecone().index<ChunkMetadata>(PINECONE_INDEX);
        handle = namespace ? base.namespace(namespace) : base;
    }
    return handle;
}

/** Drop every vector in the active namespace, so a script can start clean. */
export async function clearNamespace(): Promise<void> {
    if (!namespace) {
        throw new Error('Refusing to clear the default namespace. Call setNamespace() first.');
    }
    await index().deleteAll();
}

export interface UpsertRecord {
    id: string;
    values: number[];
    metadata: ChunkMetadata;
}

/** Upsert in batches — Pinecone rejects oversized single requests. */
export async function upsertVectors(records: UpsertRecord[], batchSize = 100): Promise<number> {
    const target = index();
    for (let offset = 0; offset < records.length; offset += batchSize) {
        await target.upsert({ records: records.slice(offset, offset + batchSize) });
    }
    return records.length;
}

export interface VectorHit {
    id: string;
    text: string;
    metadata: Record<string, unknown>;
    score: number;
}

export async function queryVectors(
    embedding: number[],
    topK = 20,
    filter?: Record<string, unknown>
): Promise<VectorHit[]> {
    const response = await index().query({
        vector: embedding,
        topK,
        includeMetadata: true,
        ...(filter && Object.keys(filter).length > 0 ? { filter } : {})
    });

    return (response.matches ?? []).map(match => ({
        id: match.id,
        text: String(match.metadata?.text ?? ''),
        metadata: (match.metadata ?? {}) as Record<string, unknown>,
        score: match.score ?? 0
    }));
}

export async function indexStats(): Promise<{ vectors: number; dimension: number | undefined }> {
    const stats = await index().describeIndexStats();
    return { vectors: stats.totalRecordCount ?? 0, dimension: stats.dimension };
}

export async function deleteDocument(docId: string): Promise<void> {
    await index().deleteMany({ filter: { doc_id: docId } });
}
