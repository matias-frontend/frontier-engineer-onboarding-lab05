/**
 * Query and document embeddings via Gemini.
 *
 * The lab specifies a 384-dimensional Pinecone index to match
 * all-MiniLM-L6-v2. Gemini's embedding model accepts an explicit
 * `outputDimensionality`, so we hit that exact width without shipping a local
 * ONNX runtime — see docs/DEVIATIONS.md for why the local option was rejected.
 *
 * EMBEDDING_DIM must match the Pinecone index's dimension, which is fixed at
 * creation time. Changing one without recreating the other produces upsert
 * errors, not silent corruption — which is the failure mode you want.
 */
import { GoogleGenAI } from '@google/genai';

import { EMBEDDING_DIM, GEMINI_EMBEDDING_MODEL, requireEnv } from './config.js';
import { withRetry } from './retry.js';

let client: GoogleGenAI | undefined;

function gemini(): GoogleGenAI {
    if (!client) {
        client = new GoogleGenAI({ apiKey: requireEnv('GOOGLE_API_KEY') });
    }
    return client;
}

export async function embedText(text: string): Promise<number[]> {
    const trimmed = text.trim();
    if (!trimmed) {
        throw new Error('Cannot embed empty text.');
    }

    const response = await withRetry(() =>
        gemini().models.embedContent({
            model: GEMINI_EMBEDDING_MODEL,
            contents: trimmed,
            config: { outputDimensionality: EMBEDDING_DIM }
        })
    );

    const values = response.embeddings?.[0]?.values;
    if (!values?.length) {
        throw new Error('Gemini returned no embedding values.');
    }
    if (values.length !== EMBEDDING_DIM) {
        throw new Error(
            `Expected ${EMBEDDING_DIM}-dimensional embedding, got ${values.length}. ` +
                'EMBEDDING_DIM must match the Pinecone index dimension.'
        );
    }
    return values;
}

export const embedQuery = embedText;

/**
 * Embed many texts. Requests run in bounded-concurrency waves rather than all
 * at once: a 50-chunk document fired in parallel reliably trips the free-tier
 * per-minute rate limit, and the retry backoff then serialises them anyway,
 * slower and noisier than just pacing them here.
 */
export async function embedBatch(texts: string[], concurrency = 5): Promise<number[][]> {
    const out: number[][] = new Array(texts.length);

    for (let offset = 0; offset < texts.length; offset += concurrency) {
        const slice = texts.slice(offset, offset + concurrency);
        const embedded = await Promise.all(slice.map(text => embedText(text)));
        embedded.forEach((vector, i) => {
            out[offset + i] = vector;
        });
    }

    return out;
}
