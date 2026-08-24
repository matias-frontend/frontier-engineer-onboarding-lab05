/**
 * Document chunking with sentence-boundary awareness and overlap.
 *
 * Overlap matters because a fixed-size window will otherwise cut a sentence —
 * and often the answer — in half, leaving neither chunk able to satisfy the
 * query. Overlapping means the boundary text appears whole in one of them.
 */
import { createHash } from 'node:crypto';

import { CHUNK_OVERLAP, CHUNK_SIZE } from './config.js';

export interface Chunk {
    /** Stable content hash, so re-indexing identical text reuses the same id. */
    id: string;
    text: string;
    chunk_index: number;
    start_char: number;
    end_char: number;
}

/** Only accept a sentence break past this fraction of the window. */
const MIN_BREAK_RATIO = 0.5;

export function chunkDocument(text: string, chunkSize = CHUNK_SIZE, overlap = CHUNK_OVERLAP): Chunk[] {
    const source = text.trim();
    if (!source) {
        return [];
    }
    if (overlap >= chunkSize) {
        // Otherwise the cursor never advances and this loops forever.
        throw new Error(`overlap (${overlap}) must be smaller than chunkSize (${chunkSize}).`);
    }

    const chunks: Chunk[] = [];
    let start = 0;
    let index = 0;

    while (start < source.length) {
        let end = Math.min(start + chunkSize, source.length);

        // Prefer to end on a sentence boundary, but only if it doesn't shrink
        // the chunk too far — a break at 10% would waste most of the window.
        if (end < source.length) {
            const window = source.slice(start, end);
            const lastPeriod = window.lastIndexOf('.');
            if (lastPeriod > chunkSize * MIN_BREAK_RATIO) {
                end = start + lastPeriod + 1;
            }
        }

        const body = source.slice(start, end).trim();
        if (body) {
            chunks.push({
                id: createHash('md5').update(body).digest('hex').slice(0, 16),
                text: body,
                chunk_index: index,
                start_char: start,
                end_char: end
            });
            index += 1;
        }

        if (end >= source.length) {
            break;
        }
        // Step forward by the window minus the overlap; guard against a
        // pathological sentence break leaving us stuck.
        start = Math.max(end - overlap, start + 1);
    }

    return chunks;
}
