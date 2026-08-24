/**
 * Rebuild the BM25 index from Pinecone at startup.
 *
 * The lab keeps BM25 purely in memory alongside a persistent Pinecone index.
 * That combination fails badly on any restart: the vectors survive, the keyword
 * index does not, and "hybrid" search silently degrades to vector-only while
 * still reporting itself as hybrid. On a free-tier host that spins down when
 * idle, that is the normal state rather than an edge case.
 *
 * Pinecone metadata already carries each chunk's text, so the vector store is
 * a sufficient source of truth to rebuild from. That keeps one authoritative
 * store instead of introducing a second one to keep in sync.
 */
import { bm25 } from './bm25.js';
import { index } from './pinecone.js';

export interface HydrationResult {
    documents: number;
    durationMs: number;
    truncated: boolean;
}

/**
 * Page through every vector id, fetch metadata in batches, and refill BM25.
 *
 * `limit` bounds the work so a huge index cannot make startup hang forever;
 * when hit, `truncated` is true and the caller should say so rather than
 * pretending the keyword index is complete.
 */
export async function rehydrateBm25(limit = 10_000): Promise<HydrationResult> {
    const startedAt = Date.now();
    const target = index();

    bm25.clear();

    let paginationToken: string | undefined;
    let loaded = 0;
    let truncated = false;

    for (;;) {
        const page = await target.listPaginated(
            paginationToken ? { paginationToken, limit: 100 } : { limit: 100 }
        );

        const ids = (page.vectors ?? []).map(vector => vector.id).filter((id): id is string => Boolean(id));
        if (ids.length > 0) {
            const fetched = await target.fetch({ ids });
            for (const [id, record] of Object.entries(fetched.records ?? {})) {
                const metadata = record.metadata;
                const text = typeof metadata?.text === 'string' ? metadata.text : '';
                if (text) {
                    bm25.addDocument(id, text, metadata as Record<string, unknown>);
                    loaded += 1;
                }
            }
        }

        if (loaded >= limit) {
            truncated = true;
            break;
        }

        paginationToken = page.pagination?.next;
        if (!paginationToken) {
            break;
        }
    }

    return { documents: loaded, durationMs: Date.now() - startedAt, truncated };
}
