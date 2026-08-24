/**
 * Retry with exponential backoff for Gemini calls.
 *
 * Three distinct failures hide behind similar-looking status codes, and they
 * need different responses:
 *
 *   503 "high demand"      transient, clears in seconds        -> retry
 *   429 per-minute limit   transient, clears in seconds        -> retry
 *   429 per-day quota      will not clear for hours            -> fail now
 *   4xx client errors      will fail identically every time    -> fail now
 *
 * Retrying the last two only burns time and quota, so `isDailyQuotaExhausted`
 * short-circuits them.
 */
import { isDailyQuotaExhausted } from './quota.js';

const RETRYABLE_STATUS = new Set([408, 429, 500, 502, 503, 504]);

function statusOf(error: unknown): number | undefined {
    if (typeof error !== 'object' || error === null) {
        return undefined;
    }
    const candidate = error as { status?: unknown; code?: unknown };
    for (const value of [candidate.status, candidate.code]) {
        if (typeof value === 'number') {
            return value;
        }
    }
    const message = (error as { message?: unknown }).message;
    if (typeof message === 'string') {
        const match = message.match(/"code"\s*:\s*(\d{3})/);
        if (match?.[1]) {
            return Number(match[1]);
        }
    }
    return undefined;
}

export function isRetryable(error: unknown): boolean {
    // A spent daily quota is a 429 that retrying cannot fix.
    if (isDailyQuotaExhausted(error)) {
        return false;
    }
    const status = statusOf(error);
    if (status !== undefined) {
        return RETRYABLE_STATUS.has(status);
    }
    // No status usually means a transport failure (DNS, socket reset, abort).
    const name = (error as { name?: unknown } | null)?.name;
    return name === 'AbortError' || name === 'TypeError' || name === 'FetchError';
}

export interface RetryOptions {
    attempts?: number;
    baseDelayMs?: number;
    maxDelayMs?: number;
    onRetry?: (attempt: number, delayMs: number, error: unknown) => void;
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export async function withRetry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
    const {
        attempts = 4,
        baseDelayMs = 1000,
        maxDelayMs = 20_000,
        onRetry = (attempt, delay, error) =>
            console.warn(
                `[retry] attempt ${attempt} failed, retrying in ${delay}ms:`,
                error instanceof Error ? error.message.slice(0, 160) : error
            )
    } = options;

    let lastError: unknown;
    for (let attempt = 1; attempt <= attempts; attempt += 1) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            if (attempt === attempts || !isRetryable(error)) {
                throw error;
            }
            // Full jitter: concurrent calls that are all rate-limited must not
            // retry in lockstep and trip the same limit again together.
            const ceiling = Math.min(maxDelayMs, baseDelayMs * 2 ** (attempt - 1));
            const delay = Math.round(Math.random() * ceiling);
            onRetry(attempt, delay, error);
            await sleep(delay);
        }
    }
    throw lastError;
}
