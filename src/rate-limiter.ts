/**
 * Sliding-window rate limiter.
 *
 * A fixed-window counter lets a client send 2x the limit across a window
 * boundary (all of minute one's budget at :59, all of minute two's at :01).
 * Storing timestamps and filtering by age costs more memory but has no such
 * seam, which matters when the limit exists to protect a paid API.
 *
 * In-memory by design: this is per-process state and does NOT hold across
 * replicas or restarts. A multi-instance deployment needs Redis; see
 * docs/DEVIATIONS.md.
 */

export interface RateLimitConfig {
    requestsPerMinute: number;
    requestsPerHour: number;
    tokensPerMinute: number;
}

export const DEFAULT_LIMITS: RateLimitConfig = {
    requestsPerMinute: Number(process.env.RATE_LIMIT_RPM ?? 20),
    requestsPerHour: Number(process.env.RATE_LIMIT_RPH ?? 200),
    tokensPerMinute: Number(process.env.RATE_LIMIT_TPM ?? 100_000)
};

export interface RateLimitResult {
    ok: boolean;
    /** Which limit tripped, when one did. */
    limit?: 'minute' | 'hour' | 'tokens';
    error?: string;
    /** Seconds until the caller may retry — drives the Retry-After header. */
    retryAfterSeconds?: number;
    remaining: { minute: number; hour: number };
}

const MINUTE_MS = 60_000;
const HOUR_MS = 3_600_000;

export class InMemoryRateLimiter {
    private minuteWindows = new Map<string, number[]>();
    private hourWindows = new Map<string, number[]>();
    /** [timestamp, tokenCount] pairs — each request spends a different amount. */
    private tokenWindows = new Map<string, Array<[number, number]>>();

    constructor(private config: RateLimitConfig = DEFAULT_LIMITS) {}

    /** Drop entries older than the window and return what remains. */
    private prune(store: Map<string, number[]>, clientId: string, windowMs: number, now: number): number[] {
        const kept = (store.get(clientId) ?? []).filter(ts => now - ts < windowMs);
        store.set(clientId, kept);
        return kept;
    }

    checkRateLimit(clientId: string, now = Date.now()): RateLimitResult {
        const minute = this.prune(this.minuteWindows, clientId, MINUTE_MS, now);
        const hour = this.prune(this.hourWindows, clientId, HOUR_MS, now);

        const remaining = {
            minute: Math.max(0, this.config.requestsPerMinute - minute.length),
            hour: Math.max(0, this.config.requestsPerHour - hour.length)
        };

        if (minute.length >= this.config.requestsPerMinute) {
            const oldest = minute[0]!;
            return {
                ok: false,
                limit: 'minute',
                error: `Rate limit exceeded: ${this.config.requestsPerMinute} requests per minute.`,
                retryAfterSeconds: Math.max(1, Math.ceil((MINUTE_MS - (now - oldest)) / 1000)),
                remaining
            };
        }

        if (hour.length >= this.config.requestsPerHour) {
            const oldest = hour[0]!;
            return {
                ok: false,
                limit: 'hour',
                error: `Rate limit exceeded: ${this.config.requestsPerHour} requests per hour.`,
                retryAfterSeconds: Math.max(1, Math.ceil((HOUR_MS - (now - oldest)) / 1000)),
                remaining
            };
        }

        const tokens = this.getTokenUsage(clientId, now);
        if (tokens >= this.config.tokensPerMinute) {
            return {
                ok: false,
                limit: 'tokens',
                error: `Token budget exceeded: ${this.config.tokensPerMinute} tokens per minute.`,
                retryAfterSeconds: 60,
                remaining
            };
        }

        // Only record once the request is actually permitted — a rejected
        // request must not consume budget, or a blocked client would extend
        // its own lockout indefinitely by retrying.
        minute.push(now);
        hour.push(now);
        return {
            ok: true,
            remaining: { minute: remaining.minute - 1, hour: remaining.hour - 1 }
        };
    }

    recordTokens(clientId: string, tokenCount: number, now = Date.now()): void {
        const kept = (this.tokenWindows.get(clientId) ?? []).filter(([ts]) => now - ts < MINUTE_MS);
        kept.push([now, tokenCount]);
        this.tokenWindows.set(clientId, kept);
    }

    getTokenUsage(clientId: string, now = Date.now()): number {
        const kept = (this.tokenWindows.get(clientId) ?? []).filter(([ts]) => now - ts < MINUTE_MS);
        this.tokenWindows.set(clientId, kept);
        return kept.reduce((sum, [, count]) => sum + count, 0);
    }

    /** Test seam. */
    reset(): void {
        this.minuteWindows.clear();
        this.hourWindows.clear();
        this.tokenWindows.clear();
    }

    stats(): { clients: number; config: RateLimitConfig } {
        const clients = new Set([...this.minuteWindows.keys(), ...this.hourWindows.keys()]);
        return { clients: clients.size, config: this.config };
    }
}

export const rateLimiter = new InMemoryRateLimiter();
