/**
 * Gemini quota detection.
 *
 * A 429 means two different things. A per-minute rate limit clears in seconds
 * and is worth waiting out. An exhausted per-day quota will not clear for
 * hours, so retrying it only delays an inevitable failure.
 *
 * The daily quota is charged **per project per model**
 * (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), so switching
 * GEMINI_MODEL is an immediate workaround — worth telling the caller, since it
 * is not obvious from the raw error.
 */

function messageOf(error: unknown): string {
    if (typeof error === 'object' && error !== null) {
        const message = (error as { message?: unknown }).message;
        if (typeof message === 'string') {
            return message;
        }
    }
    return String(error);
}

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
    const match = messageOf(error).match(/"code"\s*:\s*(\d{3})/);
    return match?.[1] ? Number(match[1]) : undefined;
}

export function isDailyQuotaExhausted(error: unknown): boolean {
    if (statusOf(error) !== 429) {
        return false;
    }
    return /per\s*day|requestsperday|generaterequestsperday/i.test(messageOf(error));
}

/** Which model the quota error was raised against, when the message says. */
export function modelFromQuotaError(error: unknown): string | null {
    return messageOf(error).match(/model:\s*([\w.-]+)/)?.[1] ?? null;
}

export function quotaGuidance(error: unknown): string {
    const model = modelFromQuotaError(error);
    return [
        `Gemini free-tier daily quota is exhausted${model ? ` for ${model}` : ''}.`,
        '',
        'The daily quota is per project PER MODEL, so the quickest fix is to switch models:',
        '  GEMINI_MODEL=gemini-3.7-flash npm run smoke:e2e',
        '',
        'Other options: wait for the daily reset, or use an API key from a different',
        'Google Cloud project. Check your limits at https://aistudio.google.com/rate-limit',
        '',
        'Note: embeddings draw on a separate quota and are usually still available.'
    ].join('\n');
}
