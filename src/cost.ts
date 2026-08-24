/**
 * Per-request cost and token accounting.
 *
 * Prices are a configuration input, not a fact this code can know — they change
 * and they differ per model, so they live in env vars with documented defaults
 * and every reported figure is labelled an estimate. A cost number presented as
 * authoritative when it is actually a stale constant is worse than no number.
 */

export interface ModelPricing {
    /** USD per million input tokens. */
    inputPerMillion: number;
    /** USD per million output tokens. */
    outputPerMillion: number;
}

/**
 * Defaults are placeholders for free-tier use, where actual spend is zero.
 * Override via GEMINI_INPUT_PRICE / GEMINI_OUTPUT_PRICE with your tier's real
 * numbers before treating any of this as a spend figure.
 */
export const PRICING: ModelPricing = {
    inputPerMillion: Number(process.env.GEMINI_INPUT_PRICE ?? 0.075),
    outputPerMillion: Number(process.env.GEMINI_OUTPUT_PRICE ?? 0.30)
};

export interface UsageRecord {
    model: string;
    promptTokens: number;
    outputTokens: number;
    estimatedCostUsd: number;
    at: number;
}

export function estimateCost(promptTokens: number, outputTokens: number, pricing: ModelPricing = PRICING): number {
    return (promptTokens / 1_000_000) * pricing.inputPerMillion + (outputTokens / 1_000_000) * pricing.outputPerMillion;
}

export class CostTracker {
    private records: UsageRecord[] = [];
    private maxRecords = 10_000;

    record(model: string, promptTokens: number, outputTokens: number, at = Date.now()): UsageRecord {
        const entry: UsageRecord = {
            model,
            promptTokens,
            outputTokens,
            estimatedCostUsd: estimateCost(promptTokens, outputTokens),
            at
        };
        this.records.push(entry);
        if (this.records.length > this.maxRecords) {
            this.records.splice(0, this.records.length - this.maxRecords);
        }
        return entry;
    }

    /** Tokens saved by cache hits — the cache's actual payoff, in money. */
    private savedTokens = { prompt: 0, output: 0 };

    recordCacheSaving(promptTokens: number, outputTokens: number): void {
        this.savedTokens.prompt += promptTokens;
        this.savedTokens.output += outputTokens;
    }

    summary(now = Date.now()) {
        const totalCost = this.records.reduce((sum, r) => sum + r.estimatedCostUsd, 0);
        const promptTokens = this.records.reduce((sum, r) => sum + r.promptTokens, 0);
        const outputTokens = this.records.reduce((sum, r) => sum + r.outputTokens, 0);

        const lastHour = this.records.filter(r => now - r.at < 3_600_000);
        const hourlyCost = lastHour.reduce((sum, r) => sum + r.estimatedCostUsd, 0);

        const byModel: Record<string, { calls: number; costUsd: number; tokens: number }> = {};
        for (const record of this.records) {
            const bucket = (byModel[record.model] ??= { calls: 0, costUsd: 0, tokens: 0 });
            bucket.calls += 1;
            bucket.costUsd += record.estimatedCostUsd;
            bucket.tokens += record.promptTokens + record.outputTokens;
        }

        return {
            note: 'Estimates. Derived from configured per-token prices, not billed amounts.',
            calls: this.records.length,
            prompt_tokens: promptTokens,
            output_tokens: outputTokens,
            total_tokens: promptTokens + outputTokens,
            estimated_cost_usd: Number(totalCost.toFixed(6)),
            last_hour_cost_usd: Number(hourlyCost.toFixed(6)),
            // Straight-line extrapolation from the last hour. Honest only if
            // traffic is steady; labelled so nobody reads it as a forecast.
            projected_monthly_usd: Number((hourlyCost * 24 * 30).toFixed(4)),
            estimated_savings_from_cache_usd: Number(
                estimateCost(this.savedTokens.prompt, this.savedTokens.output).toFixed(6)
            ),
            by_model: byModel,
            pricing_used: PRICING
        };
    }

    reset(): void {
        this.records = [];
        this.savedTokens = { prompt: 0, output: 0 };
    }
}

export const costTracker = new CostTracker();
