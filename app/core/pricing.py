from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_million: float
    output_usd_per_million: float


# Rates verified against multiple current sources as of August 2026.
# OpenAI does not expose a pricing API — this table is a hardcoded
# snapshot that WILL drift out of date as prices change. Recheck against
# https://openai.com/api/pricing before trusting real spend numbers for
# anything beyond a demo/portfolio context.
_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_usd_per_million=2.50, output_usd_per_million=10.00),
    "gpt-4o-mini": ModelPricing(input_usd_per_million=0.15, output_usd_per_million=0.60),
}


def get_pricing(model: str) -> ModelPricing | None:
    return _PRICING.get(model)


def compute_cost_micros(model: str, prompt_tokens: int, completion_tokens: int) -> int | None:
    """Cost in integer micro-dollars (millionths of a dollar), or None if
    `model` isn't in the pricing table. None is a distinct case from 0 —
    callers must not silently treat "unknown price" as "free"; the budget
    tracker logs a warning and skips incrementing spend rather than
    guessing at a number.
    """
    pricing = get_pricing(model)
    if pricing is None:
        return None

    cost_usd = (
        prompt_tokens * pricing.input_usd_per_million + completion_tokens * pricing.output_usd_per_million
    ) / 1_000_000
    return round(cost_usd * 1_000_000)
