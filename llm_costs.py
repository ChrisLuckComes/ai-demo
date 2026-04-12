from typing import Optional


MODEL_PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
}


def estimate_cost(
    model_name: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(model_name)
    if pricing is None:
        return None
    if input_tokens is None and output_tokens is None:
        return None

    normalized_input = max(0, input_tokens or 0)
    normalized_output = max(0, output_tokens or 0)
    cost = (
        normalized_input * pricing["input"] + normalized_output * pricing["output"]
    ) / 1_000_000
    return round(cost, 8)
