"""Cost from token usage. Prices in USD per million tokens, checked 2026-09-16 on
https://platform.claude.com/docs/en/about-claude/pricing. Unknown model → cost 0 and known=False."""

import os

PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (5.0, 25.0),
    "jev-1.13.0": (0.042, 0.0),  # TypeSafe, docs.typesafe.ai/models 2026-09-22: input only
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
    key = model.split("/")[-1]  # OpenRouter ids look like anthropic/claude-sonnet-4-6
    if key not in PRICES_USD_PER_MTOK:
        return 0.0, False
    pin, pout = PRICES_USD_PER_MTOK[key]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000, True


def usd_to_eur(usd: float) -> float:
    return usd * float(os.environ.get("EUR_PER_USD", "0.86"))
