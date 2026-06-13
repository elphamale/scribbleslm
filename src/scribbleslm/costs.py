"""Per-model cost accounting.

Records spend keyed by MODEL identity (not just per-source totals) so a charge is
explainable with one GROUP BY query. Local models cost $0.

Prices are USD per 1M tokens (input, output). Embedding models bill input only.
NOTE: free-token allotments (context-3/4-lite: 200M free; law-2: 50M free) are NOT
modelled here — this records list-price spend, which over-states real cost while you
are inside the free tier. Good enough to answer "what would this cost / why a charge".
"""
from __future__ import annotations

import sqlite3

# model -> (usd_per_1M_input, usd_per_1M_output)
PRICING: dict[str, tuple[float, float]] = {
    "voyage-context-3": (0.18, 0.0),
    "voyage-4-lite": (0.02, 0.0),
    "voyage-3.5": (0.06, 0.0),          # legacy, no free tier
    "voyage-law-2": (0.12, 0.0),
    # DeepSeek context LLM (private path); approximate published rates
    "deepseek-v4-flash": (0.27, 1.10),
    "deepseek-chat": (0.27, 1.10),
}


def price_for(model: str) -> tuple[float, float]:
    if model in PRICING:
        return PRICING[model]
    if model.startswith("bge") or "gguf" in model:
        return (0.0, 0.0)  # local
    if model.startswith("voyage"):
        return (0.10, 0.0)  # unknown voyage model — conservative estimate
    if model.startswith("deepseek"):
        return (0.27, 1.10)
    return (0.0, 0.0)


def compute_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    pin, pout = price_for(model)
    return input_tokens / 1_000_000 * pin + output_tokens / 1_000_000 * pout


def record_cost(conn: sqlite3.Connection, source_id: int | None, model: str, *,
                api_calls: int = 0, input_tokens: int = 0, output_tokens: int = 0,
                cache_hit_tokens: int = 0) -> None:
    conn.execute(
        "INSERT INTO cost_log(source_id, model, api_calls, input_tokens, output_tokens, "
        "cache_hit_tokens, cost_usd) VALUES(?,?,?,?,?,?,?)",
        (source_id, model, api_calls, input_tokens, output_tokens, cache_hit_tokens,
         compute_cost(model, input_tokens, output_tokens)),
    )
    conn.commit()


def source_cost_breakdown(conn: sqlite3.Connection, source_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT model, SUM(api_calls) calls, SUM(input_tokens) in_tok, "
        "SUM(output_tokens) out_tok, SUM(cache_hit_tokens) cache_tok, "
        "ROUND(SUM(cost_usd), 6) cost_usd FROM cost_log WHERE source_id = ? "
        "GROUP BY model ORDER BY cost_usd DESC",
        (source_id,),
    ).fetchall()
    return [dict(r) for r in rows]
