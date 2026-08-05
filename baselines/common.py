from __future__ import annotations

from typing import Any


def aggregate_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token, latency, truncation, and memory telemetry."""

    input_tokens = sum(int(call["input_tokens"]) for call in calls)
    output_tokens = sum(int(call["output_tokens"]) for call in calls)
    elapsed_seconds = sum(float(call["elapsed_seconds"]) for call in calls)
    return {
        "model_calls": len(calls),
        "agent_calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "average_output_tokens_per_call": round(output_tokens / len(calls), 2),
        "average_elapsed_seconds_per_call": round(
            elapsed_seconds / len(calls), 4
        ),
        "hit_token_limit_count": sum(
            int(bool(call.get("hit_token_limit"))) for call in calls
        ),
        "peak_cuda_memory_mb": max(
            (
                float(call["peak_cuda_memory_mb"])
                for call in calls
                if call.get("peak_cuda_memory_mb") is not None
            ),
            default=None,
        ),
    }
