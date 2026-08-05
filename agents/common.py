from __future__ import annotations

from typing import Any

from core.llm_runtime import generate_answer, load_model
from utils.answer import extract_declared_answer


AGENT_MAX_INPUT_TOKENS = 3072


def reasoning_excerpt(
    response: str,
    *,
    max_tokens: int,
    head_ratio: float = 0.35,
) -> str:
    """Keep a bounded head/tail excerpt for downstream agents."""

    tokenizer, _ = load_model()
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return response
    head_size = max(1, int(max_tokens * head_ratio))
    tail_size = max_tokens - head_size
    return (
        tokenizer.decode(token_ids[:head_size], skip_special_tokens=True)
        + "\n...[middle omitted]...\n"
        + tokenizer.decode(token_ids[-tail_size:], skip_special_tokens=True)
    )


def run_agent(
    *,
    role: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    seed: int,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_input_tokens: int = AGENT_MAX_INPUT_TOKENS,
) -> dict[str, Any]:
    """Run one role and normalize its telemetry."""

    generation = generate_answer(
        messages=messages,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        enable_thinking=False,
    )
    raw_response = generation["text"]
    return {
        "role": role,
        "seed": seed,
        "raw_response": raw_response,
        "declared_answer": extract_declared_answer(raw_response),
        "input_tokens": int(generation["input_tokens"]),
        "output_tokens": int(generation["output_tokens"]),
        "total_tokens": int(generation["total_tokens"]),
        "elapsed_seconds": float(generation["elapsed_seconds"]),
        "tokens_per_second": generation.get("tokens_per_second"),
        "max_new_tokens": int(generation["max_new_tokens"]),
        "max_input_tokens": max_input_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "enable_thinking": False,
        "hit_token_limit": bool(generation["hit_token_limit"]),
        "peak_cuda_memory_mb": generation.get("peak_cuda_memory_mb"),
    }
