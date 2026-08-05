from __future__ import annotations

from typing import Any

from core.config import MAX_NEW_TOKENS
from core.llm_runtime import (
    generate_answer,
    load_model,
)

SOLVER_MAX_TOKENS = 1536
VERIFIER_MAX_TOKENS = 1536
CRITIC_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 1024

def _truncate_for_context(
    text: str,
    max_tokens: int,
) -> str:
    """
    缩短传给下游Agent的回答。

    保留前60%和后40%，既保留主要推导，
    又保留通常位于结尾的最终答案。
    """

    tokenizer, _ = load_model()

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(token_ids) <= max_tokens:
        return text

    head_size = int(max_tokens * 0.6)
    tail_size = max_tokens - head_size

    head_text = tokenizer.decode(
        token_ids[:head_size],
        skip_special_tokens=True,
    )

    tail_text = tokenizer.decode(
        token_ids[-tail_size:],
        skip_special_tokens=True,
    )

    return (
        head_text
        + "\n\n...[middle reasoning truncated]...\n\n"
        + tail_text
    )

def _run_agent(
    role: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> dict[str, Any]:
    
    result = generate_answer(
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        seed=42,
    )

    output_tokens = int(result["output_tokens"])

    return {
        "role": role,
        "raw_response": result["text"],
        "input_tokens": int(result["input_tokens"]),
        "output_tokens": output_tokens,
        "total_tokens": int(result["total_tokens"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "tokens_per_second": result.get("tokens_per_second"),
        "max_new_tokens": int(
            result.get("max_new_tokens", max_new_tokens)
        ),
        "hit_token_limit": bool(
            result.get(
                "hit_token_limit",
                output_tokens >= max_new_tokens,
            )
        ),
    }


def run_solver_a(problem: str) -> dict[str, Any]:
    return _run_agent(
        role="solver_a",
        max_new_tokens=SOLVER_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": """
You are Solver A in a multi-agent mathematics system.
Solve the problem using a direct and systematic derivation.
Check calculations and all original constraints.
""".strip(),
            },
            {
                "role": "user",
                "content": f"""
Solve the following mathematics problem.

Problem:
{problem}

Use only necessary steps and no more than 650 words.
Do not restart or repeat the reasoning.
Place the final answer inside \\boxed{{}} on the last line.
""".strip(),
            },
        ],
    )


def run_solver_b(problem: str) -> dict[str, Any]:
    return _run_agent(
        role="solver_b",
        max_new_tokens=SOLVER_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": """
You are Solver B in a multi-agent mathematics system.
Solve independently as a skeptical mathematical analyst.
Prefer an alternative method when practical.
Check hidden constraints, factors of two, signs, boundary cases,
and whether the result satisfies the original problem.
""".strip(),
            },
            {
                "role": "user",
                "content": f"""
Independently solve the following mathematics problem.

Problem:
{problem}

Use no more than 650 words.
Do not assume another solver's answer.
Place the final answer inside \\boxed{{}} on the last line.
""".strip(),
            },
        ],
    )


def run_critic(
    problem: str,
    solver_a_response: str,
    solver_b_response: str,
    
) -> dict[str, Any]:
     
    solver_a_context = _truncate_for_context(
        solver_a_response,
        max_tokens=700,
    )
    solver_b_context = _truncate_for_context(
        solver_b_response,
        max_tokens=700,
    )

    return _run_agent(
        role="critic",
        max_new_tokens=512,
        messages=[
            {
                "role": "system",
                "content": """
You are the Critic in a multi-agent mathematics system.
Compare two independent solutions and locate concrete mathematical errors.
Do not favor an answer because it is longer or more confident.
Do not use \\boxed{}.
""".strip(),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

Solver A:
--- BEGIN A ---
{solver_a_context}
--- END A ---

Solver B:
--- BEGIN B ---
{solver_b_context}
--- END B ---

Return:

VERDICT: AGREE / PREFER_A / PREFER_B / BOTH_WRONG / UNCERTAIN
A_ISSUES: ...
B_ISSUES: ...
DECISIVE_CHECK: ...
RECOMMENDATION: ...

Use no more than 450 words.
Do not provide a boxed answer.
""".strip(),
            },
        ],
    )


def run_independent_verifier(
    problem: str,
) -> dict[str, Any]:
    """
    Verifier只看原题，不看Solver与Critic，避免被错误答案锚定。
    """

    return _run_agent(
        role="independent_verifier",
        max_new_tokens=VERIFIER_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": """
You are an independent mathematical Verifier.

Recompute the problem from scratch.
Pay special attention to numerical factors, algebraic transformations,
constraints, exceptional cases, and substitution checks.
You have not seen any other agent's answer.
""".strip(),
            },
            {
                "role": "user",
                "content": f"""
Independently verify and solve this mathematics problem.

Problem:
{problem}

Use a concise derivation of no more than 650 words.
Place your independently verified answer inside \\boxed{{}}
on the last line.
""".strip(),
            },
        ],
    )


def run_judge(
    problem: str,
    solver_a_response: str,
    solver_b_response: str,
    critic_response: str,
    verifier_response: str,
) -> dict[str, Any]:
    solver_a_context = _truncate_for_context(
        solver_a_response,
        max_tokens=500,
    )

    solver_b_context = _truncate_for_context(
        solver_b_response,
        max_tokens=500,
    )

    critic_context = _truncate_for_context(
        critic_response,
        max_tokens=350,
    )

    verifier_context = _truncate_for_context(
        verifier_response,
        max_tokens=500,
    )

    return _run_agent(
        role="judge",
        max_new_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": """
You are the final Judge in a multi-agent mathematics system.

The previous agents may all be wrong.
Compare their reasoning, independently check the decisive step,
and submit one best-supported answer.
Do not decide by simple majority alone.
""".strip(),
            },
            {
                "role": "user",
                "content": f"""
Problem:
{problem}

Solver A:
--- BEGIN A ---
{solver_a_context}
--- END A ---

Solver B:
--- BEGIN B ---
{solver_b_context}
--- END B ---

Critic:
--- BEGIN CRITIC ---
{critic_context}
--- END CRITIC ---

Independent Verifier:
--- BEGIN VERIFIER ---
{verifier_context}
--- END VERIFIER ---

Resolve disagreements using mathematical verification.
Produce a concise final derivation.
Place the final answer inside \\boxed{{}} on the last line.
""".strip(),
            },
        ],
    )