from __future__ import annotations

from typing import Any

from core.config import MAX_NEW_TOKENS
from core.llm_runtime import generate_answer


DEFAULT_AGENT_SEED = 42


def _run_agent(
    role: str,
    messages: list[dict[str, str]],
    seed: int = DEFAULT_AGENT_SEED,
) -> dict[str, Any]:
    """
    调用一个逻辑Agent。

    所有Agent复用llm_runtime中的同一模型实例，
    不会重复加载模型权重。
    """

    result = generate_answer(
        messages=messages,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        seed=seed,
    )

    output_tokens = int(result["output_tokens"])

    return {
        "role": role,
        "seed": seed,
        "raw_response": result["text"],
        "input_tokens": int(result["input_tokens"]),
        "output_tokens": output_tokens,
        "total_tokens": int(result["total_tokens"]),
        "elapsed_seconds": float(
            result["elapsed_seconds"]
        ),
        "tokens_per_second": result.get(
            "tokens_per_second"
        ),
        "max_new_tokens": int(
            result.get(
                "max_new_tokens",
                MAX_NEW_TOKENS,
            )
        ),
        "hit_token_limit": bool(
            result.get(
                "hit_token_limit",
                output_tokens >= MAX_NEW_TOKENS,
            )
        ),
    }


def run_solver(
    problem: str,
) -> dict[str, Any]:
    """
    Solver：独立完成初始解答。
    """

    system_prompt = """
You are the Solver in a multi-agent mathematics system.

Construct a rigorous and efficient solution independently.
Pay attention to all problem conditions, edge cases, and calculations.
Do not assume that later agents will repair your mistakes.
""".strip()

    user_prompt = f"""
Solve the following mathematics problem.

Problem:
{problem}

Provide a focused step-by-step derivation using only the necessary steps.
Check the main calculations and the original constraints.
Do not restart or repeat the solution.
Use no more than 700 words.

Place your proposed final answer inside \\boxed{{}} on the last line.
""".strip()

    return _run_agent(
        role="solver",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )


def run_critic(
    problem: str,
    solver_response: str,
) -> dict[str, Any]:
    """
    Critic：审查Solver的解答，重点发现错误。
    """

    system_prompt = """
You are the Critic in a multi-agent mathematics system.

Audit the Solver's proposed solution.
Your task is to identify specific mathematical weaknesses, not to praise it.

Check:
1. Whether every inference is justified.
2. Whether calculations and transformations are valid.
3. Whether all conditions and edge cases are handled.
4. Whether the proposed answer actually follows from the reasoning.

Do not accept the solution merely because it appears confident.
Do not use \\boxed{} in your response.
""".strip()

    user_prompt = f"""
Review the following problem and proposed solution.

Problem:
{problem}

Solver response:
--- BEGIN SOLVER RESPONSE ---
{solver_response}
--- END SOLVER RESPONSE ---

Return a concise critique in this exact structure:

VERDICT: CORRECT / MINOR_ERROR / MAJOR_ERROR / UNCERTAIN
VALID_STEPS: ...
IDENTIFIED_ISSUES: ...
REQUIRED_CORRECTION: ...

Use no more than 450 words.
Do not provide a boxed final answer.
""".strip()

    return _run_agent(
        role="critic",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )


def run_verifier(
    problem: str,
    solver_response: str,
    critic_response: str,
) -> dict[str, Any]:
    """
    Verifier：独立验证关键数学步骤。

    Critic负责“找问题”，Verifier负责“验证真假”。
    """

    system_prompt = """
You are the Verifier in a multi-agent mathematics system.

Independently verify the mathematical correctness of the proposed solution.
Treat both the Solver and Critic as potentially fallible.

Recompute the decisive steps when necessary.
Check the proposed result against the original problem conditions.
Distinguish a stylistic weakness from a genuine mathematical error.

Do not use \\boxed{} in your response.
""".strip()

    user_prompt = f"""
Verify the following mathematics problem, solution, and critique.

Problem:
{problem}

Solver response:
--- BEGIN SOLVER RESPONSE ---
{solver_response}
--- END SOLVER RESPONSE ---

Critic response:
--- BEGIN CRITIC RESPONSE ---
{critic_response}
--- END CRITIC RESPONSE ---

Return a concise verification report in this exact structure:

VERIFICATION: VALID / CORRECTABLE / INVALID / UNCERTAIN
CHECKED_FACTS: ...
CONFIRMED_ERRORS: ...
CORRECTED_REASONING: ...
PROPOSED_FINAL_ANSWER: ...

The proposed final answer must be plain text or LaTeX,
but must not use \\boxed{{}}.
Use no more than 550 words.
""".strip()

    return _run_agent(
        role="verifier",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )


def run_judge(
    problem: str,
    solver_response: str,
    critic_response: str,
    verifier_response: str,
) -> dict[str, Any]:
    """
    Judge：综合前面三个Agent并提交唯一最终答案。
    """

    system_prompt = """
You are the final Judge in a multi-agent mathematics system.

Use the Solver, Critic, and Verifier outputs as evidence,
but do not treat any of them as automatically correct.

Independently resolve any disagreement through mathematical verification.
You may accept the original solution, repair it, or replace it.
Your responsibility is to submit one final answer.
""".strip()

    user_prompt = f"""
Determine the final answer to the following mathematics problem.

Problem:
{problem}

Solver:
--- BEGIN SOLVER ---
{solver_response}
--- END SOLVER ---

Critic:
--- BEGIN CRITIC ---
{critic_response}
--- END CRITIC ---

Verifier:
--- BEGIN VERIFIER ---
{verifier_response}
--- END VERIFIER ---

Produce a concise final derivation.
Do not repeat the entire discussion.
Explicitly resolve any important disagreement.
Even if earlier agents are uncertain, submit your best-supported result.

Place the final answer inside \\boxed{{}} on the last line.
""".strip()

    return _run_agent(
        role="judge",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )