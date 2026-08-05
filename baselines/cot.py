from typing import Any

from core.config import ENABLE_THINKING, MAX_INPUT_TOKENS, MAX_NEW_TOKENS
from core.llm_runtime import generate_answer


COT_SYSTEM_PROMPT = """
You are a mathematics problem solver.
Solve each problem using explicit step-by-step reasoning.
Follow the requested output format exactly.
""".strip()


COT_USER_TEMPLATE = """
Solve the following mathematics problem.

Problem:
{problem}

Provide a concise step-by-step derivation using only the necessary steps.
Check the main calculations, but do not restart or repeat the solution.
Even if uncertain, always finish with your best final answer.
End with exactly:
FINAL_ANSWER: \\boxed{{your answer}}
""".strip()


def run_cot(
    problem: str,
    seed: int = 42,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": COT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": COT_USER_TEMPLATE.format(
                problem=problem,
            ),
        },
    ]

    return generate_answer(
        messages=messages,
        max_new_tokens=MAX_NEW_TOKENS,
        max_input_tokens=MAX_INPUT_TOKENS,
        do_sample=False,
        seed=seed,
        enable_thinking=ENABLE_THINKING,
    )
