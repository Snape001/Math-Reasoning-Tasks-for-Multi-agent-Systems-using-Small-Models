from core.config import ENABLE_THINKING, MAX_INPUT_TOKENS, MAX_NEW_TOKENS
from core.llm_runtime import generate_answer


def run_direct(
    problem: str,
    seed: int = 42,
) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Direct Answer baseline in a mathematics "
                "evaluation. Follow the requested output format exactly."
            ),
        },
        {
            "role": "user",
            "content": f"""
Answer the following mathematics problem directly.

Problem:
{problem}

Do not provide a derivation, explanation, verification, or alternative answer.
Output exactly one line:
FINAL_ANSWER: \\boxed{{your answer}}
""".strip(),
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
