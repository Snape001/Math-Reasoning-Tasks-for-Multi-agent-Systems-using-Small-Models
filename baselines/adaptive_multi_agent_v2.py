from __future__ import annotations

from typing import Any

from agents.math_agents_v2 import (
    run_critic,
    run_independent_verifier,
    run_judge,
    run_solver_b,
)
from baselines.cot import (
    COT_SYSTEM_PROMPT,
    COT_USER_TEMPLATE,
)
from core.config import MAX_NEW_TOKENS
from core.llm_runtime import generate_answer
from utils.answer import (
    extract_boxed_answer,
    normalize_answer,
)


def _wrap_router_call(
    generation_result: dict[str, Any],
) -> dict[str, Any]:
    output_tokens = int(
        generation_result["output_tokens"]
    )

    return {
        "role": "router_solver_a",
        "seed": 42,
        "raw_response": generation_result["text"],
        "input_tokens": int(
            generation_result["input_tokens"]
        ),
        "output_tokens": output_tokens,
        "total_tokens": int(
            generation_result["total_tokens"]
        ),
        "elapsed_seconds": float(
            generation_result["elapsed_seconds"]
        ),
        "tokens_per_second": (
            generation_result.get(
                "tokens_per_second"
            )
        ),
        "max_new_tokens": int(
            generation_result.get(
                "max_new_tokens",
                MAX_NEW_TOKENS,
            )
        ),
        "hit_token_limit": bool(
            generation_result.get(
                "hit_token_limit",
                output_tokens >= MAX_NEW_TOKENS,
            )
        ),
        "entropy": generation_result.get(
            "entropy",
            {},
        ),
    }


def run_router_v2(
    problem: str,
) -> dict[str, Any]:
    """
    第一阶段：
    单智能体CoT，同时记录生成概率熵。
    """

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

    generation_result = generate_answer(
        messages=messages,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        seed=42,
        record_entropy=True,
        entropy_stride=8,
    )

    router_call = _wrap_router_call(
        generation_result
    )

    predicted_answer = extract_boxed_answer(
        router_call["raw_response"]
    )

    entropy = router_call["entropy"]

    return {
        "predicted_answer": predicted_answer,
        "normalized_predicted_answer": (
            normalize_answer(predicted_answer)
        ),
        "route_score": entropy.get(
            "route_score"
        ),
        "entropy": entropy,
        "router_call": router_call,
        "input_tokens": router_call[
            "input_tokens"
        ],
        "output_tokens": router_call[
            "output_tokens"
        ],
        "total_tokens": router_call[
            "total_tokens"
        ],
        "elapsed_seconds": router_call[
            "elapsed_seconds"
        ],
        "hit_token_limit": router_call[
            "hit_token_limit"
        ],
    }


def _aggregate_calls(
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    total_input_tokens = sum(
        int(call["input_tokens"])
        for call in calls
    )
    total_output_tokens = sum(
        int(call["output_tokens"])
        for call in calls
    )
    total_tokens = sum(
        int(call["total_tokens"])
        for call in calls
    )
    total_elapsed_seconds = sum(
        float(call["elapsed_seconds"])
        for call in calls
    )

    return {
        "model_calls": len(calls),
        "agent_calls": calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "elapsed_seconds": round(
            total_elapsed_seconds,
            4,
        ),
        "average_output_tokens_per_call": round(
            total_output_tokens / len(calls),
            2,
        ),
        "average_elapsed_seconds_per_call": round(
            total_elapsed_seconds / len(calls),
            4,
        ),
        "hit_token_limit_count": sum(
            int(call["hit_token_limit"])
            for call in calls
        ),
    }


def run_adaptive_multi_agent_v2(
    problem: str,
    entropy_threshold: float,
) -> dict[str, Any]:
    """
    Entropy-Gated Adaptive Orchestration。

    简单题：
        Router CoT直接输出，1次调用。

    高不确定题：
        Router/Solver A
        + Solver B
        + Independent Verifier
        + Critic
        + Judge
        共5次调用。
    """

    router = run_router_v2(problem)

    router_answer = router[
        "predicted_answer"
    ]
    route_score = router["route_score"]

    if router_answer is None:
        should_escalate = True
        route_reason = "missing_boxed_answer"

    elif router["hit_token_limit"]:
        should_escalate = True
        route_reason = "router_token_limit"

    elif route_score is None:
        should_escalate = True
        route_reason = "missing_entropy"

    elif route_score > entropy_threshold:
        should_escalate = True
        route_reason = "high_entropy"

    else:
        should_escalate = False
        route_reason = "low_entropy"

    router_call = router["router_call"]

    # 高置信度：直接采用Router CoT答案
    if not should_escalate:
        totals = _aggregate_calls(
            [router_call]
        )

        return {
            "architecture": (
                "entropy_router->single_agent"
            ),
            "architecture_version": "v2",
            "route": "single_agent",
            "route_reason": route_reason,
            "entropy_threshold": (
                entropy_threshold
            ),
            "route_score": route_score,
            "entropy": router["entropy"],

            "predicted_answer": router_answer,
            "router_answer": router_answer,
            "solver_b_answer": None,
            "verifier_answer": None,

            **totals,
        }

    # 低置信度：Router结果直接作为Solver A
    solver_b = run_solver_b(problem)

    independent_verifier = (
        run_independent_verifier(problem)
    )

    critic = run_critic(
        problem=problem,
        solver_a_response=router_call[
            "raw_response"
        ],
        solver_b_response=solver_b[
            "raw_response"
        ],
    )

    judge = run_judge(
        problem=problem,
        solver_a_response=router_call[
            "raw_response"
        ],
        solver_b_response=solver_b[
            "raw_response"
        ],
        critic_response=critic[
            "raw_response"
        ],
        verifier_response=(
            independent_verifier[
                "raw_response"
            ]
        ),
    )

    final_answer = extract_boxed_answer(
        judge["raw_response"]
    )

    solver_b_answer = extract_boxed_answer(
        solver_b["raw_response"]
    )
    verifier_answer = extract_boxed_answer(
        independent_verifier["raw_response"]
    )

    calls = [
        router_call,
        solver_b,
        independent_verifier,
        critic,
        judge,
    ]

    totals = _aggregate_calls(calls)

    return {
        "architecture": (
            "entropy_router->"
            "solver_b+independent_verifier"
            "->critic->judge"
        ),
        "architecture_version": "v2",
        "route": "multi_agent",
        "route_reason": route_reason,
        "entropy_threshold": entropy_threshold,
        "route_score": route_score,
        "entropy": router["entropy"],

        "predicted_answer": final_answer,
        "router_answer": router_answer,
        "solver_b_answer": solver_b_answer,
        "verifier_answer": verifier_answer,

        **totals,
    }
