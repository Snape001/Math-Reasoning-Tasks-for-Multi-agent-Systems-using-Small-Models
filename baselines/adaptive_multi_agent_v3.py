from __future__ import annotations

from typing import Any

from baselines.router_v3 import run_router_solver
from baselines.multi_agent_v3 import run_multi_agent_v3


def run_adaptive_multi_agent_v3(
    problem: str,
    *,
    entropy_threshold: float,
    base_seed: int = 42,
) -> dict[str, Any]:
    """Route easy items directly and hard items into the V3 collaboration."""

    router = run_router_solver(problem, seed=base_seed)
    answer = router["predicted_answer"]
    route_score = router["route_score"]
    lowered_problem = problem.lower()
    dense_visual = (
        "[asy]" in lowered_problem and lowered_problem.count("label(") >= 12
    ) or r"\begin{tabular}" in problem

    if answer is None:
        should_escalate = True
        route_reason = "missing_answer"
    elif router["hit_token_limit_count"]:
        should_escalate = True
        route_reason = "router_token_limit"
    elif route_score is None:
        should_escalate = True
        route_reason = "missing_entropy"
    elif dense_visual:
        # Dense tables make entropy noisy and add a known transcription failure
        # mode when many downstream agents repeatedly reinterpret the same cells.
        should_escalate = False
        route_reason = "dense_visual_router_preserved"
    elif route_score > entropy_threshold:
        should_escalate = True
        route_reason = "high_entropy"
    else:
        should_escalate = False
        route_reason = "low_entropy"

    if not should_escalate:
        return {
            "architecture": "entropy_router->single_agent",
            "architecture_version": "v3",
            "route": "single_agent",
            "route_reason": route_reason,
            "entropy_threshold": entropy_threshold,
            "route_score": route_score,
            "entropy": router["entropy"],
            "predicted_answer": answer,
            "final_answer_source": "router",
            "router_answer": answer,
            **{
                key: router[key]
                for key in (
                    "model_calls",
                    "agent_calls",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "elapsed_seconds",
                    "average_output_tokens_per_call",
                    "average_elapsed_seconds_per_call",
                    "hit_token_limit_count",
                    "peak_cuda_memory_mb",
                )
            },
        }

    collaboration = run_multi_agent_v3(
        problem,
        router_call=router["router_call"],
        base_seed=base_seed,
    )
    return {
        **collaboration,
        "architecture": "entropy_router->" + collaboration["architecture"],
        "route": "multi_agent",
        "route_reason": route_reason,
        "entropy_threshold": entropy_threshold,
        "route_score": route_score,
        "entropy": router["entropy"],
        "router_answer": answer,
    }
