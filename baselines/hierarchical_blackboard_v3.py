from __future__ import annotations

from typing import Any

from agents.math_agents_v3 import (
    run_blackboard_decomposer_v3,
    run_blackboard_integrator_v3,
    run_blackboard_plan_auditor_v3,
    run_blackboard_repairer_v3,
    run_blackboard_verifier_v3,
    run_blackboard_worker_v3,
    run_table_calculator_v3,
    run_table_mapping_auditor_v3,
    run_table_transcriber_v3,
)
from baselines.common import aggregate_calls
from utils.answer import exact_match, extract_declared_answer, normalize_answer


INVALID_ANSWERS = {"", "missing", "unknown", "none", "n/a"}


def _valid_answer(answer: str | None) -> bool:
    normalized = normalize_answer(answer)
    return normalized is not None and normalized.lower() not in INVALID_ANSWERS


def _extract_field(text: str, field: str) -> str | None:
    target = field.upper().replace("_", " ") + ":"
    for line in text.splitlines():
        stripped = line.strip().replace("*", "")
        normalized = stripped.upper().replace("_", " ")
        if normalized.startswith(target):
            return stripped.split(":", 1)[1].strip() or None
    return None


def _answer(call: dict[str, Any] | None) -> str | None:
    if call is None:
        return None
    return call.get("declared_answer") or extract_declared_answer(
        call.get("raw_response")
    )


def run_hierarchical_blackboard_v3(
    problem: str,
    *,
    base_seed: int = 42,
) -> dict[str, Any]:
    """Solve by dependent role specialization, not independent-sample voting."""

    decomposer = run_blackboard_decomposer_v3(problem, seed=base_seed + 8)
    plan_auditor = run_blackboard_plan_auditor_v3(
        problem,
        decomposer["raw_response"],
        seed=base_seed + 9,
    )
    plan = (
        decomposer["raw_response"]
        + "\n\nCONSTRAINT_AUDIT (takes precedence):\n"
        + plan_auditor["raw_response"]
    )
    subgoals = [
        _extract_field(plan_auditor["raw_response"], f"AUDITED_SUBGOAL_{index}")
        or _extract_field(decomposer["raw_response"], f"SUBGOAL_{index}")
        or f"Complete ordered subgoal {index} from TEAM_PLAN."
        for index in range(1, 4)
    ]

    workers: list[dict[str, Any]] = []
    published: list[str] = []
    for index, subgoal in enumerate(subgoals, start=1):
        worker = run_blackboard_worker_v3(
            problem,
            plan,
            subgoal,
            "\n\n".join(published),
            worker_id=index,
            seed=base_seed + 9 + index,
        )
        workers.append(worker)
        published.append(worker["raw_response"])

    integrator = run_blackboard_integrator_v3(
        problem,
        plan,
        published,
        seed=base_seed + 13,
    )
    verifier = run_blackboard_verifier_v3(
        problem,
        plan,
        published,
        integrator["raw_response"],
        seed=base_seed + 14,
    )

    integrator_answer = _answer(integrator)
    verifier_answer = _answer(verifier)
    verifier_verdict = _extract_field(verifier["raw_response"], "VERDICT")
    verifier_accepts = (
        bool(verifier_verdict)
        and verifier_verdict.upper().startswith("PASS")
        and _valid_answer(integrator_answer)
        and _valid_answer(verifier_answer)
        and exact_match(integrator_answer, verifier_answer)
    )

    repairer: dict[str, Any] | None = None
    repairer_answer: str | None = None
    if not verifier_accepts:
        repairer = run_blackboard_repairer_v3(
            problem,
            plan,
            published,
            integrator["raw_response"],
            verifier["raw_response"],
            seed=base_seed + 15,
        )
        repairer_answer = _answer(repairer)

    if verifier_accepts:
        predicted_answer = integrator_answer
        final_answer_source = "verified_blackboard_integrator"
        workflow_branch = "blackboard_verified"
    elif _valid_answer(repairer_answer):
        predicted_answer = repairer_answer
        final_answer_source = "blackboard_repairer"
        workflow_branch = "blackboard_repaired"
    elif _valid_answer(verifier_answer):
        predicted_answer = verifier_answer
        final_answer_source = "blackboard_verifier_fallback"
        workflow_branch = "blackboard_unverified_fallback"
    elif _valid_answer(integrator_answer):
        predicted_answer = integrator_answer
        final_answer_source = "blackboard_integrator_fallback"
        workflow_branch = "blackboard_unverified_fallback"
    else:
        predicted_answer = None
        final_answer_source = "blackboard_no_valid_answer"
        workflow_branch = "blackboard_failed"

    calls = [decomposer, plan_auditor, *workers, integrator, verifier]
    if repairer is not None:
        calls.append(repairer)
    return {
        "architecture": (
            "decomposer->constraint_auditor->worker_1->worker_2(shared_blackboard)"
            "->worker_3(shared_blackboard)->integrator"
            "->independent_verifier->conditional_repairer"
        ),
        "architecture_version": "v3",
        "predicted_answer": predicted_answer,
        "normalized_predicted_answer": normalize_answer(predicted_answer),
        "final_answer_source": final_answer_source,
        "workflow_branch": workflow_branch,
        "blackboard_plan": plan,
        "blackboard_plan_audit": plan_auditor["raw_response"],
        "blackboard_subgoals": subgoals,
        "blackboard_worker_responses": published,
        "blackboard_worker_answers": [_answer(worker) for worker in workers],
        "blackboard_integrator_answer": integrator_answer,
        "blackboard_verifier_answer": verifier_answer,
        "blackboard_verifier_verdict": verifier_verdict,
        "blackboard_verifier_accepts": verifier_accepts,
        "blackboard_repairer_answer": repairer_answer,
        **aggregate_calls(calls),
    }


def run_dense_table_team_v3(
    problem: str,
    *,
    router_call: dict[str, Any] | None = None,
    base_seed: int = 42,
) -> dict[str, Any]:
    """Use a lossless transcriber/calculator pipeline for dense visual tables."""

    transcriber = run_table_transcriber_v3(problem, seed=base_seed + 8)
    mapping_auditor = run_table_mapping_auditor_v3(
        problem,
        transcriber["raw_response"],
        seed=base_seed + 9,
    )
    audited_transcript = (
        transcriber["raw_response"]
        + "\n\nMAPPING_AUDIT (takes precedence):\n"
        + mapping_auditor["raw_response"]
    )
    calculator = run_table_calculator_v3(
        problem,
        audited_transcript,
        seed=base_seed + 10,
    )
    answer = _answer(calculator)
    calls = [
        *([router_call] if router_call is not None else []),
        transcriber,
        mapping_auditor,
        calculator,
    ]
    return {
        "architecture": (
            "router(optional)->table_transcriber->mapping_auditor"
            "->table_calculator"
        ),
        "architecture_version": "v3",
        "predicted_answer": answer if _valid_answer(answer) else None,
        "normalized_predicted_answer": normalize_answer(answer),
        "final_answer_source": "table_calculator" if _valid_answer(answer) else "table_team_failed",
        "workflow_branch": "dense_table_specialist_team",
        "table_transcript": transcriber["raw_response"],
        "table_mapping_audit": mapping_auditor["raw_response"],
        "table_calculator_answer": answer,
        **aggregate_calls(calls),
    }
