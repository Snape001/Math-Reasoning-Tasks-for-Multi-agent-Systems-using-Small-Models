from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.adaptive_multi_agent_v4 import run_adaptive_multi_agent_v4
from baselines.multi_agent_v4 import run_multi_agent_v4
from baselines.router_v3 import run_router_solver
from core.config import ENABLE_THINKING, EXPERIMENT_VERSION, REPETITION_PENALTY
from core.model_spec import validate_model_requirement
from utils.answer import exact_match, extract_boxed_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METHOD_BY_MODE = {
    "probe": "answer_first_router_probe_v4",
    "adaptive": "multi_agent_v4",
    "full": "multi_agent_v4_full",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行 JSON 错误：{error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象。")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 V4 证据验证的两阶段多智能体数学推理框架。"
    )
    parser.add_argument("--mode", choices=["probe", "adaptive", "full"], required=True)
    parser.add_argument("--input", type=Path, default=Path("data/dev_20.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--router-config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--problem-id",
        action="append",
        help="只运行指定 problem_id；可重复传入。",
    )
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0。")

    input_file = resolve_path(args.input)
    output_file = resolve_path(args.output)
    if not input_file.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_file}")

    model_metadata = validate_model_requirement(strict=args.strict_model)
    threshold: float | None = None
    if args.mode == "adaptive":
        if args.router_config is None:
            raise ValueError("adaptive 模式必须提供 --router-config。")
        config_path = resolve_path(args.router_config)
        with config_path.open("r", encoding="utf-8") as file:
            threshold = float(json.load(file)["entropy_threshold"])

    problems = read_jsonl(input_file)
    if args.problem_id:
        selected_ids = set(args.problem_id)
        problems = [
            item
            for item in problems
            if str(item.get("id") or item.get("unique_id")) in selected_ids
        ]
        found_ids = {
            str(item.get("id") or item.get("unique_id")) for item in problems
        }
        missing = selected_ids - found_ids
        if missing:
            raise ValueError(f"输入数据中不存在：{sorted(missing)}")
    if args.limit is not None:
        problems = problems[: args.limit]

    expected_method = METHOD_BY_MODE[args.mode]
    completed = {
        (
            str(row.get("problem_id")),
            int(row.get("run_id", 0)),
            str(row.get("mode")),
            str(row.get("method")),
            str(row.get("experiment_version")),
        )
        for row in read_jsonl(output_file)
    }

    print(f"模式：{args.mode}")
    print(f"题目数：{len(problems)}")
    print(f"模型检查合规：{model_metadata['requirement_compliant']}")
    if threshold is not None:
        print(f"熵阈值：{threshold}")

    for index, item in enumerate(problems, start=1):
        problem_id = str(item.get("id") or item["unique_id"])
        key = (
            problem_id,
            args.run_id,
            args.mode,
            expected_method,
            EXPERIMENT_VERSION,
        )
        if key in completed:
            print(f"[{index}/{len(problems)}] 跳过：{problem_id}")
            continue

        print(f"\n[{index}/{len(problems)}] {problem_id}")
        if args.mode == "probe":
            result = run_router_solver(item["problem"], seed=args.seed)
            route = "probe"
        elif args.mode == "adaptive":
            assert threshold is not None
            result = run_adaptive_multi_agent_v4(
                item["problem"],
                entropy_threshold=threshold,
                base_seed=args.seed,
            )
            route = result["route"]
        else:
            result = run_multi_agent_v4(item["problem"], base_seed=args.seed)
            route = "multi_agent"

        reference_answer = extract_boxed_answer(item["answer"]) or item["answer"]
        predicted_answer = result["predicted_answer"]
        record = {
            "problem_id": problem_id,
            "level": int(item["level"]),
            "subject": item["subject"],
            "method": expected_method,
            "experiment_version": EXPERIMENT_VERSION,
            "mode": args.mode,
            "run_id": args.run_id,
            "base_seed": args.seed,
            "architecture": result.get("architecture"),
            "architecture_version": result.get("architecture_version", "v4"),
            "generation_config": {
                "adaptive_per_role_token_budgets": True,
                "enable_thinking": ENABLE_THINKING,
                "repetition_penalty": REPETITION_PENALTY,
                "conditional_third_reviewer": True,
            },
            "predicted_answer": predicted_answer,
            "reference_answer": reference_answer,
            "correct": exact_match(predicted_answer, reference_answer),
            "route": route,
            "route_reason": result.get("route_reason"),
            "route_score": result.get("route_score"),
            "entropy_threshold": result.get("entropy_threshold"),
            "entropy": result.get("entropy"),
            "final_answer_source": result.get("final_answer_source"),
            "workflow_branch": result.get("workflow_branch"),
            "router_answer": result.get("router_answer"),
            "router_proposal_answer": result.get("router_proposal_answer"),
            "candidate_answers": result.get("candidate_answers"),
            "vote_groups": result.get("vote_groups"),
            "majority_answer": result.get("majority_answer"),
            "logic_reviewer_answer": result.get("logic_reviewer_answer"),
            "logic_reviewer_verdict": result.get("logic_reviewer_verdict"),
            "logic_reviewer_response": result.get("logic_reviewer_response"),
            "calculation_reviewer_answer": result.get(
                "calculation_reviewer_answer"
            ),
            "calculation_reviewer_verdict": result.get(
                "calculation_reviewer_verdict"
            ),
            "calculation_reviewer_response": result.get(
                "calculation_reviewer_response"
            ),
            "reviewer_consensus": result.get("reviewer_consensus"),
            "arbiter_answer": result.get("arbiter_answer"),
            "arbiter_response": result.get("arbiter_response"),
            "decision_finalizer_answer": result.get(
                "decision_finalizer_answer"
            ),
            "decision_finalizer_response": result.get(
                "decision_finalizer_response"
            ),
            "model_calls": result["model_calls"],
            "agent_calls": result["agent_calls"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],
            "elapsed_seconds": result["elapsed_seconds"],
            "hit_token_limit_count": result["hit_token_limit_count"],
            "peak_cuda_memory_mb": result.get("peak_cuda_memory_mb"),
            "model_metadata": model_metadata,
        }
        append_jsonl(output_file, record)

        print(f"路由：{route} ({record['route_reason']})")
        print(f"协作分支：{record['workflow_branch']}")
        print(f"第一轮候选：{record['candidate_answers']}")
        print(f"答案来源：{record['final_answer_source']}")
        print(f"预测 / 标准：{predicted_answer} / {reference_answer}")
        print(f"正确：{record['correct']}")
        print(
            f"调用 / Token：{record['model_calls']} / "
            f"{record['input_tokens']}+{record['output_tokens']}"
        )


if __name__ == "__main__":
    main()
