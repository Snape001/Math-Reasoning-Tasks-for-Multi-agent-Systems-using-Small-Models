from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from baselines.cot import run_cot
from baselines.direct import run_direct
from core.config import (
    ENABLE_THINKING,
    EXPERIMENT_VERSION,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
)
from core.model_spec import validate_model_requirement
from utils.answer import exact_match, extract_boxed_answer, extract_declared_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一运行 Direct 或 CoT 基线。")
    parser.add_argument("--method", choices=["direct", "cot"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--strict-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = resolve(args.input)
    output_file = resolve(args.output)
    model_metadata = validate_model_requirement(strict=args.strict_model)
    runner: Callable[..., dict[str, Any]] = run_direct if args.method == "direct" else run_cot

    problems = read_jsonl(input_file)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0。")
        problems = problems[: args.limit]

    completed = {
        (
            str(row["problem_id"]),
            int(row.get("run_id", 0)),
            str(row.get("method")),
            str(row.get("experiment_version")),
        )
        for row in read_jsonl(output_file)
    } if output_file.exists() else set()

    for index, item in enumerate(problems, start=1):
        problem_id = str(item.get("id") or item["unique_id"])
        key = (
            problem_id,
            args.run_id,
            args.method,
            EXPERIMENT_VERSION,
        )
        if key in completed:
            print(f"[{index}/{len(problems)}] 跳过：{problem_id}")
            continue

        result = runner(problem=item["problem"], seed=args.seed)
        predicted = extract_declared_answer(result["text"])
        reference = extract_boxed_answer(item["answer"]) or item["answer"]
        record = {
            "problem_id": problem_id,
            "level": int(item["level"]),
            "subject": item["subject"],
            "method": args.method,
            "experiment_version": EXPERIMENT_VERSION,
            "run_id": args.run_id,
            "seed": args.seed,
            "raw_response": result["text"],
            "predicted_answer": predicted,
            "reference_answer": reference,
            "correct": exact_match(predicted, reference),
            "model_calls": 1,
            "generation_config": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_input_tokens": MAX_INPUT_TOKENS,
                "do_sample": False,
                "enable_thinking": ENABLE_THINKING,
                "repetition_penalty": REPETITION_PENALTY,
            },
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],
            "elapsed_seconds": result["elapsed_seconds"],
            "hit_token_limit": result.get("hit_token_limit", False),
            "peak_cuda_memory_mb": result.get("peak_cuda_memory_mb"),
            "model_metadata": model_metadata,
        }
        append_jsonl(output_file, record)
        print(
            f"[{index}/{len(problems)}] {problem_id} | "
            f"pred={predicted} ref={reference} correct={record['correct']}"
        )


if __name__ == "__main__":
    main()
