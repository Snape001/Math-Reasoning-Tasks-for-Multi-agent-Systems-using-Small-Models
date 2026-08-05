from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.multi_agent_v1 import (
    run_multi_agent_v1,
)
from utils.answer import (
    exact_match,
    extract_boxed_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "dev_20.jsonl"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "v1_dev_20.jsonl"
)


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}第{line_number}行JSON格式错误："
                    f"{error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}第{line_number}行"
                    "不是JSON对象。"
                )

            records.append(record)

    return records


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def resolve_path(
    path: Path,
) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_completed_ids(
    result_file: Path,
) -> set[str]:
    """
    以完整题目为单位断点续跑。

    四个Agent全部执行完成后才写入该题结果。
    """

    if not result_file.exists():
        return set()

    completed_ids: set[str] = set()

    for record in read_jsonl(result_file):
        problem_id = (
            record.get("problem_id")
            or record.get("id")
        )

        if problem_id is not None:
            completed_ids.add(str(problem_id))

    return completed_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "运行Solver-Critic-Verifier-Judge"
            "多智能体数学推理系统。"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="输入JSONL数据集。",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="输出JSONL结果。",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅运行前N题，用于调试。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_file = resolve_path(args.input)
    output_file = resolve_path(args.output)

    if not input_file.is_file():
        raise FileNotFoundError(
            f"输入文件不存在：{input_file}"
        )

    problems = read_jsonl(input_file)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit必须大于0。"
            )

        problems = problems[:args.limit]

    completed_ids = load_completed_ids(
        output_file
    )

    print(f"题目数量：{len(problems)}")
    print(
        "架构：Solver -> Critic "
        "-> Verifier -> Judge"
    )
    print("每题模型调用次数：4")
    print(
        f"已经完成：{len(completed_ids)}"
    )
    print(
        f"结果保存位置：{output_file}"
    )

    processed_count = 0
    correct_count = 0
    total_elapsed_seconds = 0.0
    total_tokens = 0

    for index, item in enumerate(
        problems,
        start=1,
    ):
        problem_id = str(
            item.get("id")
            or item["unique_id"]
        )

        if problem_id in completed_ids:
            print(
                f"[{index}/{len(problems)}] "
                f"跳过已完成：{problem_id}"
            )
            continue

        print(
            f"\n[{index}/{len(problems)}] "
            f"正在处理：{problem_id}"
        )

        print(
            f"Level {item['level']} | "
            f"{item['subject']}"
        )

        result = run_multi_agent_v1(
            problem=item["problem"],
        )

        reference_answer = (
            extract_boxed_answer(
                item["answer"]
            )
            or item["answer"]
        )

        predicted_answer = result[
            "predicted_answer"
        ]

        is_correct = exact_match(
            predicted_answer,
            reference_answer,
        )

        record = {
            "problem_id": problem_id,
            "level": int(item["level"]),
            "subject": item["subject"],

            "method": (
                "multi_agent_v1"
            ),
            "architecture": result[
                "architecture"
            ],
            "architecture_version": "v1",
            "run_id": 0,

            "predicted_answer": (
                predicted_answer
            ),
            "reference_answer": (
                reference_answer
            ),
            "correct": is_correct,

            "solver_answer": result[
                "solver_answer"
            ],
            "critic_verdict": result[
                "critic_verdict"
            ],
            "verifier_verdict": result[
                "verifier_verdict"
            ],

            "model_calls": result[
                "model_calls"
            ],
            "agent_calls": result[
                "agent_calls"
            ],

            "input_tokens": result[
                "input_tokens"
            ],
            "output_tokens": result[
                "output_tokens"
            ],
            "total_tokens": result[
                "total_tokens"
            ],
            "elapsed_seconds": result[
                "elapsed_seconds"
            ],

            "average_output_tokens_per_call": (
                result[
                    "average_output_tokens_per_call"
                ]
            ),
            "average_elapsed_seconds_per_call": (
                result[
                    "average_elapsed_seconds_per_call"
                ]
            ),

            "hit_token_limit_count": result[
                "hit_token_limit_count"
            ],
        }

        append_jsonl(
            output_file,
            record,
        )

        processed_count += 1
        correct_count += int(is_correct)
        total_elapsed_seconds += result[
            "elapsed_seconds"
        ]
        total_tokens += result[
            "total_tokens"
        ]

        print(
            "Solver初始答案：",
            result["solver_answer"],
        )
        print(
            "Critic结论：",
            result["critic_verdict"],
        )
        print(
            "Verifier结论：",
            result["verifier_verdict"],
        )
        print(
            "Judge最终答案：",
            predicted_answer,
        )
        print(
            "标准答案：",
            reference_answer,
        )
        print(
            "当前判定：",
            is_correct,
        )
        print(
            "模型调用次数：",
            result["model_calls"],
        )
        print(
            "截断调用次数：",
            result[
                "hit_token_limit_count"
            ],
        )
        print(
            "本题总耗时：",
            result["elapsed_seconds"],
            "秒",
        )

    print("\nMulti-Agent运行完成。")

    if processed_count > 0:
        print(
            f"本次完成题数：{processed_count}"
        )
        print(
            f"本次正确题数：{correct_count}"
        )
        print(
            "本次准确率："
            f"{correct_count / processed_count:.2%}"
        )
        print(
            "平均单题总耗时："
            f"{total_elapsed_seconds / processed_count:.2f}秒"
        )
        print(
            "平均单题总Token："
            f"{total_tokens / processed_count:.2f}"
        )


if __name__ == "__main__":
    main()
