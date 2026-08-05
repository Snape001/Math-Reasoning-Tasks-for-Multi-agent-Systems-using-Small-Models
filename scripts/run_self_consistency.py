from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.self_consistency import (
    run_self_consistency,
)
from core.config import (
    ENABLE_THINKING,
    EXPERIMENT_VERSION,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    SC_BASE_SEED,
    SC_NUM_SAMPLES,
    SC_TEMPERATURE,
    SC_TOP_P,
)
from core.model_spec import validate_model_requirement
from utils.answer import (
    exact_match,
    extract_boxed_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_FILE = PROJECT_ROOT / "data" / "dev_20.jsonl"


def read_jsonl(
    path: Path,
) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}第{line_number}行JSON错误："
                    f"{error}"
                ) from error

    return records


def append_jsonl(
    path: Path,
    record: dict,
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


def load_completed_ids(
    result_file: Path,
    run_id: int,
    num_samples: int,
) -> set[str]:
    """
    支持以题目为单位断点续跑。

    一道题的5次采样全部完成后才会写入结果文件。
    """

    if not result_file.exists():
        return set()

    records = read_jsonl(result_file)

    return {
        record["problem_id"]
        for record in records
        if (
            int(record.get("run_id", 0)) == run_id
            and int(record.get("num_samples", 0)) == num_samples
            and str(record.get("experiment_version")) == EXPERIMENT_VERSION
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在MATH-500开发集上运行"
            "Self-Consistency基线。"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEV_FILE,
        help="输入 JSONL 数据集。",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL；默认仍写入开发集结果文件。",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=SC_NUM_SAMPLES,
        help="每道题采样次数，默认5次。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SC_BASE_SEED,
    )

    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--strict-model",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前N道题，用于调试。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError(
            "--num-samples必须大于0。"
        )

    input_file = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    result_file = (
        args.output
        if args.output is not None
        else Path("results") / f"self_consistency_dev_n{args.num_samples}.jsonl"
    )
    if not result_file.is_absolute():
        result_file = PROJECT_ROOT / result_file

    model_metadata = validate_model_requirement(strict=args.strict_model)

    if not input_file.is_file():
        raise FileNotFoundError(
            f"数据集不存在：{input_file}"
        )

    problems = read_jsonl(input_file)

    if args.limit is not None:
        problems = problems[:args.limit]

    completed_ids = load_completed_ids(
        result_file,
        args.run_id,
        args.num_samples,
    )

    print(
        f"开发集题目数量：{len(problems)}"
    )
    print(
        f"每题采样次数：{args.num_samples}"
    )
    print(
        f"已经完成：{len(completed_ids)}"
    )
    print(
        f"结果保存位置：{result_file}"
    )

    new_correct_count = 0
    new_processed_count = 0
    total_elapsed_seconds = 0.0
    total_output_tokens = 0

    for index, item in enumerate(
        problems,
        start=1,
    ):
        problem_id = (
            item.get("id")
            or item["unique_id"]
        )

        if problem_id in completed_ids:
            print(
                f"[{index}/{len(problems)}] "
                f"跳过已完成题目：{problem_id}"
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

        result = run_self_consistency(
            problem=item["problem"],
            num_samples=args.num_samples,
            base_seed=args.seed,
            temperature=SC_TEMPERATURE,
            top_p=SC_TOP_P,
        )

        reference_answer = (
            extract_boxed_answer(item["answer"])
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
            "method": "self_consistency",
            "experiment_version": EXPERIMENT_VERSION,
            "run_id": args.run_id,
            "num_samples": result[
                "num_samples"
            ],
            "valid_samples": result[
                "valid_samples"
            ],
            "base_seed": result[
                "base_seed"
            ],
            "temperature": result[
                "temperature"
            ],
            "top_p": result["top_p"],
            "predicted_answer": (
                predicted_answer
            ),
            "reference_answer": (
                reference_answer
            ),
            "correct": is_correct,
            "vote_counts": result[
                "vote_counts"
            ],
            "candidates": result[
                "candidates"
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
            "average_output_tokens": result[
                "average_output_tokens"
            ],
            "average_elapsed_seconds": result[
                "average_elapsed_seconds"
            ],
            "hit_token_limit_count": result[
                "hit_token_limit_count"
            ],
            "model_calls": result["num_samples"],
            "generation_config": {
                "max_new_tokens_per_call": MAX_NEW_TOKENS,
                "max_input_tokens_per_call": MAX_INPUT_TOKENS,
                "do_sample": True,
                "temperature": SC_TEMPERATURE,
                "top_p": SC_TOP_P,
                "enable_thinking": ENABLE_THINKING,
                "repetition_penalty": REPETITION_PENALTY,
            },
            "peak_cuda_memory_mb": result.get("peak_cuda_memory_mb"),
            "model_metadata": model_metadata,
        }

        append_jsonl(
            result_file,
            record,
        )

        new_processed_count += 1
        new_correct_count += int(is_correct)
        total_elapsed_seconds += result[
            "elapsed_seconds"
        ]
        total_output_tokens += result[
            "output_tokens"
        ]

        print("\n候选答案：")

        for candidate in result["candidates"]:
            print(
                f"  sample={candidate['sample_id']} "
                f"seed={candidate['seed']} "
                f"answer={candidate['predicted_answer']} "
                f"tokens={candidate['output_tokens']} "
                f"truncated={candidate['hit_token_limit']}"
            )

        print(
            "票数：",
            result["vote_counts"],
        )
        print(
            "投票结果：",
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
            "本题总耗时：",
            result["elapsed_seconds"],
            "秒",
        )

    print(
        "\nSelf-Consistency开发集运行结束。"
    )

    if new_processed_count > 0:
        print(
            f"本次处理题数："
            f"{new_processed_count}"
        )
        print(
            f"本次正确题数："
            f"{new_correct_count}"
        )
        print(
            "本次准确率："
            f"{new_correct_count / new_processed_count:.2%}"
        )
        print(
            "单题平均总耗时："
            f"{total_elapsed_seconds / new_processed_count:.2f}秒"
        )
        print(
            "单题平均总输出Token："
            f"{total_output_tokens / new_processed_count:.2f}"
        )


if __name__ == "__main__":
    main()
