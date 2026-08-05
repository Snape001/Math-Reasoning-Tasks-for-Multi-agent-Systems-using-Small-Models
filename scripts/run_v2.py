from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baselines.adaptive_multi_agent_v2 import (
    run_adaptive_multi_agent_v2,
    run_router_v2,
)
from utils.answer import (
    exact_match,
    extract_boxed_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows = []

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
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}第{line_number}行错误："
                    f"{error}"
                ) from error

    return rows


def append_jsonl(
    path: Path,
    row: dict[str, Any],
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
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


def completed_ids(
    path: Path,
) -> set[str]:
    if not path.exists():
        return set()

    return {
        str(row["problem_id"])
        for row in read_jsonl(path)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["probe", "adaptive"],
        required=True,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--router-config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_file = (
        args.input
        if args.input.is_absolute()
        else PROJECT_ROOT / args.input
    )
    output_file = (
        args.output
        if args.output.is_absolute()
        else PROJECT_ROOT / args.output
    )

    threshold = None

    if args.mode == "adaptive":
        if args.router_config is None:
            raise ValueError(
                "adaptive模式必须提供"
                "--router-config。"
            )

        config_file = (
            args.router_config
            if args.router_config.is_absolute()
            else PROJECT_ROOT
            / args.router_config
        )

        with config_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            router_config = json.load(file)

        threshold = float(
            router_config["entropy_threshold"]
        )

    problems = read_jsonl(input_file)

    if args.limit is not None:
        problems = problems[:args.limit]

    done = completed_ids(output_file)

    print(f"模式：{args.mode}")
    print(f"题目数量：{len(problems)}")

    if threshold is not None:
        print(f"熵阈值：{threshold}")

    for index, item in enumerate(
        problems,
        start=1,
    ):
        problem_id = str(
            item.get("id")
            or item["unique_id"]
        )

        if problem_id in done:
            print(
                f"[{index}/{len(problems)}] "
                f"跳过：{problem_id}"
            )
            continue

        print(
            f"\n[{index}/{len(problems)}] "
            f"{problem_id}"
        )

        if args.mode == "probe":
            result = run_router_v2(
                item["problem"]
            )
            method = "entropy_router_probe"
            route = "probe"

        else:
            result = run_adaptive_multi_agent_v2(
                problem=item["problem"],
                entropy_threshold=threshold,
            )
            method = "multi_agent_v2"
            route = result["route"]

        reference_answer = (
            extract_boxed_answer(item["answer"])
            or item["answer"]
        )

        predicted_answer = result[
            "predicted_answer"
        ]

        correct = exact_match(
            predicted_answer,
            reference_answer,
        )

        record = {
            "problem_id": problem_id,
            "level": int(item["level"]),
            "subject": item["subject"],
            "method": method,
            "architecture_version": (
                "v2" if args.mode == "adaptive" else None
            ),
            "predicted_answer": predicted_answer,
            "reference_answer": reference_answer,
            "correct": correct,

            "route": route,
            "route_score": result.get(
                "route_score"
            ),
            "entropy_threshold": result.get(
                "entropy_threshold"
            ),
            "entropy": result.get("entropy"),

            "model_calls": result.get(
                "model_calls",
                1,
            ),
            "agent_calls": result.get(
                "agent_calls",
                [result.get("router_call")],
            ),

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
            "hit_token_limit_count": (
                result.get(
                    "hit_token_limit_count",
                    int(
                        result.get(
                            "hit_token_limit",
                            False,
                        )
                    ),
                )
            ),
        }

        append_jsonl(
            output_file,
            record,
        )

        print("Route score：", record["route_score"])
        print("路由：", route)
        print("模型调用：", record["model_calls"])
        print("预测：", predicted_answer)
        print("标准：", reference_answer)
        print("判定：", correct)


if __name__ == "__main__":
    main()
