from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        "--route-rate",
        type=float,
        default=0.30,
        help="计划升级到多Agent的样本比例。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 < args.route_rate < 1:
        raise ValueError(
            "--route-rate必须位于(0,1)。"
        )

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

    rows = read_jsonl(input_file)

    scores = sorted(
        float(row["route_score"])
        for row in rows
        if row.get("route_score") is not None
    )

    if not scores:
        raise ValueError("没有有效熵分数。")

    # 例如route_rate=0.30，
    # 使用70%分位点作为阈值
    quantile = 1.0 - args.route_rate

    index = math.ceil(
        quantile * len(scores)
    ) - 1

    index = max(
        0,
        min(index, len(scores) - 1),
    )

    threshold = scores[index]

    config = {
        "method": (
            "normalized_generation_entropy"
        ),
        "entropy_threshold": threshold,
        "target_route_rate": args.route_rate,
        "calibration_size": len(scores),
        "entropy_stride": 8,
        "route_rule": (
            "route_score > entropy_threshold"
        ),
        "forced_routes": [
            "missing_boxed_answer",
            "router_token_limit",
            "missing_entropy",
        ],
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    routed_count = sum(
        score > threshold
        for score in scores
    )

    print(f"样本数量：{len(scores)}")
    print(f"熵阈值：{threshold}")
    print(
        f"预计熵触发数量："
        f"{routed_count}/{len(scores)}"
    )
    print(f"配置保存：{output_file}")


if __name__ == "__main__":
    main()