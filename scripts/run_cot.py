from __future__ import annotations

import json
from pathlib import Path

from baselines.cot import run_cot
from utils.answer import (
    exact_match,
    extract_boxed_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_FILE = PROJECT_ROOT / "data" / "dev_20.jsonl"
RESULT_FILE = (
    PROJECT_ROOT / "results" / "cot_dev_v2.jsonl"
)

SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}第{line_number}行JSON格式错误：{error}"
                ) from error

    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def load_completed_ids() -> set[str]:
    """
    支持断点续跑。
    程序中断后再次运行，不会重复执行已经完成的题目。
    """

    if not RESULT_FILE.exists():
        return set()

    records = read_jsonl(RESULT_FILE)

    return {
        record["problem_id"]
        for record in records
    }


def main() -> None:
    if not DEV_FILE.is_file():
        raise FileNotFoundError(
            f"没有找到开发集：{DEV_FILE}"
        )

    problems = read_jsonl(DEV_FILE)
    completed_ids = load_completed_ids()

    print(f"开发集题目数量：{len(problems)}")
    print(f"已经完成：{len(completed_ids)}")
    print(f"结果保存位置：{RESULT_FILE}")

    for index, item in enumerate(problems, start=1):
        problem_id = item.get("id") or item["unique_id"]

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

        result = run_cot(
            problem=item["problem"],
            seed=SEED,
        )

        raw_response = result["text"]

        predicted_answer = extract_boxed_answer(
            raw_response
        )

        # MATH-500中的answer通常是最终答案，
        # 但仍兼容其中包含\\boxed{}的情况。
        reference_answer = (
            extract_boxed_answer(item["answer"])
            or item["answer"]
        )

        is_correct = exact_match(
            predicted_answer,
            reference_answer,
        )

        record = {
            "problem_id": problem_id,
            "level": int(item["level"]),
            "subject": item["subject"],
            "method": "cot",
            "run_id": 0,
            "seed": SEED,
            "raw_response": raw_response,
            "predicted_answer": predicted_answer,
            "reference_answer": reference_answer,
            "correct": is_correct,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_per_second": result.get(
                "tokens_per_second"
            ),
            "max_new_tokens": result["max_new_tokens"],
            "hit_token_limit": result["hit_token_limit"],
        }

        append_jsonl(
            RESULT_FILE,
            record,
        )

        print("模型答案：", predicted_answer)
        print("标准答案：", reference_answer)
        print("当前判定：", is_correct)
        print(
            "耗时：",
            result["elapsed_seconds"],
            "秒",
        )

    print("\nDirect开发集运行完成。")


if __name__ == "__main__":
    main()