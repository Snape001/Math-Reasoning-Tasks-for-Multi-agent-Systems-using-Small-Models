from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_FILE = DATA_DIR / "test.jsonl"
IDS_FILE = DATA_DIR / "appendix_200_ids.txt"

EVAL_FILE = DATA_DIR / "eval_200.jsonl"
DEV_FILE = DATA_DIR / "dev_20.jsonl"
TEST_FILE = DATA_DIR / "test_180.jsonl"

SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]


def main() -> None:
    all_records = read_jsonl(SOURCE_FILE)
    appendix_ids = read_ids(IDS_FILE)

    if len(all_records) != 500:
        raise ValueError(f"MATH-500 应有 500 题，实际为 {len(all_records)}")

    if len(appendix_ids) != 200:
        raise ValueError(f"附录应有 200 个 ID，实际为 {len(appendix_ids)}")

    if len(set(appendix_ids)) != 200:
        raise ValueError("附录 ID 中存在重复项")

    record_by_id = {
        record["unique_id"]: record
        for record in all_records
    }

    missing_ids = [
        item_id
        for item_id in appendix_ids
        if item_id not in record_by_id
    ]
    if missing_ids:
        raise ValueError(
            "以下 ID 未在 test.jsonl 中找到：\n"
            + "\n".join(missing_ids)
        )

    eval_records = [
        {
            "id": record_by_id[item_id]["unique_id"],
            **record_by_id[item_id],
        }
        for item_id in appendix_ids
    ]

    level_counts = Counter(
        int(record["level"])
        for record in eval_records
    )
    expected_counts = {
        1: 40,
        2: 40,
        3: 40,
        4: 40,
        5: 40,
    }

    if dict(sorted(level_counts.items())) != expected_counts:
        raise ValueError(
            f"200 题难度分布错误：{dict(level_counts)}"
        )

    write_jsonl(EVAL_FILE, eval_records)

    # 仅用于开发调试：从指定 200 题中每个 level 抽 4 题
    rng = random.Random(SEED)
    dev_records = []
    dev_ids = set()

    for level in range(1, 6):
        candidates = [
            record
            for record in eval_records
            if int(record["level"]) == level
        ]
        selected = rng.sample(candidates, 4)
        dev_records.extend(selected)
        dev_ids.update(
            record["unique_id"]
            for record in selected
        )

    dev_records.sort(
        key=lambda record: (
            int(record["level"]),
            record["unique_id"],
        )
    )
    test_records = [
        record
        for record in eval_records
        if record["unique_id"] not in dev_ids
    ]

    write_jsonl(DEV_FILE, dev_records)
    write_jsonl(TEST_FILE, test_records)

    print("数据准备完成")
    print(f"完整源数据：{len(all_records)}")
    print(f"正式评测集：{len(eval_records)}")
    print(f"开发集：{len(dev_records)}")
    print(f"内部测试集：{len(test_records)}")
    print(f"正式评测难度分布：{dict(sorted(level_counts.items()))}")


if __name__ == "__main__":
    main()