from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from utils.answer import equivalence_reason


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORING_VERSION = "math500_equivalence_v2"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON：{error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象。")
            rows.append(row)
    return rows


def parse_input(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--input 必须为 METHOD=PATH。")
    method, path = spec.split("=", 1)
    if not method.strip() or not path.strip():
        raise argparse.ArgumentTypeError("方法名和路径不能为空。")
    return method.strip(), resolve(Path(path.strip()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truncated_calls(row: dict[str, Any]) -> int:
    if "hit_token_limit_count" in row:
        return int(row.get("hit_token_limit_count") or 0)
    if "hit_token_limit" in row:
        return int(bool(row.get("hit_token_limit")))
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        return sum(int(bool(item.get("hit_token_limit"))) for item in candidates)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复核 MATH 结果中的格式及简单数学等价，不覆盖原始 JSONL。"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="METHOD=PATH，可重复传入。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/audited_fair_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    inputs = [parse_input(spec) for spec in args.input]

    audited_by_method: dict[str, list[dict[str, Any]]] = {}
    changes: list[dict[str, Any]] = []
    expected_ids: set[str] | None = None

    for method, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        raw_rows = read_jsonl(path)
        ids = {str(row.get("problem_id")) for row in raw_rows}
        if len(ids) != len(raw_rows):
            raise ValueError(f"{path} 存在重复 problem_id。")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"{path} 与其他方法的题目集合不一致。")

        audited_rows: list[dict[str, Any]] = []
        for row in raw_rows:
            original_correct = bool(row.get("correct"))
            reason = equivalence_reason(
                row.get("predicted_answer"),
                row.get("reference_answer"),
            )
            audited_correct = reason is not None
            audited = {
                **row,
                "original_correct": original_correct,
                "correct": audited_correct,
                "audit_status": (
                    "corrected_equivalent"
                    if audited_correct and not original_correct
                    else "confirmed_correct"
                    if audited_correct
                    else "confirmed_incorrect"
                ),
                "audit_reason": reason or "not_equivalent",
                "scoring_version": SCORING_VERSION,
            }
            audited_rows.append(audited)
            if original_correct != audited_correct:
                changes.append(
                    {
                        "method": method,
                        "problem_id": row.get("problem_id"),
                        "level": row.get("level"),
                        "subject": row.get("subject"),
                        "predicted_answer": row.get("predicted_answer"),
                        "reference_answer": row.get("reference_answer"),
                        "original_correct": original_correct,
                        "audited_correct": audited_correct,
                        "audit_reason": reason,
                    }
                )

        audited_by_method[method] = audited_rows
        output_path = output_dir / f"{path.stem}_audited.jsonl"
        write_jsonl(output_path, audited_rows)

    review_rows: list[dict[str, Any]] = []
    assert expected_ids is not None
    method_maps = {
        method: {str(row["problem_id"]): row for row in rows}
        for method, rows in audited_by_method.items()
    }
    for problem_id in sorted(expected_ids):
        if all(method_maps[method][problem_id]["correct"] for method in method_maps):
            continue
        first = next(iter(method_maps.values()))[problem_id]
        review: dict[str, Any] = {
            "problem_id": problem_id,
            "level": first.get("level"),
            "subject": first.get("subject"),
            "reference_answer": first.get("reference_answer"),
        }
        for method in method_maps:
            row = method_maps[method][problem_id]
            review[f"{method}_answer"] = row.get("predicted_answer")
            review[f"{method}_correct"] = row.get("correct")
            review[f"{method}_truncated_calls"] = truncated_calls(row)
        review_rows.append(review)

    write_csv(output_dir / "audit_changes.csv", changes)
    write_csv(output_dir / "cross_method_review.csv", review_rows)
    print(f"审计结果：{output_dir}")
    print(f"题目数：{len(expected_ids)}")
    print(f"修正的假阴性记录：{len(changes)}")
    for method, rows in audited_by_method.items():
        correct = sum(int(row["correct"]) for row in rows)
        corrected = sum(row["audit_status"] == "corrected_equivalent" for row in rows)
        print(f"{method}: {correct}/{len(rows)}，纠正 {corrected} 条")


if __name__ == "__main__":
    main()
