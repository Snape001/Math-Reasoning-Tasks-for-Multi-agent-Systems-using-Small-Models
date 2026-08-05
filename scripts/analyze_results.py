from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
DEFAULT_MAX_NEW_TOKENS = 4096


@dataclass
class ProblemMetrics:
    method: str
    problem_id: str
    level: int
    subject: str
    correct: bool
    has_answer: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    elapsed_seconds: float
    model_calls: int
    truncated_calls: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象。")
            rows.append(row)
    return rows


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_truncated_calls(row: dict[str, Any], max_new_tokens: int) -> int:
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        count = 0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if "hit_token_limit" in candidate:
                count += int(bool(candidate["hit_token_limit"]))
            else:
                count += int(
                    as_int(candidate.get("output_tokens")) >= max_new_tokens
                )
        return count

    if "hit_token_limit_count" in row:
        return as_int(row.get("hit_token_limit_count"))
    if "hit_token_limit" in row:
        return int(bool(row.get("hit_token_limit")))
    return int(as_int(row.get("output_tokens")) >= max_new_tokens)


def infer_model_calls(row: dict[str, Any]) -> int:
    if as_int(row.get("model_calls")) > 0:
        return as_int(row.get("model_calls"))
    candidates = row.get("candidates")
    if isinstance(candidates, list) and candidates:
        return len(candidates)
    if as_int(row.get("num_samples")) > 0:
        return as_int(row.get("num_samples"))
    return 1


def to_problem_metrics(
    row: dict[str, Any],
    method_label: str,
    max_new_tokens: int,
) -> ProblemMetrics:
    problem_id = str(row.get("problem_id") or row.get("id") or "")
    if not problem_id:
        raise ValueError("结果记录缺少 problem_id/id。")

    return ProblemMetrics(
        method=method_label,
        problem_id=problem_id,
        level=as_int(row.get("level")),
        subject=str(row.get("subject") or "Unknown"),
        correct=bool(row.get("correct")),
        has_answer=row.get("predicted_answer") is not None,
        input_tokens=as_int(row.get("input_tokens")),
        output_tokens=as_int(row.get("output_tokens")),
        total_tokens=as_int(row.get("total_tokens")),
        elapsed_seconds=as_float(row.get("elapsed_seconds")),
        model_calls=infer_model_calls(row),
        truncated_calls=infer_truncated_calls(row, max_new_tokens),
    )


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(group: Iterable[ProblemMetrics]) -> dict[str, Any]:
    rows = list(group)
    problems = len(rows)
    correct = sum(int(row.correct) for row in rows)
    answered = sum(int(row.has_answer) for row in rows)
    calls = sum(row.model_calls for row in rows)
    truncated_calls = sum(row.truncated_calls for row in rows)
    truncated_problems = sum(int(row.truncated_calls > 0) for row in rows)

    input_tokens = sum(row.input_tokens for row in rows)
    output_tokens = sum(row.output_tokens for row in rows)
    total_tokens = sum(row.total_tokens for row in rows)
    elapsed_seconds = sum(row.elapsed_seconds for row in rows)

    return {
        "problems": problems,
        "correct": correct,
        "accuracy": safe_div(correct, problems),
        "answered": answered,
        "answer_rate": safe_div(answered, problems),
        "model_calls": calls,
        "avg_calls_per_problem": safe_div(calls, problems),
        "truncated_problems": truncated_problems,
        "problem_truncation_rate": safe_div(truncated_problems, problems),
        "truncated_calls": truncated_calls,
        "call_truncation_rate": safe_div(truncated_calls, calls),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens_per_problem": safe_div(input_tokens, problems),
        "avg_output_tokens_per_problem": safe_div(output_tokens, problems),
        "avg_total_tokens_per_problem": safe_div(total_tokens, problems),
        "avg_output_tokens_per_call": safe_div(output_tokens, calls),
        "total_elapsed_seconds": elapsed_seconds,
        "avg_elapsed_seconds_per_problem": safe_div(elapsed_seconds, problems),
        "avg_elapsed_seconds_per_call": safe_div(elapsed_seconds, calls),
    }


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "--input 格式必须是 METHOD=PATH，例如 Direct=results/direct_dev_v2.jsonl"
        )
    method, raw_path = spec.split("=", 1)
    method = method.strip()
    raw_path = raw_path.strip()
    if not method or not raw_path:
        raise argparse.ArgumentTypeError("--input 的方法名和路径都不能为空。")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return method, path


def ensure_same_problem_set(method_rows: dict[str, list[ProblemMetrics]]) -> None:
    sets = {method: {row.problem_id for row in rows} for method, rows in method_rows.items()}
    if len(sets) <= 1:
        return
    methods = list(sets)
    reference_method = methods[0]
    reference_set = sets[reference_method]
    for method in methods[1:]:
        missing = sorted(reference_set - sets[method])
        extra = sorted(sets[method] - reference_set)
        if missing or extra:
            raise ValueError(
                f"题目集合不一致：{method} 相对 {reference_method} "
                f"缺少 {len(missing)} 题，多出 {len(extra)} 题。"
            )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def generate_markdown(
    summary_rows: list[dict[str, Any]],
    level_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# MATH result comparison",
        "",
        "## Overall metrics",
        "",
        "| Method | Accuracy | Answer rate | Avg input tokens/problem | Avg output tokens/problem | Avg total tokens/problem | Avg time/problem (s) | Model calls | Call truncation rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {pct(row['accuracy'])} "
            f"({row['correct']}/{row['problems']}) | {pct(row['answer_rate'])} | "
            f"{row['avg_input_tokens_per_problem']:.2f} | "
            f"{row['avg_output_tokens_per_problem']:.2f} | "
            f"{row['avg_total_tokens_per_problem']:.2f} | "
            f"{row['avg_elapsed_seconds_per_problem']:.2f} | "
            f"{row['model_calls']} | {pct(row['call_truncation_rate'])} |"
        )

    lines.extend([
        "",
        "## Accuracy by level",
        "",
        "| Method | Level | Correct | Problems | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in level_rows:
        lines.append(
            f"| {row['method']} | {row['level']} | {row['correct']} | "
            f"{row['problems']} | {pct(row['accuracy'])} |"
        )

    lines.extend([
        "",
        "## Failures",
        "",
        "| Method | Problem | Level | Subject | Has answer | Truncated calls |",
        "|---|---|---:|---|---:|---:|",
    ])
    for row in failure_rows:
        lines.append(
            f"| {row['method']} | `{row['problem_id']}` | {row['level']} | "
            f"{row['subject']} | {row['has_answer']} | {row['truncated_calls']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一分析 Direct、CoT、Self-Consistency JSONL 结果。"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "输入方法与结果路径，可重复使用。"
            "例如 --input Direct=results/direct_dev_v2.jsonl"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="用于兼容旧日志并推断是否截断，默认4096。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="分析结果输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    method_rows: dict[str, list[ProblemMetrics]] = {}
    for spec in args.input:
        method, path = parse_input_spec(spec)
        if method in method_rows:
            raise ValueError(f"方法名重复：{method}")
        if not path.is_file():
            raise FileNotFoundError(f"结果文件不存在：{path}")
        raw_rows = read_jsonl(path)
        method_rows[method] = [
            to_problem_metrics(row, method, args.max_new_tokens)
            for row in raw_rows
        ]

    ensure_same_problem_set(method_rows)

    summary_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for method, rows in method_rows.items():
        overall = summarize(rows)
        summary_rows.append({"method": method, **overall})

        by_level: dict[int, list[ProblemMetrics]] = defaultdict(list)
        by_subject: dict[str, list[ProblemMetrics]] = defaultdict(list)
        for row in rows:
            by_level[row.level].append(row)
            by_subject[row.subject].append(row)
            if not row.correct:
                failure_rows.append({
                    "method": method,
                    "problem_id": row.problem_id,
                    "level": row.level,
                    "subject": row.subject,
                    "has_answer": row.has_answer,
                    "truncated_calls": row.truncated_calls,
                    "output_tokens": row.output_tokens,
                    "elapsed_seconds": row.elapsed_seconds,
                })

        for level in sorted(by_level):
            level_rows.append({
                "method": method,
                "level": level,
                **summarize(by_level[level]),
            })
        for subject in sorted(by_subject):
            subject_rows.append({
                "method": method,
                "subject": subject,
                **summarize(by_subject[subject]),
            })

    summary_fields = [
        "method", "problems", "correct", "accuracy", "answered", "answer_rate",
        "model_calls", "avg_calls_per_problem", "truncated_problems",
        "problem_truncation_rate", "truncated_calls", "call_truncation_rate",
        "total_input_tokens", "total_output_tokens", "total_tokens",
        "avg_input_tokens_per_problem", "avg_output_tokens_per_problem",
        "avg_total_tokens_per_problem", "avg_output_tokens_per_call",
        "total_elapsed_seconds", "avg_elapsed_seconds_per_problem",
        "avg_elapsed_seconds_per_call",
    ]
    grouped_fields = ["method", "level"] + [x for x in summary_fields if x != "method"]
    subject_fields = ["method", "subject"] + [x for x in summary_fields if x != "method"]
    failure_fields = [
        "method", "problem_id", "level", "subject", "has_answer",
        "truncated_calls", "output_tokens", "elapsed_seconds",
    ]

    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "by_level.csv", level_rows, grouped_fields)
    write_csv(output_dir / "by_subject.csv", subject_rows, subject_fields)
    write_csv(output_dir / "failures.csv", failure_rows, failure_fields)

    markdown = generate_markdown(summary_rows, level_rows, failure_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"分析文件已保存到：{output_dir}")


if __name__ == "__main__":
    main()
