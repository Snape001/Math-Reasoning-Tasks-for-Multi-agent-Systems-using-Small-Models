from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row.get("problem_id") or row.get("id") or "")
            if not problem_id:
                raise ValueError(f"{path} 第 {line_number} 行缺少 problem_id。")
            if problem_id in rows:
                raise ValueError(f"{path} 中 problem_id 重复：{problem_id}")
            rows[problem_id] = row
    return rows


def exact_mcnemar_p(rescued: int, regressed: int) -> float:
    discordant = rescued + regressed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) * (0.5**discordant)
        for value in range(0, min(rescued, regressed) + 1)
    )
    return min(1.0, 2.0 * tail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="配对比较单智能体与多智能体，定位纠错和退化样本。"
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--collaboration", type=Path, required=True)
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Optionally restrict both inputs to problem IDs listed one per line.",
    )
    parser.add_argument(
        "--embedded-author",
        action="store_true",
        help="Use author_answer/author_correct in the collaboration file as baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/collaboration_analysis"),
    )
    return parser.parse_args()


def author_tokens(row: dict[str, Any]) -> int:
    """Return the generation cost of the answer-producing Author call."""

    source = str(row.get("author_source") or "")
    preferred_role = {
        "deep_author": "review_author",
        "plan_reasoner": "plan_reasoner",
    }.get(source, "router_solver_a")
    calls = list(row.get("agent_calls") or [])
    for call in calls:
        if call.get("role") == preferred_role:
            return int(call.get("total_tokens", 0))
    return 0


def embedded_author_baseline(
    collaboration: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline: dict[str, dict[str, Any]] = {}
    for problem_id, row in collaboration.items():
        if "author_correct" not in row:
            raise ValueError(
                f"{problem_id} lacks author_correct; it is not an ablation record."
            )
        baseline[problem_id] = {
            "problem_id": problem_id,
            "predicted_answer": row.get("author_answer"),
            "correct": bool(row.get("author_correct")),
            "total_tokens": author_tokens(row),
        }
    return baseline


def main() -> None:
    args = parse_args()
    collaboration = read_jsonl(resolve(args.collaboration))
    if args.embedded_author:
        if args.baseline is not None:
            raise ValueError("--embedded-author and --baseline are mutually exclusive.")
        baseline = embedded_author_baseline(collaboration)
    else:
        if args.baseline is None:
            raise ValueError("Provide --baseline or use --embedded-author.")
        baseline = read_jsonl(resolve(args.baseline))
    if args.ids_file is not None:
        ids = {
            line.strip()
            for line in resolve(args.ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing_baseline = ids - set(baseline)
        missing_collaboration = ids - set(collaboration)
        if missing_baseline or missing_collaboration:
            raise ValueError(
                "Requested IDs are missing: "
                f"baseline={sorted(missing_baseline)}, "
                f"collaboration={sorted(missing_collaboration)}"
            )
        baseline = {problem_id: baseline[problem_id] for problem_id in ids}
        collaboration = {
            problem_id: collaboration[problem_id] for problem_id in ids
        }
    if set(baseline) != set(collaboration):
        raise ValueError(
            "两种方法的题目集合不一致："
            f"baseline={len(baseline)}，collaboration={len(collaboration)}。"
        )

    transitions = Counter()
    route_stats: dict[str, Counter] = defaultdict(Counter)
    level_stats: dict[int, Counter] = defaultdict(Counter)
    details: list[dict[str, Any]] = []

    for problem_id in sorted(baseline):
        base = baseline[problem_id]
        collab = collaboration[problem_id]
        base_ok = bool(base.get("correct"))
        collab_ok = bool(collab.get("correct"))
        if not base_ok and collab_ok:
            transition = "rescued"
        elif base_ok and not collab_ok:
            transition = "regressed"
        elif base_ok:
            transition = "both_correct"
        else:
            transition = "both_wrong"
        transitions[transition] += 1

        route = str(collab.get("route") or "unspecified")
        level = int(collab.get("level", 0))
        route_stats[route][transition] += 1
        level_stats[level][transition] += 1
        details.append(
            {
                "problem_id": problem_id,
                "level": level,
                "subject": collab.get("subject"),
                "route": route,
                "route_reason": collab.get("route_reason"),
                "transition": transition,
                "baseline_answer": base.get("predicted_answer"),
                "collaboration_answer": collab.get("predicted_answer"),
                "reference_answer": collab.get("reference_answer"),
                "final_answer_source": collab.get("final_answer_source"),
                "baseline_tokens": int(base.get("total_tokens", 0)),
                "collaboration_tokens": int(collab.get("total_tokens", 0)),
            }
        )

    count = len(details)
    baseline_correct = transitions["both_correct"] + transitions["regressed"]
    collaboration_correct = transitions["both_correct"] + transitions["rescued"]
    baseline_tokens = sum(row["baseline_tokens"] for row in details)
    collaboration_tokens = sum(row["collaboration_tokens"] for row in details)
    summary = {
        "problems": count,
        "baseline_accuracy": baseline_correct / count if count else 0.0,
        "collaboration_accuracy": collaboration_correct / count if count else 0.0,
        "accuracy_delta": (
            (collaboration_correct - baseline_correct) / count if count else 0.0
        ),
        "rescued": transitions["rescued"],
        "regressed": transitions["regressed"],
        "both_correct": transitions["both_correct"],
        "both_wrong": transitions["both_wrong"],
        "mcnemar_exact_p": exact_mcnemar_p(
            transitions["rescued"], transitions["regressed"]
        ),
        "avg_baseline_tokens": baseline_tokens / count if count else 0.0,
        "avg_collaboration_tokens": collaboration_tokens / count if count else 0.0,
        "token_multiplier": (
            collaboration_tokens / baseline_tokens if baseline_tokens else None
        ),
    }

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_details.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]) if details else [])
        if details:
            writer.writeheader()
            writer.writerows(details)

    payload = {
        "summary": summary,
        "by_route": {key: dict(value) for key, value in route_stats.items()},
        "by_level": {str(key): dict(value) for key, value in level_stats.items()},
    }
    (output_dir / "paired_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    markdown = f"""# Paired collaboration analysis

| Metric | Value |
|---|---:|
| Problems | {count} |
| Baseline accuracy | {summary['baseline_accuracy']:.2%} |
| Collaboration accuracy | {summary['collaboration_accuracy']:.2%} |
| Accuracy delta | {summary['accuracy_delta']:+.2%} |
| Rescued / regressed | {summary['rescued']} / {summary['regressed']} |
| McNemar exact p | {summary['mcnemar_exact_p']:.6f} |
| Avg baseline tokens | {summary['avg_baseline_tokens']:.2f} |
| Avg collaboration tokens | {summary['avg_collaboration_tokens']:.2f} |
| Token multiplier | {summary['token_multiplier'] if summary['token_multiplier'] is not None else 'N/A'} |

`rescued` 表示单智能体错误而协作正确；`regressed` 表示单智能体正确而协作错误。
只有在配对 200 题上 rescued 稳定多于 regressed，才能声称协作机制带来净收益。
"""
    (output_dir / "paired_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
