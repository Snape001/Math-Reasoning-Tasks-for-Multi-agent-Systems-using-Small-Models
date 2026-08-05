from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import transformers

from core.config import DATA_DIR, MODEL_PATH
from core.model_spec import validate_model_requirement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查正式评测前的环境、模型和数据。")
    parser.add_argument("--strict-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")

    metadata = validate_model_requirement(strict=args.strict_model)
    datasets = {}
    for name in ("dev_20.jsonl", "eval_200.jsonl"):
        path = DATA_DIR / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        datasets[name] = {
            "count": len(rows),
            "levels": dict(sorted(Counter(int(row["level"]) for row in rows).items())),
        }

    report = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "model_path": str(MODEL_PATH),
        "model": metadata,
        "datasets": datasets,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
