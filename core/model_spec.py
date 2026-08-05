from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from core.config import MODEL_NAME, MODEL_PATH, REQUIRED_MODEL_FAMILY


def _normalized_family(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def read_model_metadata(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"模型目录中缺少 config.json：{config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    architectures = [str(item) for item in config.get("architectures", [])]
    required = _normalized_family(REQUIRED_MODEL_FAMILY)

    # MODEL_NAME 是用户声明，不能单独作为合规证据；真实 config 必须包含
    # Qwen3.5 的族名（兼容 qwen3_5 / Qwen3.5 等写法）。
    config_identity = _normalized_family(
        " ".join([str(config.get("model_type", "")), *architectures])
    )
    compliant = required in config_identity

    return {
        "declared_model_name": MODEL_NAME,
        "model_path": str(model_path.resolve()),
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "torch_dtype": config.get("torch_dtype"),
        "required_model_family": REQUIRED_MODEL_FAMILY,
        "requirement_compliant": compliant,
        "detected_identity": config_identity,
    }


def validate_model_requirement(
    model_path: Path = MODEL_PATH,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    metadata = read_model_metadata(model_path)
    if metadata["requirement_compliant"]:
        return metadata

    message = (
        "当前权重不符合试题的基座模型要求："
        f"检测到 model_type={metadata['model_type']!r}, "
        f"architectures={metadata['architectures']!r}；"
        f"要求 {REQUIRED_MODEL_FAMILY}。请将 MULTIAGENT_MODEL_PATH 指向"
        "Qwen3.5-4B 或 Qwen3.5-9B 的本地权重目录。"
    )
    if strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return metadata
