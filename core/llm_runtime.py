from __future__ import annotations

import gc
import os
import time
from typing import Any

# 减少长上下文反复分配导致的 CUDA 内存碎片。必须在导入 torch 前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from core.config import (
    COMPUTE_DTYPE,
    ENABLE_THINKING,
    LOAD_IN_4BIT,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    QUANTIZATION_TYPE,
    REPETITION_PENALTY,
    USE_DOUBLE_QUANT,
)

from core.confidence import EntropyRecorder
from core.model_spec import validate_model_requirement
from transformers import LogitsProcessorList


# 模块级缓存：确保同一个 Python 进程只加载一次模型
_MODEL: Any | None = None
_TOKENIZER: Any | None = None
_MODEL_REQUIREMENT_CHECKED = False


class ContextBudgetError(RuntimeError):
    """输入在预填充前已超过当前显存预算。"""


def release_cuda_memory() -> None:
    """释放跨 Agent 调用不再使用的临时张量与 CUDA 缓存。"""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model():
    global _MODEL, _TOKENIZER, _MODEL_REQUIREMENT_CHECKED

    # 已加载过则直接复用，避免每道题重新加载模型
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{MODEL_PATH}")

    if not (MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(
            f"模型目录中缺少 config.json：{MODEL_PATH}"
        )

    if not _MODEL_REQUIREMENT_CHECKED:
        validate_model_requirement(MODEL_PATH, strict=False)
        _MODEL_REQUIREMENT_CHECKED = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请检查PyTorch和显卡驱动。")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=QUANTIZATION_TYPE,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )

    print(f"模型目录：{MODEL_PATH}")
    print("正在加载 tokenizer...")

    _TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )

    print("正在以 4-bit 模式加载模型...")

    # 注意：必须赋值给 _MODEL
    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=COMPUTE_DTYPE,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    if _MODEL is None:
        raise RuntimeError("模型加载结果为None，请检查模型文件。")

    _MODEL.eval()

    print("模型加载完成。")
    print("显卡：", torch.cuda.get_device_name(0))

    return _TOKENIZER, _MODEL


@torch.inference_mode()
def generate_answer(
    messages: list[dict[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    record_entropy: bool = False,
    entropy_stride: int = 8,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    enable_thinking: bool = ENABLE_THINKING,
) -> dict[str, Any]:
    """
    统一的大模型推理接口。

    Direct / CoT：
        do_sample=False

    Self-Consistency：
        do_sample=True，并设置不同seed

    Adaptive Router：
        record_entropy=True
    """

    tokenizer, model = load_model()

    # 设置随机种子
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 使用模型自带的聊天模板构造输入
    template_arguments = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "enable_thinking": enable_thinking,
    }
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            **template_arguments,
        )
    except TypeError:
        # 兼容不认识 enable_thinking 参数的旧 tokenizer。
        template_arguments.pop("enable_thinking")
        inputs = tokenizer.apply_chat_template(
            messages,
            **template_arguments,
        )

    input_tokens = int(inputs["input_ids"].shape[1])
    if input_tokens > max_input_tokens:
        del inputs
        release_cuda_memory()
        raise ContextBudgetError(
            f"输入上下文为 {input_tokens} Token，超过安全预算 "
            f"{max_input_tokens} Token。请压缩下游 Agent 上下文。"
        )

    # 输入放到模型词嵌入层所在设备
    input_device = (
        model.get_input_embeddings()
        .weight
        .device
    )

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    # 必须先创建熵记录器，再加入generate参数
    entropy_recorder = None

    if record_entropy:
        entropy_recorder = EntropyRecorder(
            stride=entropy_stride,
            tail_window=8,
        )

    generation_config: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": REPETITION_PENALTY,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        "use_cache": True,
    }

    # temperature和top_p只在随机采样时传入
    if do_sample:
        generation_config.update(
            {
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    # Adaptive Router需要记录生成过程的概率熵
    if entropy_recorder is not None:
        generation_config["logits_processor"] = (
            LogitsProcessorList(
                [entropy_recorder]
            )
        )

    # CUDA是异步执行，计时前同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        output_ids = model.generate(
            **inputs,
            **generation_config,
        )
    except torch.OutOfMemoryError as error:
        del inputs
        release_cuda_memory()
        raise RuntimeError(
            "CUDA 显存不足。该调用已清理缓存；请降低 Agent 的输入或输出 "
            "Token 预算后从当前题目断点续跑。"
        ) from error

    # 等待GPU计算完成后停止计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    # 删除输入部分，只保留新生成Token
    generated_ids = output_ids[
        0,
        input_tokens:,
    ]

    output_tokens = int(
        generated_ids.shape[0]
    )

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    tokens_per_second = (
        output_tokens / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    hit_token_limit = (
        output_tokens >= max_new_tokens
    )

    result: dict[str, Any] = {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            input_tokens + output_tokens
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            4,
        ),
        "tokens_per_second": round(
            tokens_per_second,
            4,
        ),
        "max_new_tokens": max_new_tokens,
        "hit_token_limit": hit_token_limit,
        "peak_cuda_memory_mb": (
            round(torch.cuda.max_memory_allocated() / (1024**2), 2)
            if torch.cuda.is_available()
            else None
        ),
    }

    # 只有Router调用时才增加entropy字段
    if entropy_recorder is not None:
        result["entropy"] = (
            entropy_recorder.summary()
        )

    del generated_ids
    del output_ids
    del inputs
    release_cuda_memory()

    return result
