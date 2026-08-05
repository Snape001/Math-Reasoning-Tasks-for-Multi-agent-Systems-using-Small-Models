import os
from pathlib import Path

import torch


# 项目根目录：multi_agent_math
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 正式评测必须使用题目指定的 Qwen3.5。默认仍指向项目内目录，
# 但可通过环境变量切换权重，避免复制数 GB 模型文件。
MODEL_PATH = Path(
    os.environ.get(
        "MULTIAGENT_MODEL_PATH",
        str(PROJECT_ROOT / "baselines" / "models"),
    )
).expanduser()

# 数据与结果目录
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


# 模型信息
MODEL_NAME = os.environ.get(
    "MULTIAGENT_MODEL_NAME",
    "local-model",
)
REQUIRED_MODEL_FAMILY = "qwen3.5"

# 生成配置
MAX_NEW_TOKENS = 4096
MAX_INPUT_TOKENS = 4096
DEFAULT_SEED = 42
ENABLE_THINKING = False
REPETITION_PENALTY = 1.08

# 四种方法共同使用的正式实验协议版本。结果断点将同时校验这个版本，
# 避免误把旧模型或旧 Prompt 的记录当作当前实验。
EXPERIMENT_VERSION = "qwen35_math500_fair_v1"

# 8 GB 显存下的安全边界。各 Agent 使用更小的独立输出预算；这里仅作为
# 旧基线和通用接口的上限。超长协作上下文会在送入模型前被明确拒绝。

# 量化配置
LOAD_IN_4BIT = True
QUANTIZATION_TYPE = "nf4"
USE_DOUBLE_QUANT = True

# 4-bit是权重精度，模型计算使用BF16
COMPUTE_DTYPE = torch.bfloat16

# Self-Consistency配置
SC_NUM_SAMPLES = 5
SC_BASE_SEED = 42
SC_TEMPERATURE = 0.7
SC_TOP_P = 0.9
