# 小模型多智能体数学推理系统

本项目基于本地 `Qwen3.5-4B`，在固定的 MATH-500 子集上比较 Direct、CoT、Self-Consistency 与多智能体协作推理。开发集为 `data/dev_20.jsonl`，正式测试集固定为 `data/eval_200.jsonl`。

当前保留 V1、V2、V3、V4 和最终 V4.1。V4.1 是最新正式版本，在 200 题上取得 **193/200（96.5%）**。

## 环境

- Conda 环境：`multiagent`
- Python：`D:/anaconda3/envs/multiagent/python.exe`
- 模型：本地 `Qwen3.5-4B`
- 默认模型目录：`baselines/models`
- 量化：4-bit NF4，BF16 计算
- 最大输入/生成 Token：4096/4096
- 正式实验：随机种子 42，`--strict-model`

安装依赖：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m pip install -r requirements-multiagent.txt
```

也可通过环境变量 `MULTIAGENT_MODEL_PATH` 指向其他本地模型目录。

## 版本演进

### V1：线性角色协作

```text
Solver -> Critic -> Verifier -> Judge
```

先建立最基本的角色分工和消息传递：Solver 求解，Critic 定位错误，Verifier 检查关键步骤，Judge 综合证据。问题是错误会沿单一路径传播，后续 Agent 容易被前文锚定。

### V2：熵路由与独立重算

```text
Entropy Router
  ├─ 低风险：Router 直接提交
  └─ 高风险：两个独立 Solver -> Critic 比较
                             -> Verifier 独立重算
                             -> Judge 裁决
```

V2 引入生成熵路由：简单题只调用单 Agent，困难题才进入协作；同时用独立 Solver 和独立重算降低单链错误传播。该版本建立了后续版本一直保留的“按难度分配计算量”原则。

### V3：闭环验证与条件救援

```text
Entropy Router
  └─ Planner -> Primary Solver -> Adversarial Verifier
             -> Reviser（条件触发）
             -> Independent Final Checker
             -> Rescue Solver（条件触发）
             -> Rescue Finalizer（必要时收尾）
```

V3 强化真正的角色协作：Planner 明确目标量和答案类型；对抗验证器寻找反例；独立终检避免被批评意见锚定；验证失败才启动修订和救援。正常通过时跳过额外调用，以控制成本。该主干正式结果为 188/200，换种子重复运行为 189/200。

### V4：两阶段证据评审探索

V4 尝试“多名异构 Solver + 逻辑/计算双 Reviewer + 条件 Arbiter”。实验表明，角色差异过强会使 4B 模型的候选质量低于统一 CoT 采样；复杂评审也可能在已有正确候选时做出错误裁决。因此 V4 仅保留为结构探索，不作为最终正式对比结果。

### V4.1：多数锁定与分歧协作

```text
Entropy Router
  ├─ 低风险：Router 直接提交
  └─ 高风险：3 个独立同分布 CoT Solver
             ├─ 存在 2/3 多数：Conformist 锁定并提交
             └─ 无多数：Evidence Auditor
                        -> Disagreement Resolver
                        -> Constraint-aware Finalizer（必要时）
                        -> 唯一完整候选安全回退（仍未决时）
```

最终设计遵循以下思路：

1. 候选生成与 SC 使用相同 Prompt、温度和采样方式，只改变随机种子，先保证基础候选质量。
2. 2/3 多数出现后立即锁定，避免后续辩论把较可靠的多数答案带偏。
3. 只有真正分歧时才通信：Evidence Auditor 整理目标、约束和冲突证据，Resolver 针对争议点重算，必要时由 Finalizer 收尾。
4. 若完整协作链仍无答案，仅当三个 Solver 中恰好一个给出未截断且含显式 `FINAL_ANSWER` 的完整答案时采用它。

因此 V4.1 不等同于简单投票：投票只是低成本共识门；分歧样本仍经过具有不同职责和信息交互的审计、重算与提交过程。

## 可执行指令

### 20 题开发集

V1：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v1 --input data/dev_20.jsonl --output results/v1_dev_20.jsonl
```

V2：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v2 --mode adaptive --input data/dev_20.jsonl --output results/v2_dev_20.jsonl --router-config configs/router_v2.json
```

V3：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v3 --mode adaptive --input data/dev_20.jsonl --output results/v3_dev_20.jsonl --router-config configs/router_v3.json --seed 42 --run-id 0 --strict-model
```

V4：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v4 --mode adaptive --input data/dev_20.jsonl --output results/v4_dev_20.jsonl --router-config configs/router_v4.json --seed 42 --run-id 0 --strict-model
```

最终 V4.1：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v4_1 --mode adaptive --input data/dev_20.jsonl --output results/v4_1_dev_20.jsonl --router-config configs/router_v4_1.json --seed 42 --run-id 0 --strict-model
```

### 200 题正式测试

V3：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v3 --mode adaptive --input data/eval_200.jsonl --output results/v3_eval_200.jsonl --router-config configs/router_v3.json --seed 42 --run-id 0 --strict-model
```

SC@5：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_self_consistency --input data/eval_200.jsonl --output results/self_consistency_eval_200.jsonl --num-samples 5 --seed 42 --run-id 0 --strict-model
```

最终 V4.1：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v4_1 --mode adaptive --input data/eval_200.jsonl --output results/v4_1_eval_200.jsonl --router-config configs/router_v4_1.json --seed 42 --run-id 0 --strict-model
```

脚本支持断点续跑；若需要从头复现实验，请使用新的输出文件名，避免已有记录被自动跳过。

### 自动化测试

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m unittest discover -s tests -v
```

## 最终实验结果

所有方法使用同一基座模型和固定 200 题；Direct、CoT、SC@5 和 V3 采用人工格式复核后的结果。Token 为输入与输出 Token 总和。

| 方法 | 正确数 | 准确率 | 模型调用 | Token |
|---|---:|---:|---:|---:|
| Direct | 46/200 | 23.0% | 200 | 36,270 |
| CoT | 174/200 | 87.0% | 200 | 263,794 |
| Self-Consistency@5 | 187/200 | 93.5% | 1,000 | 1,362,606 |
| V3（正式） | 188/200 | 94.0% | 741 | 1,329,695 |
| V3（重复运行） | 189/200 | 94.5% | 725 | 1,278,356 |
| **V4.1（最终）** | **193/200** | **96.5%** | **558** | **947,433** |

V4.1 相比 SC@5 多做对 6 题，准确率提高 3.0 个百分点；调用次数减少 44.2%，Token 减少 30.5%。其中有5题提升直接来自投票之后的多 Agent 推理、纠错和裁决机制。
配对检验约 \(p=0.031\)，达到常用显著性标准。

### V4.1 分支结果

| 分支 | 正确/总数 |
|---|---:|
| Router 单 Agent | 99/99 |
| 三 Solver 多数锁定 | 79/79 |
| 分歧由 Resolver 解决 | 11/11 |
| 分歧由 Finalizer 收尾 | 4/6 |
| 最终仍未决 | 0/5 |

剩余 7 道错题全部为 Level 5，并且均因 Router 达到生成上限而进入多 Agent 路径。这说明当前主要瓶颈已集中在小模型对高难中间代数和组合概率题的基础推理能力，而不是简单题路由。

## 结果文件与实验边界

- 最终 V4.1：`results/v4_1_eval_200.jsonl`
- SC@5 复核：`results/self_consistency_eval_200_audited.jsonl`
- V3 复核：`results/v3_eval_200_audited.jsonl`
- V3 重跑：`results/v3_eval_200_run1.jsonl`
- Direct/CoT 复核：`results/direct_eval_200_audited.jsonl`、`results/cot_eval_200_audited.jsonl`