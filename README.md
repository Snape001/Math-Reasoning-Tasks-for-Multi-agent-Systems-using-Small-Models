# MATH-500 多智能体数学推理

项目保留三个多 Agent 版本，用于展示从线性协作、熵路由到闭环验证的优化路径。正式开发集为 `data/dev_20.jsonl`，正式测试集为 `data/eval_200.jsonl`。

## 环境

- Conda 环境：`multiagent`
- 基座模型：本地 Qwen3.5-4B
- 模型目录：`baselines/models`
- 正式实验使用 `--strict-model`

## V1：线性协作

```text
Solver -> Critic -> Verifier -> Judge
```

- Solver 独立求解。
- Critic 定位最早错误。
- Verifier 验证关键步骤。
- Judge 综合三方证据并提交答案。

运行 20 题开发集：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v1 --input data/dev_20.jsonl --output results/v1_dev_20.jsonl
```

## V2：熵路由与独立求解

```text
Entropy Router / Solver A
    -> 低熵：直接提交
    -> 高熵：Solver B + Independent Verifier
             -> Critic 比较
             -> Judge 裁决
```

- 使用生成熵判断是否升级到多 Agent。
- Solver A 与 Solver B 独立求解。
- Verifier 只读原题并独立重算，避免被候选答案锚定。
- Critic 比较候选，Judge 最终裁决。

运行 20 题开发集：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v2 --mode adaptive --input data/dev_20.jsonl --output results/v2_dev_20.jsonl --router-config configs/router_v2.json
```

## V3：闭环验证与条件救援

```text
Entropy Router
  -> Planner
  -> Primary Solver
  -> Adversarial Verifier
  -> Reviser（条件触发）
  -> Independent Final Checker
  -> Rescue Solver（条件触发）
  -> Rescue Finalizer（截断时触发）
  -> Hierarchical Blackboard（完全无答案时的最后救援）
```

V3 是当前表现最好的版本。正常通过时跳过 Reviser 和 Rescue；高风险路径不回退 Router 初猜。层级黑板只在主链完全没有有效答案时启动。

运行 20 题开发集：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v3 --mode adaptive --input data/dev_20.jsonl --output results/v3_dev_20.jsonl --router-config configs/router_v3.json --seed 42 --run-id 0 --strict-model
```

运行 200 题正式测试集：

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m scripts.run_v3 --mode adaptive --input data/eval_200.jsonl --output results/v3_eval_200.jsonl --router-config configs/router_v3.json --seed 42 --run-id 0 --strict-model
```

## 保留结果

```text
results/
  v1_dev_20.jsonl
  v2_dev_20.jsonl
  v3_dev_20.jsonl
  v3_eval_200.jsonl
  v3_eval_200_audited.jsonl
  direct_dev_20.jsonl
  direct_eval_200.jsonl
  direct_eval_200_audited.jsonl
  cot_dev_20.jsonl
  cot_eval_200.jsonl
  cot_eval_200_audited.jsonl
  self_consistency_dev_20.jsonl
  self_consistency_eval_200.jsonl
  self_consistency_eval_200_audited.jsonl
  v3_vs_self_consistency/
```

V1 和 V2 没有运行过与当前定义完全一致的正式 200 题实验，因此没有补造对应结果。V3 的正式 200 题结果为 188/200；人工格式复核后仍为 188/200。

## 公平比较

Direct、CoT、Self-Consistency 与 V3 使用相同模型、正式 200 题数据和答案匹配规则。Self-Consistency 使用 5 条独立 CoT 轨迹投票。人工复核结果保存在带 `_audited` 后缀的文件中。

## 测试

```powershell
& D:/anaconda3/envs/multiagent/python.exe -m unittest discover -s tests -v
```
