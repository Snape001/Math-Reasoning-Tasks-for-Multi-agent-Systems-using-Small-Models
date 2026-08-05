from __future__ import annotations

import math
from typing import Any

import torch
from transformers import LogitsProcessor


class EntropyRecorder(LogitsProcessor):
    """
    在生成过程中定期计算词表概率分布的归一化熵。

    H_norm = H(p) / log(|V|)

    归一化后理论范围约为[0, 1]：
    - 越低：分布越集中，模型越确定
    - 越高：分布越平坦，模型越不确定

    stride=8表示每8个生成Token测量一次，
    避免在RTX 4060上引入过高计算开销。
    """

    def __init__(
        self,
        stride: int = 8,
        tail_window: int = 8,
    ) -> None:
        if stride <= 0:
            raise ValueError("stride必须大于0。")

        self.stride = stride
        self.tail_window = tail_window
        self.step_count = 0

        # 保存的是GPU上的标量Tensor，不保存完整词表logits
        self.normalized_entropies: list[torch.Tensor] = []
        self.top1_probabilities: list[torch.Tensor] = []

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        current_step = self.step_count
        self.step_count += 1

        if current_step % self.stride != 0:
            return scores

        logits = scores.float()

        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )
        probabilities = log_probs.exp()

        # 避免0 * -inf产生NaN
        entropy_terms = torch.nan_to_num(
            probabilities * log_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        entropy = -entropy_terms.sum(dim=-1)

        vocabulary_size = logits.shape[-1]
        normalized_entropy = (
            entropy / math.log(vocabulary_size)
        )

        top1_probability = probabilities.max(
            dim=-1
        ).values

        self.normalized_entropies.append(
            normalized_entropy.mean().detach().float().cpu()
        )

        self.top1_probabilities.append(
            top1_probability.mean().detach().float().cpu()
        )

        # 只记录，不修改生成分布
        return scores

    def summary(self) -> dict[str, Any]:
        if not self.normalized_entropies:
            return {
                "measured_steps": 0,
                "entropy_stride": self.stride,
                "mean_normalized_entropy": None,
                "tail_normalized_entropy": None,
                "p90_normalized_entropy": None,
                "mean_top1_probability": None,
                "route_score": None,
            }

        entropies = torch.stack(
            self.normalized_entropies
        ).float().cpu()

        top1_probabilities = torch.stack(
            self.top1_probabilities
        ).float().cpu()

        tail_size = min(
            self.tail_window,
            len(entropies),
        )

        mean_entropy = entropies.mean().item()
        tail_entropy = (
            entropies[-tail_size:].mean().item()
        )
        p90_entropy = torch.quantile(
            entropies,
            0.9,
        ).item()

        # 全局推理不确定性 + 最后约64个Token的不确定性
        route_score = (
            0.7 * mean_entropy
            + 0.3 * tail_entropy
        )

        return {
            "measured_steps": int(len(entropies)),
            "entropy_stride": self.stride,
            "mean_normalized_entropy": round(
                mean_entropy,
                6,
            ),
            "tail_normalized_entropy": round(
                tail_entropy,
                6,
            ),
            "p90_normalized_entropy": round(
                p90_entropy,
                6,
            ),
            "mean_top1_probability": round(
                top1_probabilities.mean().item(),
                6,
            ),
            "route_score": round(
                route_score,
                6,
            ),
        }