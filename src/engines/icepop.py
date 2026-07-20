"""
IcePop 训推对齐检测模块

基于 Ring-1T 论文 (arxiv.org/abs/2510.18855) 的 IcePop 技术：
检测训练引擎和推理引擎计算的 log_prob 差异，丢弃高差异 response 的梯度贡献，
防止策略更新被噪声主导。

设计要点：
- Response 级别检测（对比序列平均 log_prob）
- 通过将异常 response 的 advantage 置零实现掩蔽（不修改 loss 函数签名）
- max_mask_ratio 硬上限保护，防止过度丢弃
- 完全通过 enabled 开关控制，禁用时代码路径不变
- TP 兼容：所有计算 per-rank 独立，不引入跨 rank 通信
"""

import torch
from dataclasses import dataclass
from typing import Dict, Tuple

from ..utils.logger import get_logger

logger = get_logger("engines.icepop")


@dataclass
class IcePopConfig:
    """IcePop 配置"""
    enabled: bool = False
    divergence_threshold: float = 0.5   # |train - infer| > threshold 则判定为异常
    max_mask_ratio: float = 0.5         # 最多掩蔽 50% responses，防止过度丢弃
    log_diagnostics: bool = True


class IcePopFilter:
    """IcePop 训推对齐检测过滤器

    核心流程：
    1. 计算训练与推理 log_probs 的绝对差异
    2. 超过阈值的 response 标记为异常
    3. 受 max_mask_ratio 限制，仅掩蔽差异最大的 top-k 个
    4. 将异常 response 的 advantage 置零，使其不贡献梯度
    """

    def __init__(self, config: IcePopConfig):
        self.config = config
        self._first_mask_logged = False  # 首次掩蔽时输出 info 级别日志

    def filter_divergent_responses(
        self,
        train_log_probs: torch.Tensor,   # [batch*K]
        infer_log_probs: torch.Tensor,   # [batch*K]
        advantages: torch.Tensor,        # [batch*K]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """检测训推差异，掩蔽异常 response 的 advantage

        Args:
            train_log_probs: 训练模式下计算的序列级 log_probs [batch*K]
            infer_log_probs: 推理模式下计算的序列级 log_probs [batch*K]
            advantages: 组内相对优势 [batch*K]

        Returns:
            filtered_advantages: 掩蔽异常 response 后的 advantages [batch*K]
            diagnostics: 诊断信息字典
        """
        # 1. 计算差异
        divergence = (train_log_probs - infer_log_probs).abs()

        # 2. 标记超过阈值的异常 response
        anomaly_mask = divergence > self.config.divergence_threshold

        # 3. 限制掩蔽比例（max_mask_ratio 硬上限保护）
        num_responses = advantages.shape[0]
        max_masked = int(num_responses * self.config.max_mask_ratio)
        num_anomalies = anomaly_mask.sum().item()

        if num_anomalies > max_masked:
            # 只在异常集合内按差异排序，保留差异最大的 max_masked 个
            anomaly_indices = anomaly_mask.nonzero(as_tuple=False).squeeze(-1)
            _, order = divergence[anomaly_indices].sort(descending=True)
            keep_indices = anomaly_indices[order[:max_masked]]

            # 重建 anomaly_mask，仅保留前 max_masked 个异常
            anomaly_mask = torch.zeros_like(anomaly_mask)
            anomaly_mask[keep_indices] = True
            num_anomalies = max_masked

        # 4. 将异常 response 的 advantage 置零
        filtered_advantages = advantages.clone()
        if num_anomalies > 0:
            filtered_advantages[anomaly_mask] = 0.0

        # 5. 诊断信息
        masked_ratio = num_anomalies / max(num_responses, 1)
        mean_divergence = divergence.mean().item()
        max_divergence = divergence.max().item()

        diagnostics = {
            "icepop_masked_ratio": masked_ratio,
            "icepop_mean_divergence": mean_divergence,
            "icepop_max_divergence": max_divergence,
            "icepop_num_masked": num_anomalies,
            "icepop_threshold": self.config.divergence_threshold,
        }

        # 6. 日志输出
        if self.config.log_diagnostics and num_anomalies > 0:
            if not self._first_mask_logged:
                logger.info(
                    "IcePop first mask event: masked %d/%d responses (%.1f%%), "
                    "mean_div=%.4f, max_div=%.4f, threshold=%.4f",
                    num_anomalies, num_responses, masked_ratio * 100,
                    mean_divergence, max_divergence, self.config.divergence_threshold,
                )
                self._first_mask_logged = True
            else:
                logger.debug(
                    "IcePop: masked %d/%d (%.1f%%), mean_div=%.4f, max_div=%.4f",
                    num_anomalies, num_responses, masked_ratio * 100,
                    mean_divergence, max_divergence,
                )

        return filtered_advantages, diagnostics
