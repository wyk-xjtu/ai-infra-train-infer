"""
训练Worker - Ray Actor封装

将TrainEngine封装为Ray Actor，支持：
- 远程调用训练步骤
- 异步权重导出
- 状态查询
- GPU资源绑定
"""
import time
from typing import Dict, List, Optional

import torch

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

from .train_engine import TrainEngine, TrainConfig, TrainMetrics


def _create_train_worker_class():
    """动态创建Ray Actor类（避免import时ray未初始化）"""
    if not _RAY_AVAILABLE:
        raise ImportError("Ray is required for TrainWorker. Install via: pip install ray")

    @ray.remote(num_gpus=2)
    class TrainWorker:
        """Ray Actor: 训练Worker

        封装TrainEngine为可远程调用的Actor，用于Disaggregated架构中
        独立GPU组上执行训练任务。
        """

        def __init__(self, config: TrainConfig):
            """
            Args:
                config: 训练配置
            """
            self.config = config
            self.engine: Optional[TrainEngine] = None
            self._initialized = False
            self._step_count = 0
            self._total_train_time = 0.0

        def initialize(self):
            """初始化训练引擎

            创建TrainEngine并执行完整初始化流程:
            - 模型加载 + TP切分
            - LoRA注入
            - 优化器创建
            """
            self.engine = TrainEngine(self.config)
            self.engine.initialize()
            self._initialized = True
            return True

        def train_step(
            self,
            prompts_tokens: List[List[int]],
            responses_group: List[List[List[int]]],
            rewards_group: List[List[float]],
        ) -> TrainMetrics:
            """执行一步GRPO训练

            Args:
                prompts_tokens: batch个prompt的token ids
                responses_group: [batch, K, seq_len] 每个prompt的K个回复token
                rewards_group: [batch, K] 每个prompt的K个回复奖励

            Returns:
                TrainMetrics: 训练指标（loss, grad_norm等）
            """
            assert self._initialized, "TrainWorker not initialized. Call initialize() first."

            t0 = time.time()
            metrics = self.engine.grpo_step(
                prompts_tokens=prompts_tokens,
                responses_group=responses_group,
                rewards_group=rewards_group,
            )
            self._total_train_time += time.time() - t0
            self._step_count += 1

            return metrics

        def export_weights(self) -> Dict[str, torch.Tensor]:
            """导出合并LoRA后的权重

            将LoRA权重合并到基础模型后导出state_dict（CPU上）
            用于传输给推理侧。

            Returns:
                state_dict: 合并后的模型权重（CPU tensors）
            """
            assert self._initialized, "TrainWorker not initialized."
            return self.engine.export_weights()

        def get_status(self) -> dict:
            """获取当前状态

            Returns:
                包含step数、显存使用、训练时间等信息的字典
            """
            status = {
                "initialized": self._initialized,
                "step_count": self._step_count,
                "total_train_time_s": self._total_train_time,
                "config": {
                    "model_path": self.config.model_path,
                    "tp_size": self.config.tp_size,
                    "lora_rank": self.config.lora_rank,
                },
            }

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                status["gpu"] = {
                    "device_id": device,
                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
                    "peak_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                }

            if self._initialized and self.engine is not None:
                status["trainable_params"] = self.engine.trainable_params_count
                status["total_params"] = self.engine.total_params_count

            return status

        def save_checkpoint(self, path: str):
            """保存训练检查点"""
            assert self._initialized, "TrainWorker not initialized."
            self.engine.save_checkpoint(path)

        def load_checkpoint(self, path: str):
            """加载训练检查点"""
            assert self._initialized, "TrainWorker not initialized."
            self.engine.load_checkpoint(path)

    return TrainWorker


# 延迟创建，避免import时ray未初始化报错
TrainWorker = None


def get_train_worker_class():
    """获取TrainWorker Ray Actor类

    Returns:
        Ray remote class: TrainWorker
    """
    global TrainWorker
    if TrainWorker is None:
        TrainWorker = _create_train_worker_class()
    return TrainWorker
