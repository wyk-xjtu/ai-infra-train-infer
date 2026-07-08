"""
Disaggregated训推分离调度器

架构:
- GPU 0-1: Training Worker (TP=2, 持续训练)
- GPU 2-3: Inference Worker (TP=2, 持续推理, vLLM)
- Orchestrator: 协调两者，管理权重版本

Pipeline调度:
- Step N: Training正在训练 | Inference用Step N-1的权重生成
- 训练完成后异步推送权重到推理侧
- 推理侧收到新权重后热更新

关键: 训练和推理可以overlap，提高GPU利用率
"""
import asyncio
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

from ..engines.train_engine import TrainConfig, TrainMetrics
from ..engines.train_worker import get_train_worker_class
from ..engines.infer_worker import get_infer_worker_class
from ..transfer.nccl_transfer import WeightTransferManager
from ..reward.gsm8k_reward import GSM8KRewardFunction
from ..data.data_pipeline import GSM8KDataPipeline, GRPOBatch
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector

logger = get_logger("orchestrator.disagg")


@dataclass
class DisaggConfig:
    """Disaggregated调度器配置"""
    model_path: str = "Qwen/Qwen3-4B"
    train_tp_size: int = 2
    infer_tp_size: int = 2
    train_gpus: List[int] = field(default_factory=lambda: [0, 1])
    infer_gpus: List[int] = field(default_factory=lambda: [2, 3])

    # 训练
    total_iterations: int = 50
    lora_rank: int = 16
    learning_rate: float = 1e-5
    use_flash_attn: bool = False
    attention_backend: str = "sdpa"  # "standard" / "sdpa" / "flash_attn"

    # 推理
    inference_backend: str = "vllm"  # "vllm" 或 "custom"（自研引擎）
    num_samples_per_prompt: int = 4
    max_tokens: int = 512
    temperature: float = 0.7

    # 权重同步
    weight_sync_backend: str = "ray_object_store"  # "nccl" or "ray_object_store"
    async_weight_update: bool = True

    # Pipeline
    overlap_training_inference: bool = True

    # 数据
    batch_size: int = 4
    max_data_samples: int = 200


@dataclass
class DisaggIterationMetrics:
    """单次迭代指标"""
    iteration: int
    train_metrics: Optional[TrainMetrics] = None
    avg_reward: float = 0.0
    accuracy: float = 0.0
    weight_version: int = 0
    train_time: float = 0.0
    infer_time: float = 0.0
    sync_time: float = 0.0
    total_time: float = 0.0
    overlap_ratio: float = 0.0  # 训练推理并行比


class DisaggOrchestrator:
    """Disaggregated训推分离调度器

    训练和推理在不同GPU组上，支持Pipeline并行。
    通过Ray Actor实现跨进程协作。
    """

    def __init__(self, config: DisaggConfig):
        if not _RAY_AVAILABLE:
            raise ImportError("Ray is required for DisaggOrchestrator. Install via: pip install ray")

        self.config = config
        self.metrics_history: List[DisaggIterationMetrics] = []

        # Workers（initialize时创建）
        self._train_worker = None
        self._infer_worker = None

        # 权重传输
        self._weight_manager: Optional[WeightTransferManager] = None
        self._weight_version = 0

        # 数据和奖励
        self._reward_fn: Optional[GSM8KRewardFunction] = None
        self._data_pipeline: Optional[GSM8KDataPipeline] = None

        # 工具
        self._metrics_collector = MetricsCollector()

        # Ray资源
        self._placement_group = None

    async def initialize(self):
        """初始化

        - 初始化Ray（如果未初始化）
        - 创建PlacementGroup绑定GPU
        - 启动Training Worker (Ray Actor)
        - 启动Inference Worker (Ray Actor, 内含vLLM)
        - 初始化WeightTransferManager
        - 加载数据
        """
        logger.info("Initializing DisaggOrchestrator...")

        # 确保Ray已初始化
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        bundles = [
            {"GPU": len(self.config.train_gpus)},  # 训练用GPU
            {"GPU": len(self.config.infer_gpus)},  # 推理用GPU
        ]
        self._placement_group = ray.util.placement_group(
            bundles=bundles,
            strategy="STRICT_PACK",
        )
        await asyncio.wrap_future(self._placement_group.ready())
        logger.info(f"PlacementGroup created: train_gpus={self.config.train_gpus}, infer_gpus={self.config.infer_gpus}")

        TrainWorkerCls = get_train_worker_class()
        train_config = TrainConfig(
            model_path=self.config.model_path,
            tp_size=self.config.train_tp_size,
            lora_rank=self.config.lora_rank,
            learning_rate=self.config.learning_rate,
            use_flash_attn=self.config.use_flash_attn,
            attention_backend=self.config.attention_backend,
            grpo_num_samples=self.config.num_samples_per_prompt,
        )
        # 在PlacementGroup的第一个bundle上启动训练Worker
        self._train_worker = TrainWorkerCls.options(
            placement_group=self._placement_group,
            placement_group_bundle_index=0,
            num_gpus=len(self.config.train_gpus),
        ).remote(train_config)

        await asyncio.wrap_future(self._train_worker.initialize.remote())
        logger.info("TrainWorker initialized")

        InferWorkerCls = get_infer_worker_class()
        self._infer_worker = InferWorkerCls.options(
            placement_group=self._placement_group,
            placement_group_bundle_index=1,
            num_gpus=len(self.config.infer_gpus),
        ).remote(
            model_path=self.config.model_path,
            tp_size=self.config.infer_tp_size,
            backend=self.config.inference_backend,
        )

        await asyncio.wrap_future(self._infer_worker.initialize.remote())
        logger.info("InferWorker initialized")

        self._weight_manager = WeightTransferManager(
            backend=self.config.weight_sync_backend,
        )
        logger.info(f"WeightTransferManager initialized (backend={self.config.weight_sync_backend})")

        self._reward_fn = GSM8KRewardFunction()

        self._data_pipeline = GSM8KDataPipeline(
            tokenizer_name=self.config.model_path,
            max_samples=self.config.max_data_samples,
        )
        logger.info(f"Data pipeline loaded: {len(self._data_pipeline)} samples")

        logger.info("DisaggOrchestrator initialization complete.")

    async def run(self) -> List[DisaggIterationMetrics]:
        """运行Pipeline训推循环

        两种模式:
        1. overlap模式: 训练和推理并行（Pipeline）
        2. 顺序模式: 训练和推理交替执行

        Returns:
            所有迭代的指标列表
        """
        logger.info(
            f"Starting DisaggOrchestrator loop: {self.config.total_iterations} iterations, "
            f"overlap={self.config.overlap_training_inference}"
        )

        batches = list(self._data_pipeline.get_batches(self.config.batch_size))

        if self.config.overlap_training_inference:
            return await self._run_pipeline(batches)
        else:
            return await self._run_sequential(batches)

    async def _run_pipeline(self, batches: List[GRPOBatch]) -> List[DisaggIterationMetrics]:
        """Pipeline模式: 训练和推理可以并行

        核心思想:
        - 在训练Step N的同时，用Step N-1的权重进行推理
        - 训练完成后异步推送权重
        """
        # 第一轮: 仅推理（用初始权重），无并行
        batch_0 = batches[0 % len(batches)]
        logger.info("[Pipeline] Iteration 0: initial inference (no training overlap)")

        self._metrics_collector.start_timer("inference")
        responses_prev = await self._remote_generate(batch_0)
        infer_time = self._metrics_collector.stop_timer("inference")

        rewards_prev = self._do_rewarding(batch_0, responses_prev)
        batch_prev = batch_0

        for iteration in range(self.config.total_iterations):
            iter_start = time.time()
            iter_metrics = DisaggIterationMetrics(iteration=iteration)

            batch_curr = batches[(iteration + 1) % len(batches)]

            # Pipeline: 训练(当前batch) 和 推理(下一batch) 并行
            self._metrics_collector.start_timer("training")
            train_future = self._remote_train_step(
                batch_prev, responses_prev, rewards_prev
            )

            # 同时推理下一batch
            self._metrics_collector.start_timer("inference")
            responses_curr = await self._remote_generate(batch_curr)
            infer_time = self._metrics_collector.stop_timer("inference")
            iter_metrics.infer_time = infer_time

            # 等待训练完成
            train_metrics = await train_future
            train_time = self._metrics_collector.stop_timer("training")
            iter_metrics.train_time = train_time
            iter_metrics.train_metrics = train_metrics

            # 计算overlap比
            overlap = min(infer_time, train_time)
            total_parallel = max(infer_time, train_time)
            iter_metrics.overlap_ratio = overlap / total_parallel if total_parallel > 0 else 0

            # 奖励计算
            rewards_curr = self._do_rewarding(batch_curr, responses_curr)
            flat_rewards = [r for group in rewards_curr for r in group]
            iter_metrics.avg_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0
            iter_metrics.accuracy = self._reward_fn.accuracy

            # 异步权重同步
            self._metrics_collector.start_timer("syncing")
            await self._sync_weights_async()
            sync_time = self._metrics_collector.stop_timer("syncing")
            iter_metrics.sync_time = sync_time

            self._weight_version += 1
            iter_metrics.weight_version = self._weight_version

            iter_metrics.total_time = time.time() - iter_start
            self.metrics_history.append(iter_metrics)

            # 记录指标
            self._metrics_collector.record("reward_mean", iter_metrics.avg_reward)
            self._metrics_collector.record("accuracy", iter_metrics.accuracy)
            if train_metrics:
                self._metrics_collector.record("loss", train_metrics.loss)

            logger.info(
                f"[Iter {iteration}/{self.config.total_iterations}] "
                f"reward={iter_metrics.avg_reward:.3f} "
                f"acc={iter_metrics.accuracy:.2%} "
                f"overlap={iter_metrics.overlap_ratio:.1%} "
                f"time={iter_metrics.total_time:.1f}s"
            )

            # 更新prev引用
            batch_prev = batch_curr
            responses_prev = responses_curr
            rewards_prev = rewards_curr

        logger.info("Pipeline training loop completed.")
        return self.metrics_history

    async def _run_sequential(self, batches: List[GRPOBatch]) -> List[DisaggIterationMetrics]:
        """顺序模式: 训练→推理→训练→... 依次执行"""
        for iteration in range(self.config.total_iterations):
            iter_start = time.time()
            iter_metrics = DisaggIterationMetrics(iteration=iteration)

            batch = batches[iteration % len(batches)]

            # 推理
            self._metrics_collector.start_timer("inference")
            responses = await self._remote_generate(batch)
            iter_metrics.infer_time = self._metrics_collector.stop_timer("inference")

            # 奖励
            rewards = self._do_rewarding(batch, responses)
            flat_rewards = [r for group in rewards for r in group]
            iter_metrics.avg_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0
            iter_metrics.accuracy = self._reward_fn.accuracy

            # 训练
            self._metrics_collector.start_timer("training")
            train_future = self._remote_train_step(batch, responses, rewards)
            train_metrics = await train_future
            iter_metrics.train_time = self._metrics_collector.stop_timer("training")
            iter_metrics.train_metrics = train_metrics

            # 权重同步
            self._metrics_collector.start_timer("syncing")
            await self._sync_weights_async()
            iter_metrics.sync_time = self._metrics_collector.stop_timer("syncing")

            self._weight_version += 1
            iter_metrics.weight_version = self._weight_version

            iter_metrics.total_time = time.time() - iter_start
            self.metrics_history.append(iter_metrics)

            self._metrics_collector.record("reward_mean", iter_metrics.avg_reward)
            self._metrics_collector.record("accuracy", iter_metrics.accuracy)

            logger.info(
                f"[Iter {iteration}/{self.config.total_iterations}] "
                f"reward={iter_metrics.avg_reward:.3f} "
                f"acc={iter_metrics.accuracy:.2%} "
                f"time={iter_metrics.total_time:.1f}s"
            )

        logger.info("Sequential training loop completed.")
        return self.metrics_history

    async def _remote_generate(self, batch: GRPOBatch) -> List[List[str]]:
        """远程调用推理Worker生成回复"""
        result_ref = self._infer_worker.generate.remote(
            prompts=batch.prompts,
            num_samples=self.config.num_samples_per_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return await asyncio.wrap_future(result_ref)

    async def _remote_train_step(
        self, batch: GRPOBatch, responses: List[List[str]], rewards: List[List[float]]
    ):
        """远程调用训练Worker执行GRPO训练步

        Returns:
            awaitable that resolves to TrainMetrics
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        responses_group_tokens: List[List[List[int]]] = []
        for response_group in responses:
            group_tokens = []
            for text in response_group:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                group_tokens.append(tokens)
            responses_group_tokens.append(group_tokens)

        result_ref = self._train_worker.train_step.remote(
            prompts_tokens=batch.prompt_tokens,
            responses_group=responses_group_tokens,
            rewards_group=rewards,
        )
        return await asyncio.wrap_future(result_ref)

    def _do_rewarding(self, batch: GRPOBatch, responses: List[List[str]]) -> List[List[float]]:
        """计算奖励"""
        all_rewards: List[List[float]] = []
        for i, response_group in enumerate(responses):
            gt = batch.ground_truths[i]
            rewards = self._reward_fn(
                responses=response_group,
                ground_truths=[gt] * len(response_group),
            )
            all_rewards.append(rewards)
        return all_rewards

    async def _sync_weights_async(self):
        """异步权重同步: 从训练Worker导出权重 → 传输 → 推理Worker更新

        支持双backend:
        - vllm/custom: 都通过 InferWorker.update_weights() 更新
          内部根据 backend 自动选择 vLLM 直接替换或自研引擎缓存清除
        """
        # 从训练Worker导出权重
        state_dict_ref = self._train_worker.export_weights.remote()
        state_dict = await asyncio.wrap_future(state_dict_ref)

        if self.config.weight_sync_backend == "ray_object_store":
            # 通过Ray Object Store传输
            await self._weight_manager.sync_weights(state_dict, is_sender=True)
            # 推理Worker更新权重（内部根据backend自动处理）
            update_ref = self._infer_worker.update_weights.remote(state_dict)
            success = await asyncio.wrap_future(update_ref)
            if not success:
                logger.warning("Inference worker weight update failed")
        else:
            # NCCL广播
            await self._weight_manager.sync_weights(state_dict, is_sender=True)

        logger.info(f"Weight sync completed (version={self._weight_version + 1})")

    async def shutdown(self):
        """清理资源: 停止Workers, 释放PlacementGroup"""
        logger.info("Shutting down DisaggOrchestrator...")

        # 终止Workers
        if self._train_worker is not None:
            ray.kill(self._train_worker)
            self._train_worker = None

        if self._infer_worker is not None:
            ray.kill(self._infer_worker)
            self._infer_worker = None

        # 释放PlacementGroup
        if self._placement_group is not None:
            ray.util.remove_placement_group(self._placement_group)
            self._placement_group = None

        logger.info("DisaggOrchestrator shut down.")

    def get_summary(self) -> dict:
        """获取运行总结

        Returns:
            包含完整运行统计的字典
        """
        if not self.metrics_history:
            return {"status": "no_data"}

        rewards = [m.avg_reward for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history]
        total_times = [m.total_time for m in self.metrics_history]
        train_times = [m.train_time for m in self.metrics_history]
        infer_times = [m.infer_time for m in self.metrics_history]
        sync_times = [m.sync_time for m in self.metrics_history]
        overlap_ratios = [m.overlap_ratio for m in self.metrics_history]

        summary = {
            "config": {
                "model_path": self.config.model_path,
                "total_iterations": self.config.total_iterations,
                "overlap": self.config.overlap_training_inference,
                "weight_sync_backend": self.config.weight_sync_backend,
                "inference_backend": self.config.inference_backend,
                "train_gpus": self.config.train_gpus,
                "infer_gpus": self.config.infer_gpus,
            },
            "results": {
                "completed_iterations": len(self.metrics_history),
                "final_accuracy": accuracies[-1] if accuracies else 0,
                "final_avg_reward": rewards[-1] if rewards else 0,
                "best_accuracy": max(accuracies) if accuracies else 0,
                "total_time_s": sum(total_times),
            },
            "timing": {
                "avg_iteration_s": sum(total_times) / len(total_times) if total_times else 0,
                "avg_train_s": sum(train_times) / len(train_times) if train_times else 0,
                "avg_infer_s": sum(infer_times) / len(infer_times) if infer_times else 0,
                "avg_sync_s": sum(sync_times) / len(sync_times) if sync_times else 0,
                "avg_overlap_ratio": sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0,
            },
            "metrics_summary": self._metrics_collector.get_summary(),
        }

        return summary
