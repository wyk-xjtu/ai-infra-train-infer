"""
Colocate训推调度器

状态机: INIT → TRAINING → SLEEPING → SYNCING → INFERRING → REWARDING → TRAINING → ...

工作流:
1. INFERRING: 调用vLLM生成rollout样本（K个回复/prompt）
2. REWARDING: 计算奖励信号
3. TRAINING: 使用TrainEngine执行GRPO训练步
4. SLEEPING: 调用vLLM sleep(level=1)释放GPU显存
5. SYNCING: 通过CUDA IPC推送更新后的权重到vLLM
6. wake up → 回到INFERRING

关键设计:
- 状态机确保各阶段正确转换
- 异常恢复：任一阶段失败可重试或回退
- Metrics收集：各阶段耗时、显存使用
"""
import asyncio
import time
import enum
import json
import math
import os
import sys
from typing import List, Dict, Optional
from dataclasses import asdict, dataclass, field

from ..engines.train_engine import TrainEngine, TrainConfig, TrainMetrics
from ..transfer.ipc_transfer import VLLMClient, IPCWeightTransfer
from ..reward.gsm8k_reward import GSM8KRewardFunction
from ..data.data_pipeline import GSM8KDataPipeline, GRPOBatch
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector
from ..utils.memory_profiler import MemoryProfiler
from ..utils.eval_utils import EvalMetrics, compute_token_f1, compute_bleu4
from ..utils.artifact_writer import ArtifactWriter

logger = get_logger("orchestrator.colocate")


class OrchestratorState(enum.Enum):
    INIT = "init"
    TRAINING = "training"
    SLEEPING = "sleeping"
    SYNCING = "syncing"
    INFERRING = "inferring"
    REWARDING = "rewarding"
    ERROR = "error"
    DONE = "done"


@dataclass
class ColocateConfig:
    """Colocate调度器配置"""
    # 模型
    model_path: str = "Qwen/Qwen3-4B"
    tp_size: int = 1
    pp_size: int = 1  # Pipeline Parallelism 度（预留，当前未实现）
    dp_size: int = 1
    max_seq_len: int = 2048
    use_flash_attn: bool = False
    attention_backend: str = "sdpa"  # "standard" / "sdpa" / "flash_attn"

    # 训练
    training_mode: str = "grpo"  # "grpo" 或 "sft"
    output_dir: str = "outputs"
    checkpoint_save_dir: str = "checkpoints"
    checkpoint_save_interval: int = 0
    checkpoint_save_enabled: bool = True
    resume_from: Optional[str] = None
    train_steps_per_iteration: int = 1  # 每轮训推循环训练几步
    total_iterations: int = 50
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    amp_dtype: str = "bf16"  # 支持 "fp32" / "bf16" / "fp16"
    gradient_accumulation_steps: int = 4
    grpo_beta: float = 0.1
    grpo_clip_eps: float = 0.2

    # LR Scheduler
    lr_scheduler_type: str = "cosine"   # "cosine" / "linear" / "constant"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1

    # ZeRO 配置
    zero_stage: int = 1                 # ZeRO 阶段: 1=优化器分片, 2=优化器+梯度分片, 3=全分片

    # 推理
    inference_backend: str = "vllm"  # "vllm" 或 "custom"（自研引擎）
    inference_preview_enabled: bool = True  # 训练循环内推理预览开关
    vllm_url: str = "http://localhost:8000"
    num_samples_per_prompt: int = 4  # GRPO的K
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

    # 权重同步
    sleep_level: int = 1
    ipc_chunk_size: int = 16

    # 数据
    batch_size: int = 4
    max_data_samples: int = 200
    dataset_name: str = "openai/gsm8k"
    dataset_config: Optional[str] = "main"
    dataset_split: str = "train"
    prompt_template: str = "qwen3_math"
    prompt_field: str = "question"
    input_field: Optional[str] = None
    response_field: str = "answer"
    answer_field: Optional[str] = "answer"

    # 评估
    eval_max_tokens: int = 128
    eval_num_prompts: int = 3

    # Profiling 配置
    memory_profile_interval: int = 100      # 每 N 步显存快照
    enable_profiling: bool = False           # 通信/计算 profiling 开关
    profile_interval: int = 100             # profiling 采样间隔

    # 通信-计算 overlap
    enable_comm_overlap: bool = False       # 启用通信与计算重叠（tp_size > 1 时生效）

    # 精度配置传递
    amp_dtype: str = "bf16"                 # 传递给 TrainConfig

    # 重试
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class IterationMetrics:
    """单次迭代的指标"""
    iteration: int
    state_durations: Dict[str, float] = field(default_factory=dict)
    train_metrics: Optional[TrainMetrics] = None
    avg_reward: float = 0.0
    accuracy: float = 0.0
    total_time: float = 0.0


class ColocateOrchestrator:
    """Colocate训推共卡调度器

    训练和推理共用同一组GPU，通过sleep/wake机制交替执行。
    """

    def __init__(self, config: ColocateConfig):
        self.config = config
        self.state = OrchestratorState.INIT
        self._state_start_time = time.time()
        self.metrics_history: List[IterationMetrics] = []

        # 组件（initialize时创建）
        self._train_engine: Optional[TrainEngine] = None
        self._vllm_client: Optional[VLLMClient] = None
        self._ipc_transfer: Optional[IPCWeightTransfer] = None
        self._reward_fn: Optional[GSM8KRewardFunction] = None
        self._data_pipeline: Optional[GSM8KDataPipeline] = None
        self._tokenizer = None

        # 自研推理引擎（backend="custom"时使用）
        self._custom_engine = None

        # 工具
        self._metrics_collector = MetricsCollector()
        self._memory_profiler = MemoryProfiler()
        self._comm_profiler = None
        self._compute_profiler = None
        self._tb_writer = None
        self._metrics_file = None
        self._metrics_jsonl_path: Optional[str] = None
        self._artifact_writer: Optional[ArtifactWriter] = None
        self._eval_snapshots: Dict[str, List[Dict]] = {}
        self._run_started_at = time.time()
        self._last_checkpoint_path: Optional[str] = None
        self._last_mfu: float = 0.0
        self._last_throughput: float = 0.0

    async def initialize(self):
        """初始化所有组件

        - TrainEngine: 模型加载 + TP + LoRA + Optimizer
        - VLLMClient + IPCWeightTransfer: 权重同步通道 (vllm backend)
        - InferenceEngine: 自研推理引擎 (custom backend)
        - GSM8KRewardFunction: 奖励计算
        - GSM8KDataPipeline: 数据加载
        """
        logger.info("Initializing ColocateOrchestrator...")

        effective_grpo_beta = self.config.grpo_beta if self.config.training_mode == "grpo" else 0.0
        # 计算总训练步数（传给 TrainConfig 用于 LR Scheduler）
        total_training_steps = self.config.total_iterations // self.config.gradient_accumulation_steps
        train_config = TrainConfig(
            model_path=self.config.model_path,
            tp_size=self.config.tp_size,
            dp_size=self.config.dp_size,
            max_seq_len=self.config.max_seq_len,
            use_flash_attn=self.config.use_flash_attn,
            attention_backend=self.config.attention_backend,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_target_modules=self.config.lora_target_modules,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            use_amp=self.config.amp_dtype != "fp32",
            amp_dtype=self.config.amp_dtype,
            grpo_num_samples=self.config.num_samples_per_prompt,
            grpo_beta=effective_grpo_beta,
            grpo_clip_eps=self.config.grpo_clip_eps,
            enable_comm_overlap=self.config.enable_comm_overlap,
            zero_stage=self.config.zero_stage,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            min_lr_ratio=self.config.min_lr_ratio,
            total_training_steps=total_training_steps,
        )
        self._train_engine = TrainEngine(train_config)
        self._train_engine.initialize()
        # 绑定 metrics collector
        self._train_engine.metrics_collector = self._metrics_collector
        if self.config.resume_from:
            success = self._train_engine.load_checkpoint(self.config.resume_from)
            if success:
                logger.info(f"Resumed training state from {self.config.resume_from}")
            else:
                logger.warning(f"Failed to resume from {self.config.resume_from}; starting from scratch")
        logger.info(f"TrainEngine initialized: {self._train_engine.trainable_params_count} trainable params")

        if self.config.inference_backend == "custom" and self.config.training_mode != "sft":
            from ..inference.engine import InferenceEngine, InferenceConfig

            infer_config = InferenceConfig(
                max_num_batched_tokens=2048,
                max_num_sequences=256,
                enable_prefix_caching=False,
                enable_cuda_graph=True,
            )
            self._custom_engine = InferenceEngine(infer_config)
            self._custom_engine.initialize()
            logger.info("Custom InferenceEngine initialized")
        elif self.config.inference_backend == "custom":
            logger.info("SFT custom mode uses TrainEngine directly for local preview")
        else:
            self._vllm_client = VLLMClient(
                base_url=self.config.vllm_url,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )
            self._ipc_transfer = IPCWeightTransfer(
                vllm_client=self._vllm_client,
                chunk_size=self.config.ipc_chunk_size,
            )
            logger.info(f"VLLMClient initialized: {self.config.vllm_url}")

        self._reward_fn = GSM8KRewardFunction()

        logger.info(
            "Loading dataset: name=%s, config=%s, split=%s, max_samples=%s",
            self.config.dataset_name,
            self.config.dataset_config,
            self.config.dataset_split,
            self.config.max_data_samples,
        )
        dp_rank = self._train_engine.parallel_ctx.dp_rank if self._train_engine.parallel_ctx else 0
        dp_size = self._train_engine.parallel_ctx.dp_size if self._train_engine.parallel_ctx else 1
        self._data_pipeline = GSM8KDataPipeline(
            tokenizer_name=self.config.model_path,
            dataset_name=self.config.dataset_name,
            dataset_config=self.config.dataset_config,
            split=self.config.dataset_split,
            max_samples=self.config.max_data_samples,
            prompt_template=self.config.prompt_template,
            prompt_field=self.config.prompt_field,
            input_field=self.config.input_field,
            response_field=self.config.response_field,
            answer_field=self.config.answer_field,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
        self._tokenizer = self._data_pipeline.tokenizer
        logger.info(f"Data pipeline loaded: {len(self._data_pipeline)} samples")

        self._memory_profiler.snapshot("after_init")

        if self.config.enable_profiling:
            from ..profiling.comm_profiler import CommProfiler
            from ..profiling.compute_profiler import ComputeProfiler
            self._comm_profiler = CommProfiler(enabled=True)
            self._compute_profiler = ComputeProfiler()
            logger.info("Performance profiling enabled (interval=%d)", self.config.profile_interval)

        os.makedirs(self.config.output_dir, exist_ok=True)

        rank = self._train_engine.parallel_ctx.rank if self._train_engine.parallel_ctx else 0
        world_size = self._train_engine.parallel_ctx.world_size if self._train_engine.parallel_ctx else 1
        self._artifact_writer = ArtifactWriter(
            output_dir=self.config.output_dir,
            rank=rank,
            world_size=world_size,
        )

        self._write_config_snapshot()
        self._metrics_jsonl_path = os.path.join(self.config.output_dir, "metrics.jsonl")
        logger.info(f"Metrics JSONL logging to {self._metrics_jsonl_path} (rank={rank})")

        # TensorBoard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = os.path.join(self.config.output_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard logging to {tb_log_dir}")
        except Exception as e:
            logger.warning(f"TensorBoard not available (skip): {e}")

        await self._transition(OrchestratorState.INIT)
        logger.info("ColocateOrchestrator initialization complete.")

    async def run(self) -> List[IterationMetrics]:
        """运行完整的训推循环

        主循环逻辑:
        for i in range(total_iterations):
            1. 获取batch
            2. INFERRING: 调用vLLM生成K个回复
            3. REWARDING: 计算奖励
            4. TRAINING: GRPO训练步
            5. SLEEPING: vLLM sleep
            6. SYNCING: IPC推送权重
            7. wake up vLLM
            8. 记录metrics

        Returns:
            所有迭代的指标列表
        """
        logger.info(f"Starting training loop: {self.config.total_iterations} iterations")
        batches = list(self._data_pipeline.get_batches(self.config.batch_size))
        batch_idx = 0

        if self.config.eval_num_prompts > 0:
            await self._do_eval(stage="before")

        for iteration in range(self.config.total_iterations):
            iter_start = time.perf_counter()
            iter_metrics = IterationMetrics(iteration=iteration)

            try:
                # 获取当前batch（循环使用数据）
                batch = batches[batch_idx % len(batches)]
                batch_idx += 1

                if self.config.training_mode == "sft":
                    await self._transition(OrchestratorState.TRAINING)
                    self._metrics_collector.start_timer("training")
                    train_metrics = await self._do_sft_training(batch)
                    train_time = self._metrics_collector.stop_timer("training")
                    iter_metrics.state_durations["training"] = train_time
                    iter_metrics.train_metrics = train_metrics

                    if self._tb_writer and train_metrics:
                        self._tb_writer.add_scalar("train/loss", train_metrics.loss, iteration)
                        self._tb_writer.add_scalar("train/grad_norm", train_metrics.grad_norm, iteration)
                        self._tb_writer.add_scalar("train/learning_rate", train_metrics.learning_rate, iteration)
                        self._tb_writer.add_scalar("train/step", train_metrics.step, iteration)

                    if self.config.inference_backend == "custom" and self.config.max_tokens > 0 and self.config.inference_preview_enabled:
                        await self._transition(OrchestratorState.INFERRING)
                        self._metrics_collector.start_timer("inference")
                        await self._do_inference(batch, num_samples=1)
                        infer_time = self._metrics_collector.stop_timer("inference")
                        iter_metrics.state_durations["inferring"] = infer_time
                else:
                    await self._transition(OrchestratorState.INFERRING)
                    self._metrics_collector.start_timer("inference")
                    responses = await self._do_inference(batch)
                    infer_time = self._metrics_collector.stop_timer("inference")
                    iter_metrics.state_durations["inferring"] = infer_time

                    await self._transition(OrchestratorState.REWARDING)
                    self._metrics_collector.start_timer("rewarding")
                    rewards = self._do_rewarding(batch, responses)
                    reward_time = self._metrics_collector.stop_timer("rewarding")
                    iter_metrics.state_durations["rewarding"] = reward_time

                    # 计算平均奖励
                    flat_rewards = [r for group in rewards for r in group]
                    iter_metrics.avg_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0
                    iter_metrics.accuracy = self._reward_fn.accuracy

                    await self._transition(OrchestratorState.TRAINING)
                    self._metrics_collector.start_timer("training")
                    train_metrics = await self._do_training(batch, responses, rewards)
                    train_time = self._metrics_collector.stop_timer("training")
                    iter_metrics.state_durations["training"] = train_time
                    iter_metrics.train_metrics = train_metrics

                    if self._tb_writer and train_metrics:
                        self._tb_writer.add_scalar("train/loss", train_metrics.loss, iteration)
                        self._tb_writer.add_scalar("train/policy_loss", train_metrics.policy_loss, iteration)
                        self._tb_writer.add_scalar("train/kl_divergence", train_metrics.kl_divergence, iteration)
                        self._tb_writer.add_scalar("train/grad_norm", train_metrics.grad_norm, iteration)
                        self._tb_writer.add_scalar("train/reward_mean", iter_metrics.avg_reward, iteration)
                        self._tb_writer.add_scalar("train/accuracy", iter_metrics.accuracy, iteration)

                await self._transition(OrchestratorState.SLEEPING)
                self._metrics_collector.start_timer("sleeping")
                if self.config.inference_backend == "vllm":
                    await self._vllm_client.sleep(level=self.config.sleep_level)
                # custom backend 无需sleep（共用内存，直接更新）
                sleep_time = self._metrics_collector.stop_timer("sleeping")
                iter_metrics.state_durations["sleeping"] = sleep_time

                await self._transition(OrchestratorState.SYNCING)
                self._metrics_collector.start_timer("syncing")
                await self._do_weight_sync()
                sync_time = self._metrics_collector.stop_timer("syncing")
                iter_metrics.state_durations["syncing"] = sync_time

                if self.config.inference_backend == "vllm":
                    await self._vllm_client.wake_up(tags=["weights", "kv_cache"])
                # custom backend 无需wake_up

                # 记录本轮总时间
                iter_metrics.total_time = max(time.perf_counter() - iter_start, 1e-9)
                self.metrics_history.append(iter_metrics)

                # 记录指标
                self._metrics_collector.record("reward_mean", iter_metrics.avg_reward)
                self._metrics_collector.record("accuracy", iter_metrics.accuracy)
                if train_metrics:
                    self._metrics_collector.record("loss", train_metrics.loss)

                self._write_metrics_jsonl(iter_metrics)

                if self.config.training_mode == "sft":
                    logger.info(
                        f"[Iter {iteration}/{self.config.total_iterations}] "
                        f"mode=sft loss={train_metrics.loss:.4f} "
                        f"step={train_metrics.step} "
                        f"time={iter_metrics.total_time:.1f}s"
                    )
                else:
                    logger.info(
                        f"[Iter {iteration}/{self.config.total_iterations}] "
                        f"mode=grpo reward={iter_metrics.avg_reward:.3f} "
                        f"acc={iter_metrics.accuracy:.2%} "
                        f"loss={train_metrics.loss:.4f} "
                        f"time={iter_metrics.total_time:.1f}s"
                    )

                self._save_checkpoint_if_needed(iteration)

                # 显存 Profiling
                if iteration % self.config.memory_profile_interval == 0:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        mem_allocated = _torch.cuda.memory_allocated() / 1e9
                        mem_total = _torch.cuda.get_device_properties(0).total_memory / 1e9
                        utilization = mem_allocated / mem_total if mem_total > 0 else 0
                        self._memory_profiler.snapshot(f"step_{iteration}")
                        if utilization > 0.9:
                            logger.warning(
                                "[Memory] High utilization: %.1f%% (%.1f/%.1f GB)",
                                utilization * 100, mem_allocated, mem_total,
                            )
                        else:
                            logger.info(
                                "[Memory] Step %d: %.1f/%.1f GB (%.0f%%)",
                                iteration, mem_allocated, mem_total, utilization * 100,
                            )
                    else:
                        self._memory_profiler.snapshot(f"step_{iteration}")

                # 通信/计算 Profiling 采样
                if self.config.enable_profiling and iteration % self.config.profile_interval == 0:
                    elapsed = iter_metrics.total_time
                    # 估算 tokens 吞吐量
                    batch_tokens = self.config.batch_size * self.config.max_seq_len
                    tokens_per_sec = batch_tokens / elapsed if elapsed > 0 else 0

                    # 通信占比
                    comm_ratio = 0.0
                    if self._comm_profiler is not None:
                        comm_summary = self._comm_profiler.get_summary()
                        comm_ratio = comm_summary.get("comm_ratio", 0.0)

                    # MFU 估算（Megatron-LM standard）
                    # Full SFT: per_gpu_flops = forward × 3 / (tp × pp)
                    # LoRA: per_gpu_flops = forward × 2 / (tp × pp)（skip dW for frozen params）
                    mfu = 0.0
                    if self._compute_profiler is not None and elapsed > 0:
                        # 从实际模型对象动态提取架构参数
                        model = self._train_engine.model
                        num_layers = getattr(model, 'num_layers', None)
                        hidden_size = getattr(model, 'hidden_size', None)
                        lm_head = model.lm_head if hasattr(model, 'lm_head') else None
                        if lm_head is not None:
                            if hasattr(lm_head, 'original_layer'):
                                lm_head = lm_head.original_layer
                            vocab_size = lm_head.out_features if hasattr(lm_head, 'out_features') else None
                        else:
                            vocab_size = None
                        intermediate_size = None
                        if hasattr(model, 'layers') and len(model.layers) > 0:
                            layer0 = model.layers[0]
                            if hasattr(layer0, 'mlp') and hasattr(layer0.mlp, 'gate_proj'):
                                gate_proj = layer0.mlp.gate_proj
                                if hasattr(gate_proj, 'original_layer'):
                                    gate_proj = gate_proj.original_layer
                                intermediate_size = gate_proj.out_features if hasattr(gate_proj, 'out_features') else None

                        # 任一参数缺失则整体回退到简化公式，与 _write_run_summary 逻辑一致
                        if num_layers is not None and hidden_size is not None \
                                and intermediate_size is not None and vocab_size is not None:
                            # Megatron-LM standard（不含 Attention O(n²) 项）
                            per_layer_flops = 2 * self.config.batch_size * self.config.max_seq_len * (
                                4 * hidden_size * hidden_size
                                + 3 * hidden_size * intermediate_size
                            )
                            forward_flops = num_layers * per_layer_flops
                            forward_flops += 2 * self.config.batch_size * self.config.max_seq_len * hidden_size * vocab_size
                        else:
                            logger.warning("Model arch attributes incomplete for per-step MFU, using simplified formula")
                            model_params = sum(p.numel() for p in model.parameters())
                            forward_flops = 2 * model_params * self.config.max_seq_len * self.config.batch_size

                        # Megatron-LM standard: per_gpu_flops = forward × multiplier / (tp × pp)
                        # Full SFT: 3× (1 forward + 2 backward: dX + dW)
                        # LoRA: 2× (1 forward + 1 backward: dX only, frozen W skips dW)
                        tp_size = max(self.config.tp_size, 1)
                        pp_size = max(getattr(self.config, 'pp_size', 1), 1)
                        is_lora = getattr(self.config, 'lora_rank', 0) > 0
                        training_multiplier = 2 if is_lora else 3
                        per_gpu_flops = forward_flops * training_multiplier / (tp_size * pp_size)

                        # 优先使用 training-only 时间
                        step_time = iter_metrics.state_durations.get("training", elapsed)

                        mfu = self._compute_profiler.compute_mfu(
                            actual_time_s=step_time,
                            flops=int(per_gpu_flops),
                            device="H20",
                            num_devices=1,  # 已经是 per-GPU 值了
                        )

                    logger.info(
                        "[Perf] Step %d: MFU=%.1f%%, comm_ratio=%.1f%%, throughput=%.0f tokens/s",
                        iteration, mfu * 100, comm_ratio * 100, tokens_per_sec,
                    )
                    self._last_mfu = mfu * 100
                    self._last_throughput = tokens_per_sec
                    self._metrics_collector.record("mfu", mfu)
                    self._metrics_collector.record("comm_ratio", comm_ratio)
                    self._metrics_collector.record("throughput_tokens_per_sec", tokens_per_sec)

            except Exception as e:
                await self._handle_error(e, self.state.value)
                # 错误后尝试恢复: 确保vLLM处于wake状态
                if self.config.inference_backend == "vllm" and self._vllm_client:
                    try:
                        await self._vllm_client.wake_up()
                    except Exception:
                        pass

        if self.config.eval_num_prompts > 0:
            await self._do_eval(stage="after")
        self._save_checkpoint(final=True)
        self._save_loss_curve()
        self._write_run_summary()

        # 保存显存时间线
        if self._memory_profiler is not None:
            timeline = self._memory_profiler.get_timeline()
            if timeline:
                timeline_data = [
                    {
                        "timestamp": snap.timestamp,
                        "stage": snap.stage,
                        "allocated_mb": snap.allocated_mb,
                        "reserved_mb": snap.reserved_mb,
                        "peak_mb": snap.peak_mb,
                        "device_id": snap.device_id,
                    }
                    for snap in timeline
                ]
                self._artifact_writer.write_rank_json("memory_timeline.json", timeline_data)
                logger.info("Memory timeline saved (per-rank)")

        # 生成完整显存报告
        try:
            memory_report = self._train_engine.generate_memory_report()
            self._artifact_writer.write_main_json("memory_report.json", memory_report)
            logger.info("Memory report saved (rank0 only)")
        except Exception as e:
            logger.warning("Failed to generate memory report: %s", e)

        await self._transition(OrchestratorState.DONE)
        logger.info("Training loop completed.")
        return self.metrics_history

    async def _transition(self, new_state: OrchestratorState):
        """状态转换，记录耗时

        Args:
            new_state: 目标状态
        """
        now = time.time()
        old_state = self.state
        duration = now - self._state_start_time

        if old_state != new_state:
            logger.debug(f"State transition: {old_state.value} -> {new_state.value} (duration={duration:.2f}s)")

        self.state = new_state
        self._state_start_time = now

    async def _do_inference(self, batch: GRPOBatch, num_samples: Optional[int] = None) -> List[List[str]]:
        """调用推理引擎生成回复

        根据 inference_backend 配置选择:
        - "vllm": 通过HTTP调用vLLM的/v1/completions API
        - "custom": 使用当前训练模型做本地自回归生成

        Args:
            batch: 包含prompts的GRPOBatch

        Returns:
            [batch_size, K] 的回复列表
        """
        if self.config.inference_backend == "custom":
            return self._do_inference_custom(batch, num_samples=num_samples)
        else:
            return await self._do_inference_vllm(batch)

    def _do_inference_custom(self, batch: GRPOBatch, num_samples: Optional[int] = None) -> List[List[str]]:
        """使用当前训练模型生成回复

        对batch中每个prompt，生成K个回复。
        这条路径用于本地验证真实 rollout -> reward -> train 闭环，不依赖 vLLM。
        """
        assert self._train_engine is not None, "TrainEngine not initialized."
        assert self._tokenizer is not None, "Tokenizer not initialized."

        all_responses: List[List[str]] = []
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        sample_count = num_samples or self.config.num_samples_per_prompt

        for prompt_tokens in batch.prompt_tokens:
            responses = []
            for _ in range(sample_count):
                output_tokens = self._train_engine.generate(
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    eos_token_id=eos_token_id,
                )
                text = self._tokenizer.decode(output_tokens, skip_special_tokens=True)
                responses.append(text)
            all_responses.append(responses)

        if all_responses and all_responses[0]:
            preview = all_responses[0][0].replace("\n", "\\n")[:120]
            logger.info(f"Custom rollout sample: {ascii(preview)}")

        return all_responses

    async def _do_sft_training(self, batch: GRPOBatch) -> TrainMetrics:
        """执行 SFT teacher-forcing 训练步。"""
        assert self._tokenizer is not None, "Tokenizer not initialized."
        assert batch.reference_responses is not None, "SFT requires reference responses."

        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        response_tokens: List[List[int]] = []
        for text in batch.reference_responses:
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            if eos_token_id is not None:
                tokens = tokens + [eos_token_id]
            response_tokens.append(tokens)

        metrics = None
        for _ in range(self.config.train_steps_per_iteration):
            metrics = self._train_engine.sft_step(
                prompts_tokens=batch.prompt_tokens,
                response_tokens=response_tokens,
            )
        return metrics

    async def _do_eval(self, stage: str) -> Dict:
        """Evaluate fixed prompts with loss, perplexity, exact match, and token F1."""
        if not self._tokenizer or not self._train_engine:
            return {}

        batches = list(self._data_pipeline.get_batches(self.config.eval_num_prompts))
        if not batches:
            logger.warning("No data available for eval")
            return {}

        batch = batches[0]
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        references = batch.reference_responses or batch.ground_truths
        response_tokens = []
        for text in references:
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            if eos_id is not None:
                tokens = tokens + [eos_id]
            response_tokens.append(tokens)

        eval_loss = None
        perplexity = None
        if batch.reference_responses:
            eval_loss = self._train_engine.eval_sft_loss(batch.prompt_tokens, response_tokens)
            perplexity = math.exp(min(eval_loss, 20.0))

        logger.info("=" * 60)
        logger.info(f"Evaluation ({stage}):")
        logger.info("=" * 60)

        samples = []
        exact_matches = []
        token_f1s = []
        bleu_scores = []

        for i in range(len(batch.prompts)):
            prompt_text = batch.prompts[i]
            prompt_tokens = batch.prompt_tokens[i]
            reference = references[i]

            output_tokens = self._train_engine.generate(
                prompt_tokens=prompt_tokens,
                max_new_tokens=self.config.eval_max_tokens,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=eos_id,
            )
            generated = self._tokenizer.decode(output_tokens, skip_special_tokens=True)
            token_f1 = compute_token_f1(generated, reference)
            bleu = compute_bleu4(generated, reference)
            exact_match = self._normalize_text(generated) == self._normalize_text(reference)
            exact_matches.append(1.0 if exact_match else 0.0)
            token_f1s.append(token_f1)
            bleu_scores.append(bleu)

            sample = {
                "index": i,
                "prompt": prompt_text,
                "generated": generated,
                "reference": reference,
                "generated_tokens": len(output_tokens),
                "exact_match": exact_match,
                "token_f1": token_f1,
                "bleu4": bleu,
            }
            samples.append(sample)

            logger.info(f"--- Sample {i + 1} ---")
            logger.info(f"Prompt:")
            logger.info(f"  {prompt_text[:400]}")
            logger.info(f"Generated ({len(output_tokens)} tokens):")
            logger.info(f"  {generated[:400]}")
            logger.info(f"Reference:")
            logger.info(f"  {reference[:400]}")
            logger.info(f"Metrics: exact_match={exact_match}, token_f1={token_f1:.4f}, bleu4={bleu:.4f}")

            if self._tb_writer:
                tb_step = self._train_engine.step_count
                self._tb_writer.add_text(f"eval_{stage}/sample_{i}/prompt", prompt_text, tb_step)
                self._tb_writer.add_text(f"eval_{stage}/sample_{i}/generated", generated, tb_step)
                self._tb_writer.add_text(f"eval_{stage}/sample_{i}/reference", reference, tb_step)

        metrics = {
            "stage": stage,
            "step": self._train_engine.step_count,
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
            "token_f1": sum(token_f1s) / len(token_f1s) if token_f1s else 0.0,
            "bleu4": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            "num_samples": len(samples),
        }

        logger.info(
            "Eval metrics: loss=%s ppl=%s exact_match=%.4f token_f1=%.4f",
            f"{eval_loss:.4f}" if eval_loss is not None else "n/a",
            f"{perplexity:.4f}" if perplexity is not None else "n/a",
            metrics["exact_match"],
            metrics["token_f1"],
        )

        if self._tb_writer:
            tb_step = self._train_engine.step_count
            if eval_loss is not None:
                self._tb_writer.add_scalar(f"eval/{stage}_loss", eval_loss, tb_step)
                self._tb_writer.add_scalar(f"eval/{stage}_perplexity", perplexity, tb_step)
            self._tb_writer.add_scalar(f"eval/{stage}_exact_match", metrics["exact_match"], tb_step)
            self._tb_writer.add_scalar(f"eval/{stage}_token_f1", metrics["token_f1"], tb_step)

        payload = {"metrics": metrics, "samples": samples}
        self._eval_snapshots[stage] = samples
        self._write_json(os.path.join(self.config.output_dir, f"eval_{stage}.json"), payload)
        if stage == "after" and "before" in self._eval_snapshots:
            self._write_fixed_prompt_comparison()

        logger.info("=" * 60)
        return metrics

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _write_json(self, path: str, payload: Dict):
        """写入全局JSON文件（仅rank0执行）"""
        if self._artifact_writer and not self._artifact_writer.is_main:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_config_snapshot(self):
        self._artifact_writer.write_main_json("config_snapshot.json", asdict(self.config))

    def _load_eval_metrics(self, stage: str) -> Optional[Dict]:
        path = os.path.join(self.config.output_dir, f"eval_{stage}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("metrics")
        except Exception as e:
            logger.warning(f"Failed to read eval metrics from {path}: {e}")
            return None

    def _collect_artifact_paths(self) -> Dict[str, str]:
        candidates = {
            "metrics_jsonl": self._metrics_jsonl_path,
            "loss_curve_csv": os.path.join(self.config.output_dir, "loss_curve.csv"),
            "loss_curve_png": os.path.join(self.config.output_dir, "loss_curve.png"),
            "eval_before": os.path.join(self.config.output_dir, "eval_before.json"),
            "eval_after": os.path.join(self.config.output_dir, "eval_after.json"),
            "fixed_prompt_comparison": os.path.join(self.config.output_dir, "fixed_prompt_comparison.json"),
            "config_snapshot": os.path.join(self.config.output_dir, "config_snapshot.json"),
            "latest_checkpoint": os.path.join(self.config.checkpoint_save_dir, "latest")
            if self.config.checkpoint_save_dir
            else None,
            "final_checkpoint": os.path.join(self.config.checkpoint_save_dir, "final")
            if self.config.checkpoint_save_dir
            else None,
        }
        return {
            key: path
            for key, path in candidates.items()
            if path and os.path.exists(path)
        }

    def _collect_system_info(self) -> Dict:
        info = {
            "python": sys.version.split()[0],
            "platform": os.name,
        }
        try:
            import torch

            info["torch"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_version"] = torch.version.cuda
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
        except Exception as e:
            info["torch_error"] = str(e)
        return info

    def _write_run_summary(self):
        summary = self.get_summary()
        summary["status"] = "completed" if self.metrics_history else "no_data"
        summary["mode"] = self.config.training_mode
        summary["run"] = {
            "started_at_unix": self._run_started_at,
            "ended_at_unix": time.time(),
            "output_dir": self.config.output_dir,
            "resume_from": self.config.resume_from,
            "last_checkpoint_path": self._last_checkpoint_path,
        }
        summary["system"] = self._collect_system_info()
        summary["eval"] = {
            "before": self._load_eval_metrics("before"),
            "after": self._load_eval_metrics("after"),
        }
        summary["artifacts"] = self._collect_artifact_paths()

        try:
            model_params = sum(p.numel() for p in self._train_engine.model.parameters())
            seq_len = self.config.max_seq_len
            batch_size = self.config.batch_size
            tp_size = max(self.config.tp_size, 1)
            pp_size = max(getattr(self.config, 'pp_size', 1), 1)  # 预留 PP
            dp_size = max(self.config.dp_size, 1)

            # step_time: 优先使用 training-only 时间（排除 syncing 等非训练开销）
            if self.metrics_history and self.metrics_history[0].state_durations.get("training") is not None:
                training_times = [m.state_durations.get("training", m.total_time) for m in self.metrics_history]
                avg_step_time = sum(training_times) / len(training_times)
            else:
                avg_step_time = summary["results"]["avg_iteration_time_s"]

            # 从模型获取架构参数
            model = self._train_engine.model
            num_layers = getattr(model, 'num_layers', None)
            hidden_size = getattr(model, 'hidden_size', None)
            lm_head = model.lm_head if hasattr(model, 'lm_head') else None
            if lm_head is not None:
                if hasattr(lm_head, 'original_layer'):
                    lm_head = lm_head.original_layer
                vocab_size = lm_head.out_features if hasattr(lm_head, 'out_features') else None
            else:
                vocab_size = None
            intermediate_size = None
            if hasattr(model, 'layers') and len(model.layers) > 0:
                layer0 = model.layers[0]
                if hasattr(layer0, 'mlp') and hasattr(layer0.mlp, 'gate_proj'):
                    gate_proj = layer0.mlp.gate_proj
                    if hasattr(gate_proj, 'original_layer'):
                        gate_proj = gate_proj.original_layer
                    intermediate_size = gate_proj.out_features if hasattr(gate_proj, 'out_features') else None

            # 如果模型属性不可用，从配置文件推断
            if num_layers is None or hidden_size is None or intermediate_size is None or vocab_size is None:
                logger.warning("Model arch attributes incomplete, falling back to config-based estimation")
                forward_flops = 2 * model_params * seq_len * batch_size
            else:
                # Megatron-LM standard FLOPs 计算（不含 Attention O(n²) 项，与 6*P*B*S 近似对齐）
                #   QKV + O projections: 4 个 linear [H, H] → 8H²
                #   MLP SwiGLU (gate + up + down): 3 个 linear [H, intermediate] → 6 × H × intermediate
                # 注：Megatron 标准不计入 Attention score 的 O(n²) 项（2×S×H），
                #     因为对大模型该项占比极小（<5%），且 Megatron 源码也不含此项。
                per_layer_flops = 2 * batch_size * seq_len * (
                    4 * hidden_size * hidden_size          # QKV + O projections
                    + 3 * hidden_size * intermediate_size  # MLP SwiGLU (gate + up + down)
                )

                forward_flops = num_layers * per_layer_flops
                forward_flops += 2 * batch_size * seq_len * hidden_size * vocab_size  # embedding + lm_head

            device_peak_tflops = 148.0

            # Megatron-LM standard: per_gpu_flops = forward × multiplier / (tp × pp)
            # Full SFT: 3× (1 forward + 2 backward: dX + dW)
            # LoRA: 2× (1 forward + 1 backward: dX only, frozen W skips dW)
            is_lora = getattr(self.config, 'lora_rank', 0) > 0
            training_multiplier = 2 if is_lora else 3
            per_gpu_training_flops = forward_flops * training_multiplier / (tp_size * pp_size)

            actual_tflops = per_gpu_training_flops / avg_step_time / 1e12
            mfu = actual_tflops / device_peak_tflops * 100

            # MFU 物理上限保护：超过 99.9% 说明公式或测量存在系统性偏差
            if mfu > 99.9:
                logger.warning(
                    "MFU=%.1f%% exceeds physical limit, capping at 99.9%%. "
                    "Possible causes: step_time excludes overhead, or FLOPs formula overestimates.",
                    mfu
                )
                mfu = 99.9

            world_size = tp_size * pp_size * dp_size
            tokens_per_second = batch_size * seq_len * world_size / avg_step_time
            samples_per_second = batch_size * world_size / avg_step_time

            summary["performance"] = {
                "mfu_percent": round(mfu, 2),
                "per_gpu_training_flops": int(per_gpu_training_flops),
                "forward_flops_per_step": int(forward_flops),
                "actual_tflops_per_gpu": round(actual_tflops, 2),
                "device_peak_tflops": device_peak_tflops,
                "world_size": world_size,
                "tp_size": tp_size,
                "pp_size": pp_size,
                "dp_size": dp_size,
                "tokens_per_second": round(tokens_per_second, 0),
                "samples_per_second": round(samples_per_second, 2),
                "avg_step_time_s": round(avg_step_time, 4),
                "model_params": model_params,
                "model_arch": {
                    "num_layers": num_layers,
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "vocab_size": vocab_size,
                    "seq_len": seq_len,
                },
                "note": (
                    f"Megatron-LM standard: MFU = (forward_flops × {training_multiplier}) / (tp×pp) / (step_time × peak). "
                    f"H20 BF16 peak=148T. world_size={world_size} "
                    f"(dp={dp_size}, tp={tp_size}, pp={pp_size})"
                ),
            }

            logger.info(
                "Performance: MFU=%.2f%%, tokens/s=%d, samples/s=%.1f, actual=%.1f TFLOPS/GPU, step_time=%.3fs",
                mfu, tokens_per_second, samples_per_second, actual_tflops, avg_step_time
            )

            summary["performance"]["primary_metric"] = "tokens_per_second" if is_lora else "mfu"

            if is_lora:
                summary["performance"]["mfu_note"] = (
                    "LoRA: 2×forward FLOPs (forward + backward_dX, skip dW for frozen params). "
                    "Directly comparable with other LoRA runs."
                )
            else:
                summary["performance"]["mfu_note"] = (
                    "Megatron-LM standard: 3×forward FLOPs / (tp×pp) / (step_time × peak). Directly comparable."
                )
        except Exception as e:
            logger.warning("Failed to compute MFU: %s", e)

        self._artifact_writer.write_main_json("run_summary.json", summary)
        logger.info("Run summary saved (rank0 only)")

    def _write_fixed_prompt_comparison(self):
        before = self._eval_snapshots.get("before", [])
        after = self._eval_snapshots.get("after", [])
        comparisons = []
        for b, a in zip(before, after):
            comparisons.append({
                "index": b["index"],
                "prompt": b["prompt"],
                "reference": b["reference"],
                "before_generated": b["generated"],
                "after_generated": a["generated"],
                "before_token_f1": b["token_f1"],
                "after_token_f1": a["token_f1"],
                "before_exact_match": b["exact_match"],
                "after_exact_match": a["exact_match"],
            })
        self._write_json(
            os.path.join(self.config.output_dir, "fixed_prompt_comparison.json"),
            {"samples": comparisons},
        )

    def _write_metrics_jsonl(self, iter_metrics: IterationMetrics):
        if self._artifact_writer is None:
            return
        train = iter_metrics.train_metrics
        row = {
            "iteration": iter_metrics.iteration,
            "mode": self.config.training_mode,
            "total_time_s": iter_metrics.total_time,
            "state_durations": iter_metrics.state_durations,
            "avg_reward": iter_metrics.avg_reward,
            "accuracy": iter_metrics.accuracy,
            "loss": train.loss if train else None,
            "policy_loss": train.policy_loss if train else None,
            "kl_divergence": train.kl_divergence if train else None,
            "grad_norm": train.grad_norm if train else None,
            "learning_rate": train.learning_rate if train else None,
            "optimizer_step": train.step if train else None,
        }
        # 如果 profiling 启用且有数据
        if self.config.enable_profiling and hasattr(self, '_last_mfu'):
            row["mfu_percent"] = self._last_mfu
            row["throughput_tokens_per_sec"] = self._last_throughput
        if train and hasattr(train, 'comm_time_ms') and train.comm_time_ms is not None:
            row["comm_time_ms"] = train.comm_time_ms
        self._artifact_writer.write_rank_jsonl("metrics.jsonl", row)

    def _save_checkpoint_if_needed(self, iteration: int):
        if not self.config.checkpoint_save_enabled:
            if iteration == 0:
                logger.info("Checkpoint saving disabled (checkpoint.save_enabled=false)")
            return
        interval = self.config.checkpoint_save_interval
        if interval <= 0:
            return
        if (iteration + 1) % interval == 0:
            self._save_checkpoint(iteration=iteration)

    def _save_checkpoint(self, iteration: Optional[int] = None, final: bool = False):
        if not self.config.checkpoint_save_enabled:
            if final:
                logger.info("Skipping final checkpoint (checkpoint.save_enabled=false)")
            return
        if self._train_engine is None:
            return
        # 仅 rank0 保存 checkpoint（LoRA 权重各 rank 相同）
        if self._artifact_writer and not self._artifact_writer.is_main:
            return
        save_dir = self.config.checkpoint_save_dir
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        if final:
            path = os.path.join(save_dir, "final")
        elif iteration is not None:
            path = os.path.join(save_dir, f"iter_{iteration + 1}")
        else:
            path = os.path.join(save_dir, "latest")
        self._train_engine.save_checkpoint(path)
        self._train_engine.save_checkpoint(os.path.join(save_dir, "latest"))
        self._last_checkpoint_path = path
        logger.info(f"Checkpoint saved: {path}")

    def _save_loss_curve(self):
        if not self._artifact_writer.is_main:
            return
        rows = []
        for item in self.metrics_history:
            if item.train_metrics is None:
                continue
            rows.append((item.iteration, item.train_metrics.loss, item.train_metrics.step))

        if not rows:
            return

        csv_text = "iteration,loss,optimizer_step\n"
        for iteration, loss, step in rows:
            csv_text += f"{iteration},{loss},{step}\n"
        self._artifact_writer.write_main_text("loss_curve.csv", csv_text)
        logger.info("Loss curve CSV saved")

        try:
            import matplotlib.pyplot as plt
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            plt.figure(figsize=(8, 4.5))
            plt.plot(xs, ys, marker="o", linewidth=1.5)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            png_path = os.path.join(self.config.output_dir, "loss_curve.png")
            plt.savefig(png_path, dpi=160)
            plt.close()
            logger.info(f"Loss curve plot saved to {png_path}")
        except Exception as e:
            logger.warning(f"Could not render loss_curve.png; CSV is still available: {e}")

    async def _do_inference_vllm(self, batch: GRPOBatch) -> List[List[str]]:
        """通过vLLM HTTP API生成回复（原有逻辑）"""
        import httpx

        all_responses: List[List[str]] = []
        client = self._vllm_client._get_client()

        for prompt in batch.prompts:
            # 调用vLLM completions API
            payload = {
                "model": self.config.model_path,
                "prompt": prompt,
                "n": self.config.num_samples_per_prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": 0.95,
            }

            resp = await client.post("/v1/completions", json=payload)
            resp.raise_for_status()
            result = resp.json()

            # 提取K个回复
            completions = [choice["text"] for choice in result["choices"]]
            all_responses.append(completions)

        return all_responses

    def _do_rewarding(self, batch: GRPOBatch, responses: List[List[str]]) -> List[List[float]]:
        """计算奖励

        Args:
            batch: 包含ground_truths的GRPOBatch
            responses: [batch_size, K] 的回复列表

        Returns:
            [batch_size, K] 的奖励列表
        """
        all_rewards: List[List[float]] = []

        for i, response_group in enumerate(responses):
            gt = batch.ground_truths[i]
            # 对同一prompt的K个回复计算奖励
            rewards = self._reward_fn(
                responses=response_group,
                ground_truths=[gt] * len(response_group),
            )
            all_rewards.append(rewards)

        return all_rewards

    async def _do_training(
        self,
        batch: GRPOBatch,
        responses: List[List[str]],
        rewards: List[List[float]],
    ) -> TrainMetrics:
        """执行GRPO训练步

        Args:
            batch: GRPOBatch（含prompt_tokens）
            responses: [batch_size, K] 回复文本
            rewards: [batch_size, K] 奖励分数

        Returns:
            TrainMetrics
        """
        assert self._tokenizer is not None, "Tokenizer not initialized."

        responses_group_tokens: List[List[List[int]]] = []
        for response_group in responses:
            group_tokens = []
            for text in response_group:
                tokens = self._tokenizer.encode(text, add_special_tokens=False)
                group_tokens.append(tokens)
            responses_group_tokens.append(group_tokens)

        # 执行训练步
        metrics = None
        for _ in range(self.config.train_steps_per_iteration):
            metrics = self._train_engine.grpo_step(
                prompts_tokens=batch.prompt_tokens,
                responses_group=responses_group_tokens,
                rewards_group=rewards,
            )

        return metrics

    async def _do_weight_sync(self):
        """权重同步: export → 推送到推理引擎

        根据 backend 选择不同的同步策略:
        - vllm: 导出权重 → IPC推送到vLLM
        - custom: 直接调用 engine.on_weights_updated()

        注意: sleep已在调用前执行，wake_up在调用后执行。
        """
        if self.config.inference_backend == "custom":
            # 自研引擎: 直接通知权重更新（清除Prefix Cache + CUDA Graph）
            if self._custom_engine is None:
                self._memory_profiler.snapshot(f"after_sync_step{self._train_engine.step_count}")
                logger.info("Custom backend uses TrainEngine weights directly; no sync needed")
                return
            self._custom_engine.on_weights_updated()
            self._memory_profiler.snapshot(f"after_sync_step{self._train_engine.step_count}")
            logger.info("Custom engine notified of weight update (caches invalidated)")
            return

        # vLLM后端: 优化后的IPC权重同步流程
        # 优化: GPU 上直接 export + merge LoRA，避免 GPU→CPU→GPU 往返拷贝
        import torch

        sync_t0 = time.perf_counter()

        # 尝试 GPU 直出路径（省去 2 次 CPU 中转拷贝）
        device = torch.device("cuda:0")
        try:
            # 优化路径: export_weights(device="cuda") 直接输出 GPU tensors
            # LoRA merge 也在 GPU 上完成，无需 CPU 中转
            export_t0 = time.perf_counter()
            gpu_state_dict = self._train_engine.export_weights(device="cuda")
            export_time = time.perf_counter() - export_t0
            logger.info(f"[WeightSync] GPU-direct export completed in {export_time:.3f}s "
                        f"({len(gpu_state_dict)} params)")
        except torch.cuda.OutOfMemoryError:
            logger.warning("[WeightSync] GPU OOM during direct export, falling back to CPU intermediate path")
            torch.cuda.empty_cache()
            export_t0 = time.perf_counter()
            state_dict = self._train_engine.export_weights(device="cpu")
            gpu_state_dict = {k: v.to(device) for k, v in state_dict.items()}
            del state_dict
            export_time = time.perf_counter() - export_t0
            logger.info(f"[WeightSync] CPU-fallback export completed in {export_time:.3f}s "
                        f"({len(gpu_state_dict)} params)")

        # 构建IPC handles（输入已是 GPU tensor，无额外拷贝）
        ipc_t0 = time.perf_counter()
        handles = self._ipc_transfer.build_ipc_handles(gpu_state_dict)
        ipc_time = time.perf_counter() - ipc_t0

        # 分步完成传输协议
        transfer_t0 = time.perf_counter()
        await self._vllm_client.init_weight_transfer()
        await self._vllm_client.start_weight_update(is_checkpoint_format=True)

        # 分块发送
        chunks = self._ipc_transfer._chunk_handles(handles)
        for chunk in chunks:
            payload = self._ipc_transfer._build_http_payload(chunk)
            await self._vllm_client.update_weights(
                payload, weight_version=self._ipc_transfer.weight_version
            )

        await self._vllm_client.finish_weight_update()
        transfer_time = time.perf_counter() - transfer_t0

        # 释放GPU上的临时权重
        self._ipc_transfer.release_refs()
        del gpu_state_dict

        total_sync_time = time.perf_counter() - sync_t0
        self._memory_profiler.snapshot(f"after_sync_step{self._train_engine.step_count}")
        logger.info(
            f"[WeightSync] Completed (version={self._ipc_transfer.weight_version}) "
            f"total={total_sync_time:.3f}s "
            f"[export={export_time:.3f}s, ipc_build={ipc_time:.3f}s, transfer={transfer_time:.3f}s]"
        )

    async def _handle_error(self, error: Exception, stage: str):
        """错误处理与恢复

        策略:
        - 记录错误
        - 转入ERROR状态
        - 尝试恢复（如sleep失败→重试wake_up）

        Args:
            error: 捕获的异常
            stage: 出错时的阶段名
        """
        await self._transition(OrchestratorState.ERROR)
        logger.error(f"Error in stage '{stage}': {error}", exc_info=True)

        # 对于sync阶段的错误，尝试唤醒vLLM
        if stage in ("sleeping", "syncing") and self._vllm_client is not None:
            for retry in range(self.config.max_retries):
                try:
                    logger.info(f"Recovery attempt {retry + 1}: waking up vLLM...")
                    await self._vllm_client.wake_up()
                    logger.info("Recovery successful: vLLM woken up")
                    break
                except Exception as wake_err:
                    logger.warning(f"Wake-up retry {retry + 1} failed: {wake_err}")
                    await asyncio.sleep(self.config.retry_delay)

    def get_summary(self) -> dict:
        """获取运行总结（用于Benchmark报告）

        Returns:
            包含完整运行统计的字典
        """
        if not self.metrics_history:
            return {"status": "no_data"}

        rewards = [m.avg_reward for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history]
        total_times = [m.total_time for m in self.metrics_history]

        summary = {
            "config": {
                "model_path": self.config.model_path,
                "training_mode": self.config.training_mode,
                "inference_backend": self.config.inference_backend,
                "dataset_name": self.config.dataset_name,
                "dataset_config": self.config.dataset_config,
                "dataset_split": self.config.dataset_split,
                "max_data_samples": self.config.max_data_samples,
                "total_iterations": self.config.total_iterations,
                "batch_size": self.config.batch_size,
                "num_samples_per_prompt": self.config.num_samples_per_prompt,
                "max_seq_len": self.config.max_seq_len,
                "max_tokens": self.config.max_tokens,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "lora_target_modules": self.config.lora_target_modules,
                "learning_rate": self.config.learning_rate,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "grpo_beta": self.config.grpo_beta,
                "grpo_clip_eps": self.config.grpo_clip_eps,
                "checkpoint_save_dir": self.config.checkpoint_save_dir,
                "checkpoint_save_interval": self.config.checkpoint_save_interval,
            },
            "results": {
                "completed_iterations": len(self.metrics_history),
                "final_accuracy": accuracies[-1] if accuracies else 0,
                "final_avg_reward": rewards[-1] if rewards else 0,
                "best_accuracy": max(accuracies) if accuracies else 0,
                "avg_iteration_time_s": sum(total_times) / len(total_times) if total_times else 0,
                "total_time_s": sum(total_times),
            },
            "stage_avg_durations": {},
            "memory": {
                "peak_mb": self._memory_profiler.peak_memory_mb(),
            },
            "metrics_summary": self._metrics_collector.get_summary(),
        }

        # 计算各阶段平均耗时
        stage_totals: Dict[str, List[float]] = {}
        for m in self.metrics_history:
            for stage, dur in m.state_durations.items():
                if stage not in stage_totals:
                    stage_totals[stage] = []
                stage_totals[stage].append(dur)

        for stage, durations in stage_totals.items():
            summary["stage_avg_durations"][stage] = sum(durations) / len(durations)

        return summary

    async def shutdown(self):
        """清理资源"""
        if self._artifact_writer:
            self._artifact_writer.close()
            self._artifact_writer = None
        if self._metrics_file:
            self._metrics_file.close()
            self._metrics_file = None
        if self._tb_writer:
            self._tb_writer.close()
        if self._vllm_client:
            await self._vllm_client.close()
        logger.info("ColocateOrchestrator shut down.")
