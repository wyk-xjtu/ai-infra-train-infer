"""
训练引擎 - 集成Mini-Megatron TP + Mini-ZeRO + LoRA + GRPO

职责:
1. 模型加载: 从HuggingFace checkpoint加载，用TP切分
2. LoRA注入: 对指定层添加LoRA adapter
3. GRPO训练: 实现Group Relative Policy Optimization
4. 权重导出: 合并LoRA后导出给推理侧

使用方式:
    engine = TrainEngine(model_path="Qwen/Qwen3-4B", tp_size=2, lora_rank=16)
    engine.initialize()
    
    for batch in dataloader:
        # GRPO训练步
        metrics = engine.grpo_step(prompts, responses_group, rewards_group)
    
    # 导出权重给推理
    merged_weights = engine.export_weights()
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from ..distributed import ParallelContext, ParallelTransformerModel, ZeROOptimizer, ZeROStage2Optimizer, ZeROStage3Optimizer
from ..distributed.lora import apply_lora, merge_lora_weights, get_lora_state_dict
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector


logger = get_logger("engines.train")


@dataclass
class TrainConfig:
    model_path: str
    tp_size: int = 1
    dp_size: int = 1
    max_seq_len: int = 2048
    use_flash_attn: bool = False
    attention_backend: str = "sdpa"  # "standard" / "sdpa" / "flash_attn"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bf16"  # 支持 "fp32" / "bf16" / "fp16"
    gradient_accumulation_steps: int = 4
    # GRPO参数
    grpo_num_samples: int = 4   # 每个prompt生成K个回复
    grpo_beta: float = 0.1      # KL散度系数
    grpo_clip_eps: float = 0.2  # PPO-style clipping
    # Tokenizer相关
    pad_token_id: int = 0       # 默认值，initialize时从tokenizer覆盖
    # ZeRO 配置
    zero_stage: int = 1                 # ZeRO 阶段: 1=优化器分片, 2=优化器+梯度分片, 3=全分片
    # 通信-计算 overlap
    enable_comm_overlap: bool = False   # 当 tp_size > 1 时启用通信与计算重叠
    min_prompt_len: int = 16            # 序列截断时保留的最少prompt token数
    min_advantage_scale: float = 0.01   # K=1退化时的minimum advantage
    # LR Scheduler
    lr_scheduler_type: str = "cosine"   # "cosine" / "linear" / "constant"
    warmup_steps: int = 100             # warmup 步数
    min_lr_ratio: float = 0.1           # 最终 lr = learning_rate * min_lr_ratio
    total_training_steps: int = 0       # 由外部传入（orchestrator 计算）


@dataclass
class TrainMetrics:
    loss: float
    policy_loss: float
    kl_divergence: float
    grad_norm: float
    learning_rate: float
    step: int


class TrainEngine:
    """集成Mini-Megatron的训练引擎，支持GRPO算法"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.parallel_ctx: Optional[ParallelContext] = None
        self.model: Optional[ParallelTransformerModel] = None
        self.ref_model: Optional[ParallelTransformerModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[GradScaler] = None
        self._amp_enabled = False
        self.step_count = 0
        self._accumulation_count = 0
        self.metrics_collector: Optional[MetricsCollector] = None
        self.lr_scheduler = None
        self._old_log_probs_buffer: Optional[torch.Tensor] = None  # GRPO importance sampling buffer

    def initialize(self, parallel_context: Optional[ParallelContext] = None):
        """初始化模型、LoRA、优化器

        步骤:
        1. 初始化或接收ParallelContext
        2. 创建ParallelTransformerModel并加载HF权重
        3. 冻结原始参数，注入LoRA
        4. 创建ZeROOptimizer（仅优化LoRA参数）
        5. 配置AMP autocast
        """
        if parallel_context is not None:
            self.parallel_ctx = parallel_context
        else:
            self.parallel_ctx = ParallelContext.init_distributed(
                tp_size=self.config.tp_size,
                dp_size=self.config.dp_size,
                backend="nccl",
            )

        model_config = self._load_model_config(self.config.model_path)

        # 通信-计算 overlap: 仅在 tp_size > 1 且配置启用时创建 StreamManager
        stream_manager = None
        enable_overlap = self.config.enable_comm_overlap and self.parallel_ctx.tp_size > 1
        if enable_overlap:
            from ..distributed.stream_manager import StreamManager
            device = torch.device(f"cuda:{self.parallel_ctx.local_rank}") if torch.cuda.is_available() else None
            stream_manager = StreamManager(device=device)
            logger.info("StreamManager created for comm-compute overlap (tp_size=%d)", self.parallel_ctx.tp_size)

        self.model = ParallelTransformerModel(
            vocab_size=model_config["vocab_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            num_kv_heads=model_config["num_kv_heads"],
            head_dim=model_config["head_dim"],
            intermediate_size=model_config["intermediate_size"],
            parallel_context=self.parallel_ctx,
            max_position_embeddings=model_config.get("max_position_embeddings", 32768),
            rms_norm_eps=model_config.get("rms_norm_eps", 1e-6),
            rope_theta=model_config.get("rope_theta", 1000000.0),
            tie_word_embeddings=model_config.get("tie_word_embeddings", False),
            use_flash_attn=self.config.use_flash_attn,
            attention_backend=self.config.attention_backend,
            enable_comm_overlap=enable_overlap,
            stream_manager=stream_manager,
        )

        # 加载HF checkpoint权重
        from ..distributed.tensor_parallel import load_from_hf_checkpoint
        load_from_hf_checkpoint(self.model, self.config.model_path, self.parallel_ctx)

        # 将模型移动到对应GPU
        device = torch.device(f"cuda:{self.parallel_ctx.local_rank}")
        logger.info(
            "CUDA status: available=%s, device_count=%s, target_device=%s",
            torch.cuda.is_available(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
            device if torch.cuda.is_available() else "cpu",
        )
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            logger.info("Base model moved to %s", device)
        else:
            logger.warning("CUDA is not visible to this process; training will run on CPU.")

        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            logger.info("pad_token_id loaded from tokenizer: %d", self.config.pad_token_id)
        except Exception as e:
            logger.warning("Could not load pad_token_id from tokenizer, using default 0: %s", e)

        # 深拷贝当前模型（apply LoRA之前的状态）作为参考策略
        # ref_model不需要梯度，不注入LoRA
        if self.config.grpo_beta > 0:
            import copy
            self.ref_model = copy.deepcopy(self.model)
            # deepcopy后显式绑定设备，防止跨设备计算错误
            # 某些PyTorch版本/设备组合下deepcopy可能不保留device
            if torch.cuda.is_available():
                ref_device = next(self.model.parameters()).device
                self.ref_model = self.ref_model.to(ref_device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.info(f"ref_model created on {next(self.ref_model.parameters()).device}, "
                        f"memory: {sum(p.numel() for p in self.ref_model.parameters()) * 2 / 1e6:.1f}MB")
        else:
            self.ref_model = None

        if self.config.lora_rank > 0:
            lora_config = {
                "rank": self.config.lora_rank,
                "alpha": self.config.lora_alpha,
                "target_modules": self.config.lora_target_modules,
                "dropout": 0.0,
            }
            self.model = apply_lora(self.model, lora_config, self.parallel_ctx)
        else:
            logger.info("LoRA disabled (rank=0), full parameter training")

        # apply_lora 会新建 LoRA 参数；如果 base model 已经在 GPU 上，
        # 这些新参数默认仍在 CPU。这里再次迁移整个模型，保证原始层与 LoRA 层设备一致。
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        self._assert_single_device()
        self._log_device_state("after_lora")

        # autocast 只影响计算精度，不改变参数存储dtype；必须显式转换才能减少显存占用
        # LoRA 训练中 base model 冻结参数不参与梯度更新，可安全降精度存储
        # 全量 SFT (lora_rank=0) 时，所有参数都 requires_grad=True，也需要转精度以省显存
        if self.config.amp_dtype != "fp32" and torch.cuda.is_available():
            cast_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
            frozen_count = 0
            frozen_numel = 0
            trainable_cast_count = 0
            trainable_cast_numel = 0
            for param in self.model.parameters():
                if not param.requires_grad:
                    param.data = param.data.to(cast_dtype)
                    frozen_count += 1
                    frozen_numel += param.numel()
                elif self.config.lora_rank == 0:
                    # 全量 SFT: 所有参数转低精度（用 autocast 保证计算精度）
                    param.data = param.data.to(cast_dtype)
                    trainable_cast_count += 1
                    trainable_cast_numel += param.numel()
            if frozen_count > 0:
                logger.info(
                    "Frozen parameters cast to %s: %d tensors (%d params), "
                    "memory savings: ~%.1f GB",
                    self.config.amp_dtype, frozen_count, frozen_numel,
                    frozen_numel * 2 / 1e9,
                )
            if trainable_cast_count > 0:
                logger.info(
                    "Full SFT: all trainable params cast to %s: %d tensors (%d params), "
                    "memory savings: ~%.1f GB",
                    self.config.amp_dtype, trainable_cast_count, trainable_cast_numel,
                    trainable_cast_numel * 2 / 1e9,
                )
            # ref_model 也转为低精度（不参与训练，纯推理用）
            if self.ref_model is not None:
                for param in self.ref_model.parameters():
                    param.data = param.data.to(cast_dtype)
                logger.info("ref_model parameters also cast to %s", self.config.amp_dtype)

        # ZeRO作用域选择:
        # - dp_size > 1: ZeRO 在 dp_group 上操作（梯度在 DP rank 间同步）
        # - dp_size == 1 且 tp_size > 1: ZeRO 在 tp_group 上操作（向后兼容）
        # - 都是 1: 使用普通 AdamW
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.parallel_ctx.dp_size > 1 or self.parallel_ctx.tp_size > 1:
            # 选择 ZeRO stage
            if self.config.zero_stage == 3:
                self.optimizer = ZeROStage3Optimizer(
                    params=iter(trainable_params),
                    optimizer_class=torch.optim.AdamW,
                    parallel_context=self.parallel_ctx,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                zero_group = "dp_group" if self.parallel_ctx.dp_size > 1 else "tp_group"
                logger.info(
                    "Optimizer initialized: Mini-ZeRO Stage-3 AdamW on %s (size=%s)",
                    zero_group,
                    self.parallel_ctx.dp_size if self.parallel_ctx.dp_size > 1 else self.parallel_ctx.tp_size,
                )
            elif self.config.zero_stage == 2:
                self.optimizer = ZeROStage2Optimizer(
                    params=iter(trainable_params),
                    optimizer_class=torch.optim.AdamW,
                    parallel_context=self.parallel_ctx,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                zero_group = "dp_group" if self.parallel_ctx.dp_size > 1 else "tp_group"
                logger.info(
                    "Optimizer initialized: Mini-ZeRO Stage-2 AdamW on %s (size=%s)",
                    zero_group,
                    self.parallel_ctx.dp_size if self.parallel_ctx.dp_size > 1 else self.parallel_ctx.tp_size,
                )
            else:
                # ZeRO-1 (默认)
                self.optimizer = ZeROOptimizer(
                    params=iter(trainable_params),
                    optimizer_class=torch.optim.AdamW,
                    parallel_context=self.parallel_ctx,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                zero_group = "dp_group" if self.parallel_ctx.dp_size > 1 else "tp_group"
                logger.info(
                    "Optimizer initialized: Mini-ZeRO Stage-1 AdamW on %s (size=%s)",
                    zero_group,
                    self.parallel_ctx.dp_size if self.parallel_ctx.dp_size > 1 else self.parallel_ctx.tp_size,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            logger.info("Optimizer initialized: AdamW (single-process local mode)")

        self.lr_scheduler = None
        if self.config.lr_scheduler_type != "constant" and self.config.total_training_steps > 0:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            total_steps = self.config.total_training_steps
            warmup_steps = min(self.config.warmup_steps, total_steps)

            # 确定 scheduler 作用的 optimizer 对象
            base_optimizer = (
                self.optimizer.local_optimizer
                if isinstance(self.optimizer, (ZeROOptimizer, ZeROStage2Optimizer, ZeROStage3Optimizer))
                else self.optimizer
            )

            warmup_scheduler = LinearLR(
                base_optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )

            if self.config.lr_scheduler_type == "cosine":
                # Cosine decay
                decay_scheduler = CosineAnnealingLR(
                    base_optimizer,
                    T_max=max(total_steps - warmup_steps, 1),
                    eta_min=self.config.learning_rate * self.config.min_lr_ratio,
                )
            else:
                # Linear decay
                decay_scheduler = LinearLR(
                    base_optimizer,
                    start_factor=1.0,
                    end_factor=self.config.min_lr_ratio,
                    total_iters=max(total_steps - warmup_steps, 1),
                )

            # 组合 warmup + decay
            self.lr_scheduler = SequentialLR(
                base_optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )
            logger.info(
                "LR Scheduler: %s, warmup=%d, total_steps=%d, min_lr=%.2e",
                self.config.lr_scheduler_type, warmup_steps, total_steps,
                self.config.learning_rate * self.config.min_lr_ratio,
            )

        # ZeRO optimizer already owns gradient synchronization; keep AMP in autocast-only mode.
        # torch.amp.autocast 在新版 PyTorch 中必须显式指定 device_type。
        if self.config.amp_dtype == "fp32":
            self._amp_enabled = False
            self._amp_dtype = torch.float32
        else:
            self._amp_enabled = torch.cuda.is_available()
            self._amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16

        self.scaler = None  # 保持无 GradScaler（bf16 不需要 loss scaling）
        if self._amp_enabled:
            logger.info(f"AMP enabled: dtype={self.config.amp_dtype}")
        else:
            logger.info("AMP disabled (fp32 mode)")

        self.model.train()

    def _truncate_prompt_response(
        self,
        prompt: List[int],
        response: List[int],
    ) -> Tuple[List[int], List[int]]:
        """截断prompt+response使总长不超过max_seq_len

        策略：
        1. 优先保证response完整（response是训练目标）
        2. 如果response本身超长，截断response并保留最少prompt（至少16 tokens）
        3. 如果prompt+response超长，截断prompt尾部
        """
        max_seq_len = self.config.max_seq_len
        # min_prompt_len 不能超过实际 prompt 长度，否则短 prompt 会被扩成无效切片。
        min_prompt_len = min(self.config.min_prompt_len, len(prompt))

        total = len(prompt) + len(response)

        if total <= max_seq_len:
            return prompt, response  # 不需要截断

        # response超长：截断response，保留最少prompt
        if len(response) >= max_seq_len - min_prompt_len:
            truncated_response = response[:max_seq_len - min_prompt_len]
            truncated_prompt = prompt[:min_prompt_len]
            return truncated_prompt, truncated_response

        # 正常情况：截断prompt尾部，保留完整response
        available_for_prompt = max_seq_len - len(response)
        truncated_prompt = prompt[:available_for_prompt]
        return truncated_prompt, response

    def _log_device_state(self, stage: str):
        """Log the real model device and CUDA memory seen by this process."""
        param = next(self.model.parameters())
        trainable_numel = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_numel = sum(p.numel() for p in self.model.parameters())

        logger.info(
            "Model device [%s]: first_param=%s, dtype=%s, trainable=%s / total=%s",
            stage,
            param.device,
            param.dtype,
            f"{trainable_numel:,}",
            f"{total_numel:,}",
        )

        if torch.cuda.is_available():
            memory_device = param.device if param.is_cuda else torch.device("cuda:0")
            logger.info(
                "CUDA memory [%s]: allocated=%.1f MB, reserved=%.1f MB, peak=%.1f MB",
                stage,
                torch.cuda.memory_allocated(memory_device) / (1024**2),
                torch.cuda.memory_reserved(memory_device) / (1024**2),
                torch.cuda.max_memory_allocated(memory_device) / (1024**2),
            )

    def _assert_single_device(self):
        """初始化阶段检查参数设备一致性，避免前向时才暴露 CUDA/CPU 混用错误。"""
        param_devices = {p.device for p in self.model.parameters()}
        buffer_devices = {b.device for b in self.model.buffers()}
        devices = param_devices | buffer_devices
        if len(devices) > 1:
            details = ", ".join(str(d) for d in sorted(devices, key=str))
            raise RuntimeError(f"Model parameters/buffers are split across devices: {details}")

    def _load_model_config(self, model_path: str) -> Dict:
        """从HuggingFace模型目录加载配置

        Raises:
            FileNotFoundError: config.json不存在
            ValueError: config.json缺少必要字段
        """
        import json
        import os

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model config not found: {config_path}. "
                f"Please provide a valid HuggingFace model directory."
            )

        with open(config_path, "r") as f:
            hf_config = json.load(f)

        # 校验必要字段存在
        required_keys = {
            "hidden_size", "num_attention_heads", "num_hidden_layers",
            "intermediate_size", "vocab_size",
        }
        missing_keys = required_keys - set(hf_config.keys())
        if missing_keys:
            raise ValueError(
                f"config.json is missing required fields: {sorted(missing_keys)}. "
                f"Cannot safely initialize model with unknown architecture parameters."
            )

        return {
            "vocab_size": hf_config["vocab_size"],
            "hidden_size": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
            "head_dim": hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"]),
            "intermediate_size": hf_config["intermediate_size"],
            "max_position_embeddings": hf_config.get("max_position_embeddings", 32768),
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
        }

    def grpo_step(
        self,
        prompts_tokens: List[List[int]],
        responses_group: List[List[List[int]]],  # [batch, K, seq_len]
        rewards_group: List[List[float]],         # [batch, K]
    ) -> TrainMetrics:
        """GRPO训练步骤

        GRPO (Group Relative Policy Optimization):
        1. 对每个prompt的K个回复，计算组内相对优势:
           advantage_i = (reward_i - mean(rewards)) / std(rewards)
        2. 计算当前策略的log_prob
        3. Policy gradient loss:
           L = -mean(advantage * log_prob)
        4. 可选: 加KL正则项防止偏离太远

        Args:
            prompts_tokens: batch个prompt的token ids
            responses_group: 每个prompt对应K个response的token ids
            rewards_group: 每个prompt对应K个response的reward分数

        Returns:
            TrainMetrics: 训练指标
        """
        device = next(self.model.parameters()).device
        batch_size = len(prompts_tokens)
        num_samples = len(responses_group[0])  # K

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0

        # 对每个prompt的K个response计算GRPO loss
        all_input_ids = []
        all_labels = []
        all_advantages = []

        for i in range(batch_size):
            prompt = prompts_tokens[i]
            rewards = rewards_group[i]

            # 计算组内相对优势 (Group Relative Advantage)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            reward_mean = rewards_tensor.mean()
            reward_std = rewards_tensor.std(unbiased=False)
            # 避免除以0
            if reward_std < 1e-8:
                # K=1或所有reward相同时，使用绝对reward信号
                # baseline设为0（假设reward范围是[-1, 1]）
                advantages = rewards_tensor.clone()
                # 如果所有advantage的绝对值都<epsilon，设为微小正值避免完全无梯度
                # 使用配置化的min_advantage_scale
                if advantages.abs().max() < 1e-8:
                    advantages = torch.ones_like(rewards_tensor) * self.config.min_advantage_scale
            else:
                advantages = (rewards_tensor - reward_mean) / (reward_std + 1e-8)

            for k in range(num_samples):
                prompt_trunc, response = self._truncate_prompt_response(prompt, responses_group[i][k])
                # 构造完整序列: prompt + response
                full_seq = prompt_trunc + response
                # Labels: prompt部分mask为-100，只计算response部分的loss
                labels = [-100] * len(prompt_trunc) + response

                all_input_ids.append(full_seq)
                all_labels.append(labels)
                all_advantages.append(advantages[k].item())

        # Pad序列构造batch
        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_labels = []

        for seq, lab in zip(all_input_ids, all_labels):
            pad_len = max_len - len(seq)
            padded_input_ids.append(seq + [self.config.pad_token_id] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=device)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=device)

        # ZeRO-3: forward 前 AllGather 完整参数
        if isinstance(self.optimizer, ZeROStage3Optimizer):
            self.optimizer.pre_forward()

        # 计算log_probs
        if self._amp_enabled:
            with autocast(device_type=device.type, dtype=self._amp_dtype):
                log_probs = self._compute_log_probs(input_ids_tensor, labels_tensor)
                # 计算ref_log_probs（用于KL约束）
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_log_probs = self._compute_log_probs_with_model(self.ref_model, input_ids_tensor, labels_tensor)
                    else:
                        ref_log_probs = None
                if self._old_log_probs_buffer is None:
                    old_log_probs = log_probs.detach()
                else:
                    old_log_probs = self._old_log_probs_buffer
                loss, loss_info = self._compute_grpo_loss(log_probs, advantages_tensor, old_log_probs, ref_log_probs)
        else:
            log_probs = self._compute_log_probs(input_ids_tensor, labels_tensor)
            # 计算ref_log_probs（用于KL约束）
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_log_probs = self._compute_log_probs_with_model(self.ref_model, input_ids_tensor, labels_tensor)
                else:
                    ref_log_probs = None
            if self._old_log_probs_buffer is None:
                old_log_probs = log_probs.detach()
            else:
                old_log_probs = self._old_log_probs_buffer
            loss, loss_info = self._compute_grpo_loss(log_probs, advantages_tensor, old_log_probs, ref_log_probs)

        self._old_log_probs_buffer = log_probs.detach().clone()

        # 梯度累积
        scaled_loss = loss / self.config.gradient_accumulation_steps

        # autocast-only模式：无需scaler.scale()，直接backward
        scaled_loss.backward()

        # ZeRO-3: backward 后释放完整参数
        if isinstance(self.optimizer, ZeROStage3Optimizer):
            self.optimizer.post_backward()

        self._accumulation_count += 1

        # 累积满gradient_accumulation_steps步后执行优化器更新
        grad_norm = 0.0
        if self._accumulation_count >= self.config.gradient_accumulation_steps:
            # autocast-only模式：不再有GradScaler分支
            # 梯度裁剪：ZeRO-2/3 使用专用接口（p.grad 已被清空）
            if isinstance(self.optimizer, (ZeROStage2Optimizer, ZeROStage3Optimizer)):
                grad_norm = self.optimizer.clip_grad_norm(self.config.max_grad_norm)
            else:
                grad_norm = self._clip_grad_norm()
            if grad_norm < 0:
                # 梯度异常（NaN/Inf），跳过optimizer更新
                logger.warning("Skipping optimizer step due to gradient anomaly")
                self.optimizer.zero_grad()
                self._accumulation_count = 0
                # 不递增 step_count
            else:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self._accumulation_count = 0
                self.step_count += 1  # 仅在真正更新权重时递增
                # 显存水位监控
                if self.step_count % 10 == 0 and torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / 1e9
                    logger.info("[Step %d] peak_memory=%.2fGB", self.step_count, peak_mem)
                    if self.metrics_collector is not None:
                        self.metrics_collector.record("peak_memory_gb", peak_mem)

        total_loss = loss.item()
        total_policy_loss = loss_info["policy_loss"]
        total_kl = loss_info["kl_divergence"]

        # 获取当前动态 lr
        current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.config.learning_rate

        # 记录学习率到 metrics
        if self.metrics_collector is not None:
            self.metrics_collector.record("learning_rate", current_lr)

        return TrainMetrics(
            loss=total_loss,
            policy_loss=total_policy_loss,
            kl_divergence=total_kl,
            grad_norm=grad_norm,
            learning_rate=current_lr,
            step=self.step_count,
        )

    def sft_step(
        self,
        prompts_tokens: List[List[int]],
        response_tokens: List[List[int]],
    ) -> TrainMetrics:
        """SFT teacher-forcing 训练步骤。

        prompt 部分 label 置为 -100，只对标准回复 token 计算交叉熵。
        """
        assert self.model is not None, "TrainEngine is not initialized."
        assert self.optimizer is not None, "Optimizer is not initialized."

        device = next(self.model.parameters()).device
        all_input_ids = []
        all_labels = []

        for prompt, response in zip(prompts_tokens, response_tokens):
            prompt, response = self._truncate_prompt_response(prompt, response)
            full_seq = prompt + response
            labels = [-100] * len(prompt) + response
            all_input_ids.append(full_seq)
            all_labels.append(labels)

        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_labels = []

        for seq, labels in zip(all_input_ids, all_labels):
            pad_len = max_len - len(seq)
            padded_input_ids.append(seq + [self.config.pad_token_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=device)

        # ZeRO-3: forward 前 AllGather 完整参数
        if isinstance(self.optimizer, ZeROStage3Optimizer):
            self.optimizer.pre_forward()

        if self._amp_enabled:
            with autocast(device_type=device.type, dtype=self._amp_dtype):
                loss = self._compute_sft_loss(input_ids_tensor, labels_tensor)
        else:
            loss = self._compute_sft_loss(input_ids_tensor, labels_tensor)

        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        # ZeRO-3: backward 后释放完整参数
        if isinstance(self.optimizer, ZeROStage3Optimizer):
            self.optimizer.post_backward()

        self._accumulation_count += 1
        grad_norm = 0.0
        if self._accumulation_count >= self.config.gradient_accumulation_steps:
            # 梯度裁剪：ZeRO-2/3 使用专用接口（p.grad 已被清空）
            if isinstance(self.optimizer, (ZeROStage2Optimizer, ZeROStage3Optimizer)):
                grad_norm = self.optimizer.clip_grad_norm(self.config.max_grad_norm)
            else:
                grad_norm = self._clip_grad_norm()
            if grad_norm < 0:
                # 梯度异常（NaN/Inf），跳过optimizer更新
                logger.warning("Skipping optimizer step due to gradient anomaly")
                self.optimizer.zero_grad()
                self._accumulation_count = 0
                # 不递增 step_count
            else:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self._accumulation_count = 0
                self.step_count += 1  # 仅在真正更新权重时递增
                # 显存水位监控
                if self.step_count % 10 == 0 and torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / 1e9
                    logger.info("[Step %d] peak_memory=%.2fGB", self.step_count, peak_mem)
                    if self.metrics_collector is not None:
                        self.metrics_collector.record("peak_memory_gb", peak_mem)

        loss_value = loss.item()

        # 获取当前动态 lr
        current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.config.learning_rate

        # 记录学习率到 metrics
        if self.metrics_collector is not None:
            self.metrics_collector.record("learning_rate", current_lr)

        return TrainMetrics(
            loss=loss_value,
            policy_loss=loss_value,
            kl_divergence=0.0,
            grad_norm=grad_norm,
            learning_rate=current_lr,
            step=self.step_count,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """使用当前训练模型做本地自回归生成。

        这是 Windows/custom 后端的本地验证路径：不依赖 vLLM，不使用 KV cache，
        每步重新前向完整序列，速度较慢但能验证 rollout -> reward -> train 闭环。
        """
        assert self.model is not None, "TrainEngine is not initialized."

        # ZeRO-3: 使用 eval_mode 上下文管理器安全访问完整参数
        _zero3 = isinstance(self.optimizer, ZeROStage3Optimizer) if hasattr(self, 'optimizer') and self.optimizer is not None else False
        if _zero3:
            return self._generate_with_zero3_eval(prompt_tokens, max_new_tokens, temperature, top_p, eos_token_id)

        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()

        sequence = list(prompt_tokens)
        generated: List[int] = []

        try:
            for _ in range(max_new_tokens):
                input_context = sequence[-self.config.max_seq_len :] if self.config.max_seq_len > 0 else sequence
                input_ids = torch.tensor([input_context], dtype=torch.long, device=device)

                if self._amp_enabled:
                    with autocast(device_type=device.type, dtype=self._amp_dtype):
                        logits = self.model(input_ids)[:, -1, :]
                else:
                    logits = self.model(input_ids)[:, -1, :]

                next_token = self._sample_next_token(
                    logits=logits,
                    temperature=temperature,
                    top_p=top_p,
                )
                token_id = int(next_token.item())

                if eos_token_id is not None and token_id == eos_token_id:
                    break

                generated.append(token_id)
                sequence.append(token_id)
        finally:
            if was_training:
                self.model.train()

        return generated

    @torch.no_grad()
    def _generate_with_zero3_eval(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: Optional[int],
    ) -> List[int]:
        """ZeRO-3 专用生成路径：使用 eval_mode() 保证参数完整可用。"""
        device = next(self.model.parameters()).device
        was_training = self.model.training

        with self.optimizer.eval_mode():
            self.model.eval()
            sequence = list(prompt_tokens)
            generated: List[int] = []

            for _ in range(max_new_tokens):
                input_context = sequence[-self.config.max_seq_len :] if self.config.max_seq_len > 0 else sequence
                input_ids = torch.tensor([input_context], dtype=torch.long, device=device)

                if self._amp_enabled:
                    with autocast(device_type=device.type, dtype=self._amp_dtype):
                        logits = self.model(input_ids)[:, -1, :]
                else:
                    logits = self.model(input_ids)[:, -1, :]

                next_token = self._sample_next_token(
                    logits=logits,
                    temperature=temperature,
                    top_p=top_p,
                )
                token_id = int(next_token.item())

                if eos_token_id is not None and token_id == eos_token_id:
                    break

                generated.append(token_id)
                sequence.append(token_id)

        if was_training:
            self.model.train()

        return generated

    @torch.no_grad()
    def eval_sft_loss(
        self,
        prompts_tokens: List[List[int]],
        response_tokens: List[List[int]],
    ) -> float:
        """Evaluate teacher-forcing SFT loss without updating weights."""
        assert self.model is not None, "TrainEngine is not initialized."

        # ZeRO-3: 使用 eval_mode 上下文管理器安全访问完整参数
        _zero3 = isinstance(self.optimizer, ZeROStage3Optimizer) if hasattr(self, 'optimizer') and self.optimizer is not None else False

        was_training = self.model.training
        self.model.eval()
        try:
            if _zero3:
                with self.optimizer.eval_mode():
                    return self._eval_sft_loss_forward(prompts_tokens, response_tokens)
            else:
                return self._eval_sft_loss_forward(prompts_tokens, response_tokens)
        finally:
            if was_training:
                self.model.train()

    def _eval_sft_loss_forward(
        self,
        prompts_tokens: List[List[int]],
        response_tokens: List[List[int]],
    ) -> float:
        """eval_sft_loss 的实际前向计算逻辑（抽取以支持 ZeRO-3 eval_mode 包裹）。"""
        device = next(self.model.parameters()).device
        all_input_ids = []
        all_labels = []

        for prompt, response in zip(prompts_tokens, response_tokens):
            prompt, response = self._truncate_prompt_response(prompt, response)
            full_seq = prompt + response
            labels = [-100] * len(prompt) + response
            all_input_ids.append(full_seq)
            all_labels.append(labels)

        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_labels = []

        for seq, labels in zip(all_input_ids, all_labels):
            pad_len = max_len - len(seq)
            padded_input_ids.append(seq + [self.config.pad_token_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=device)
        if self._amp_enabled:
            with autocast(device_type=device.type, dtype=self._amp_dtype):
                loss = self._compute_sft_loss(input_ids_tensor, labels_tensor)
        else:
            loss = self._compute_sft_loss(input_ids_tensor, labels_tensor)
        return float(loss.item())

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """从最后一步 logits 中采样一个 token。"""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        logits = logits.float() / max(temperature, 1e-5)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            logits = filtered_logits

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _compute_log_probs(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """计算模型对给定序列的log概率

        input_ids: [batch, seq_len]
        labels: [batch, seq_len] (prompt部分mask为-100)

        return: [batch] 每个样本的平均log_prob
        """
        # 前向推理获取logits
        logits = self.model(input_ids)  # [batch, seq_len, vocab_size]

        # Shift: logits[:-1] 对应 labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab]
        shift_labels = labels[:, 1:].contiguous()       # [batch, seq_len-1]

        # 计算每个token的log_prob
        log_probs_all = F.log_softmax(shift_logits, dim=-1)  # [batch, seq_len-1, vocab]

        # 收集每个位置真实token的log_prob
        # [batch, seq_len-1]
        token_log_probs = log_probs_all.gather(
            dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

        # 创建mask：labels != -100 的位置才计算
        mask = (shift_labels != -100).float()  # [batch, seq_len-1]

        # 对mask后的token_log_probs求平均
        # 每个样本的平均log_prob
        token_log_probs = token_log_probs * mask
        seq_log_probs = token_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)  # [batch]

        return seq_log_probs

    def _compute_log_probs_with_model(
        self, model: 'ParallelTransformerModel', input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """使用指定模型计算log概率（复用log_prob逻辑，用于ref_model）

        Args:
            model: 用于计算的模型（如ref_model）
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] (prompt部分mask为-100)

        Returns:
            [batch] 每个样本的平均log_prob
        """
        logits = model(input_ids)  # [batch, seq_len, vocab_size]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs_all = F.log_softmax(shift_logits, dim=-1)

        token_log_probs = log_probs_all.gather(
            dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask
        seq_log_probs = token_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)

        return seq_log_probs

    def _compute_sft_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算 SFT 交叉熵损失。"""
        logits = self.model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """完整 GRPO Loss = PPO Clipping + KL Regularization

        Args:
            log_probs: [batch*K] 当前策略的 log probabilities (requires_grad)
            advantages: [batch*K] 组内相对优势 (detached)
            old_log_probs: [batch*K] 上一轮策略的 log probabilities (detached, 用于 importance sampling)
            ref_log_probs: [batch*K] 参考策略的 log probabilities (detached, 用于 KL penalty)

        Returns:
            loss: 标量loss
            info: 包含诊断信息的字典
        """
        ratio = torch.exp(log_probs - old_log_probs.detach())

        # 安全监控: ratio 异常时输出警告
        with torch.no_grad():
            ratio_max = ratio.max().item()
            ratio_min = ratio.min().item()
            if ratio_max > 10.0 or ratio_min < 0.1:
                logger.warning(
                    "GRPO ratio out of safe range: min=%.4f, max=%.4f (safe=[0.1, 10.0])",
                    ratio_min, ratio_max,
                )

        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.grpo_clip_eps,
            1.0 + self.config.grpo_clip_eps,
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        kl_div = torch.tensor(0.0, device=log_probs.device)
        if ref_log_probs is not None and self.config.grpo_beta > 0:
            # KL(policy || ref) = E[log(policy) - log(ref)]
            kl_div = (log_probs - ref_log_probs).mean()
            # KL应该是非负的，但由于估计误差可能为负，clamp到0
            kl_div = kl_div.clamp(min=0.0)

        # Total loss
        loss = policy_loss + self.config.grpo_beta * kl_div

        # 诊断信息
        with torch.no_grad():
            info = {
                "policy_loss": policy_loss.item(),
                "kl_divergence": kl_div.item(),
                "ratio_mean": ratio.mean().item(),
                "ratio_max": ratio_max,
                "ratio_min": ratio_min,
                "clipped_fraction": (
                    (ratio - 1.0).abs() > self.config.grpo_clip_eps
                ).float().mean().item(),
            }

        return loss, info

    def _clip_grad_norm(self) -> float:
        """裁剪梯度范数

        多卡TP场景下，各rank持有不同的参数分片，
        梯度范数应为全局范数而非local范数。
        解决方案：计算local范数后AllReduce聚合为全局范数再clip。
        单卡时(world_size=1)退化为标准clip_grad_norm。
        """
        import time

        trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not trainable_params:
            return 0.0

        # 检测 NaN/Inf 梯度
        has_nan = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in trainable_params
        )
        if has_nan:
            logger.warning("NaN/Inf detected in gradients, skipping this step")
            return -1.0  # 特殊返回值表示异常

        # [TP梯度同步] 仅对 replicated 参数做 AllReduce SUM
        # 根因：LoRAColumnParallel 的 lora_A 在各 rank 相同，但 backward 时
        # 各 rank 只从自己的 B_local 切片获得部分梯度。
        # 完整梯度 = Σ(各rank部分梯度)，需要 SUM（不是 AVG）。
        if self.parallel_ctx is not None and self.parallel_ctx.tp_size > 1:
            import torch.distributed as dist
            for p in trainable_params:
                if p.grad is not None and getattr(p, '_tp_replicated', False):
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.parallel_ctx.tp_group)

        # 计算local梯度范数的平方和
        total_norm_sq = sum(p.grad.data.float().norm(2).item() ** 2 for p in trainable_params)
        total_norm_sq = torch.tensor(total_norm_sq, device=next(iter(trainable_params)).device)

        # 多卡时AllReduce梯度范数平方和
        # 根因：纯DP场景(tp=1,dp>1)时tp_group只有自己，不是全局norm。
        # 需要根据并行模式选择正确的通信组进行梯度范数聚合。
        if self.parallel_ctx is not None and self.parallel_ctx.world_size > 1:
            import torch.distributed as dist
            # 选择正确的 norm 通信组
            if self.parallel_ctx.dp_size > 1:
                norm_group = self.parallel_ctx.dp_group
            else:
                norm_group = self.parallel_ctx.tp_group
            t0 = time.perf_counter()
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM, group=norm_group)
            comm_time_ms = (time.perf_counter() - t0) * 1000
            if self.metrics_collector is not None:
                self.metrics_collector.record("comm_time_ms", comm_time_ms)

        total_norm = total_norm_sq.sqrt().item()

        # Clip
        max_norm = self.config.max_grad_norm
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in trainable_params:
                p.grad.data.mul_(clip_coef)

        # 记录梯度范数到 metrics
        if self.metrics_collector is not None:
            self.metrics_collector.record("grad_norm", total_norm)

        return total_norm


    def export_weights(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """导出合并LoRA后的完整权重（用于传输给推理侧）

        调用merge_lora_weights将LoRA权重合并到原始权重。
        LoRA merge 在 GPU 上完成（模型参数本身在 GPU）。

        Args:
            device: 输出 tensor 所在设备。
                - "cpu": 传统路径，合并后移到 CPU（节省显存，但多一次拷贝）
                - "cuda": 直接返回 GPU tensor（省去 GPU→CPU→GPU 往返拷贝）

        Returns:
            合并后的 state_dict，tensor 位于指定 device 上
        """
        self.model.eval()
        # merge_lora_weights 在 GPU 上完成（模型参数已在 GPU）
        merged_state = merge_lora_weights(self.model)

        if device == "cpu":
            # 传统路径: 将权重移到CPU以释放GPU显存
            result = {k: v.cpu() for k, v in merged_state.items()}
        else:
            # 优化路径: 保留在 GPU 上，避免 GPU→CPU 拷贝
            result = merged_state

        self.model.train()
        return result

    def save_checkpoint(self, path: str):
        """保存训练检查点（仅LoRA参数 + optimizer state）"""
        import os
        os.makedirs(path, exist_ok=True)

        # 保存LoRA参数
        lora_state = get_lora_state_dict(self.model)
        torch.save(lora_state, os.path.join(path, "lora_weights.pt"))

        # 保存optimizer状态
        optimizer_state = {
            "type": "zero" if isinstance(self.optimizer, (ZeROOptimizer, ZeROStage2Optimizer, ZeROStage3Optimizer)) else "adamw",
            "step_count": self.step_count,
            "accumulation_count": self._accumulation_count,
        }
        if isinstance(self.optimizer, (ZeROOptimizer, ZeROStage2Optimizer, ZeROStage3Optimizer)):
            optimizer_state["local_shard"] = self.optimizer.local_shard.data.cpu()
            optimizer_state["local_optimizer_state"] = self.optimizer.local_optimizer.state_dict()
        else:
            optimizer_state["optimizer_state"] = self.optimizer.state_dict()
        torch.save(optimizer_state, os.path.join(path, "optimizer_state.pt"))

        # 保存scaler状态
        if self.scaler is not None:
            torch.save(self.scaler.state_dict(), os.path.join(path, "scaler_state.pt"))

        logger.info("[Rank %d] Checkpoint saved to %s", self.parallel_ctx.rank, path)

    def load_checkpoint(self, path: str) -> bool:
        """加载训练检查点

        Returns:
            True if at least one component (LoRA or optimizer) loaded successfully,
            False if path does not exist or all loads failed.
        """
        import os

        # 路径不存在时安全返回
        if not os.path.exists(path):
            logger.warning("Checkpoint path does not exist, skipping load: %s", path)
            return False

        lora_loaded = False
        optimizer_loaded = False

        # ZeRO-3 防护：如果参数已释放为空壳，先恢复形状再加载权重
        if isinstance(self.optimizer, ZeROStage3Optimizer):
            if any(p.numel() == 0 for p in self.optimizer.params):
                self.optimizer.pre_forward()

        # 加载LoRA参数
        lora_path = os.path.join(path, "lora_weights.pt")
        if os.path.exists(lora_path):
            try:
                lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)
                # 加载LoRA参数到模型
                model_state = self.model.state_dict()
                loaded_count = 0
                for k, v in lora_state.items():
                    if k in model_state:
                        model_state[k].copy_(v.to(model_state[k].device))
                        loaded_count += 1
                logger.info(
                    "[Rank %d] LoRA weights loaded from %s (%d parameters)",
                    self.parallel_ctx.rank, lora_path, loaded_count,
                )
                lora_loaded = True
            except Exception as e:
                logger.error("Failed to load LoRA weights from %s: %s", lora_path, e)

        # 加载optimizer状态
        optim_path = os.path.join(path, "optimizer_state.pt")
        if os.path.exists(optim_path):
            try:
                optim_state = torch.load(optim_path, map_location="cpu", weights_only=True)
                if isinstance(self.optimizer, (ZeROOptimizer, ZeROStage2Optimizer, ZeROStage3Optimizer)) and optim_state.get("type") == "zero":
                    self.optimizer.local_shard.data.copy_(optim_state["local_shard"])
                    self.optimizer.local_optimizer.load_state_dict(optim_state["local_optimizer_state"])
                    # ZeRO-3 关键修复：同步 flat_param_shard（通信用）
                    if isinstance(self.optimizer, ZeROStage3Optimizer):
                        self.optimizer.flat_param_shard.copy_(
                            self.optimizer.local_shard.data.to(self.optimizer._buffer_dtype)
                        )
                elif not isinstance(self.optimizer, (ZeROOptimizer, ZeROStage2Optimizer, ZeROStage3Optimizer)) and optim_state.get("optimizer_state") is not None:
                    self.optimizer.load_state_dict(optim_state["optimizer_state"])
                    self._move_optimizer_state_to_model_device()

                # 验证并恢复 step_count / accumulation_count
                if "step_count" in optim_state:
                    self.step_count = optim_state["step_count"]
                else:
                    logger.warning("'step_count' missing in optimizer state, using current default %d", self.step_count)

                if "accumulation_count" in optim_state:
                    self._accumulation_count = optim_state["accumulation_count"]
                else:
                    logger.warning("'accumulation_count' missing in optimizer state, using current default %d", self._accumulation_count)

                logger.info("[Rank %d] Optimizer state loaded from %s", self.parallel_ctx.rank, optim_path)
                optimizer_loaded = True
            except Exception as e:
                logger.error("Failed to load optimizer state from %s: %s", optim_path, e)

        # 加载scaler状态
        scaler_path = os.path.join(path, "scaler_state.pt")
        if self.scaler is not None and os.path.exists(scaler_path):
            try:
                scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=True)
                self.scaler.load_state_dict(scaler_state)
            except Exception as e:
                logger.error("Failed to load scaler state from %s: %s", scaler_path, e)

        return lora_loaded or optimizer_loaded

    def _move_optimizer_state_to_model_device(self):
        """Move AdamW optimizer state tensors to the same device as model params."""
        if self.model is None or self.optimizer is None:
            return
        device = next(self.model.parameters()).device
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def generate_memory_report(self) -> dict:
        """生成完整的显存分析报告

        基于当前训练状态（模型、优化器）生成显存 breakdown 报告，
        不执行额外的 train step，避免干扰训练主流程。

        Returns:
            dict: 完整显存分析报告
        """
        try:
            from ..utils.memory_profiler import MemoryReport

            config = {
                "model_name": self.config.model_path,
                "seq_len": self.config.max_seq_len,
                "batch_size": 1,
                "parallel_strategy": f"TP={self.config.tp_size}, DP={self.config.dp_size}",
                "lora_rank": self.config.lora_rank,
            }

            reporter = MemoryReport(self.model, self.optimizer, config)
            report = reporter.generate_report_without_step()

            # 补充峰值信息（基于训练过程中 PyTorch 记录的全局峰值）
            if torch.cuda.is_available():
                device = next(self.model.parameters()).device
                report["peak_moments"] = {
                    "overall_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                    "current_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
                    "current_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
                }

            return report
        except Exception as e:
            logger.warning("Failed to generate memory report: %s", e)
            return {"error": str(e)}

    @property
    def trainable_params_count(self) -> int:
        """可训练参数数量"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def total_params_count(self) -> int:
        """总参数数量"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
