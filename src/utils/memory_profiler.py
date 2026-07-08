"""
GPU显存监控工具

提供显存快照、峰值追踪和报告功能:
- 在关键阶段前后拍摄显存快照
- 追踪显存使用峰值
- 生成可读的显存使用报告
"""
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch


@dataclass
class MemorySnapshot:
    """GPU显存快照"""
    timestamp: float
    stage: str
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    device_id: int = 0


class MemoryProfiler:
    """GPU显存Profile工具

    用法:
        profiler = MemoryProfiler()
        profiler.snapshot("before_training")
        # ... training ...
        profiler.snapshot("after_training")
        print(profiler.report())
    """

    def __init__(self, device_id: int = -1):
        """
        Args:
            device_id: 监控的GPU设备ID，-1表示自动使用当前设备（推荐）
                       注意：-1时在首次snapshot()调用时才解析，确保set_device已执行
        """
        # 不在此处解析 device_id — set_device 可能尚未调用（Megatron-LM 风格懒加载）
        self.device_id = device_id
        self._snapshots: List[MemorySnapshot] = []
        self._start_time = time.time()

    def snapshot(self, stage: str) -> MemorySnapshot:
        """拍摄当前显存快照

        Args:
            stage: 阶段名称（如 "before_training", "after_sync"）

        Returns:
            MemorySnapshot: 快照数据
        """
        if not torch.cuda.is_available():
            snap = MemorySnapshot(
                timestamp=time.time() - self._start_time,
                stage=stage,
                allocated_mb=0.0,
                reserved_mb=0.0,
                peak_mb=0.0,
                device_id=self.device_id,
            )
            self._snapshots.append(snap)
            return snap

        # 懒更新 device_id：确保使用实际的 current_device（Megatron-LM 风格）
        if self.device_id == -1:
            self.device_id = torch.cuda.current_device()

        device = torch.device(f"cuda:{self.device_id}")
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        snap = MemorySnapshot(
            timestamp=time.time() - self._start_time,
            stage=stage,
            allocated_mb=allocated,
            reserved_mb=reserved,
            peak_mb=peak,
            device_id=self.device_id,
        )
        self._snapshots.append(snap)
        return snap

    def report(self) -> str:
        """生成可读的显存使用报告

        Returns:
            格式化的报告字符串
        """
        if not self._snapshots:
            return "No memory snapshots recorded."

        lines = [
            f"{'='*60}",
            f"  GPU Memory Profile Report (device={self.device_id})",
            f"{'='*60}",
            f"  {'Stage':<25} {'Alloc(MB)':>10} {'Reserved(MB)':>13} {'Peak(MB)':>10} {'Time(s)':>8}",
            f"  {'-'*25} {'-'*10} {'-'*13} {'-'*10} {'-'*8}",
        ]

        for snap in self._snapshots:
            lines.append(
                f"  {snap.stage:<25} {snap.allocated_mb:>10.1f} "
                f"{snap.reserved_mb:>13.1f} {snap.peak_mb:>10.1f} "
                f"{snap.timestamp:>8.2f}"
            )

        lines.append(f"{'='*60}")
        lines.append(f"  Peak memory: {self.peak_memory_mb():.1f} MB")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def get_timeline(self) -> List[MemorySnapshot]:
        """获取所有快照的时间线

        Returns:
            按时间排序的快照列表
        """
        return list(self._snapshots)

    def peak_memory_mb(self) -> float:
        """获取记录到的峰值显存（MB）

        Returns:
            峰值显存使用量
        """
        if not self._snapshots:
            return 0.0
        return max(snap.peak_mb for snap in self._snapshots)

    def reset(self):
        """重置所有快照和峰值追踪"""
        self._snapshots.clear()
        self._start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)

    def delta(self, stage_a: str, stage_b: str) -> float:
        """计算两个阶段之间的显存变化（MB）

        Args:
            stage_a: 起始阶段名
            stage_b: 结束阶段名

        Returns:
            显存变化量（正数表示增加）
        """
        snap_a = next((s for s in self._snapshots if s.stage == stage_a), None)
        snap_b = next((s for s in self._snapshots if s.stage == stage_b), None)
        if snap_a is None or snap_b is None:
            return 0.0
        return snap_b.allocated_mb - snap_a.allocated_mb


# MemoryBreakdown: 显存按用途分类分析


class MemoryBreakdown:
    """显存breakdown分析

    将GPU显存按用途分类：
    - 模型权重 (Model Weights)
    - 优化器状态 (Optimizer States): Adam的m和v
    - 梯度 (Gradients)
    - 激活值 (Activations): 前向传播的中间结果
    - KV Cache (推理时)
    - 碎片/预留 (Fragmentation/Reserved)

    技术要点：
    - 4B参数 FP16模型权重 ≈ 8GB (4B * 2 bytes)
    - Adam优化器状态 ≈ 2x权重大小（m和v各一份FP32 = 4B * 4bytes * 2 = 32GB）
    - LoRA大幅减少可训练参数 → 优化器状态从32GB降到~几十MB
    - ZeRO-1分片优化器状态：每卡只存1/N
    - 激活值与batch_size和seq_len成正比：≈ 2 * L * B * S * H * dtype_bytes
    - KV Cache: 2 * L * B * S * H * dtype_bytes（K和V各一份）
    - 为什么实际显存比理论高？→ CUDA内存碎片 + PyTorch预分配策略
    """

    def __init__(self, model=None, optimizer=None):
        """
        Args:
            model: nn.Module实例（可选，用于实际分析）
            optimizer: 优化器实例（可选）
        """
        self.model = model
        self.optimizer = optimizer

    def analyze(self) -> dict:
        """分析当前显存分布（需要实际模型和CUDA可用）

        技术要点：
        - torch.cuda.memory_allocated() vs memory_reserved()？
          → allocated: 实际被tensor占用的显存
          → reserved: PyTorch缓存分配器预留的显存（包含碎片）
          → reserved - allocated = 碎片/缓存

        Returns:
            各类别的MB数和占比
        """
        result = {
            "model_weights_mb": 0.0,
            "optimizer_states_mb": 0.0,
            "gradients_mb": 0.0,
            "total_allocated_mb": 0.0,
            "total_reserved_mb": 0.0,
            "fragmentation_mb": 0.0,
        }

        if self.model is not None:
            model_bytes = sum(
                p.nelement() * p.element_size() for p in self.model.parameters()
            )
            result["model_weights_mb"] = model_bytes / (1024 * 1024)

            grad_bytes = sum(
                p.grad.nelement() * p.grad.element_size()
                for p in self.model.parameters()
                if p.grad is not None
            )
            result["gradients_mb"] = grad_bytes / (1024 * 1024)

        # 优化器状态
        if self.optimizer is not None:
            optim_bytes = 0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    state = self.optimizer.state.get(p, {})
                    for v in state.values():
                        if isinstance(v, torch.Tensor):
                            optim_bytes += v.nelement() * v.element_size()
            result["optimizer_states_mb"] = optim_bytes / (1024 * 1024)

        # CUDA总体统计
        if torch.cuda.is_available():
            result["total_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            result["total_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            result["fragmentation_mb"] = (
                result["total_reserved_mb"] - result["total_allocated_mb"]
            )

        return result

    def estimate(self, model_params: int, batch_size: int, seq_len: int,
                 num_layers: int = 32, hidden_size: int = 4096,
                 lora_rank: int = 0, tp_size: int = 1, zero_stage: int = 0,
                 dtype_bytes: int = 2, num_kv_heads: int = 8,
                 head_dim: int = 128) -> dict:
        """静态估算显存需求（不需要实际加载模型）

        技术要点：
        - 训练显存 = 权重 + 优化器 + 梯度 + 激活
        - 推理显存 = 权重 + KV Cache
        - LoRA为什么省显存？→ 可训练参数少 → 优化器状态和梯度都少
        - TP为什么省显存？→ 每卡只存1/tp_size的权重（和对应的优化器状态）
        - ZeRO-1为什么省？→ 优化器状态分到N张卡，每卡1/N

        Args:
            model_params: 模型总参数量
            batch_size: batch大小
            seq_len: 序列长度
            num_layers: Transformer层数
            hidden_size: 隐藏维度
            lora_rank: LoRA秩（0表示全参数训练）
            tp_size: 张量并行度
            zero_stage: ZeRO阶段 (0=无, 1=优化器分片, 2=梯度+优化器分片)
            dtype_bytes: 数据类型字节数（FP16=2, FP32=4, BF16=2）
            num_kv_heads: KV头数（用于KV Cache估算）
            head_dim: 每个头的维度

        Returns:
            各类别的显存估算（MB）和总计
        """
        # TP: 每卡约存 1/tp_size（近似，实际embedding和norm不切分）
        weight_bytes = model_params * dtype_bytes / tp_size
        weight_mb = weight_bytes / (1024 * 1024)

        if lora_rank > 0:
            # LoRA参数量 ≈ 2 * rank * hidden_size * num_linear_layers
            # 每层约4个linear (q,k,v,o) + 3个MLP linear
            num_lora_layers = num_layers * 7
            trainable_params = 2 * lora_rank * hidden_size * num_lora_layers
        else:
            trainable_params = model_params / tp_size

        optim_bytes_per_param = 8  # FP32 m + FP32 v = 4 + 4 = 8 bytes
        optim_bytes = trainable_params * optim_bytes_per_param

        # ZeRO-1: 优化器状态分N份
        if zero_stage >= 1:
            optim_bytes /= tp_size  # 简化：用tp_size近似world_size
        optim_mb = optim_bytes / (1024 * 1024)

        # 梯度与可训练参数同精度（通常FP16或BF16）
        grad_bytes = trainable_params * dtype_bytes
        if zero_stage >= 2:
            grad_bytes /= tp_size
        grad_mb = grad_bytes / (1024 * 1024)

        # 简化估算：每层激活 ≈ B * S * H * dtype_bytes * activation_factor
        # activation_factor ≈ 10-14（考虑attention scores, QKV, 中间结果等）
        activation_factor = 12  # 经验值
        activation_bytes = (
            num_layers * batch_size * seq_len * hidden_size * dtype_bytes * activation_factor
        )
        activation_mb = activation_bytes / (1024 * 1024)

        # KV Cache per layer = 2 * B * S * num_kv_heads * head_dim * dtype_bytes
        kv_cache_bytes = (
            2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * dtype_bytes
        )
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        subtotal = weight_mb + optim_mb + grad_mb + activation_mb
        fragmentation_mb = subtotal * 0.12  # 约12%碎片

        total_training_mb = weight_mb + optim_mb + grad_mb + activation_mb + fragmentation_mb
        total_inference_mb = weight_mb + kv_cache_mb + fragmentation_mb * 0.5

        return {
            "model_weights_mb": round(weight_mb, 1),
            "optimizer_states_mb": round(optim_mb, 1),
            "gradients_mb": round(grad_mb, 1),
            "activations_mb": round(activation_mb, 1),
            "kv_cache_mb": round(kv_cache_mb, 1),
            "fragmentation_mb": round(fragmentation_mb, 1),
            "total_training_mb": round(total_training_mb, 1),
            "total_inference_mb": round(total_inference_mb, 1),
            "trainable_params": int(trainable_params),
            "trainable_ratio": trainable_params / model_params if model_params > 0 else 0,
            "config": {
                "model_params": model_params,
                "lora_rank": lora_rank,
                "tp_size": tp_size,
                "zero_stage": zero_stage,
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def report(self, estimate_result: Optional[Dict] = None) -> str:
        """生成显存breakdown报告

        技术要点：
        - 这个报告可以直接用于回答"你的模型需要多少显存"的问题
        - 通过LoRA + ZeRO的组合看显存节省效果

        Args:
            estimate_result: estimate()方法的返回值。None时使用analyze()。

        Returns:
            格式化的显存报告
        """
        if estimate_result is None:
            estimate_result = self.analyze()

        # 确定总显存和各部分
        total_mb = estimate_result.get("total_training_mb",
                   estimate_result.get("total_allocated_mb", 0))

        lines = []
        lines.append("┌" + "─" * 45 + "┐")
        lines.append(f"│{'GPU显存Breakdown':^37}        │")
        if total_mb > 0:
            total_gb = total_mb / 1024
            lines.append(f"│{'(总计: ' + f'{total_gb:.1f}GB)':^37}        │")
        lines.append("├" + "─" * 45 + "┤")

        # 各部分
        items = [
            ("模型权重", "model_weights_mb"),
            ("优化器状态", "optimizer_states_mb"),
            ("梯度", "gradients_mb"),
            ("激活值", "activations_mb"),
            ("KV Cache", "kv_cache_mb"),
            ("碎片/预留", "fragmentation_mb"),
        ]

        for label, key in items:
            val = estimate_result.get(key, 0)
            if val > 0:
                pct = val / total_mb * 100 if total_mb > 0 else 0
                if val >= 1024:
                    val_str = f"{val/1024:.1f} GB"
                else:
                    val_str = f"{val:.0f} MB"
                lines.append(f"│  {label:<12} │ {val_str:>10} {pct:>5.1f}%     │")

        lines.append("├" + "─" * 45 + "┤")

        # 训练/推理总计
        train_mb = estimate_result.get("total_training_mb", 0)
        infer_mb = estimate_result.get("total_inference_mb", 0)
        if train_mb > 0:
            lines.append(f"│  训练总计: {train_mb/1024:.1f} GB" + " " * 21 + "│")
        if infer_mb > 0:
            lines.append(f"│  推理总计: {infer_mb/1024:.1f} GB" + " " * 21 + "│")

        # 可训练参数信息
        trainable_ratio = estimate_result.get("trainable_ratio", 0)
        if trainable_ratio > 0 and trainable_ratio < 1:
            lines.append(f"│  可训练比例: {trainable_ratio:.4%} (LoRA)" + " " * 10 + "│")

        lines.append("└" + "─" * 45 + "┘")
        return "\n".join(lines)


# MemoryReport: 完整显存分析报告生成器


class MemoryReport:
    """完整显存分析报告生成器

    在训练结束后收集：
    - 静态显存分布（权重、梯度、optimizer states）
    - 动态显存分布（per-layer activations）
    - 三个峰值点（forward_peak, backward_peak, step_peak）
    - 框架与通信开销
    """

    def __init__(self, model, optimizer, config: dict):
        """
        Args:
            model: 模型实例
            optimizer: 优化器实例
            config: 训练配置字典 {model_name, total_params, trainable_params,
                    precision, seq_len, batch_size, parallel_strategy, lora_rank, ...}
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self._layer_memory = {}  # per-layer activation memory
        self._hooks = []

    def profile_step(self, input_ids, labels, train_step_fn):
        """运行一个带完整 profiling 的 training step

        Args:
            input_ids: 输入 tensor
            labels: 标签 tensor
            train_step_fn: callable，执行一步训练（接受 input_ids, labels）

        Returns:
            dict: 完整报告
        """
        device = next(self.model.parameters()).device

        report = {
            "model_spec": self._get_model_spec(),
            "static_memory": self._get_static_memory(),
            "dynamic_memory": {},
            "framework_overhead": {},
            "peak_moments": {},
        }

        # 注册 per-layer hooks
        self._register_layer_hooks()

        # Phase 1: Forward peak
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        mem_before_forward = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            _ = self.model(input_ids)

        torch.cuda.synchronize(device)
        forward_peak = torch.cuda.max_memory_allocated(device)
        mem_after_forward = torch.cuda.memory_allocated(device)

        report["peak_moments"]["forward_peak_gb"] = forward_peak / 1e9
        report["peak_moments"]["forward_activation_gb"] = (mem_after_forward - mem_before_forward) / 1e9

        # Phase 2: Backward peak
        torch.cuda.reset_peak_memory_stats(device)

        # 运行完整 train step (forward + backward + optimizer)
        self.model.train()
        logits = self.model(input_ids)
        # 简单 loss 用于触发 backward
        if labels is not None:
            import torch.nn.functional as F
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss.backward()

        torch.cuda.synchronize(device)
        backward_peak = torch.cuda.max_memory_allocated(device)
        report["peak_moments"]["backward_peak_gb"] = backward_peak / 1e9

        # Phase 3: Step peak
        torch.cuda.reset_peak_memory_stats(device)
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        torch.cuda.synchronize(device)
        step_peak = torch.cuda.max_memory_allocated(device)
        report["peak_moments"]["step_peak_gb"] = step_peak / 1e9

        # 收集动态项
        report["dynamic_memory"] = self._get_dynamic_memory()
        report["framework_overhead"] = self._get_framework_overhead(device)

        self._remove_hooks()

        return report

    def generate_report_without_step(self):
        """不执行 training step，仅基于当前状态生成静态报告"""
        device = next(self.model.parameters()).device

        report = {
            "model_spec": self._get_model_spec(),
            "static_memory": self._get_static_memory(),
            "dynamic_memory": {"note": "Run profile_step() for per-layer activation data"},
            "framework_overhead": self._get_framework_overhead(device),
            "peak_moments": {"note": "Run profile_step() for peak annotations"},
        }
        return report

    def _get_model_spec(self):
        """模型规格与训练配置"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # 检测精度分布
        dtypes = {}
        for p in self.model.parameters():
            dtype_name = str(p.dtype)
            dtypes[dtype_name] = dtypes.get(dtype_name, 0) + p.numel()

        return {
            "model_name": self.config.get("model_name", "unknown"),
            "total_params": total_params,
            "total_params_readable": f"{total_params/1e9:.2f}B",
            "trainable_params": trainable_params,
            "trainable_params_readable": f"{trainable_params/1e6:.1f}M",
            "frozen_params": frozen_params,
            "precision_distribution": {k: f"{v/1e6:.1f}M" for k, v in dtypes.items()},
            "seq_len": self.config.get("seq_len", 0),
            "batch_size": self.config.get("batch_size", 0),
            "parallel_strategy": self.config.get("parallel_strategy", ""),
            "lora_rank": self.config.get("lora_rank", 0),
        }

    def _get_static_memory(self):
        """静态项账本"""
        # 按 requires_grad 和 dtype 分类
        frozen_bf16_bytes = 0
        frozen_fp32_bytes = 0
        trainable_bytes = 0
        grad_bytes = 0

        for p in self.model.parameters():
            size = p.numel() * p.element_size()
            if p.requires_grad:
                trainable_bytes += size
                if p.grad is not None:
                    grad_bytes += p.grad.numel() * p.grad.element_size()
            else:
                if p.dtype == torch.bfloat16 or p.dtype == torch.float16:
                    frozen_bf16_bytes += size
                else:
                    frozen_fp32_bytes += size

        # Optimizer states
        optimizer_bytes = 0
        if self.optimizer is not None:
            try:
                if hasattr(self.optimizer, 'local_optimizer'):
                    # ZeRO optimizer - check local optimizer
                    for group in self.optimizer.local_optimizer.param_groups:
                        for p in group["params"]:
                            state = self.optimizer.local_optimizer.state.get(p, {})
                            for v in state.values():
                                if isinstance(v, torch.Tensor):
                                    optimizer_bytes += v.numel() * v.element_size()
                elif hasattr(self.optimizer, 'param_groups'):
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            state = self.optimizer.state.get(p, {})
                            for v in state.values():
                                if isinstance(v, torch.Tensor):
                                    optimizer_bytes += v.numel() * v.element_size()
            except Exception:
                pass  # optimizer state access failure should not block report

        return {
            "frozen_weights_bf16_gb": frozen_bf16_bytes / 1e9,
            "frozen_weights_fp32_gb": frozen_fp32_bytes / 1e9,
            "trainable_weights_gb": trainable_bytes / 1e9,
            "gradients_gb": grad_bytes / 1e9,
            "optimizer_states_gb": optimizer_bytes / 1e9,
            "static_total_gb": (frozen_bf16_bytes + frozen_fp32_bytes + trainable_bytes + grad_bytes + optimizer_bytes) / 1e9,
        }

    def _register_layer_hooks(self):
        """注册 per-layer forward hooks 测量激活内存"""
        self._layer_memory = {}

        for name, module in self.model.named_modules():
            # 只 hook transformer layers 和关键层
            if any(keyword in name for keyword in ['layers.', 'embed', 'lm_head', 'norm']):
                hook = module.register_forward_hook(
                    self._make_memory_hook(name)
                )
                self._hooks.append(hook)

    def _make_memory_hook(self, layer_name):
        """创建记录显存增量的 hook"""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self._layer_memory[layer_name] = {
                    "output_mb": output.numel() * output.element_size() / 1e6,
                    "dtype": str(output.dtype),
                    "shape": list(output.shape),
                }
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                self._layer_memory[layer_name] = {
                    "output_mb": output[0].numel() * output[0].element_size() / 1e6,
                    "dtype": str(output[0].dtype),
                    "shape": list(output[0].shape),
                }
        return hook

    def _remove_hooks(self):
        """移除所有 hooks"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _get_dynamic_memory(self):
        """动态项账本（基于 hooks 收集的数据）"""
        layers_data = {}
        embedding_mb = 0
        logits_mb = 0
        transformer_total_mb = 0

        for name, info in self._layer_memory.items():
            if 'embed' in name:
                embedding_mb += info["output_mb"]
            elif 'lm_head' in name:
                logits_mb += info["output_mb"]
            elif 'layers.' in name:
                transformer_total_mb += info["output_mb"]
                layers_data[name] = info

        return {
            "embedding_mb": embedding_mb,
            "transformer_layers_total_mb": transformer_total_mb,
            "logits_head_mb": logits_mb,
            "per_layer_detail": layers_data,
            "layer_count": len([k for k in layers_data if 'layers.' in k]),
        }

    def _get_framework_overhead(self, device):
        """框架与通信开销"""
        if not torch.cuda.is_available():
            return {
                "pytorch_caching_gb": 0.0,
                "cuda_context_estimated_gb": 0.0,
                "nccl_buffers_estimated_gb": 0.0,
                "total_overhead_gb": 0.0,
            }

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        # PyTorch caching = reserved - allocated
        caching_gb = (reserved - allocated) / 1e9

        # CUDA context（经验值，H20 typical ~0.8GB）
        cuda_context_gb = 0.8
        # NCCL buffers（经验值）
        nccl_buffers_gb = 0.2

        return {
            "pytorch_caching_gb": caching_gb,
            "cuda_context_estimated_gb": cuda_context_gb,
            "nccl_buffers_estimated_gb": nccl_buffers_gb,
            "total_overhead_gb": caching_gb + cuda_context_gb + nccl_buffers_gb,
            "device_total_memory_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
        }
