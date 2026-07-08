"""
Mini-ZeRO Stage 3: 参数 + 梯度 + 优化器状态 全分片 (Full Sharding)

核心思想:
- 在 ZeRO-2 基础上，模型参数也进行分片——每卡只常驻 1/N 的模型参数
- Forward 前 AllGather 恢复完整参数，forward+backward 完成后释放
- Backward 后 ReduceScatter 梯度，只保留本 rank 负责的梯度 shard
- Optimizer step 只更新本地 shard，无需全量参数

与 ZeRO-1/2 的区别:
- ZeRO-1: 完整参数 + 完整梯度 + 分片 optimizer
- ZeRO-2: 完整参数 + 分片梯度（hook 触发 ReduceScatter） + 分片 optimizer
- ZeRO-3: 分片参数 + 分片梯度 + 分片 optimizer

显存收益 (以 8B bf16 模型 8卡为例):
- 模型权重: 16GB → 2GB/卡（1/N 常驻）
- 梯度: 同 ZeRO-2，ReduceScatter 后只保留 1/N
- 优化器状态: 同 ZeRO-1/2，只管理 1/N
- Forward/backward 期间需要临时 AllGather 完整参数（峰值 = 全量参数）

MVP 实现策略:
- pre_forward(): AllGather 全量参数到模型（forward+backward 共用）
- backward 完成后: ReduceScatter 梯度
- post_step(): 释放完整参数，只保留 shard
- 显存节省体现在: optimizer 不占全量空间 + 训练步之间参数被释放

使用方式:
    optimizer = ZeROStage3Optimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        parallel_context=ctx,
        lr=1e-4, weight_decay=0.01
    )
    # 训练循环中:
    optimizer.pre_forward()       # AllGather 完整参数
    loss = model(input)
    loss.backward()               # hook 自动 ReduceScatter 梯度
    optimizer.post_backward()     # 释放完整参数
    optimizer.step()
    optimizer.zero_grad()
"""

from typing import Optional, Type, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist

from .parallel_context import ParallelContext
from ..utils.logger import get_logger

logger = get_logger("distributed.zero_v3")


class ZeROStage3Optimizer:
    """Mini-ZeRO Stage 3 优化器

    参数 + 梯度 + 优化器状态全分片。
    每卡只常驻 1/N 的模型参数，forward/backward 时按需 AllGather 完整参数，
    用完立即释放。梯度通过 ReduceScatter 分片后只保留本 rank 负责的部分。

    显存收益:
    - 模型权重从 N*dtype → dtype/N（8B bf16 从 16GB → 2GB/8卡）
    - 结合分片 optimizer：总显存极大降低
    """

    def __init__(
        self,
        params: Iterator[nn.Parameter],
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        parallel_context: ParallelContext = None,
        comm_group: "Optional[dist.ProcessGroup]" = None,
        **optimizer_kwargs,
    ):
        """
        Args:
            params: 模型参数迭代器
            optimizer_class: 优化器类（默认 AdamW）
            parallel_context: 并行上下文（提供 world_size 和通信组）
            comm_group: 显式指定通信组。若为 None，则自动选择:
                        - dp_size > 1 时使用 dp_group（梯度在 DP rank 间同步）
                        - 否则使用 tp_group（向后兼容）
            **optimizer_kwargs: 传递给优化器的参数（lr, weight_decay 等）
        """
        self.parallel_context = parallel_context

        # 确定通信组和对应的 world_size / rank
        if comm_group is not None:
            self.group = comm_group
        elif parallel_context and parallel_context.dp_size > 1:
            # DP模式：ZeRO 在 dp_group 上操作（梯度在 DP rank 间同步）
            self.group = parallel_context.dp_group
        elif parallel_context:
            # 纯TP模式（向后兼容）：ZeRO 在 tp_group 上操作
            self.group = parallel_context.tp_group
        else:
            self.group = None

        # 从通信组获取 world_size 和 rank
        if self.group is not None:
            self.world_size = dist.get_world_size(self.group)
            self.rank = dist.get_rank(self.group)
        else:
            self.world_size = 1
            self.rank = 0

        # 收集所有需要优化的参数
        self.params = list(params)
        if len(self.params) == 0:
            raise ValueError("ZeROStage3Optimizer: no trainable parameters found")
        self.param_count = sum(p.numel() for p in self.params)

        # 记录每个参数的原始 shape 和 numel（释放后需要恢复）
        self._param_shapes = [p.shape for p in self.params]
        self._param_numels = [p.numel() for p in self.params]

        # 对齐到 world_size 的整数倍（padding）
        self.padded_size = self._pad_to_divisible(self.param_count, self.world_size)
        self._buffer_dtype = self.params[0].dtype  # 跟随模型参数 dtype（bf16）
        self._device = self.params[0].device

        # 创建完整的 flat buffer 并拷贝参数数据
        flat_full = torch.zeros(self.padded_size, dtype=self._buffer_dtype, device=self._device)
        offset = 0
        for p in self.params:
            numel = p.numel()
            flat_full[offset:offset + numel].copy_(p.data.view(-1).to(self._buffer_dtype))
            offset += numel

        self.shard_size = self.padded_size // self.world_size
        self.shard_start = self.rank * self.shard_size
        self.shard_end = self.shard_start + self.shard_size

        # 这是 ZeRO-3 的核心：每卡只常驻 1/N 的参数
        self.flat_param_shard = flat_full[self.shard_start:self.shard_end].clone()

        # local_shard 使用 fp32（AdamW master weights 需要高精度）
        self.local_shard = self.flat_param_shard.clone().detach().float().requires_grad_(True)
        self.local_optimizer = optimizer_class([self.local_shard], **optimizer_kwargs)

        self._local_grad_acc = torch.zeros(
            self.shard_size, dtype=torch.float32, device=self._device
        )

        # 状态标记
        self._params_gathered = False  # 当前模型是否持有完整参数
        self._grad_hooks = []

        self._grad_count = 0
        self._register_grad_hooks()

        # 初始状态下模型参数被释放，只有 forward 前 AllGather 才恢复
        self._release_full_params()

        # 释放临时 full buffer
        del flat_full

        logger.info(
            "ZeROStage3Optimizer initialized: params=%d, padded=%d, shard_size=%d, "
            "rank=%d/%d, buffer_dtype=%s",
            self.param_count, self.padded_size, self.shard_size,
            self.rank, self.world_size, self._buffer_dtype,
        )

    def _pad_to_divisible(self, size: int, divisor: int) -> int:
        """将 size 向上对齐到 divisor 的整数倍"""
        return size + (divisor - size % divisor) % divisor

    def _register_grad_hooks(self):
        """为每个参数注册 post_accumulate_grad_hook

        当所有参数梯度就绪时，触发 ReduceScatter。
        """
        for i, p in enumerate(self.params):
            hook = p.register_post_accumulate_grad_hook(
                lambda param, idx=i: self._on_grad_ready(idx, param)
            )
            self._grad_hooks.append(hook)

    def _on_grad_ready(self, idx: int, param: nn.Parameter):
        """参数梯度就绪回调

        当所有参数梯度都就绪时，触发 ReduceScatter。
        """
        self._grad_count += 1
        if self._grad_count == len(self.params):
            self._reduce_scatter_grads()

    def _reduce_scatter_grads(self):
        """将所有参数梯度 flatten → ReduceScatter → 累加到本地 shard

        核心流程:
        1. 收集所有参数梯度到临时 flat buffer（瞬时分配）
        2. ReduceScatter: 每个 rank 只拿自己 shard 的聚合梯度
        3. 取 MEAN（除以 world_size）
        4. 累加到 _local_grad_acc（支持梯度累积）
        5. 释放各参数的 .grad（核心显存节省）
        6. 重置 grad_count
        """
        # 收集所有梯度到临时 flat buffer
        flat_grad = torch.zeros(
            self.padded_size, dtype=self._buffer_dtype, device=self._device
        )
        offset = 0
        for p in self.params:
            numel = p.numel()
            if p.grad is not None:
                flat_grad[offset:offset + numel].copy_(p.grad.data.view(-1).to(self._buffer_dtype))
            offset += numel

        if self.world_size > 1 and self.group is not None:
            # ReduceScatter: 每个 rank 只拿自己 shard 的聚合梯度
            local_grad = torch.empty(
                self.shard_size, dtype=self._buffer_dtype, device=self._device
            )
            dist.reduce_scatter_tensor(
                local_grad, flat_grad, op=dist.ReduceOp.SUM, group=self.group
            )
            # MEAN 语义：SUM聚合后除以 world_size
            local_grad = local_grad / self.world_size
        else:
            # 单卡模式：直接取对应分片
            local_grad = flat_grad[self.shard_start:self.shard_end]

        # 累加到 _local_grad_acc（支持 gradient accumulation）
        # local_grad 是 bf16/fp16，累加到 fp32 accumulator
        self._local_grad_acc.add_(local_grad.float())

        # 释放各参数的梯度（核心显存节省）
        for p in self.params:
            p.grad = None

        # 重置计数器，为下一次 backward 做准备
        self._grad_count = 0

    def pre_forward(self):
        """训练步 forward 前调用：AllGather 恢复完整参数到模型

        从各 rank 收集完整的 flat 参数，然后写回模型的各个参数 tensor。
        Forward + backward 共用这份完整参数（backward 需要 weight 做 chain rule）。
        """
        if self._params_gathered:
            return  # 已经 gather 过，跳过（gradient accumulation 场景）

        if self.world_size > 1 and self.group is not None:
            # AllGather: 从所有 rank 收集完整参数
            full_flat = torch.empty(
                self.padded_size, dtype=self._buffer_dtype, device=self._device
            )
            dist.all_gather_into_tensor(
                full_flat, self.flat_param_shard, group=self.group
            )
        else:
            # 单卡模式：用 shard 直接填充
            full_flat = torch.zeros(
                self.padded_size, dtype=self._buffer_dtype, device=self._device
            )
            full_flat[self.shard_start:self.shard_end].copy_(self.flat_param_shard)

        # 将 full_flat 写回模型参数
        offset = 0
        for i, p in enumerate(self.params):
            numel = self._param_numels[i]
            shape = self._param_shapes[i]
            p.data = full_flat[offset:offset + numel].view(shape).to(p.dtype)
            offset += numel

        self._params_gathered = True

    def post_backward(self):
        """训练步 backward 后调用：释放完整参数，只保留 shard

        Backward 完成后，完整参数不再需要，释放以节省显存。
        梯度已通过 hook 自动 ReduceScatter 到本地 shard。
        """
        if not self._params_gathered:
            return  # 尚未 gather，无需释放

        self._release_full_params()
        self._params_gathered = False

    def pre_eval(self):
        """eval/inference 前调用：AllGather 恢复完整参数到模型

        功能与 pre_forward() 相同（AllGather 完整参数），但语义上用于
        eval/generate 路径，不涉及 backward hook 注册。
        幂等：重复调用不会出错（已 gather 则直接跳过）。
        """
        if self._params_gathered:
            return  # 已经 gather 过，跳过

        if self.world_size > 1 and self.group is not None:
            full_flat = torch.empty(
                self.padded_size, dtype=self._buffer_dtype, device=self._device
            )
            dist.all_gather_into_tensor(
                full_flat, self.flat_param_shard, group=self.group
            )
        else:
            full_flat = torch.zeros(
                self.padded_size, dtype=self._buffer_dtype, device=self._device
            )
            full_flat[self.shard_start:self.shard_end].copy_(self.flat_param_shard)

        # 将 full_flat 写回模型参数
        offset = 0
        for i, p in enumerate(self.params):
            numel = self._param_numels[i]
            shape = self._param_shapes[i]
            p.data = full_flat[offset:offset + numel].view(shape).to(p.dtype)
            offset += numel

        self._params_gathered = True

    def post_eval(self):
        """eval/inference 后调用：释放完整参数，恢复参数到分片状态

        幂等：重复调用不会出错（未 gather 则直接跳过）。
        """
        if not self._params_gathered:
            return  # 尚未 gather，无需释放

        self._release_full_params()
        self._params_gathered = False

    @contextmanager
    def eval_mode(self):
        """上下文管理器：在 eval/inference 时安全访问完整参数

        使用方式:
            with optimizer.eval_mode():
                model.eval()
                output = model(input_ids)

        保证进入时参数完整可用，退出时正确释放。
        """
        self.pre_eval()
        try:
            yield
        finally:
            self.post_eval()

    def _release_full_params(self):
        """释放模型中的完整参数，替换为空壳 tensor

        节省显存：模型参数从 full_size 降为 0（实际数据在 flat_param_shard）。
        """
        for p in self.params:
            p.data = torch.empty(0, dtype=self._buffer_dtype, device=self._device)

    def step(self):
        """ZeRO-3 优化器步骤

        流程:
        1. 使用 _local_grad_acc（已通过 hook 在 backward 时累积完成）更新 local_shard
        2. 将更新后的 fp32 master weights 写回 flat_param_shard（bf16）
        3. 清零梯度累加器
        
        注意：不需要 AllGather 全量参数（下次 forward 时按需 gather）
        """
        if self.local_shard.grad is None:
            self.local_shard.grad = torch.zeros_like(self.local_shard)
        self.local_shard.grad.copy_(self._local_grad_acc)  # fp32 → fp32 in-place
        self.local_optimizer.step()

        self.flat_param_shard.copy_(self.local_shard.data.to(self._buffer_dtype))

        self._local_grad_acc.zero_()

    def zero_grad(self):
        """清零所有参数梯度

        ZeRO-3 中梯度在 ReduceScatter 后已被释放（p.grad = None），
        此方法主要用于确保清理状态和重置计数器。
        """
        for p in self.params:
            if p.grad is not None:
                p.grad = None  # 直接释放
        self._grad_count = 0
        self._local_grad_acc.zero_()

    def clip_grad_norm(self, max_norm: float) -> float:
        """ZeRO-3 专用梯度裁剪（基于 _local_grad_acc）

        ZeRO-3 的 _reduce_scatter_grads 清空了所有 p.grad = None，
        因此不能依赖 p.grad 读取梯度。本方法直接操作 _local_grad_acc。

        Returns:
            total_norm: 全局梯度范数（正常值）
            -1.0: 检测到 NaN/Inf
        """
        if torch.isnan(self._local_grad_acc).any() or torch.isinf(self._local_grad_acc).any():
            logger.warning("NaN/Inf detected in ZeRO-3 local gradients")
            return -1.0

        local_norm_sq = self._local_grad_acc.float().norm(2).item() ** 2
        total_norm_sq = torch.tensor(local_norm_sq, device=self._device)

        if self.world_size > 1 and self.group is not None:
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM, group=self.group)

        total_norm = total_norm_sq.sqrt().item()

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            self._local_grad_acc.mul_(clip_coef)

        return total_norm

    @property
    def state_memory_usage(self) -> int:
        """返回当前 rank 的优化器状态显存占用（字节）

        说明:
        - Adam 每个参数有 2 个 fp32 状态 (exp_avg, exp_avg_sq)
        - 当前 rank 只管理 shard_size 个参数
        - 总占用 = shard_size * 2 * 4 bytes (fp32)
        """
        total_bytes = 0
        for state in self.local_optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.numel() * v.element_size()
        return total_bytes

    @property
    def total_optimizer_state_savings(self) -> str:
        """返回相比标准优化器的显存节省描述"""
        standard_size = self.param_count * 2 * 4  # 2 states * fp32
        zero_size = self.shard_size * 2 * 4
        # ZeRO-3 额外节省: 参数也分片（训练步之间只保留 1/N）
        param_dtype_size = torch.finfo(self._buffer_dtype).bits // 8
        full_param_mem = self.param_count * param_dtype_size
        shard_param_mem = self.shard_size * param_dtype_size
        savings = (1 - zero_size / standard_size) * 100
        return (
            f"Standard Adam: {standard_size / 1024**2:.1f} MB | "
            f"ZeRO-3 (rank {self.rank}): {zero_size / 1024**2:.1f} MB | "
            f"Optimizer savings: {savings:.1f}% | "
            f"Param memory: {full_param_mem / 1024**2:.1f} MB → "
            f"{shard_param_mem / 1024**2:.1f} MB/rank"
        )

    def __repr__(self) -> str:
        return (
            f"ZeROStage3Optimizer(params={self.param_count}, shard_size={self.shard_size}, "
            f"rank={self.rank}/{self.world_size}, "
            f"optimizer={self.local_optimizer.__class__.__name__})"
        )
