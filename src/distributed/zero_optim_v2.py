"""
Mini-ZeRO Stage 2: 优化器状态 + 梯度分片 (Optimizer States + Gradients Partitioning)

核心思想:
- 在 ZeRO-1 基础上，梯度也进行分片——不保留全量 flat_grad buffer
- backward 时通过 gradient hook 检测所有参数梯度就绪，触发 ReduceScatter
- ReduceScatter 后每个 rank 只保留 1/N 的梯度 shard，立即释放各参数的 .grad

与 ZeRO-1 的区别:
- ZeRO-1: 保留完整 flat_grad buffer（常驻显存），step() 时才 ReduceScatter
- ZeRO-2: 无 flat_grad 常驻 buffer，backward 完成后立即 ReduceScatter 并释放梯度

显存收益:
- 节省 flat_grad 的全量 buffer（8B 模型 bf16 = 16GB）
- 梯度在 ReduceScatter 后只保留 1/N 的 shard（fp32，用于 AdamW 更新）

梯度累积支持:
- 每次 backward 完成后，立即 ReduceScatter 并累加到本地 shard
- step() 时使用累积后的 local_grad 更新参数
- 这样每次 backward 后都能释放全量梯度，显存峰值不随累积步数增长

使用方式:
    optimizer = ZeROStage2Optimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        parallel_context=ctx,
        lr=1e-4, weight_decay=0.01
    )
    loss.backward()  # hook 自动触发 ReduceScatter
    optimizer.step()
    optimizer.zero_grad()
"""

from typing import Optional, Type, Iterator

import torch
import torch.nn as nn
import torch.distributed as dist

from .parallel_context import ParallelContext
from ..utils.logger import get_logger

logger = get_logger("distributed.zero_v2")


class ZeROStage2Optimizer:
    """Mini-ZeRO Stage 2 优化器

    将参数平坦化后按 rank 分片，每个 rank 只维护自己分片的优化器状态。
    与 ZeRO-1 不同，梯度通过 backward hook 在 backward 完成后立即 ReduceScatter，
    不保留全量 flat_grad buffer，节省显存。
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
        else:
            # [X-2修复] dp_size<=1：ZeRO 退化为本地优化器（不分片）。
            # 不再回退 tp_group，避免对 TP 切分参数做语义错误的 ReduceScatter/AllGather。
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
            raise ValueError("ZeROStage2Optimizer: no trainable parameters found")
        self.param_count = sum(p.numel() for p in self.params)

        # Step 1: 创建平坦化的参数 buffer（所有 rank 一致）
        # 对齐到 world_size 的整数倍（padding）
        # flat_params 使用模型参数的 dtype（bf16/fp16）用于通信
        # local_shard 保持 fp32（AdamW master weights 需要高精度）
        self.padded_size = self._pad_to_divisible(self.param_count, self.world_size)
        self._buffer_dtype = self.params[0].dtype  # 跟随模型参数 dtype
        self.flat_params = torch.zeros(
            self.padded_size, dtype=self._buffer_dtype, device=self.params[0].device
        )
        # 将参数数据拷贝到 flat buffer
        self._copy_params_to_flat()

        # Step 2: 计算每个 rank 负责的分片范围
        self.shard_size = self.padded_size // self.world_size
        self.shard_start = self.rank * self.shard_size
        self.shard_end = self.shard_start + self.shard_size

        # Step 3: 为本地分片创建 optimizer
        # local_shard 使用 fp32（AdamW master weights 需要高精度）
        self.local_shard = self.flat_params[self.shard_start:self.shard_end].clone().detach().float().requires_grad_(True)
        self.local_optimizer = optimizer_class([self.local_shard], **optimizer_kwargs)

        # ZeRO-2 核心：不创建 flat_grad buffer，通过 hook 机制处理梯度
        # _local_grad_acc: 本地梯度 shard 累加器（fp32），支持 gradient accumulation
        self._local_grad_acc = torch.zeros(
            self.shard_size, dtype=torch.float32, device=self.params[0].device
        )
        self._grad_count = 0
        self._hooks = []

        # 注册 gradient hooks
        self._register_hooks()

        logger.info(
            "ZeROStage2Optimizer initialized: params=%d, padded=%d, shard_size=%d, "
            "rank=%d/%d, buffer_dtype=%s",
            self.param_count, self.padded_size, self.shard_size,
            self.rank, self.world_size, self._buffer_dtype,
        )

    def _pad_to_divisible(self, size: int, divisor: int) -> int:
        """将 size 向上对齐到 divisor 的整数倍"""
        return size + (divisor - size % divisor) % divisor

    def _copy_params_to_flat(self):
        """将模型参数数据拷贝到平坦化 buffer（保持 buffer dtype）"""
        offset = 0
        for p in self.params:
            numel = p.numel()
            self.flat_params[offset:offset + numel].copy_(p.data.view(-1).to(self._buffer_dtype))
            offset += numel

    def _copy_flat_to_params(self):
        """将平坦化 buffer 的数据拷贝回模型参数"""
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(
                self.flat_params[offset:offset + numel].view(p.shape).to(p.dtype)
            )
            offset += numel

    def _register_hooks(self):
        """为每个参数注册 post_accumulate_grad_hook

        PyTorch 2.1+ 的 register_post_accumulate_grad_hook 在梯度累积完成后调用，
        适合检测所有参数梯度是否就绪。
        """
        for i, p in enumerate(self.params):
            hook = p.register_post_accumulate_grad_hook(
                lambda param, idx=i: self._on_grad_ready(idx, param)
            )
            self._hooks.append(hook)

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
        # [X-1修复] TP 组内先对 replicated 参数梯度 AllReduce(SUM)，
        # 保证完整梯度 = Σ(各TP rank部分梯度)，再交由下游 DP ReduceScatter 做 DP 平均。
        # TP 切分参数不带 _tp_replicated 标记，天然不受影响。
        pc = self.parallel_context
        if pc is not None and pc.tp_size > 1 and pc.tp_group is not None:
            for p in self.params:
                if p.grad is not None and getattr(p, "_tp_replicated", False):
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=pc.tp_group)

        flat_grad = torch.zeros(
            self.padded_size, dtype=self._buffer_dtype, device=self.params[0].device
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
                self.shard_size, dtype=self._buffer_dtype, device=flat_grad.device
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

    def step(self):
        """ZeRO-2 优化器步骤

        流程:
        1. 使用 _local_grad_acc（已通过 hook 在 backward 时累积完成）更新 local_shard
        2. AllGather 恢复完整参数
        3. 写回模型参数
        4. 清零梯度累加器
        """
        # Step 1: 将累积的梯度 shard 赋给 local_shard 并更新
        if self.local_shard.grad is None:
            self.local_shard.grad = torch.zeros_like(self.local_shard)
        self.local_shard.grad.copy_(self._local_grad_acc)  # fp32 → fp32 in-place
        self.local_optimizer.step()

        # Step 2: 将更新后的 local shard (fp32) 写回 flat_params (bf16/fp16)
        self.flat_params[self.shard_start:self.shard_end].copy_(
            self.local_shard.data.to(self._buffer_dtype)
        )

        if self.world_size > 1 and self.group is not None:
            # Step 3: AllGather 参数
            # 每个 rank 广播自己更新后的 shard，所有 rank 恢复完整参数
            dist.all_gather_into_tensor(
                self.flat_params,
                self.flat_params[self.shard_start:self.shard_end].contiguous(),
                group=self.group
            )

        # Step 4: 将更新后的 flat_params 拷贝回模型参数
        self._copy_flat_to_params()

        # Step 5: 清零梯度累加器，为下一轮准备
        self._local_grad_acc.zero_()

    def zero_grad(self):
        """清零所有参数梯度

        ZeRO-2 中梯度在 ReduceScatter 后已被释放（p.grad = None），
        此方法主要用于确保清理状态和重置计数器。
        """
        for p in self.params:
            if p.grad is not None:
                p.grad = None  # 直接释放（不 zero_，节省显存）
        self._grad_count = 0
        # 注意：不清零 _local_grad_acc，因为它在 step() 末尾已被清零
        # 如果 zero_grad 在 step() 之前被调用（异常路径），也需要清零
        self._local_grad_acc.zero_()

    def clip_grad_norm(self, max_norm: float) -> float:
        """ZeRO-2 专用梯度裁剪（基于 _local_grad_acc）

        ZeRO-2 的 _reduce_scatter_grads 清空了所有 p.grad = None，
        因此不能依赖 p.grad 读取梯度。本方法直接操作 _local_grad_acc。

        Returns:
            total_norm: 全局梯度范数（正常值）
            -1.0: 检测到 NaN/Inf
        """
        # [X-4修复] 若 backward 结束后仍有部分参数未触发 hook（部分参数无梯度），
        # 强制 flush 一次 ReduceScatter，避免梯度滞留导致本步更新丢失。
        if 0 < self._grad_count < len(self.params):
            logger.warning("Partial grads ready (%d/%d), forcing reduce-scatter flush",
                           self._grad_count, len(self.params))
            self._reduce_scatter_grads()

        # 1. NaN/Inf 检测
        if torch.isnan(self._local_grad_acc).any() or torch.isinf(self._local_grad_acc).any():
            logger.warning("NaN/Inf detected in ZeRO-2 local gradients")
            return -1.0

        # 2. 计算本 rank 的局部梯度范数平方
        local_norm_sq = self._local_grad_acc.float().norm(2).item() ** 2
        total_norm_sq = torch.tensor(local_norm_sq, device=self._local_grad_acc.device)

        # 3. AllReduce 聚合全局范数（所有 rank 的 shard 范数之和 = 全局范数）
        if self.world_size > 1 and self.group is not None:
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM, group=self.group)

        total_norm = total_norm_sq.sqrt().item()

        # 4. 裁剪
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
        # ZeRO-2 额外节省: 不保留 flat_grad buffer
        flat_grad_savings = self.padded_size * (torch.finfo(self._buffer_dtype).bits // 8)
        savings = (1 - zero_size / standard_size) * 100
        return (
            f"Standard Adam: {standard_size / 1024**2:.1f} MB | "
            f"ZeRO-2 (rank {self.rank}): {zero_size / 1024**2:.1f} MB | "
            f"Optimizer savings: {savings:.1f}% | "
            f"Gradient buffer savings: {flat_grad_savings / 1024**2:.1f} MB"
        )

    def __repr__(self) -> str:
        return (
            f"ZeROStage2Optimizer(params={self.param_count}, shard_size={self.shard_size}, "
            f"rank={self.rank}/{self.world_size}, "
            f"optimizer={self.local_optimizer.__class__.__name__})"
        )
