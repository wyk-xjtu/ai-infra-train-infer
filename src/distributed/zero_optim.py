"""
Mini-ZeRO Stage 1: 优化器状态分片 (Optimizer State Partitioning)

核心思想:
- 每个 rank 只持有 1/world_size 的优化器状态 (Adam 的 momentum m 和 variance v)
- forward/backward: 正常执行（所有 rank 都有完整参数和完整梯度）
- optimizer.step():
  1. ReduceScatter 梯度 → 每个 rank 得到自己负责的那部分参数的**聚合**梯度
  2. 本地 optimizer 更新自己负责的参数分片
  3. AllGather 更新后的参数 → 所有 rank 恢复完整参数

显存节省分析 (以 Adam 为例):
- 标准 Adam: 每个参数需 2 个 float32 状态 (m, v) → 8 bytes/param
- ZeRO-1: 每个 rank 只存 8 * params / world_size 字节的优化器状态
- 对于 7B 模型 4卡: 优化器状态从 56GB → 14GB/卡

设计决策:
1. 参数平坦化 (flatten): 将所有参数拼成一个连续 buffer，便于切分和通信
   - 避免逐参数通信的 overhead
   - 参数必须按相同顺序排列（所有 rank 一致）
2. 使用 torch 原生的 reduce_scatter_tensor / all_gather_into_tensor 实现高效通信
3. 混合精度: 参数/梯度可以是 fp16/bf16，但优化器状态始终是 fp32（保证数值稳定）

使用方式:
    optimizer = ZeROOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        parallel_context=ctx,
        lr=1e-4, weight_decay=0.01
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""

from typing import Optional, Type, Iterator

import torch
import torch.nn as nn
import torch.distributed as dist

from .parallel_context import ParallelContext
from ..utils.logger import get_logger

logger = get_logger("distributed.zero_v1")


class ZeROOptimizer:
    """Mini-ZeRO Stage 1 优化器

    将参数平坦化后按 rank 分片，每个 rank 只维护自己分片的优化器状态。
    通过 ReduceScatter + 本地更新 + AllGather 三步完成一次优化器步骤。
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
        self.param_count = sum(p.numel() for p in self.params)

        # 对齐到 world_size 的整数倍（padding）
        # [显存优化] flat_params/flat_grad 使用模型参数的 dtype（bf16/fp16）而非 fp32
        # 这两个 buffer 仅用于通信（ReduceScatter/AllGather），不需要 fp32 精度
        # local_shard 仍保持 fp32（AdamW master weights 需要高精度）
        self.padded_size = self._pad_to_divisible(self.param_count, self.world_size)
        self._buffer_dtype = self.params[0].dtype  # 跟随模型参数 dtype
        self.flat_params = torch.zeros(
            self.padded_size, dtype=self._buffer_dtype, device=self.params[0].device
        )
        # 将参数数据拷贝到 flat buffer
        self._copy_params_to_flat()

        self.shard_size = self.padded_size // self.world_size
        self.shard_start = self.rank * self.shard_size
        self.shard_end = self.shard_start + self.shard_size

        # local_shard 使用 fp32（AdamW master weights 需要高精度）
        # 即使 flat_params 是 bf16，local_shard 仍转为 fp32
        self.local_shard = self.flat_params[self.shard_start:self.shard_end].clone().detach().float().requires_grad_(True)
        self.local_optimizer = optimizer_class([self.local_shard], **optimizer_kwargs)

        # 为梯度准备 flat buffer（与 flat_params 同 dtype，用于通信）
        self.flat_grad = torch.zeros(
            self.padded_size, dtype=self._buffer_dtype, device=self.params[0].device
        )

        logger.info(
            "ZeROStage1Optimizer initialized: params=%d, padded=%d, shard_size=%d, "
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

    def _collect_grads_to_flat(self):
        """将所有参数的梯度收集到 flat_grad buffer"""
        self.flat_grad.zero_()
        offset = 0
        for p in self.params:
            numel = p.numel()
            if p.grad is not None:
                self.flat_grad[offset:offset + numel].copy_(p.grad.data.view(-1).to(self._buffer_dtype))
            offset += numel

    def step(self):
        """ZeRO-1 优化器步骤

        三步流程:
        1. ReduceScatter: 梯度聚合并分发 → 每个 rank 得到自己分片的完整梯度
        2. 本地 optimizer.step(): 用聚合梯度更新本地参数分片
        3. AllGather: 参数同步 → 所有 rank 恢复完整的更新后参数
        """
        self._collect_grads_to_flat()

        if self.world_size > 1 and self.group is not None:
            # 输入: flat_grad [padded_size] (每个 rank 有自己的局部梯度)
            # 效果等价于: AllReduce → 取自己负责的 shard
            # 但通信量减半: AllReduce = 2*(N-1)/N * data, ReduceScatter = (N-1)/N * data
            local_grad = torch.empty(
                self.shard_size, dtype=self._buffer_dtype, device=self.flat_grad.device
            )
            dist.reduce_scatter_tensor(
                local_grad, self.flat_grad, op=dist.ReduceOp.SUM, group=self.group
            )
            # ReduceScatter 后取平均（MEAN 语义）
            # 根因：SUM聚合后梯度规模随DP线性增长，需要平均化以保证
            # 不同DP规模下训练等价（固定LR场景）
            local_grad = local_grad / self.world_size
        else:
            # 单卡模式：直接取对应分片
            local_grad = self.flat_grad[self.shard_start:self.shard_end].clone()

        # 只更新当前 rank 负责的参数分片
        # 直接原地拷贝梯度到 local_shard.grad（自动 dtype 转换，不创建中间 tensor）
        if self.local_shard.grad is None:
            self.local_shard.grad = torch.zeros_like(self.local_shard)
        self.local_shard.grad.copy_(local_grad)  # bf16 → fp32 in-place
        self.local_optimizer.step()

        # 将更新后的 local shard (fp32) 写回 flat_params (bf16/fp16)
        self.flat_params[self.shard_start:self.shard_end].copy_(self.local_shard.data.to(self._buffer_dtype))

        if self.world_size > 1 and self.group is not None:
            dist.all_gather_into_tensor(
                self.flat_params, self.flat_params[self.shard_start:self.shard_end].contiguous(),
                group=self.group
            )

        self._copy_flat_to_params()

    def zero_grad(self):
        """清零所有参数梯度"""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        self.flat_grad.zero_()

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
        savings = (1 - zero_size / standard_size) * 100
        return (
            f"Standard Adam: {standard_size / 1024**2:.1f} MB | "
            f"ZeRO-1 (rank {self.rank}): {zero_size / 1024**2:.1f} MB | "
            f"Savings: {savings:.1f}%"
        )

    def __repr__(self) -> str:
        return (
            f"ZeROOptimizer(params={self.param_count}, shard_size={self.shard_size}, "
            f"rank={self.rank}/{self.world_size}, "
            f"optimizer={self.local_optimizer.__class__.__name__})"
        )
