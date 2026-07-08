"""
并行上下文管理 (Parallel Context Manager)

职责:
- 初始化分布式环境 (torch.distributed)
- 创建和管理张量并行 (TP) 和数据并行 (DP) 进程组
- 提供全局统一的访问接口，避免各模块分散调用 dist API

设计决策:
1. 使用单例模式的静态工厂方法 init_distributed()，确保全局只有一个并行上下文
2. TP进程组在 world_size > 1 时创建；单卡模式下 tp_group 为 None，所有通信操作退化为 no-op
3. DP进程组在 dp_size > 1 时创建；将相同 tp_rank 的进程组成一个 dp_group
4. 支持 torchrun 启动（环境变量自动注入 RANK, WORLD_SIZE, LOCAL_RANK）

进程组布局 (rank % tp_size = tp_rank, rank // tp_size = dp_rank):
    例如 8 GPU, tp_size=2, dp_size=4:
    - TP groups: [0,1], [2,3], [4,5], [6,7]
    - DP groups: [0,2,4,6], [1,3,5,7]

使用方式:
    ctx = ParallelContext.init_distributed(tp_size=2, dp_size=4, backend="nccl")
    print(ctx.tp_rank, ctx.tp_size, ctx.dp_rank, ctx.dp_size)
"""

import os
import socket
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


# 全局单例
_PARALLEL_CONTEXT: Optional["ParallelContext"] = None


class ParallelContext:
    """并行上下文：封装进程组信息，提供统一的分布式状态查询接口。"""

    def __init__(
        self,
        tp_size: int,
        tp_group: Optional[ProcessGroup],
        dp_size: int = 1,
        dp_group: Optional[ProcessGroup] = None,
        pp_size: int = 1,
        pp_group: Optional[ProcessGroup] = None,
    ):
        """
        Args:
            tp_size: 张量并行大小
            tp_group: TP通信组（None表示单卡模式）
            dp_size: 数据并行大小
            dp_group: DP通信组（None表示无数据并行）
            pp_size: 流水线并行大小
            pp_group: PP通信组（None表示无流水线并行）
        """
        self._tp_size = tp_size
        self._tp_group = tp_group
        self._dp_size = dp_size
        self._dp_group = dp_group
        self._pp_size = pp_size
        self._pp_group = pp_group
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._rank = dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def init_distributed(
        tp_size: int = 1,
        dp_size: int = 1,
        pp_size: int = 1,
        backend: str = "nccl",
        init_single_process: bool = False,
    ) -> "ParallelContext":
        """初始化分布式环境，创建TP、DP、PP进程组。

        进程组布局 (rank 公式):
            global_rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank

        - TP组: 连续 tp_size 个 rank (stride=1)
          例如 world_size=8, tp_size=2 → TP groups: [0,1], [2,3], [4,5], [6,7]
        - PP组: stride=tp_size, pp_size 个成员 (仅 pp_size>1 时创建)
          例如 8卡 pp2tp2dp2 → PP groups: [0,2], [1,3], [4,6], [5,7]
        - DP组: stride=pp_size*tp_size, dp_size 个成员
          例如 8卡 pp2tp2dp2 → DP groups: [0,4], [1,5], [2,6], [3,7]
        - 当 pp_size=1 时，DP组 stride 退化为 tp_size，与旧布局完全一致。

        Args:
            tp_size: 张量并行度
            dp_size: 数据并行度
            pp_size: 流水线并行度 (pp_size * tp_size * dp_size 必须等于 world_size)
            backend: 通信后端，默认 "nccl"（GPU），CPU调试可用 "gloo"
            init_single_process: 是否在单进程下初始化（用于测试）

        Returns:
            初始化完成的 ParallelContext 实例
        """
        global _PARALLEL_CONTEXT

        # 从环境变量获取分布式信息（torchrun 会自动设置）
        if not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))

            if world_size > 1 or init_single_process:
                init_method = None
                if world_size == 1 and init_single_process:
                    backend = "gloo"
                    init_method = f"tcp://127.0.0.1:{_find_free_port()}?use_libuv=0"
                dist.init_process_group(
                    backend=backend,
                    world_size=world_size,
                    rank=rank,
                    init_method=init_method,
                )
                if backend == "nccl":
                    torch.cuda.set_device(local_rank)
            else:
                # 单卡模式：不初始化分布式，直接返回
                _PARALLEL_CONTEXT = ParallelContext(
                    tp_size=1, tp_group=None, dp_size=1, dp_group=None,
                    pp_size=1, pp_group=None,
                )
                return _PARALLEL_CONTEXT

        world_size = dist.get_world_size()
        assert world_size % tp_size == 0, (
            f"world_size({world_size}) must be divisible by tp_size({tp_size})"
        )
        assert pp_size * tp_size * dp_size == world_size, (
            f"pp_size({pp_size}) * tp_size({tp_size}) * dp_size({dp_size}) "
            f"must equal world_size({world_size})"
        )

        # 创建TP进程组：每 tp_size 个连续 rank 为一组
        tp_group: Optional[ProcessGroup] = None
        for start in range(0, world_size, tp_size):
            ranks = list(range(start, start + tp_size))
            group = dist.new_group(ranks)
            if dist.get_rank() in ranks:
                tp_group = group

        # 创建PP进程组：stride=tp_size，pp_size 个成员
        # PP group 包含相同 (dp_rank, tp_rank) 但不同 pp_rank 的 rank
        pp_group: Optional[ProcessGroup] = None
        if pp_size > 1:
            for dp_idx in range(dp_size):
                for tp_rank_idx in range(tp_size):
                    base = dp_idx * (pp_size * tp_size) + tp_rank_idx
                    ranks = [base + pp_idx * tp_size for pp_idx in range(pp_size)]
                    group = dist.new_group(ranks)
                    if dist.get_rank() in ranks:
                        pp_group = group

        # 创建DP进程组：stride=pp_size*tp_size，dp_size 个成员
        # DP group 包含相同 (pp_rank, tp_rank) 但不同 dp_rank 的 rank
        # 当 pp_size=1 时，stride 退化为 tp_size，与旧布局一致。
        dp_group: Optional[ProcessGroup] = None
        if dp_size > 1:
            for pp_idx in range(pp_size):
                for tp_rank_idx in range(tp_size):
                    base = pp_idx * tp_size + tp_rank_idx
                    ranks = [
                        base + dp_idx * (pp_size * tp_size) for dp_idx in range(dp_size)
                    ]
                    group = dist.new_group(ranks)
                    if dist.get_rank() in ranks:
                        dp_group = group

        _PARALLEL_CONTEXT = ParallelContext(
            tp_size=tp_size, tp_group=tp_group,
            dp_size=dp_size, dp_group=dp_group,
            pp_size=pp_size, pp_group=pp_group,
        )
        return _PARALLEL_CONTEXT

    @staticmethod
    def get_instance() -> "ParallelContext":
        """获取全局并行上下文实例"""
        if _PARALLEL_CONTEXT is None:
            return ParallelContext.init_distributed(tp_size=1, dp_size=1, backend="gloo", init_single_process=True)
        return _PARALLEL_CONTEXT

    @property
    def tp_rank(self) -> int:
        """当前进程在TP组内的rank"""
        if self._tp_group is None:
            return 0
        return dist.get_rank(self._tp_group)

    @property
    def tp_size(self) -> int:
        """张量并行度"""
        return self._tp_size

    @property
    def tp_group(self) -> Optional[ProcessGroup]:
        """TP通信组（单卡模式为None）"""
        return self._tp_group

    @property
    def dp_rank(self) -> int:
        """当前进程在DP组内的rank (rank // tp_size)"""
        if self._dp_group is None:
            return 0
        return dist.get_rank(self._dp_group)

    @property
    def dp_size(self) -> int:
        """数据并行度"""
        return self._dp_size

    @property
    def dp_group(self) -> Optional[ProcessGroup]:
        """DP通信组（dp_size<=1时为None）"""
        return self._dp_group

    @property
    def pp_size(self) -> int:
        """流水线并行度"""
        return self._pp_size

    @property
    def pp_group(self) -> Optional[ProcessGroup]:
        """PP通信组（pp_size<=1时为None）"""
        return self._pp_group

    @property
    def pp_rank(self) -> int:
        """当前进程在PP组内的rank（stage id）；无PP组时为0"""
        if self._pp_group is None:
            return 0
        return dist.get_rank(self._pp_group)

    @property
    def prev_rank(self) -> Optional[int]:
        """PP 前一个 stage 的全局 rank（= global_rank - tp_size）。

        当 pp_size<=1 或本进程处于第一个 stage（pp_rank==0）时返回 None。
        """
        if self._pp_size <= 1 or self.pp_rank == 0:
            return None
        return self._rank - self._tp_size

    @property
    def next_rank(self) -> Optional[int]:
        """PP 后一个 stage 的全局 rank（= global_rank + tp_size）。

        当 pp_size<=1 或本进程处于最后一个 stage（pp_rank==pp_size-1）时返回 None。
        """
        if self._pp_size <= 1 or self.pp_rank == self._pp_size - 1:
            return None
        return self._rank + self._tp_size

    @property
    def world_size(self) -> int:
        """全局world_size"""
        return self._world_size

    @property
    def rank(self) -> int:
        """全局rank"""
        return self._rank

    @property
    def local_rank(self) -> int:
        """本地rank（同节点内）"""
        return int(os.environ.get("LOCAL_RANK", "0"))

    def __repr__(self) -> str:
        return (
            f"ParallelContext(rank={self.rank}, world_size={self.world_size}, "
            f"tp_rank={self.tp_rank}, tp_size={self.tp_size}, "
            f"dp_rank={self.dp_rank}, dp_size={self.dp_size}, "
            f"pp_rank={self.pp_rank}, pp_size={self.pp_size})"
        )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
