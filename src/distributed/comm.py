"""
分布式通信原语封装 (Communication Primitives)

对 torch.distributed 的高层封装，提供:
- 类型安全的通信操作
- 单卡模式下的 no-op 退化（group=None 时直接返回）
- 用于反向传播的自定义 autograd Function（Megatron 风格）

设计决策:
1. 所有通信函数接受 group 参数，为 None 时退化为 identity（支持单卡调试）
2. autograd.Function 实现前向/反向的通信配对：
   - _CopyToParallelRegion: 前向 Identity, 反向 AllReduce（用于 ColumnParallel 输入）
   - _ReduceFromParallelRegion: 前向 AllReduce, 反向 Identity（用于 RowParallel 输出）
   这确保了每个 Transformer 层仅需最少通信量。

参考: Megatron-LM mappings.py 中的 f/g 算子对
"""

import logging
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

logger = logging.getLogger(__name__)


def all_reduce(
    tensor: torch.Tensor,
    group: Optional[ProcessGroup],
    op: ReduceOp = ReduceOp.SUM,
) -> torch.Tensor:
    """AllReduce: 所有 rank 的 tensor 求和（或其他归约），结果广播给所有 rank。

    Args:
        tensor: 输入张量（in-place 修改）
        group: 进程组，None 时为 no-op
        op: 归约操作，默认求和

    Returns:
        归约后的张量（与输入共享存储）
    """
    if group is None:
        return tensor
    dist.all_reduce(tensor, op=op, group=group)
    return tensor


def all_gather(tensor: torch.Tensor, group: Optional[ProcessGroup]) -> torch.Tensor:
    """AllGather: 收集所有 rank 的 tensor，沿第 0 维拼接。

    输入 tensor shape: [*, D]
    输出 tensor shape: [* * world_size, D]（沿第0维拼接）

    实际实现: 沿最后一维拼接更常用于TP（gather output_dim），
    这里提供通用版本，沿 dim=0 拼接。
    """
    if group is None:
        return tensor
    world_size = dist.get_world_size(group)
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor, group=group)
    return torch.cat(gather_list, dim=0)


def all_gather_last_dim(
    tensor: torch.Tensor, group: Optional[ProcessGroup]
) -> torch.Tensor:
    """AllGather 并沿最后一维拼接（用于 ColumnParallel gather_output）。

    输入 shape: [..., D/tp_size]
    输出 shape: [..., D]
    """
    if group is None:
        return tensor
    world_size = dist.get_world_size(group)
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor.contiguous(), group=group)
    return torch.cat(gather_list, dim=-1)


def reduce_scatter(
    tensor: torch.Tensor,
    group: Optional[ProcessGroup],
    op: ReduceOp = ReduceOp.SUM,
) -> torch.Tensor:
    """ReduceScatter: 先归约再分发，每个 rank 得到结果的 1/world_size。

    输入 tensor shape: [N, ...]  (N 必须能被 world_size 整除)
    输出 tensor shape: [N/world_size, ...]

    用于 ZeRO 梯度聚合。
    """
    if group is None:
        return tensor
    world_size = dist.get_world_size(group)
    assert tensor.shape[0] % world_size == 0
    chunk_size = tensor.shape[0] // world_size
    output = torch.empty(
        chunk_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
    )
    dist.reduce_scatter_tensor(output, tensor, op=op, group=group)
    return output


def broadcast(
    tensor: torch.Tensor, src: int, group: Optional[ProcessGroup]
) -> torch.Tensor:
    """Broadcast: 从 src rank 广播 tensor 到组内所有 rank。"""
    if group is None:
        return tensor
    dist.broadcast(tensor, src=src, group=group)
    return tensor


class _CopyToParallelRegion(torch.autograd.Function):
    """前向: Identity (输入直接传递给各 rank)
    反向: AllReduce (梯度需要聚合，因为各 rank 对同一输入计算了不同的列切片)

    用于 ColumnParallelLinear 的输入端:
    - 前向时每个 rank 都拿到完整输入 x（不需要通信）
    - 反向时 dL/dx 需要 AllReduce（因为 loss 对 x 的梯度 = sum of 各 rank 的局部梯度）
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Optional[ProcessGroup]) -> torch.Tensor:  # type: ignore
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore
        # 反向: AllReduce 梯度
        if ctx.group is not None:
            # [TP-4修复] 先 clone 再 AllReduce，避免 gradient checkpointing(use_reentrant=False)
            # 下 autograd 复用 grad tensor 时被原地修改覆盖。
            grad_output = grad_output.clone()
            dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """前向: AllReduce (聚合各 rank 的部分结果得到完整输出)
    反向: Identity (梯度直接传回，因为每个 rank 的输入是独立的分片)

    用于 RowParallelLinear 的输出端:
    - 前向时各 rank 的局部结果 y_i = x_i @ W_i 需要 AllReduce 得到 y = sum(y_i)
    - 反向时 dL/dy_i = dL/dy（梯度直接传给各 rank，无需通信）
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Optional[ProcessGroup]) -> torch.Tensor:  # type: ignore
        ctx.group = group
        if group is not None:
            dist.all_reduce(tensor, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore
        # 反向: Identity，梯度直接传回
        return grad_output, None


class _GatherFromParallelRegion(torch.autograd.Function):
    """前向: AllGather (沿最后一维拼接各 rank 的部分输出)
    反向: Split (将梯度按 tp_size 切分回各 rank)

    用于 ColumnParallelLinear 的 gather_output=True 模式。
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Optional[ProcessGroup]) -> torch.Tensor:  # type: ignore
        ctx.group = group
        if group is None:
            return tensor
        return all_gather_last_dim(tensor, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore
        if ctx.group is None:
            return grad_output, None
        # 将梯度按最后一维切分，只保留当前 rank 对应的部分
        world_size = dist.get_world_size(ctx.group)
        rank = dist.get_rank(ctx.group)
        chunks = grad_output.chunk(world_size, dim=-1)
        return chunks[rank].contiguous(), None


def copy_to_parallel_region(
    tensor: torch.Tensor, group: Optional[ProcessGroup]
) -> torch.Tensor:
    """将输入标记为进入并行区域（前向 Identity，反向 AllReduce）"""
    return _CopyToParallelRegion.apply(tensor, group)


def reduce_from_parallel_region(
    tensor: torch.Tensor, group: Optional[ProcessGroup]
) -> torch.Tensor:
    """将输出从并行区域聚合（前向 AllReduce，反向 Identity）"""
    return _ReduceFromParallelRegion.apply(tensor, group)


def gather_from_parallel_region(
    tensor: torch.Tensor, group: Optional[ProcessGroup]
) -> torch.Tensor:
    """将输出从并行区域 AllGather（前向 Gather，反向 Split）"""
    return _GatherFromParallelRegion.apply(tensor, group)


class AsyncAllReduce:
    """异步AllReduce — 通信计算Overlap的核心

    原理：
    - 标准AllReduce: 在default stream上同步执行，计算必须等待通信完成
    - 异步AllReduce: 在独立的comm stream上执行，compute stream可以继续计算

    使用场景（Transformer层内的overlap）:
    1. Attention的RowParallel输出AllReduce时，MLP的LayerNorm可以同时计算
    2. MLP的RowParallel输出AllReduce时，下一层的LayerNorm可以同时计算

    技术要点：
    - overlap的上限是什么？→ min(计算时间, 通信时间)决定了能隐藏多少
    - 什么时候overlap无效？→ 当通信时间远大于可overlap的计算时间
    - 实际能隐藏多少？→ 典型Transformer中约隐藏30-50%的AllReduce延迟
    - Ring AllReduce的通信量？→ 2*(N-1)/N * data_size，与rank数几乎无关
    """

    def __init__(self, group: Optional[ProcessGroup], stream_manager, timeout_sec: float = 300.0):
        """
        Args:
            group: 进程组
            stream_manager: StreamManager实例，提供comm_stream
            timeout_sec: 通信超时时间（秒），默认300秒
        """
        self.group = group
        self.stream_manager = stream_manager
        self.timeout_sec = timeout_sec

    def start(self, tensor: torch.Tensor) -> 'AsyncAllReduce.AsyncHandle':
        """发起异步AllReduce，返回handle

        实现步骤：
        1. 在comm stream上record当前compute stream的进度（确保tensor已就绪）
        2. 在comm stream上执行AllReduce
        3. record完成event，返回handle

        技术要点：
        - 为什么要先sync_compute_to_comm？
          → 确保tensor的计算已完成，否则AllReduce读到的是未完成的数据
        - 为什么AllReduce后不立即同步？
          → 延迟同步让compute stream继续执行不依赖通信结果的计算
        """
        if self.group is None or self.stream_manager is None or not self.stream_manager._enabled:
            # 单卡模式或无CUDA：同步执行
            return AsyncAllReduce.AsyncHandle(tensor=tensor, event=None, stream=None)

        # 确保compute stream上产生tensor的操作已完成
        self.stream_manager.sync_compute_to_comm()

        # 在comm stream上执行AllReduce
        with self.stream_manager.comm_scope():
            dist.all_reduce(tensor, op=ReduceOp.SUM, group=self.group)
            # record完成event
            done_event = self.stream_manager.comm_stream.record_event()

        return AsyncAllReduce.AsyncHandle(
            tensor=tensor,
            event=done_event,
            stream=self.stream_manager.comm_stream,
            timeout_sec=self.timeout_sec,
            start_time=time.perf_counter(),
        )

    class AsyncHandle:
        """异步操作handle

        技术要点：
        - handle的作用？→ 允许调用方选择何时等待通信完成
        - 延迟wait的好处？→ wait之前的计算与通信并行执行（overlap）
        - 超时防护？→ 避免单个rank故障导致整个训练无限挂起
        """

        def __init__(self, tensor: torch.Tensor, event, stream,
                     timeout_sec: float = 300.0, start_time=None):
            self.tensor = tensor
            self._event = event
            self._stream = stream
            self._timeout_sec = timeout_sec
            self._start_time = start_time or time.perf_counter()
            self._warned = False  # 确保 warning 只触发一次

        def wait(self):
            """等待AllReduce完成，带超时检测

            在compute stream上插入对comm event的等待，
            之后compute stream的操作可以安全使用AllReduce的结果。

            超时检测策略：
            - 快速路径：event已完成时直接同步，零额外开销
            - 慢速路径：polling + 超时检测，10ms间隔
            - 超过50% timeout时发出一次warning
            - 超时后抛出RuntimeError
            """
            if self._event is None:
                return  # 单卡模式直接返回

            # 快速路径：event已完成，直接同步
            if self._event.query():
                torch.cuda.current_stream().wait_event(self._event)
                return

            # 慢速路径：polling + 超时检测
            while not self._event.query():
                elapsed = time.perf_counter() - self._start_time
                if elapsed > self._timeout_sec:
                    raise RuntimeError(
                        f"AsyncAllReduce timeout after {elapsed:.1f}s "
                        f"(limit: {self._timeout_sec}s). "
                        f"Possible causes: rank failure, NCCL deadlock, "
                        f"network partition. Check all ranks are alive."
                    )
                if not self._warned and elapsed > self._timeout_sec * 0.5:
                    logger.warning(
                        f"AsyncAllReduce waiting for {elapsed:.1f}s "
                        f"(timeout at {self._timeout_sec}s)"
                    )
                    self._warned = True
                time.sleep(0.01)  # 10ms polling interval

            # 完成后同步 stream
            torch.cuda.current_stream().wait_event(self._event)

        @property
        def is_done(self) -> bool:
            """检查通信是否已完成（非阻塞查询）"""
            if self._event is None:
                return True
            return self._event.query()


def all_reduce_async(
    tensor: torch.Tensor,
    group: Optional[ProcessGroup],
    stream_manager,
    timeout_sec: float = 300.0,
) -> AsyncAllReduce.AsyncHandle:
    """便捷函数：发起异步AllReduce

    用法:
        handle = all_reduce_async(tensor, group, stream_manager)
        # ... 这里可以做不依赖tensor结果的计算（被overlap掉的部分）...
        handle.wait()  # 同步点：之后才能使用tensor

    Args:
        tensor: 要AllReduce的张量（in-place修改）
        group: 进程组
        stream_manager: StreamManager实例
        timeout_sec: 通信超时时间（秒），默认300秒

    Returns:
        AsyncHandle: 异步操作句柄
    """
    async_op = AsyncAllReduce(group=group, stream_manager=stream_manager, timeout_sec=timeout_sec)
    return async_op.start(tensor)


#
# 设计要点:
# - 统一使用 torch.distributed.batch_isend_irecv([P2POp...]) 后 wait()，
#   而非裸 dist.send/dist.recv。原因：1F1B 稳态下相邻 stage 会同时
#   "send 前向 activation" 与 "send 反向 grad"，若用同步 send/recv 且
#   两侧都先 send，会造成双向阻塞死锁；batch_isend_irecv 会把一批
#   收发操作打包后由后端统一调度，避免顺序依赖导致的死锁。
# - peer（prev_rank / next_rank）为全局 rank（ParallelContext 提供的
#   global_rank ∓ tp_size），与 torch 点对点接口的 rank 语义一致。
#   （无法在无通信组时凭空生成远端数据）。


def _batch_p2p(ops: list) -> None:
    """执行一批 P2POp 并等待全部完成。"""
    if not ops:
        return
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


def send_forward(
    tensor: torch.Tensor,
    next_rank: Optional[int],
    pp_group: Optional[ProcessGroup],
) -> None:
    """前向：发送 activation 到下一个 stage（next_rank）。

    Args:
        tensor: 要发送的 hidden_states
        next_rank: 下一个 stage 的全局 rank；None（最后一个 stage）时 no-op
        pp_group: PP 通信组；None 时 no-op
    """
    if pp_group is None or next_rank is None:
        return
    op = dist.P2POp(dist.isend, tensor.contiguous(), next_rank, group=pp_group)
    _batch_p2p([op])


def recv_forward(
    shape,
    dtype: torch.dtype,
    device,
    prev_rank: Optional[int],
    pp_group: Optional[ProcessGroup],
) -> torch.Tensor:
    """前向：从上一个 stage（prev_rank）接收 activation。

    Args:
        shape: 接收张量的形状
        dtype: 接收张量的 dtype
        device: 接收张量所在设备
        prev_rank: 上一个 stage 的全局 rank
        pp_group: PP 通信组；None 时抛错（无通信组无法接收）

    Returns:
        接收到的张量
    """
    if pp_group is None:
        raise RuntimeError(
            "recv_forward requires a valid pp_group, but got None "
            "(no pipeline parallel group available)."
        )
    tensor = torch.empty(shape, dtype=dtype, device=device)
    op = dist.P2POp(dist.irecv, tensor, prev_rank, group=pp_group)
    _batch_p2p([op])
    return tensor


def send_backward(
    grad: torch.Tensor,
    prev_rank: Optional[int],
    pp_group: Optional[ProcessGroup],
) -> None:
    """反向：发送梯度到上一个 stage（prev_rank）。

    Args:
        grad: 要发送的 grad_hidden_states
        prev_rank: 上一个 stage 的全局 rank；None（第一个 stage）时 no-op
        pp_group: PP 通信组；None 时 no-op
    """
    if pp_group is None or prev_rank is None:
        return
    op = dist.P2POp(dist.isend, grad.contiguous(), prev_rank, group=pp_group)
    _batch_p2p([op])


def recv_backward(
    shape,
    dtype: torch.dtype,
    device,
    next_rank: Optional[int],
    pp_group: Optional[ProcessGroup],
) -> torch.Tensor:
    """反向：从下一个 stage（next_rank）接收梯度。

    Args:
        shape: 接收梯度的形状
        dtype: 接收梯度的 dtype
        device: 接收梯度所在设备
        next_rank: 下一个 stage 的全局 rank
        pp_group: PP 通信组；None 时抛错（无通信组无法接收）

    Returns:
        接收到的梯度张量
    """
    if pp_group is None:
        raise RuntimeError(
            "recv_backward requires a valid pp_group, but got None "
            "(no pipeline parallel group available)."
        )
    tensor = torch.empty(shape, dtype=dtype, device=device)
    op = dist.P2POp(dist.irecv, tensor, next_rank, group=pp_group)
    _batch_p2p([op])
    return tensor


def send_forward_recv_backward(
    output_tensor: torch.Tensor,
    grad_shape,
    dtype: torch.dtype,
    device,
    next_rank: Optional[int],
    pp_group: Optional[ProcessGroup],
) -> Optional[torch.Tensor]:
    """组合原语（Megatron 风格）：一次性向 next_rank 发送前向 activation，
    同时从 next_rank 接收反向 grad。二者打包进同一 batch_isend_irecv，
    避免稳态双向通信死锁。

    Returns:
        接收到的 grad（形状 grad_shape）；当 pp_group/next_rank 为 None 时返回 None。
    """
    if pp_group is None or next_rank is None:
        return None
    recv_grad = torch.empty(grad_shape, dtype=dtype, device=device)
    ops = [
        dist.P2POp(dist.isend, output_tensor.contiguous(), next_rank, group=pp_group),
        dist.P2POp(dist.irecv, recv_grad, next_rank, group=pp_group),
    ]
    _batch_p2p(ops)
    return recv_grad


def send_backward_recv_forward(
    input_grad: torch.Tensor,
    act_shape,
    dtype: torch.dtype,
    device,
    prev_rank: Optional[int],
    pp_group: Optional[ProcessGroup],
) -> Optional[torch.Tensor]:
    """组合原语（Megatron 风格）：一次性向 prev_rank 发送反向 grad，
    同时从 prev_rank 接收前向 activation。二者打包进同一 batch_isend_irecv，
    避免稳态双向通信死锁。

    Returns:
        接收到的 activation（形状 act_shape）；当 pp_group/prev_rank 为 None 时返回 None。
    """
    if pp_group is None or prev_rank is None:
        return None
    recv_act = torch.empty(act_shape, dtype=dtype, device=device)
    ops = [
        dist.P2POp(dist.isend, input_grad.contiguous(), prev_rank, group=pp_group),
        dist.P2POp(dist.irecv, recv_act, prev_rank, group=pp_group),
    ]
    _batch_p2p(ops)
    return recv_act
