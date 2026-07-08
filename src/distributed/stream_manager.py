"""
CUDA Stream 管理器

核心思想：
- 默认情况下，所有CUDA操作在default stream上串行执行
- 通过创建独立的compute stream和comm stream，可以让计算和通信并行
- AllReduce可以在comm stream上异步执行，同时compute stream继续下一层的前向计算
"""

import torch
import torch.cuda as cuda
from typing import Optional, Dict
from contextlib import contextmanager
import time


class StreamManager:
    """多CUDA Stream管理器

    管理以下stream:
    - compute_stream: 主计算stream（前向/反向的矩阵乘法）
    - comm_stream: 通信stream（AllReduce/AllGather等集合通信）
    - transfer_stream: 数据传输stream（CPU↔GPU, GPU↔GPU权重传输）
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: CUDA设备。None时使用当前设备。
        """
        self.device = device or (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._enabled = torch.cuda.is_available()

        if self._enabled:
            # compute stream: 默认优先级(0)
            self.compute_stream = cuda.Stream(device=self.device, priority=0)
            # comm stream: 高优先级(-1)，确保通信不被计算kernel阻塞
            self.comm_stream = cuda.Stream(device=self.device, priority=-1)
            # transfer stream: 用于CPU↔GPU数据搬运
            self.transfer_stream = cuda.Stream(device=self.device, priority=0)
        else:
            self.compute_stream = None
            self.comm_stream = None
            self.transfer_stream = None

        # 统计信息
        self._compute_ops = 0
        self._comm_ops = 0
        self._transfer_ops = 0
        self._sync_events = 0

    @contextmanager
    def compute_scope(self):
        """在compute stream中执行计算"""
        if not self._enabled:
            yield
            return
        with cuda.stream(self.compute_stream):
            self._compute_ops += 1
            yield

    @contextmanager
    def comm_scope(self):
        """在comm stream中执行通信"""
        if not self._enabled:
            yield
            return
        with cuda.stream(self.comm_stream):
            self._comm_ops += 1
            yield

    @contextmanager
    def transfer_scope(self):
        """在transfer stream中执行数据传输

        用途：权重从训练引擎搬运到推理引擎（CPU→GPU或GPU→GPU）
        独立stream确保传输不阻塞计算和通信。
        """
        if not self._enabled:
            yield
            return
        with cuda.stream(self.transfer_stream):
            self._transfer_ops += 1
            yield

    def sync_comm_to_compute(self):
        """等待comm stream完成，然后compute stream才继续

        实现：在comm stream上record一个event，compute stream wait这个event。
        这样compute stream后续的操作会等comm完成后才执行，
        但CPU不会被阻塞（非阻塞同步）。

        使用场景：AllReduce完成后，下一步计算需要使用AllReduce的结果
        """
        if not self._enabled:
            return
        event = self.comm_stream.record_event()
        self.compute_stream.wait_event(event)
        self._sync_events += 1

    def sync_compute_to_comm(self):
        """等待compute stream完成，然后comm stream才继续

        使用场景：计算产生了需要通信的tensor，通信必须等计算完成
        例如：RowParallel的local matmul完成后，才能发起AllReduce
        """
        if not self._enabled:
            return
        event = self.compute_stream.record_event()
        self.comm_stream.wait_event(event)
        self._sync_events += 1

    def sync_compute_to_transfer(self):
        """等待compute完成后再开始传输"""
        if not self._enabled:
            return
        event = self.compute_stream.record_event()
        self.transfer_stream.wait_event(event)
        self._sync_events += 1

    def sync_transfer_to_compute(self):
        """等待传输完成后再开始计算"""
        if not self._enabled:
            return
        event = self.transfer_stream.record_event()
        self.compute_stream.wait_event(event)
        self._sync_events += 1

    def sync_all(self):
        """全局同步所有stream"""
        if not self._enabled:
            return
        self.compute_stream.synchronize()
        self.comm_stream.synchronize()
        self.transfer_stream.synchronize()

    def get_stream_stats(self) -> dict:
        """获取各stream的使用统计

        Returns:
            包含各stream操作计数和同步事件数的字典
        """
        return {
            "compute_ops": self._compute_ops,
            "comm_ops": self._comm_ops,
            "transfer_ops": self._transfer_ops,
            "sync_events": self._sync_events,
            "enabled": self._enabled,
            "device": str(self.device),
        }

    def reset_stats(self):
        """重置统计计数"""
        self._compute_ops = 0
        self._comm_ops = 0
        self._transfer_ops = 0
        self._sync_events = 0

    def __repr__(self) -> str:
        if not self._enabled:
            return "StreamManager(disabled - no CUDA)"
        return (
            f"StreamManager(device={self.device}, "
            f"compute_ops={self._compute_ops}, "
            f"comm_ops={self._comm_ops}, "
            f"sync_events={self._sync_events})"
        )
