"""
通信性能Profiler

测量和分析分布式通信的性能指标：
- AllReduce/AllGather/ReduceScatter 延迟
- 实际带宽 vs 理论带宽
- 通信量统计（bytes sent/received）
- 通信占总时间的比例
- Overlap效率（异步通信实际隐藏了多少延迟）

"""

import torch
import torch.cuda as cuda
import torch.distributed as dist
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import time


@dataclass
class CommRecord:
    """单次通信记录"""
    op_type: str           # "all_reduce", "all_gather", "reduce_scatter"
    data_size_bytes: int   # 通信数据量
    latency_ms: float      # 延迟（毫秒）
    bandwidth_gbps: float  # 实际带宽 (GB/s)
    overlap_hidden_ms: float = 0.0  # overlap隐藏的延迟
    timestamp: float = 0.0
    world_size: int = 1    # 参与的rank数


class CommProfiler:
    """通信性能Profiler

    用法:
        profiler = CommProfiler()
        with profiler.profile_op("all_reduce", tensor):
            dist.all_reduce(tensor, group=group)
        print(profiler.report())
    """

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: 是否启用profiling。生产环境可关闭以避免开销。
        """
        self.enabled = enabled
        self.records: List[CommRecord] = []
        self._total_compute_time_ms: float = 0.0  # 用于计算通信占比

    @contextmanager
    def profile_op(self, op_type: str, tensor: torch.Tensor, world_size: int = 1):
        """Profile一次通信操作

        使用CUDA Event精确测量GPU上的通信延迟。

        Args:
            op_type: 操作类型 ("all_reduce", "all_gather", "reduce_scatter")
            tensor: 通信的tensor（用于计算数据量）
            world_size: 进程组大小
        """
        if not self.enabled:
            yield
            return

        data_size = tensor.nelement() * tensor.element_size()

        if torch.cuda.is_available():
            # CUDA Event精确计时
            start_event = cuda.Event(enable_timing=True)
            end_event = cuda.Event(enable_timing=True)

            start_event.record()
            yield
            end_event.record()

            # 同步以获取精确时间
            cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            # CPU fallback（用于调试）
            start_t = time.perf_counter()
            yield
            latency_ms = (time.perf_counter() - start_t) * 1000

        # 计算实际带宽
        bandwidth_gbps = self.compute_bandwidth(data_size, latency_ms, op_type, world_size)

        record = CommRecord(
            op_type=op_type,
            data_size_bytes=data_size,
            latency_ms=latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            timestamp=time.time(),
            world_size=world_size,
        )
        self.records.append(record)

    def record_compute_time(self, compute_ms: float):
        """记录计算时间，用于计算通信占比"""
        self._total_compute_time_ms += compute_ms

    def compute_bandwidth(self, data_size: int, latency_ms: float,
                          op_type: str = "all_reduce", world_size: int = 1) -> float:
        """计算实际带宽 (GB/s)

        Args:
            data_size: 原始数据大小(bytes)
            latency_ms: 延迟(ms)
            op_type: 操作类型
            world_size: 进程组大小

        Returns:
            实际带宽 (GB/s)
        """
        if latency_ms <= 0:
            return 0.0

        # 根据操作类型计算有效传输量（Ring算法）
        n = max(world_size, 2)
        if op_type == "all_reduce":
            # Ring AllReduce: 2 * (N-1)/N * data_size
            effective_bytes = 2 * (n - 1) / n * data_size
        elif op_type == "all_gather":
            # AllGather: (N-1)/N * data_size * N = (N-1) * data_size
            effective_bytes = (n - 1) * data_size
        elif op_type == "reduce_scatter":
            # ReduceScatter: (N-1)/N * data_size
            effective_bytes = (n - 1) / n * data_size
        else:
            effective_bytes = data_size

        # bytes → GB, ms → s
        bandwidth_gbps = (effective_bytes / 1e9) / (latency_ms / 1000)
        return bandwidth_gbps

    def get_summary(self) -> dict:
        """获取通信性能摘要

        Returns:
            - 各操作类型的平均延迟
            - 平均带宽利用率
            - 通信占总时间比例
            - 带宽分布（P50/P95/P99）
        """
        if not self.records:
            return {"status": "no_records"}

        summary = {
            "total_records": len(self.records),
            "total_comm_time_ms": sum(r.latency_ms for r in self.records),
            "total_data_transferred_gb": sum(r.data_size_bytes for r in self.records) / 1e9,
        }

        # 按操作类型分组统计
        by_type: Dict[str, List[CommRecord]] = {}
        for r in self.records:
            if r.op_type not in by_type:
                by_type[r.op_type] = []
            by_type[r.op_type].append(r)

        type_stats = {}
        for op_type, records in by_type.items():
            latencies = sorted([r.latency_ms for r in records])
            bandwidths = [r.bandwidth_gbps for r in records]

            type_stats[op_type] = {
                "count": len(records),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p50_latency_ms": latencies[len(latencies) // 2],
                "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
                "p99_latency_ms": latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)],
                "avg_bandwidth_gbps": sum(bandwidths) / len(bandwidths),
                "max_bandwidth_gbps": max(bandwidths),
                "total_bytes": sum(r.data_size_bytes for r in records),
            }
        summary["by_op_type"] = type_stats

        # 通信占比
        total_comm_ms = summary["total_comm_time_ms"]
        total_time_ms = total_comm_ms + self._total_compute_time_ms
        if total_time_ms > 0:
            summary["comm_ratio"] = total_comm_ms / total_time_ms
        else:
            summary["comm_ratio"] = 0.0

        return summary

    def get_theoretical_bandwidth(self, device: str = "A100") -> float:
        """获取理论带宽 (GB/s)

        Args:
            device: 设备型号

        Returns:
            理论双向带宽 (GB/s)
        """
        bandwidth_map = {
            "A100_NVLINK": 600.0,   # A100 NVLink (8 links * 75 GB/s)
            "A100_PCIE": 64.0,      # A100 PCIe Gen4 x16
            "A100": 600.0,          # 默认NVLink
            "H100_NVLINK": 900.0,   # H100 NVLink
            "H100": 900.0,
            "4090_PCIE": 64.0,      # RTX 4090 PCIe Gen4
            "4090": 64.0,
            "3090_PCIE": 32.0,      # RTX 3090 PCIe Gen4 (x16 = 32 GB/s)
            "3090": 32.0,
            "4060": 18.0,           # RTX 4060 PCIe Gen4 x8
            "4060_PCIE": 18.0,
        }
        return bandwidth_map.get(device.upper(), 64.0)

    def report(self) -> str:
        """生成可读的性能报告

        输出示例：
        ┌──────────────────────────────────────────────────────┐
        │           通信性能 Profiling Report                    │
        ├──────────────────────────────────────────────────────┤
        │ AllReduce x 100                                      │
        │   Avg Latency: 0.45ms  |  P99: 1.2ms               │
        │   Avg Bandwidth: 120 GB/s (NVLink利用率: 20%)        │
        │   Total Data: 3.2 GB                                 │
        ├──────────────────────────────────────────────────────┤
        │ 通信占比: 35% (通信 12.5s / 总计 35.7s)              │
        │ 优化建议: 通信占比>30%, 建议增大batch或使用overlap    │
        └──────────────────────────────────────────────────────┘
        """
        summary = self.get_summary()
        if "status" in summary:
            return "No communication records to report."

        lines = []
        lines.append("┌" + "─" * 58 + "┐")
        lines.append("│" + "通信性能 Profiling Report".center(50) + "        │")
        lines.append("├" + "─" * 58 + "┤")

        # 按操作类型输出
        for op_type, stats in summary.get("by_op_type", {}).items():
            lines.append(f"│  {op_type} x {stats['count']:<45}│")
            lines.append(
                f"│    Avg Latency: {stats['avg_latency_ms']:.3f}ms  "
                f"|  P99: {stats['p99_latency_ms']:.3f}ms"
                + " " * 10 + "│"
            )
            lines.append(
                f"│    Avg Bandwidth: {stats['avg_bandwidth_gbps']:.1f} GB/s"
                + " " * 25 + "│"
            )
            data_gb = stats['total_bytes'] / 1e9
            lines.append(f"│    Total Data: {data_gb:.2f} GB" + " " * 32 + "│")
            lines.append("├" + "─" * 58 + "┤")

        # 通信占比
        comm_ratio = summary.get("comm_ratio", 0.0)
        total_comm = summary["total_comm_time_ms"] / 1000
        lines.append(f"│  通信占比: {comm_ratio:.1%}" + " " * 40 + "│")
        lines.append(f"│  总通信时间: {total_comm:.2f}s" + " " * 35 + "│")

        # 优化建议
        if comm_ratio > 0.3:
            lines.append("│  ⚠ 建议: 通信占比>30%, 考虑增大batch/使用overlap" + " " * 5 + "│")
        elif comm_ratio > 0.1:
            lines.append("│  ✓ 通信占比适中，系统较平衡" + " " * 22 + "│")
        else:
            lines.append("│  ✓ 通信占比低，计算密集型workload" + " " * 17 + "│")

        lines.append("└" + "─" * 58 + "┘")
        return "\n".join(lines)

    def reset(self):
        """重置所有记录"""
        self.records.clear()
        self._total_compute_time_ms = 0.0
