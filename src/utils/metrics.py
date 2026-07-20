"""
性能指标收集模块

收集:
- 各阶段耗时（通过Timer）
- 吞吐量(tokens/s, iterations/hour)
- 奖励统计
- 支持TensorBoard写入
"""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TimerRecord:
    """单次计时记录"""
    name: str
    start_time: float
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        """计时时长（秒）"""
        if self.end_time <= 0:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class MetricsCollector:
    """性能指标收集器

    收集:
    - 各阶段耗时
    - 吞吐量(tokens/s, iterations/hour)
    - 奖励统计
    - 通信耗时 (comm_time_ms)
    - 梯度范数 (grad_norm)
    - 峰值显存 (peak_memory_gb)
    - 当前学习率 (learning_rate)

    用法:
        collector = MetricsCollector()
        collector.start_timer("training")
        # ... training ...
        duration = collector.stop_timer("training")
        collector.record("reward_mean", 0.75)
        collector.record_step(grad_norm=1.2, learning_rate=1e-5, peak_memory_gb=12.3)
        summary = collector.get_summary()
    """

    def __init__(self):
        self._active_timers: Dict[str, TimerRecord] = {}
        self._completed_timers: Dict[str, List[TimerRecord]] = {}
        self._metrics: Dict[str, List[float]] = {}
        self._start_time: float = time.time()

    def start_timer(self, name: str):
        """开始计时

        Args:
            name: 计时器名称（如 "training", "inference", "weight_sync"）
        """
        self._active_timers[name] = TimerRecord(
            name=name, start_time=time.time()
        )

    def stop_timer(self, name: str) -> float:
        """停止计时并返回耗时

        Args:
            name: 计时器名称

        Returns:
            float: 耗时（秒）

        Raises:
            KeyError: 如果计时器不存在
        """
        if name not in self._active_timers:
            raise KeyError(f"Timer '{name}' not started")

        record = self._active_timers.pop(name)
        record.end_time = time.time()

        if name not in self._completed_timers:
            self._completed_timers[name] = []
        self._completed_timers[name].append(record)

        return record.duration

    def record(self, key: str, value: float):
        """记录一个指标值

        Args:
            key: 指标名称（如 "reward_mean", "loss", "accuracy"）
            value: 指标值
        """
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value)

    def record_step(
        self,
        comm_time_ms: Optional[float] = None,
        grad_norm: Optional[float] = None,
        peak_memory_gb: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """批量记录一步训练的可观测性指标

        Args:
            comm_time_ms: AllReduce 通信耗时（毫秒）
            grad_norm: 梯度范数
            peak_memory_gb: 峰值显存（GB）
            learning_rate: 当前学习率
        """
        if comm_time_ms is not None:
            self.record("comm_time_ms", comm_time_ms)
        if grad_norm is not None:
            self.record("grad_norm", grad_norm)
        if peak_memory_gb is not None:
            self.record("peak_memory_gb", peak_memory_gb)
        if learning_rate is not None:
            self.record("learning_rate", learning_rate)

    def get_summary(self) -> Dict:
        """生成性能摘要

        Returns:
            包含各阶段平均耗时、指标统计等的字典
        """
        total_elapsed = time.time() - self._start_time
        summary: Dict = {
            "total_elapsed_s": total_elapsed,
        }

        # Timer统计
        timer_stats = {}
        for name, records in self._completed_timers.items():
            durations = [r.duration for r in records]
            timer_stats[name] = {
                "count": len(durations),
                "total_s": sum(durations),
                "mean_s": sum(durations) / len(durations) if durations else 0,
                "min_s": min(durations) if durations else 0,
                "max_s": max(durations) if durations else 0,
            }
        summary["timers"] = timer_stats

        # Metrics统计
        metrics_stats = {}
        for key, values in self._metrics.items():
            metrics_stats[key] = {
                "count": len(values),
                "mean": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "last": values[-1] if values else 0,
            }
        summary["metrics"] = metrics_stats

        # 吞吐量
        if "iteration" in self._completed_timers:
            n_iters = len(self._completed_timers["iteration"])
            if total_elapsed > 0:
                summary["throughput"] = {
                    "iterations_total": n_iters,
                    "iterations_per_hour": n_iters / total_elapsed * 3600,
                }

        return summary

    def to_tensorboard(self, writer, step: int):
        """写入TensorBoard

        Args:
            writer: torch.utils.tensorboard.SummaryWriter实例
            step: 当前全局步数
        """
        # 写入最新metrics
        for key, values in self._metrics.items():
            if values:
                writer.add_scalar(f"metrics/{key}", values[-1], step)

        # 写入timer耗时（最新一次）
        for name, records in self._completed_timers.items():
            if records:
                writer.add_scalar(
                    f"timers/{name}_duration_s", records[-1].duration, step
                )

    def reset(self):
        """重置所有收集的指标"""
        self._active_timers.clear()
        self._completed_timers.clear()
        self._metrics.clear()
        self._start_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """从创建到现在的总耗时（秒）"""
        return time.time() - self._start_time
