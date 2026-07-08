"""
调度Gantt图生成器

可视化训推循环的各阶段时间分布：
- 训练阶段（前向/反向/优化器）
- 通信阶段（AllReduce/权重同步）
- 推理阶段（Prefill/Decode）
- 空闲/等待时间
- Sleep/Wake开销

输出格式：
- ASCII Gantt图（终端显示）
- JSON数据（Chrome Trace格式，可在chrome://tracing中查看）
- 利用率统计

技术要点：
- 通过Gantt图一眼看出瓶颈在哪
- GPU idle时间占比 → 调度效率
- Colocate vs Disaggregated的时间线对比
- Chrome Trace格式 → 工业界标准profiling可视化
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import time


# 各阶段的显示字符和分类
_STAGE_CHARS = {
    "training": ("█", "compute"),
    "forward": ("█", "compute"),
    "backward": ("▓", "compute"),
    "optimizer": ("▒", "compute"),
    "inference": ("█", "compute"),
    "prefill": ("█", "compute"),
    "decode": ("▓", "compute"),
    "all_reduce": ("▒", "comm"),
    "sync": ("▒", "comm"),
    "weight_sync": ("▒", "comm"),
    "sleep": ("░", "overhead"),
    "wake": ("░", "overhead"),
    "idle": (" ", "idle"),
    "reward": ("▒", "compute"),
}


@dataclass
class ScheduleEvent:
    """调度事件

    技术要点：
    - 为什么记录device？→ 多GPU场景下需要区分各卡的时间线
    - metadata有什么用？→ 存放通信量、batch_size等附加信息
    """
    name: str           # "training", "inference", "sync", "sleep", "wake", "idle"
    start_time: float   # 绝对时间戳（秒）
    end_time: float     # 绝对时间戳（秒）
    device: str = "GPU:0"  # 设备标识
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """持续时间（毫秒）"""
        return (self.end_time - self.start_time) * 1000

    @property
    def duration_s(self) -> float:
        """持续时间（秒）"""
        return self.end_time - self.start_time

    @property
    def category(self) -> str:
        """事件分类（compute/comm/overhead/idle）"""
        return _STAGE_CHARS.get(self.name, ("?", "other"))[1]


class ScheduleVisualizer:
    """调度时间线可视化器

    技术要点：
    - 为什么需要可视化？→ 定位性能瓶颈最直观的方式
    - GPU利用率怎么算？→ (compute时间 + comm时间) / 总时间
    - Colocate的idle问题？→ 训练和推理交替，切换开销（sleep/wake）是纯开销
    - 怎么减少idle？→ 更好的调度策略（pipeline、异步切换）

    用法:
        viz = ScheduleVisualizer()
        with viz.track("training"):
            do_training()
        with viz.track("sync"):
            do_weight_sync()
        print(viz.render_ascii())
        print(viz.report())
    """

    def __init__(self):
        self.events: List[ScheduleEvent] = []
        self._current_events: Dict[str, float] = {}  # key: "name|device" -> start_time
        self._base_time: Optional[float] = None

    def begin_event(self, name: str, device: str = "GPU:0", **metadata):
        """标记事件开始

        Args:
            name: 事件名称
            device: 设备标识
            **metadata: 附加信息（如batch_size, data_bytes等）
        """
        now = time.time()
        if self._base_time is None:
            self._base_time = now

        key = f"{name}|{device}"
        self._current_events[key] = now

    def end_event(self, name: str, device: str = "GPU:0"):
        """标记事件结束

        Args:
            name: 事件名称
            device: 设备标识
        """
        now = time.time()
        key = f"{name}|{device}"

        if key not in self._current_events:
            return

        start_time = self._current_events.pop(key)
        event = ScheduleEvent(
            name=name,
            start_time=start_time - (self._base_time or start_time),
            end_time=now - (self._base_time or start_time),
            device=device,
        )
        self.events.append(event)

    @contextmanager
    def track(self, name: str, device: str = "GPU:0", **metadata):
        """上下文管理器，自动记录事件

        用法:
            with viz.track("training", device="GPU:0", batch_size=4):
                do_training()

        Args:
            name: 事件名称
            device: 设备标识
            **metadata: 附加信息
        """
        now = time.time()
        if self._base_time is None:
            self._base_time = now

        start = now
        try:
            yield
        finally:
            end = time.time()
            event = ScheduleEvent(
                name=name,
                start_time=start - self._base_time,
                end_time=end - self._base_time,
                device=device,
                metadata=metadata,
            )
            self.events.append(event)

    def add_event(self, name: str, start_time: float, end_time: float,
                  device: str = "GPU:0", **metadata):
        """手动添加事件（用于从已有数据重建时间线）

        Args:
            name: 事件名称
            start_time: 开始时间（相对时间，秒）
            end_time: 结束时间（相对时间，秒）
            device: 设备标识
        """
        event = ScheduleEvent(
            name=name,
            start_time=start_time,
            end_time=end_time,
            device=device,
            metadata=metadata,
        )
        self.events.append(event)

    def render_ascii(self, width: int = 80) -> str:
        """生成ASCII Gantt图

        技术要点：
        - ASCII Gantt图的价值？→ SSH远程开发时无法看图片，纯文本可视化必不可少
        - 通过字符密度一眼判断各阶段占比

        示例输出：
        GPU:0 |████TRAIN████|░SLEEP░|▒SYNC▒|████INFER████|▒REWARD▒|
        Time:  0s          5s      6s     7s           12s       13s

        Args:
            width: ASCII图宽度（字符数）

        Returns:
            ASCII Gantt图字符串
        """
        if not self.events:
            return "No events recorded."

        # 获取所有设备
        devices = sorted(set(e.device for e in self.events))
        total_time = max(e.end_time for e in self.events)

        if total_time <= 0:
            return "No events with duration."

        lines = []
        lines.append(f"{'─' * (width + 10)}")
        lines.append(f"  调度时间线 (总时间: {total_time:.1f}s)")
        lines.append(f"{'─' * (width + 10)}")

        for device in devices:
            device_events = sorted(
                [e for e in self.events if e.device == device],
                key=lambda e: e.start_time
            )

            # 构建时间线字符数组
            timeline = [" "] * width
            labels: List[Tuple[int, int, str]] = []

            for event in device_events:
                start_pos = int(event.start_time / total_time * width)
                end_pos = int(event.end_time / total_time * width)
                start_pos = max(0, min(start_pos, width - 1))
                end_pos = max(start_pos + 1, min(end_pos, width))

                char = _STAGE_CHARS.get(event.name, ("?", "other"))[0]
                for i in range(start_pos, end_pos):
                    timeline[i] = char

                # 放置标签
                label = event.name.upper()[:6]
                mid = (start_pos + end_pos) // 2
                label_start = max(start_pos, mid - len(label) // 2)
                if end_pos - start_pos > len(label) + 2:
                    labels.append((label_start, end_pos, label))

            # 渲染设备行
            timeline_str = "".join(timeline)
            lines.append(f"  {device:<6}|{timeline_str}|")

            # 渲染标签行
            label_line = [" "] * width
            for ls, le, label in labels:
                for i, ch in enumerate(label):
                    pos = ls + i
                    if pos < width:
                        label_line[pos] = ch
            lines.append(f"  {'':6} {' '.join([''] + [])}{' ' * 1}{''.join(label_line)}")

        # 时间轴
        lines.append(f"  {'Time:':<6} {'0s':<{width // 4}}{f'{total_time/4:.1f}s':<{width // 4}}"
                     f"{f'{total_time/2:.1f}s':<{width // 4}}{f'{total_time:.1f}s':>{width // 4}}")
        lines.append(f"{'─' * (width + 10)}")

        return "\n".join(lines)

    def render_json(self) -> str:
        """导出为Chrome Trace格式

        技术要点：
        - Chrome Trace Event Format是什么？
          → Google的标准profiling格式，可在chrome://tracing中打开
          → PyTorch Profiler、TensorBoard都用这个格式
          → 支持多线程/多设备的时间线对齐
        - 字段含义：
          → ph: "X" = complete event, "B"/"E" = begin/end
          → ts: 时间戳（微秒）
          → dur: 持续时间（微秒）
          → pid: process id (这里映射到device)
          → cat: category

        Returns:
            JSON字符串（Chrome Trace Event Format）
        """
        trace_events = []

        for event in self.events:
            trace_event = {
                "name": event.name,
                "cat": event.category,
                "ph": "X",  # Complete event
                "ts": int(event.start_time * 1e6),  # 微秒
                "dur": int(event.duration_s * 1e6),  # 微秒
                "pid": event.device,
                "tid": event.device,
                "args": event.metadata,
            }
            trace_events.append(trace_event)

        return json.dumps({"traceEvents": trace_events}, indent=2)

    def get_utilization_stats(self) -> dict:
        """计算各阶段时间占比和GPU利用率

        技术要点：
        - GPU利用率 = (compute + comm) / total
        - 调度效率 = 1 - (idle + overhead) / total
        - 如果overhead(sleep/wake)占比高 → Colocate的切换开销大
        - 如果comm占比高 → 通信是瓶颈，考虑增大batch或overlap

        Returns:
            各阶段耗时占比、GPU active时间占比、瓶颈识别
        """
        if not self.events:
            return {"status": "no_events"}

        total_time = max(e.end_time for e in self.events)
        if total_time <= 0:
            return {"status": "no_duration"}

        # 按类别统计
        category_time: Dict[str, float] = {
            "compute": 0.0,
            "comm": 0.0,
            "overhead": 0.0,
            "idle": 0.0,
            "other": 0.0,
        }
        stage_time: Dict[str, float] = {}

        for event in self.events:
            cat = event.category
            dur = event.duration_s
            category_time[cat] = category_time.get(cat, 0.0) + dur
            stage_time[event.name] = stage_time.get(event.name, 0.0) + dur

        # GPU active = compute + comm
        active_time = category_time["compute"] + category_time["comm"]
        gpu_utilization = active_time / total_time if total_time > 0 else 0.0

        # 找瓶颈
        bottleneck = max(stage_time, key=stage_time.get) if stage_time else "unknown"

        return {
            "total_time_s": total_time,
            "gpu_utilization": gpu_utilization,
            "category_breakdown": {
                cat: {"time_s": t, "ratio": t / total_time}
                for cat, t in category_time.items()
                if t > 0
            },
            "stage_breakdown": {
                name: {"time_s": t, "ratio": t / total_time}
                for name, t in sorted(stage_time.items(), key=lambda x: -x[1])
            },
            "bottleneck": bottleneck,
            "bottleneck_ratio": stage_time.get(bottleneck, 0) / total_time if total_time > 0 else 0,
        }

    def compare(self, other: 'ScheduleVisualizer') -> str:
        """对比两种调度方案的时间线

        技术要点：
        - Colocate vs Disaggregated的trade-off？
          → Colocate: GPU共享，有sleep/wake开销，但省显存
          → Disaggregated: 专用GPU，无切换开销，但需要跨机通信
        - 什么场景Colocate更好？→ 小模型、单机多卡、sleep/wake开销可控
        - 什么场景Disaggregated更好？→ 大模型、推理延迟敏感、多机场景

        Args:
            other: 另一个ScheduleVisualizer实例

        Returns:
            对比报告字符串
        """
        stats_a = self.get_utilization_stats()
        stats_b = other.get_utilization_stats()

        lines = []
        lines.append("┌" + "─" * 55 + "┐")
        lines.append("│" + "调度方案对比".center(49) + "      │")
        lines.append("├" + "─" * 27 + "┬" + "─" * 27 + "┤")
        lines.append(f"│  {'指标':<12} {'方案A':>10}  │  {'方案B':>10}          │")
        lines.append("├" + "─" * 27 + "┼" + "─" * 27 + "┤")

        time_a = stats_a.get("total_time_s", 0)
        time_b = stats_b.get("total_time_s", 0)
        lines.append(f"│  {'总时间':<10} {time_a:>8.1f}s   │  {time_b:>8.1f}s          │")

        # GPU利用率
        util_a = stats_a.get("gpu_utilization", 0)
        util_b = stats_b.get("gpu_utilization", 0)
        lines.append(f"│  {'GPU利用率':<8} {util_a:>9.1%}   │  {util_b:>9.1%}          │")

        # 瓶颈
        bn_a = stats_a.get("bottleneck", "-")
        bn_b = stats_b.get("bottleneck", "-")
        lines.append(f"│  {'瓶颈':<10} {bn_a:>10}  │  {bn_b:>10}          │")

        lines.append("├" + "─" * 27 + "┴" + "─" * 27 + "┤")

        # 结论
        if time_a < time_b:
            speedup = time_b / time_a if time_a > 0 else 0
            lines.append(f"│  结论: 方案A快 {speedup:.2f}x" + " " * 32 + "│")
        elif time_b < time_a:
            speedup = time_a / time_b if time_b > 0 else 0
            lines.append(f"│  结论: 方案B快 {speedup:.2f}x" + " " * 32 + "│")
        else:
            lines.append(f"│  结论: 两方案耗时相近" + " " * 28 + "│")

        lines.append("└" + "─" * 55 + "┘")
        return "\n".join(lines)

    def report(self) -> str:
        """生成完整的调度分析报告

        Returns:
            格式化的调度分析报告
        """
        stats = self.get_utilization_stats()
        if "status" in stats:
            return "No schedule data to report."

        lines = []
        lines.append("┌" + "─" * 55 + "┐")
        lines.append("│" + "调度性能分析报告".center(47) + "        │")
        lines.append("├" + "─" * 55 + "┤")

        # 总体统计
        lines.append(f"│  总时间: {stats['total_time_s']:.2f}s" + " " * 35 + "│")
        lines.append(f"│  GPU利用率: {stats['gpu_utilization']:.1%}" + " " * 33 + "│")
        lines.append(f"│  瓶颈阶段: {stats['bottleneck']} ({stats['bottleneck_ratio']:.1%})" + " " * 20 + "│")
        lines.append("├" + "─" * 55 + "┤")

        # 各阶段详情
        lines.append(f"│  {'阶段':<15} {'时间(s)':>8} {'占比':>8}" + " " * 15 + "│")
        lines.append("│  " + "─" * 40 + " " * 12 + "│")

        for name, info in stats.get("stage_breakdown", {}).items():
            lines.append(
                f"│  {name:<15} {info['time_s']:>8.2f} {info['ratio']:>7.1%}" + " " * 15 + "│"
            )

        lines.append("├" + "─" * 55 + "┤")

        # 优化建议
        util = stats["gpu_utilization"]
        if util < 0.5:
            lines.append("│  ⚠ GPU利用率<50%，存在大量空闲时间" + " " * 14 + "│")
            lines.append("│    建议: 减少sleep/wake开销或使用pipeline" + " " * 8 + "│")
        elif util < 0.7:
            lines.append("│  △ GPU利用率一般，有优化空间" + " " * 19 + "│")
        else:
            lines.append("│  ✓ GPU利用率良好" + " " * 31 + "│")

        lines.append("└" + "─" * 55 + "┘")
        return "\n".join(lines)

    def reset(self):
        """重置所有事件"""
        self.events.clear()
        self._current_events.clear()
        self._base_time = None
