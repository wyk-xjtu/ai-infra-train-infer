"""
SwiftSync — 增量权重同步子系统

基于 Ring-1T 论文的 AState 技术思想，实现 LoRA delta 增量传输 + 推理侧 Double Buffering。
传输量减少 95%+（仅传 LoRA delta），权重切换零停顿（原子 buffer swap）。

核心设计:
- DeltaComputer: 在 GPU 上计算 LoRA 参数增量（当前 - 前一版本）
- DoubleBufferedLoRA: 推理侧双缓冲管理，active/shadow 交替使用
- SwiftSyncTransfer: 主控协调器，统一增量/全量同步接口

传输优化:
- 增量传输: 仅传 LoRA delta（rank=64 时约为全量的 0.8%，delta 更小）
- 稀疏优化预留: delta_threshold 过滤极小增量（未来可启用）
- 定期校准: 每 N 步强制全量同步，防止浮点累积误差
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================
# 配置与数据结构
# ============================================================


@dataclass
class SwiftSyncConfig:
    """SwiftSync 增量同步配置

    Attributes:
        enabled: 是否启用增量同步（False 时退化为全量同步）
        fallback_full_every: 每 N 步强制全量同步校准，防止浮点累积误差
        delta_threshold: 低于此值的 delta 可忽略（稀疏优化预留，当前未启用）
        enable_double_buffer: 启用推理侧双缓冲（零停顿切换）
        async_transfer: 异步传输模式（非阻塞）
    """

    enabled: bool = False
    fallback_full_every: int = 10
    delta_threshold: float = 1e-8
    enable_double_buffer: bool = True
    async_transfer: bool = True


@dataclass
class DeltaVersion:
    """增量同步的版本信息

    Attributes:
        version: 单调递增的版本号
        timestamp: 同步完成的 Unix 时间戳
        num_params: delta 中包含的参数张量数量
        delta_bytes: delta 总字节数（用于带宽统计）
    """

    version: int
    timestamp: float
    num_params: int
    delta_bytes: int


@dataclass
class SyncStats:
    """同步统计信息"""

    total_delta_syncs: int = 0
    total_full_syncs: int = 0
    total_delta_bytes: int = 0
    total_full_bytes: int = 0
    last_sync_time: float = 0.0
    last_sync_type: str = ""
    cumulative_time: float = 0.0


# ============================================================
# DeltaComputer — 增量计算器
# ============================================================


class DeltaComputer:
    """计算 LoRA 参数的增量 delta

    在 GPU 上执行逐参数减法，避免 CPU-GPU 数据搬运开销。
    支持定期全量同步校准以防止浮点累积误差。
    """

    def __init__(self, config: SwiftSyncConfig):
        """
        Args:
            config: SwiftSync 配置
        """
        self.config = config
        self._step_count: int = 0

    def compute_delta(
        self, current_lora: Dict[str, Tensor], prev_lora: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """计算当前 LoRA 参数与前一版本的差值

        在 GPU 上逐参数做减法，结果保留在原设备上。
        仅对两份 state_dict 中同名参数计算 delta。

        Args:
            current_lora: 当前 LoRA state_dict（训练后）
            prev_lora: 上一版本 LoRA state_dict（上次同步时的快照）

        Returns:
            delta_dict: {param_name: delta_tensor}，delta = current - prev
        """
        delta_dict: Dict[str, Tensor] = {}

        for name, current_param in current_lora.items():
            if name not in prev_lora:
                # 新增参数，delta 就是参数本身
                logger.debug(f"New parameter in delta: {name}")
                delta_dict[name] = current_param.detach().clone()
                continue

            prev_param = prev_lora[name]

            # 确保在同一设备上计算
            if current_param.device != prev_param.device:
                prev_param = prev_param.to(current_param.device)

            # GPU 上做减法
            delta = current_param.detach() - prev_param.detach()
            delta_dict[name] = delta

        self._step_count += 1
        logger.debug(
            f"Computed delta for {len(delta_dict)} params "
            f"(step={self._step_count})"
        )
        return delta_dict

    def should_full_sync(self, step: int) -> bool:
        """判断是否需要全量同步（定期校准）

        每隔 fallback_full_every 步强制全量同步，
        防止增量累加导致的浮点误差积累。

        Args:
            step: 当前训练步数

        Returns:
            True 表示应执行全量同步
        """
        return step > 0 and step % self.config.fallback_full_every == 0

    @staticmethod
    def compute_delta_bytes(delta: Dict[str, Tensor]) -> int:
        """计算 delta 的总字节数

        Args:
            delta: delta 参数字典

        Returns:
            总字节数
        """
        total = 0
        for tensor in delta.values():
            total += tensor.nelement() * tensor.element_size()
        return total


# ============================================================
# DoubleBufferedLoRA — 推理侧双缓冲
# ============================================================


class DoubleBufferedLoRA:
    """推理侧 LoRA 双缓冲管理

    维护 active 和 shadow 两份 LoRA 权重缓冲。
    同步时写 shadow buffer，完成后原子交换 active/shadow 指针。
    推理始终从 active buffer 读取，实现零停顿权重切换。

    线程安全: 使用 threading.Lock 保证 swap_buffers() 的原子性，
    确保推理线程和同步线程不会发生数据竞争。
    """

    def __init__(self, initial_lora_state: Dict[str, Tensor]):
        """初始化双缓冲，两个 buffer 均从 initial_lora_state 拷贝

        Args:
            initial_lora_state: 初始 LoRA 权重（深拷贝到两个 buffer）
        """
        # active: 推理正在使用的权重
        self._active: Dict[str, Tensor] = {
            k: v.detach().clone() for k, v in initial_lora_state.items()
        }
        # shadow: 接收增量更新的缓冲
        self._shadow: Dict[str, Tensor] = {
            k: v.detach().clone() for k, v in initial_lora_state.items()
        }
        # 版本号（每次 swap 递增）
        self._version: int = 0
        # 原子交换锁
        self._lock = threading.Lock()

        logger.info(
            f"DoubleBufferedLoRA initialized with {len(initial_lora_state)} params, "
            f"version=0"
        )

    def apply_delta_to_shadow(self, delta: Dict[str, Tensor]) -> None:
        """将 delta 应用到 shadow buffer（原位更新）

        shadow[k] += delta[k]，在 GPU 上执行 in-place 加法。

        Args:
            delta: 增量参数字典 {param_name: delta_tensor}
        """
        applied = 0
        for name, delta_tensor in delta.items():
            if name in self._shadow:
                # 确保设备一致
                if delta_tensor.device != self._shadow[name].device:
                    delta_tensor = delta_tensor.to(self._shadow[name].device)
                # 原位加法，避免额外显存分配
                self._shadow[name].add_(delta_tensor)
                applied += 1
            else:
                logger.warning(
                    f"Delta key '{name}' not found in shadow buffer, skipping"
                )

        logger.debug(f"Applied delta to shadow: {applied}/{len(delta)} params")

    def set_full_state(self, state: Dict[str, Tensor]) -> None:
        """全量同步时直接设置 shadow buffer 内容

        用于定期校准，覆盖 shadow buffer 的全部内容。

        Args:
            state: 完整的 LoRA state_dict
        """
        for name, tensor in state.items():
            if name in self._shadow:
                self._shadow[name].copy_(tensor.detach())
            else:
                # 新参数，直接添加
                self._shadow[name] = tensor.detach().clone()

        logger.debug(f"Set full state to shadow: {len(state)} params")

    def swap_buffers(self) -> int:
        """原子交换 active 和 shadow buffer

        使用 threading.Lock 保证交换操作的原子性。
        交换后 shadow 变为 active（供推理使用），
        原 active 变为新的 shadow（接收下次更新）。

        Returns:
            新的 active 版本号
        """
        with self._lock:
            # 指针交换（O(1) 操作，不拷贝数据）
            self._active, self._shadow = self._shadow, self._active
            self._version += 1
            new_version = self._version

        logger.info(f"Buffer swap completed, active version={new_version}")
        return new_version

    def get_active_state(self) -> Dict[str, Tensor]:
        """获取当前 active buffer 的权重（供推理使用）

        线程安全: 返回的是 active buffer 的引用字典，
        推理侧应在使用完毕前不触发新的 swap。

        Returns:
            当前 active buffer 的参数字典
        """
        with self._lock:
            return dict(self._active)

    @property
    def version(self) -> int:
        """当前 active buffer 的版本号"""
        with self._lock:
            return self._version


# ============================================================
# SwiftSyncTransfer — 增量同步主控
# ============================================================


class SwiftSyncTransfer:
    """增量权重同步主控

    协调 DeltaComputer 和 DoubleBufferedLoRA，提供统一的同步接口。
    支持增量同步（delta）和全量同步（full）两种模式，
    自动根据步数决定是否触发全量校准。

    使用方式:
        config = SwiftSyncConfig(enabled=True)
        transfer = SwiftSyncTransfer(config)
        transfer.initialize(initial_lora_state)

        # 训练循环中
        delta = transfer.delta_computer.compute_delta(current, prev)
        if transfer.delta_computer.should_full_sync(step):
            await transfer.sync_full(current, infer_worker)
        else:
            await transfer.sync_delta(delta, infer_worker)
    """

    def __init__(self, config: SwiftSyncConfig):
        """
        Args:
            config: SwiftSync 配置
        """
        self.config = config
        self.delta_computer = DeltaComputer(config)
        self._version: int = 0
        self._double_buffer: Optional[DoubleBufferedLoRA] = None
        self._prev_lora_snapshot: Optional[Dict[str, Tensor]] = None
        self._stats = SyncStats()
        self._initialized: bool = False

        logger.info(
            f"SwiftSyncTransfer created (enabled={config.enabled}, "
            f"fallback_full_every={config.fallback_full_every}, "
            f"double_buffer={config.enable_double_buffer}, "
            f"async={config.async_transfer})"
        )

    def initialize(self, initial_lora_state: Dict[str, Tensor]) -> None:
        """初始化同步状态

        设置初始 LoRA 快照和双缓冲。应在第一次同步前调用。

        Args:
            initial_lora_state: 初始 LoRA 权重（训练开始时的状态）
        """
        # 保存初始快照（深拷贝，用于计算第一次 delta）
        self._prev_lora_snapshot = {
            k: v.detach().clone() for k, v in initial_lora_state.items()
        }

        # 初始化双缓冲
        if self.config.enable_double_buffer:
            self._double_buffer = DoubleBufferedLoRA(initial_lora_state)

        self._initialized = True
        logger.info(
            f"SwiftSyncTransfer initialized with {len(initial_lora_state)} params"
        )

    async def sync_delta(
        self, delta: Dict[str, Tensor], infer_worker: Any
    ) -> DeltaVersion:
        """增量同步：将 delta 应用到推理侧

        流程:
        1. 将 delta 应用到双缓冲的 shadow buffer
        2. 通知推理 worker 更新权重（mode="delta"）
        3. 执行 buffer swap，切换到新版本

        Args:
            delta: LoRA 参数增量（current - previous）
            infer_worker: 推理 worker 实例（需实现 update_weights 方法）

        Returns:
            DeltaVersion: 本次同步的版本信息
        """
        t0 = time.perf_counter()
        delta_bytes = DeltaComputer.compute_delta_bytes(delta)

        # Step 1: 应用 delta 到双缓冲 shadow
        if self._double_buffer is not None:
            self._double_buffer.apply_delta_to_shadow(delta)

        # Step 2: 通知推理侧更新权重
        if self.config.async_transfer:
            await self._async_update_weights(infer_worker, delta, mode="delta")
        else:
            self._sync_update_weights(infer_worker, delta, mode="delta")

        # Step 3: Buffer swap（原子切换）
        if self._double_buffer is not None:
            self._double_buffer.swap_buffers()

        # 更新版本和统计
        self._version += 1
        elapsed = time.perf_counter() - t0
        self._stats.total_delta_syncs += 1
        self._stats.total_delta_bytes += delta_bytes
        self._stats.last_sync_time = elapsed
        self._stats.last_sync_type = "delta"
        self._stats.cumulative_time += elapsed

        version_info = DeltaVersion(
            version=self._version,
            timestamp=time.time(),
            num_params=len(delta),
            delta_bytes=delta_bytes,
        )

        logger.info(
            f"Delta sync completed: version={self._version}, "
            f"params={len(delta)}, bytes={delta_bytes}, "
            f"time={elapsed:.4f}s"
        )
        return version_info

    async def sync_full(
        self, full_state: Dict[str, Tensor], infer_worker: Any
    ) -> DeltaVersion:
        """全量同步：校准推理侧权重

        用于定期校准，直接覆盖推理侧的 LoRA 权重，
        消除增量累加可能产生的浮点误差。

        流程:
        1. 将完整状态设置到 shadow buffer
        2. 通知推理 worker 全量更新（mode="full"）
        3. 执行 buffer swap
        4. 更新本地 snapshot（下次增量的基准）

        Args:
            full_state: 完整的 LoRA state_dict
            infer_worker: 推理 worker 实例

        Returns:
            DeltaVersion: 本次同步的版本信息
        """
        t0 = time.perf_counter()
        full_bytes = DeltaComputer.compute_delta_bytes(full_state)

        # Step 1: 全量设置 shadow buffer
        if self._double_buffer is not None:
            self._double_buffer.set_full_state(full_state)

        # Step 2: 通知推理侧全量更新
        if self.config.async_transfer:
            await self._async_update_weights(infer_worker, full_state, mode="full")
        else:
            self._sync_update_weights(infer_worker, full_state, mode="full")

        # Step 3: Buffer swap
        if self._double_buffer is not None:
            self._double_buffer.swap_buffers()

        # Step 4: 更新 snapshot（全量同步后重置基准）
        self._prev_lora_snapshot = {
            k: v.detach().clone() for k, v in full_state.items()
        }

        # 更新版本和统计
        self._version += 1
        elapsed = time.perf_counter() - t0
        self._stats.total_full_syncs += 1
        self._stats.total_full_bytes += full_bytes
        self._stats.last_sync_time = elapsed
        self._stats.last_sync_type = "full"
        self._stats.cumulative_time += elapsed

        version_info = DeltaVersion(
            version=self._version,
            timestamp=time.time(),
            num_params=len(full_state),
            delta_bytes=full_bytes,
        )

        logger.info(
            f"Full sync completed (calibration): version={self._version}, "
            f"params={len(full_state)}, bytes={full_bytes}, "
            f"time={elapsed:.4f}s"
        )
        return version_info

    async def _async_update_weights(
        self, infer_worker: Any, state: Dict[str, Tensor], mode: str
    ) -> None:
        """异步通知推理 worker 更新权重

        Args:
            infer_worker: 推理 worker（需实现 update_weights(state, mode=...)）
            state: 权重数据（delta 或 full state）
            mode: "delta" 或 "full"

        Raises:
            AttributeError: infer_worker 没有 update_weights 方法
        """
        update_fn = getattr(infer_worker, "update_weights", None)
        if update_fn is None:
            raise AttributeError(
                "infer_worker does not have 'update_weights' method, "
                "cannot proceed with weight sync"
            )

        try:
            if asyncio.iscoroutinefunction(update_fn):
                await update_fn(state, mode=mode)
            else:
                # 同步方法包装为异步执行
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, update_fn, state, mode)
        except Exception as e:
            logger.error(
                f"Failed to update weights on infer_worker "
                f"(mode={mode}): {type(e).__name__}: {e}"
            )
            raise

    def _sync_update_weights(
        self, infer_worker: Any, state: Dict[str, Tensor], mode: str
    ) -> None:
        """同步通知推理 worker 更新权重

        Args:
            infer_worker: 推理 worker
            state: 权重数据
            mode: "delta" 或 "full"

        Raises:
            AttributeError: infer_worker 没有 update_weights 方法
        """
        update_fn = getattr(infer_worker, "update_weights", None)
        if update_fn is None:
            raise AttributeError(
                "infer_worker does not have 'update_weights' method, "
                "cannot proceed with weight sync"
            )

        try:
            update_fn(state, mode=mode)
        except Exception as e:
            logger.error(
                f"Failed to update weights on infer_worker "
                f"(mode={mode}): {type(e).__name__}: {e}"
            )
            raise

    @property
    def version(self) -> int:
        """当前同步版本号"""
        return self._version

    @property
    def prev_snapshot(self) -> Optional[Dict[str, Tensor]]:
        """上一次同步的 LoRA 快照（用于外部计算 delta）"""
        return self._prev_lora_snapshot

    @property
    def double_buffer(self) -> Optional[DoubleBufferedLoRA]:
        """双缓冲实例（可能为 None）"""
        return self._double_buffer

    @property
    def stats(self) -> SyncStats:
        """同步统计信息"""
        return self._stats

    @property
    def initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized

    def get_stats_summary(self) -> Dict[str, Any]:
        """获取统计信息摘要

        Returns:
            包含同步次数、字节数、时间等信息的字典
        """
        return {
            "version": self._version,
            "initialized": self._initialized,
            "total_delta_syncs": self._stats.total_delta_syncs,
            "total_full_syncs": self._stats.total_full_syncs,
            "total_delta_bytes": self._stats.total_delta_bytes,
            "total_full_bytes": self._stats.total_full_bytes,
            "last_sync_time_s": self._stats.last_sync_time,
            "last_sync_type": self._stats.last_sync_type,
            "cumulative_sync_time_s": self._stats.cumulative_time,
            "config": {
                "enabled": self.config.enabled,
                "fallback_full_every": self.config.fallback_full_every,
                "delta_threshold": self.config.delta_threshold,
                "enable_double_buffer": self.config.enable_double_buffer,
                "async_transfer": self.config.async_transfer,
            },
        }
