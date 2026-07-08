"""
NCCL 权重广播模块（Disaggregated架构）

工作流程:
1. 训练Worker完成一步训练
2. 通过NCCL broadcast将更新后的权重广播到推理Worker
3. 推理Worker热更新模型权重
4. 继续处理推理请求

支持两种模式:
- NCCL直接广播: 低延迟，需要训练和推理进程在同一NCCL通信组
- Ray Object Store: 更灵活，训练和推理可以完全解耦

权重版本管理:
- 每次更新递增version号
- 推理侧检查version确保使用最新权重
"""

import asyncio
import hashlib
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

try:
    import ray

    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WeightVersion:
    """权重版本信息"""

    version: int = 0
    timestamp: float = field(default_factory=time.time)
    checksum: str = ""

    def increment(self, checksum: str = "") -> "WeightVersion":
        """递增版本号，返回新版本"""
        return WeightVersion(
            version=self.version + 1,
            timestamp=time.time(),
            checksum=checksum,
        )


@dataclass
class TransferStats:
    """传输统计信息"""

    total_params: int = 0
    total_bytes: int = 0
    broadcast_time: float = 0.0
    serialization_time: float = 0.0
    total_time: float = 0.0
    sync_count: int = 0  # 累计同步次数
    errors: int = 0

    @property
    def bandwidth_gbps(self) -> float:
        if self.broadcast_time <= 0:
            return 0.0
        return (self.total_bytes / 1e9) / self.broadcast_time


class NCCLWeightTransfer:
    """NCCL直接广播权重传输"""

    def __init__(
        self,
        src_rank: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Args:
            src_rank: 广播源rank（训练master）
            process_group: 包含训练和推理所有进程的通信组
                          如果为None，使用默认进程组
        """
        self.src_rank = src_rank
        self.process_group = process_group
        self._version = WeightVersion()
        self._stats = TransferStats()
        self._lock = threading.Lock()

    def broadcast_weights(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> WeightVersion:
        """将训练后的权重广播到所有推理Worker

        对state_dict中每个tensor调用dist.broadcast
        仅在src_rank上调用此方法（训练侧）
        """
        t0 = time.perf_counter()
        total_bytes = 0

        # 先广播参数数量和元信息（通过broadcast_object_list）
        param_names = list(state_dict.keys())
        meta_list: List[Optional[List[str]]] = [param_names]
        dist.broadcast_object_list(
            meta_list, src=self.src_rank, group=self.process_group
        )

        # 逐个广播tensor
        for name in param_names:
            tensor = state_dict[name].detach().contiguous()
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            dist.broadcast(tensor, src=self.src_rank, group=self.process_group)
            total_bytes += tensor.nelement() * tensor.element_size()

        broadcast_time = time.perf_counter() - t0

        checksum = self._compute_checksum(state_dict)

        # 更新版本
        with self._lock:
            self._version = self._version.increment(checksum)

        # 广播版本号给所有Worker
        version_list: List[Optional[WeightVersion]] = [self._version]
        dist.broadcast_object_list(
            version_list, src=self.src_rank, group=self.process_group
        )

        self._stats.total_params = len(param_names)
        self._stats.total_bytes = total_bytes
        self._stats.broadcast_time = broadcast_time
        self._stats.total_time = time.perf_counter() - t0
        self._stats.sync_count += 1

        logger.info(
            f"Broadcast {len(param_names)} params "
            f"({total_bytes / 1e9:.2f} GB) in {broadcast_time:.3f}s "
            f"(version={self._version.version})"
        )
        return self._version

    def receive_weights(
        self, state_dict_template: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """推理侧接收广播的权重

        使用state_dict_template作为buffer接收
        在非src_rank上调用此方法（推理侧）
        """
        t0 = time.perf_counter()

        # 接收参数名列表
        meta_list: List[Optional[List[str]]] = [None]
        dist.broadcast_object_list(
            meta_list, src=self.src_rank, group=self.process_group
        )
        param_names: List[str] = meta_list[0]  # type: ignore

        # 逐个接收tensor
        received_dict: Dict[str, torch.Tensor] = {}
        for name in param_names:
            if name in state_dict_template:
                # 使用template中的buffer进行in-place接收
                buffer = state_dict_template[name].detach().contiguous()
                if not buffer.is_cuda:
                    buffer = buffer.cuda()
            else:
                # 如果template中没有对应参数，跳过（但仍需参与broadcast）
                logger.warning(
                    f"Parameter '{name}' not in template, "
                    f"creating empty buffer"
                )
                # 需要知道形状，但此处无法获取，先跳过
                continue

            dist.broadcast(buffer, src=self.src_rank, group=self.process_group)
            received_dict[name] = buffer

        # 接收版本号
        version_list: List[Optional[WeightVersion]] = [None]
        dist.broadcast_object_list(
            version_list, src=self.src_rank, group=self.process_group
        )
        with self._lock:
            self._version = version_list[0]  # type: ignore

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Received {len(received_dict)} params in {elapsed:.3f}s "
            f"(version={self._version.version})"
        )
        return received_dict

    @property
    def current_version(self) -> WeightVersion:
        """获取当前权重版本"""
        with self._lock:
            return self._version

    @property
    def stats(self) -> TransferStats:
        return self._stats

    @staticmethod
    def _compute_checksum(state_dict: Dict[str, torch.Tensor]) -> str:
        """计算state_dict的快速校验和（只用前几个元素）"""
        md5 = hashlib.md5()
        for name in sorted(state_dict.keys()):
            tensor = state_dict[name]
            md5.update(name.encode("utf-8"))
            # 只用前64个元素做快速checksum，避免全量拷贝
            flat = tensor.detach().flatten()[:64].float().cpu().numpy().tobytes()
            md5.update(flat)
        return md5.hexdigest()


class RayObjectStoreTransfer:
    """Ray Object Store权重传输（备选方案）

    优点: 不需要训练和推理在同一NCCL组，更灵活
    缺点: 序列化/反序列化开销，需要经过CPU
    """

    def __init__(self):
        if not _RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for RayObjectStoreTransfer. "
                "Install via: pip install ray"
            )
        self._version = WeightVersion()
        self._weight_ref: Optional["ray.ObjectRef"] = None
        self._version_ref: Optional["ray.ObjectRef"] = None
        self._stats = TransferStats()
        self._lock = threading.Lock()

    def publish_weights(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> WeightVersion:
        """训练侧发布权重到Object Store

        将state_dict转为CPU tensors后put到Ray Object Store
        """
        t0 = time.perf_counter()

        # 将GPU tensors移到CPU进行序列化
        t_serial = time.perf_counter()
        cpu_state_dict = {
            name: tensor.detach().cpu() for name, tensor in state_dict.items()
        }
        serial_time = time.perf_counter() - t_serial

        total_bytes = sum(
            t.nelement() * t.element_size() for t in cpu_state_dict.values()
        )

        checksum = self._compute_checksum(cpu_state_dict)

        # 更新版本
        with self._lock:
            self._version = self._version.increment(checksum)
            # 删除旧的object ref（允许GC）
            old_ref = self._weight_ref

        # Put到Object Store
        weight_ref = ray.put(cpu_state_dict)
        version_ref = ray.put(self._version)

        with self._lock:
            self._weight_ref = weight_ref
            self._version_ref = version_ref

        self._stats.total_params = len(cpu_state_dict)
        self._stats.total_bytes = total_bytes
        self._stats.serialization_time = serial_time
        self._stats.total_time = time.perf_counter() - t0
        self._stats.sync_count += 1

        logger.info(
            f"Published {len(cpu_state_dict)} params "
            f"({total_bytes / 1e9:.2f} GB) to Object Store "
            f"in {self._stats.total_time:.3f}s "
            f"(serial: {serial_time:.3f}s, version={self._version.version})"
        )
        return self._version

    def fetch_weights(
        self, device: Optional[torch.device] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """推理侧获取最新权重

        从Object Store get并转移到指定device
        """
        with self._lock:
            weight_ref = self._weight_ref

        if weight_ref is None:
            logger.warning("No weights available in Object Store")
            return None

        t0 = time.perf_counter()

        # 从Object Store获取
        cpu_state_dict: Dict[str, torch.Tensor] = ray.get(weight_ref)

        # 转移到目标设备
        if device is not None:
            result = {
                name: tensor.to(device) for name, tensor in cpu_state_dict.items()
            }
        else:
            result = cpu_state_dict

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Fetched {len(result)} params from Object Store "
            f"in {elapsed:.3f}s (device={device})"
        )
        return result

    def has_new_version(self, current_version: int) -> bool:
        """检查是否有新版本权重"""
        with self._lock:
            return self._version.version > current_version

    def get_version_info(self) -> WeightVersion:
        """获取当前版本信息（不获取权重数据）"""
        with self._lock:
            if self._version_ref is not None:
                return ray.get(self._version_ref)
            return self._version

    @property
    def current_version(self) -> WeightVersion:
        with self._lock:
            return self._version

    @property
    def stats(self) -> TransferStats:
        return self._stats

    @staticmethod
    def _compute_checksum(state_dict: Dict[str, torch.Tensor]) -> str:
        md5 = hashlib.md5()
        for name in sorted(state_dict.keys()):
            tensor = state_dict[name]
            md5.update(name.encode("utf-8"))
            flat = tensor.detach().flatten()[:64].float().numpy().tobytes()
            md5.update(flat)
        return md5.hexdigest()


class WeightTransferManager:
    """权重传输管理器 - 统一接口

    自动选择传输后端(NCCL/RayObjectStore)并管理版本
    """

    def __init__(self, backend: str = "nccl", **kwargs):
        """
        Args:
            backend: "nccl" 或 "ray_object_store"
            **kwargs: 传递给对应后端的参数
                nccl: src_rank, process_group
                ray_object_store: (无额外参数)
        """
        self.backend_name = backend
        self._stats_history: List[TransferStats] = []

        if backend == "nccl":
            self._transfer: NCCLWeightTransfer | RayObjectStoreTransfer = (
                NCCLWeightTransfer(**kwargs)
            )
        elif backend == "ray_object_store":
            self._transfer = RayObjectStoreTransfer(**kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Supported: 'nccl', 'ray_object_store'"
            )

    async def sync_weights(
        self,
        state_dict: Dict[str, torch.Tensor],
        is_sender: bool = True,
        state_dict_template: Optional[Dict[str, torch.Tensor]] = None,
    ) -> WeightVersion:
        """统一的权重同步接口

        Args:
            state_dict: 要发送的权重（sender侧）或接收buffer模板（receiver侧）
            is_sender: 是否为发送方（训练侧）
            state_dict_template: 接收方的buffer模板（仅NCCL receiver需要）

        Returns:
            更新后的WeightVersion
        """
        # 在executor中运行，避免阻塞事件循环
        loop = asyncio.get_event_loop()

        if isinstance(self._transfer, NCCLWeightTransfer):
            if is_sender:
                version = await loop.run_in_executor(
                    None, self._transfer.broadcast_weights, state_dict
                )
            else:
                template = state_dict_template or state_dict
                received = await loop.run_in_executor(
                    None, self._transfer.receive_weights, template
                )
                # 更新state_dict in-place
                for name, tensor in received.items():
                    if name in state_dict:
                        state_dict[name].copy_(tensor)
                version = self._transfer.current_version
        elif isinstance(self._transfer, RayObjectStoreTransfer):
            if is_sender:
                version = await loop.run_in_executor(
                    None, self._transfer.publish_weights, state_dict
                )
            else:
                device = next(iter(state_dict.values())).device if state_dict else None
                fetched = await loop.run_in_executor(
                    None, self._transfer.fetch_weights, device
                )
                if fetched:
                    for name, tensor in fetched.items():
                        if name in state_dict:
                            state_dict[name].copy_(tensor)
                version = self._transfer.current_version
        else:
            raise RuntimeError(f"Unknown transfer backend: {type(self._transfer)}")

        self._stats_history.append(self._transfer.stats)
        return version

    def sync_weights_sync(
        self,
        state_dict: Dict[str, torch.Tensor],
        is_sender: bool = True,
        state_dict_template: Optional[Dict[str, torch.Tensor]] = None,
    ) -> WeightVersion:
        """同步版本的权重同步接口"""
        if isinstance(self._transfer, NCCLWeightTransfer):
            if is_sender:
                return self._transfer.broadcast_weights(state_dict)
            else:
                template = state_dict_template or state_dict
                received = self._transfer.receive_weights(template)
                for name, tensor in received.items():
                    if name in state_dict:
                        state_dict[name].copy_(tensor)
                return self._transfer.current_version
        elif isinstance(self._transfer, RayObjectStoreTransfer):
            if is_sender:
                return self._transfer.publish_weights(state_dict)
            else:
                device = next(iter(state_dict.values())).device if state_dict else None
                fetched = self._transfer.fetch_weights(device)
                if fetched:
                    for name, tensor in fetched.items():
                        if name in state_dict:
                            state_dict[name].copy_(tensor)
                return self._transfer.current_version
        else:
            raise RuntimeError(f"Unknown transfer backend: {type(self._transfer)}")

    @property
    def current_version(self) -> WeightVersion:
        """获取当前权重版本"""
        return self._transfer.current_version

    def has_new_version(self, version: int) -> bool:
        """检查是否有新版本（主要用于推理侧轮询）"""
        return self._transfer.current_version.version > version

    def get_transfer_stats(self) -> dict:
        """获取传输统计信息（延迟、带宽等）"""
        current = self._transfer.stats
        return {
            "backend": self.backend_name,
            "current_version": self._transfer.current_version.version,
            "total_syncs": current.sync_count,
            "last_transfer": {
                "total_params": current.total_params,
                "total_bytes": current.total_bytes,
                "total_bytes_human": f"{current.total_bytes / 1e9:.2f} GB",
                "broadcast_time_s": current.broadcast_time,
                "serialization_time_s": current.serialization_time,
                "total_time_s": current.total_time,
                "bandwidth_gbps": current.bandwidth_gbps,
            },
            "errors": current.errors,
            "history_count": len(self._stats_history),
        }
