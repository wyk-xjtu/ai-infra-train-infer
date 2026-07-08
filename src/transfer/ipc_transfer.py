"""
CUDA IPC 权重传输模块（Colocate架构）

工作流程:
1. 训练完成后，获取更新权重的CUDA张量
2. 为每个张量获取IPC handle (cudaIpcGetMemHandle)
3. 通过HTTP API通知vLLM开始权重更新
4. 分块发送IPC handles到vLLM
5. vLLM端通过handle重建张量并更新模型

关键API（vLLM Weight Transfer Protocol）:
- POST /sleep                    → 暂停推理，卸载权重到CPU
- POST /wake_up                  → 恢复推理
- POST /start_weight_update      → 开始权重更新
- POST /update_weights           → 发送权重块
- POST /finish_weight_update     → 完成权重更新

自研引擎兼容:
当 inference_backend="custom" 时，权重更新不走IPC流程，
而是直接调用 InferenceEngine.on_weights_updated() 清除缓存。
该逻辑在 ColocateOrchestrator._do_weight_sync() 中处理。
"""

import asyncio
import base64
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

try:
    import cloudpickle
except ImportError:
    import pickle as cloudpickle

logger = logging.getLogger(__name__)

# 异常分类: 区分可重试异常与不可重试异常

# 可重试异常 — 网络/连接类暂时性故障，重试可能恢复
RETRIABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    ConnectionRefusedError,
    OSError,  # Network-related OS errors (errno.ECONNRESET etc.)
)

# 不可重试异常 — 数据格式或逻辑错误，重试无意义
NON_RETRIABLE_EXCEPTIONS = (
    ValueError,       # 数据格式错误
    KeyError,         # 数据结构不匹配
    TypeError,        # 类型不兼容
)


@dataclass
class IPCHandle:
    """CUDA IPC handle封装"""

    name: str  # 参数名
    handle: bytes  # cudaIpcMemHandle_t 序列化 (pickled reduce_tensor output)
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device_id: int


@dataclass
class TransferStats:
    """传输统计信息"""

    total_params: int = 0
    total_bytes: int = 0
    build_handle_time: float = 0.0
    transfer_time: float = 0.0
    total_time: float = 0.0
    chunks_sent: int = 0
    errors: int = 0
    _start_time: float = field(default=0.0, repr=False)

    def start(self):
        self._start_time = time.perf_counter()

    def finish(self):
        self.total_time = time.perf_counter() - self._start_time

    @property
    def bandwidth_gbps(self) -> float:
        if self.transfer_time <= 0:
            return 0.0
        return (self.total_bytes / 1e9) / self.transfer_time


class VLLMClient:
    """vLLM HTTP控制客户端（异步）"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60,
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[Any] = None

    def _get_client(self):
        """懒初始化httpx AsyncClient"""
        if self._client is None:
            if not _HTTPX_AVAILABLE:
                raise ImportError(
                    "httpx is required for async VLLMClient. "
                    "Install via: pip install httpx"
                )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self):
        """关闭HTTP客户端"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post_with_retry(self, path: str, *,
                               json: Optional[dict] = None,
                               params: Optional[dict] = None) -> dict:
        """带重试的POST请求（区分异常类型）

        异常处理策略:
        - 可重试异常 (网络超时/连接错误): 按退避策略重试
        - 不可重试异常 (数据格式/反序列化错误): 立即报错，不重试
        - CUDA OOM: 清理缓存后重试一次
        """
        client = self._get_client()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                resp = await client.post(path, json=json, params=params)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    return {"ok": True, "raw": resp.text}

            except NON_RETRIABLE_EXCEPTIONS as e:
                # 数据格式/结构错误 → 不可重试，立即报错
                logger.error(
                    f"Non-retriable error on {path}: {type(e).__name__}: {e}. "
                    f"Not retrying."
                )
                raise

            except torch.cuda.OutOfMemoryError as e:
                # CUDA OOM → 清理缓存后重试一次
                logger.error(
                    f"CUDA OOM during request to {path}: {e}. "
                    f"Clearing cache and retrying once."
                )
                torch.cuda.empty_cache()
                if attempt == 0:
                    # 仅重试一次
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise

            except (httpx.HTTPStatusError, httpx.ConnectError,
                    httpx.TimeoutException) as e:
                # httpx 网络/HTTP错误 → 可重试
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Retriable HTTP error on {path} (attempt {attempt + 1}/{self.max_retries}): "
                        f"{type(e).__name__}: {e}. Retrying in {self.retry_delay * (attempt + 1):.1f}s..."
                    )
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except RETRIABLE_EXCEPTIONS as e:
                # 通用网络/OS错误 → 可重试
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Retriable error on {path} (attempt {attempt + 1}/{self.max_retries}): "
                        f"{type(e).__name__}: {e}. Retrying in {self.retry_delay * (attempt + 1):.1f}s..."
                    )
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                # 未知异常 → 记录并重试（保守策略）
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Unexpected error on {path} (attempt {attempt + 1}/{self.max_retries}): "
                        f"{type(e).__name__}: {e}. Retrying..."
                    )
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise ConnectionError(
            f"Failed to reach vLLM at {path} after {self.max_retries} attempts: {last_error}"
        )

    async def health_check(self) -> bool:
        """检查vLLM服务是否健康"""
        try:
            client = self._get_client()
            resp = await client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def sleep(self, level: int = 1) -> bool:
        """让vLLM进入睡眠模式

        Level 0: 仅暂停调度
        Level 1: 卸载权重到CPU + 丢弃KV Cache
        Level 2: 完全释放GPU显存
        """
        # 先清除prefix cache
        try:
            await self._post_with_retry(
                "/reset_prefix_cache",
                params={"reset_running_requests": False, "reset_external": False},
            )
        except Exception as e:
            logger.warning(f"reset_prefix_cache failed (non-fatal): {e}")

        result = await self._post_with_retry("/sleep", params={"level": level})
        logger.info(f"vLLM sleep(level={level}) -> {result}")
        return True

    async def wake_up(self, tags: Optional[List[str]] = None) -> bool:
        """唤醒vLLM"""
        params = None
        if tags:
            # 多值参数: tags=weights&tags=kv_cache
            params = {"tags": tags}
        result = await self._post_with_retry("/wake_up", params=params)
        logger.info(f"vLLM wake_up(tags={tags}) -> {result}")
        return True

    async def init_weight_transfer(self, init_info: Optional[dict] = None) -> bool:
        """初始化权重传输引擎"""
        result = await self._post_with_retry(
            "/init_weight_transfer_engine",
            json={"init_info": init_info or {}},
        )
        logger.info(f"init_weight_transfer_engine -> {result}")
        return True

    async def start_weight_update(self, is_checkpoint_format: bool = True) -> bool:
        """开始权重更新"""
        result = await self._post_with_retry(
            "/start_weight_update",
            json={"is_checkpoint_format": is_checkpoint_format},
        )
        logger.info(f"start_weight_update -> {result}")
        return True

    async def update_weights(self, update_info: dict,
                             weight_version: Optional[str] = None) -> bool:
        """发送一批权重的IPC handles"""
        body: Dict[str, Any] = {"update_info": update_info}
        if weight_version is not None:
            body["weight_version"] = weight_version
        result = await self._post_with_retry("/update_weights", json=body)
        return True

    async def finish_weight_update(self) -> bool:
        """完成权重更新"""
        result = await self._post_with_retry(
            "/finish_weight_update", json={}
        )
        logger.info(f"finish_weight_update -> {result}")
        return True


def _get_gpu_uuid() -> str:
    """获取当前GPU的UUID"""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(props.uuid)


class IPCWeightTransfer:
    """CUDA IPC权重传输引擎"""

    def __init__(self, vllm_client: VLLMClient, chunk_size: int = 16,
                 weight_version: str = "1"):
        """
        Args:
            vllm_client: vLLM HTTP客户端
            chunk_size: 每次传输多少个参数
            weight_version: 权重版本号
        """
        self.client = vllm_client
        self.chunk_size = chunk_size
        self.weight_version = weight_version
        self._stats = TransferStats()
        self._tensor_refs: List[torch.Tensor] = []  # 防止GC释放

    def build_ipc_handles(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> List[IPCHandle]:
        """将state_dict中的CUDA张量转换为IPC handles

        使用 torch.multiprocessing.reductions.reduce_tensor 获取handle
        """
        from torch.multiprocessing.reductions import reduce_tensor

        t0 = time.perf_counter()
        handles: List[IPCHandle] = []
        device_id = torch.cuda.current_device()

        for name, tensor in state_dict.items():
            if not tensor.is_cuda:
                logger.warning(f"Skipping non-CUDA tensor: {name}")
                continue

            # 确保连续存储
            weight = tensor.detach().contiguous()
            self._tensor_refs.append(weight)  # 保持引用，防止被回收

            # 获取IPC handle (reduce_tensor返回 (rebuild_fn, args))
            _, ipc_args = reduce_tensor(weight)

            handle = IPCHandle(
                name=name,
                handle=cloudpickle.dumps(ipc_args),
                shape=tuple(weight.shape),
                dtype=weight.dtype,
                device_id=device_id,
            )
            handles.append(handle)

        self._stats.build_handle_time = time.perf_counter() - t0
        self._stats.total_params = len(handles)
        self._stats.total_bytes = sum(
            t.nelement() * t.element_size()
            for t in state_dict.values()
            if t.is_cuda
        )
        logger.info(
            f"Built {len(handles)} IPC handles in "
            f"{self._stats.build_handle_time:.3f}s "
            f"({self._stats.total_bytes / 1e9:.2f} GB)"
        )
        return handles

    def _build_http_payload(self, handles: List[IPCHandle]) -> dict:
        """将IPCHandle列表构建为HTTP请求的payload

        格式与vLLM的update_weights API兼容
        """
        gpu_uuid = _get_gpu_uuid()
        names = []
        dtype_names = []
        shapes = []
        ipc_handles = []

        for h in handles:
            names.append(h.name)
            dtype_names.append(str(h.dtype).split(".")[-1])
            shapes.append(list(h.shape))
            # 每个handle是 {gpu_uuid: ipc_args} 的映射
            ipc_args = cloudpickle.loads(h.handle)
            ipc_handles.append({gpu_uuid: ipc_args})

        # 序列化ipc_handles为base64编码的pickle
        ipc_handles_pickled = base64.b64encode(
            cloudpickle.dumps(ipc_handles)
        ).decode("utf-8")

        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles_pickled": ipc_handles_pickled,
        }

    async def push_weights(self, state_dict: Dict[str, torch.Tensor],
                           sleep_level: int = 0) -> bool:
        """完整的权重推送流程

        1. sleep vLLM (暂停调度)
        2. build IPC handles
        3. init + start_weight_update
        4. 分块发送 (chunk_size个参数一组)
        5. finish_weight_update
        6. wake_up vLLM

        异常处理:
        - CUDA OOM: 清理缓存后重试一次构建 IPC handles
        - 网络错误: 由 _post_with_retry 内部重试处理
        - 其他异常: 记录错误并尝试唤醒 vLLM 恢复服务
        """
        self._stats = TransferStats()
        self._stats.start()
        self._tensor_refs.clear()

        try:
            logger.info("Step 1: Putting vLLM to sleep...")
            await self.client.sleep(level=sleep_level)

            logger.info("Step 2: Building IPC handles...")
            try:
                handles = self.build_ipc_handles(state_dict)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "CUDA OOM while building IPC handles. "
                    "Clearing cache and retrying once..."
                )
                torch.cuda.empty_cache()
                self._tensor_refs.clear()
                handles = self.build_ipc_handles(state_dict)

            logger.info("Step 3: Initializing weight transfer...")
            await self.client.init_weight_transfer()
            await self.client.start_weight_update(is_checkpoint_format=True)

            logger.info("Step 4: Sending weight chunks...")
            t_transfer = time.perf_counter()
            chunks = self._chunk_handles(handles)
            for i, chunk in enumerate(chunks):
                payload = self._build_http_payload(chunk)
                await self.client.update_weights(
                    payload, weight_version=self.weight_version
                )
                self._stats.chunks_sent += 1
                logger.debug(
                    f"  Chunk {i + 1}/{len(chunks)} sent "
                    f"({len(chunk)} params)"
                )
            self._stats.transfer_time = time.perf_counter() - t_transfer

            logger.info("Step 5: Finishing weight update...")
            await self.client.finish_weight_update()

            logger.info("Step 6: Waking up vLLM...")
            await self.client.wake_up(tags=["weights", "kv_cache"])

            self._stats.finish()
            logger.info(
                f"Weight push completed in {self._stats.total_time:.3f}s "
                f"(transfer: {self._stats.transfer_time:.3f}s, "
                f"bandwidth: {self._stats.bandwidth_gbps:.2f} GB/s)"
            )
            return True

        except torch.cuda.OutOfMemoryError as e:
            self._stats.errors += 1
            logger.error(
                f"CUDA OOM during weight push (unrecoverable): {e}. "
                f"Clearing GPU cache."
            )
            torch.cuda.empty_cache()
            # 尝试唤醒vLLM以恢复服务
            try:
                await self.client.wake_up()
            except Exception:
                pass
            raise

        except NON_RETRIABLE_EXCEPTIONS as e:
            self._stats.errors += 1
            logger.error(
                f"Non-retriable error during weight push: "
                f"{type(e).__name__}: {e}. Aborting."
            )
            # 尝试唤醒vLLM以恢复服务
            try:
                await self.client.wake_up()
            except Exception:
                pass
            raise

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Weight push failed: {type(e).__name__}: {e}")
            # 尝试唤醒vLLM以恢复服务
            try:
                await self.client.wake_up()
            except Exception:
                pass
            raise

    def push_weights_sync(self, state_dict: Dict[str, torch.Tensor],
                          sleep_level: int = 0) -> bool:
        """同步版本的权重推送（方便非异步上下文调用）"""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已经在事件循环中，创建新线程
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(
                    asyncio.run, self.push_weights(state_dict, sleep_level)
                )
                return future.result()
        else:
            return asyncio.run(self.push_weights(state_dict, sleep_level))

    def _chunk_handles(self, handles: List[IPCHandle]) -> List[List[IPCHandle]]:
        """将handles分成chunk_size大小的块"""
        chunks = []
        for i in range(0, len(handles), self.chunk_size):
            chunks.append(handles[i: i + self.chunk_size])
        return chunks

    def verify_consistency(
        self,
        local_dict: Dict[str, torch.Tensor],
        remote_checksum: Optional[str] = None,
    ) -> bool:
        """校验权重一致性（checksum比对）"""
        local_checksum = self._compute_checksum(local_dict)
        if remote_checksum is not None:
            match = local_checksum == remote_checksum
            if not match:
                logger.error(
                    f"Checksum mismatch! local={local_checksum}, "
                    f"remote={remote_checksum}"
                )
            return match
        logger.info(f"Local checksum: {local_checksum}")
        return True

    @staticmethod
    def _compute_checksum(state_dict: Dict[str, torch.Tensor]) -> str:
        """计算state_dict的MD5校验和"""
        md5 = hashlib.md5()
        for name in sorted(state_dict.keys()):
            tensor = state_dict[name]
            # 将tensor转为bytes用于校验
            data = tensor.detach().cpu().float().numpy().tobytes()
            md5.update(name.encode("utf-8"))
            md5.update(data)
        return md5.hexdigest()

    def release_refs(self):
        """释放保持的张量引用，允许GPU显存回收"""
        self._tensor_refs.clear()

    @property
    def stats(self) -> TransferStats:
        """获取最近一次传输的统计信息"""
        return self._stats


class WeightUpdateManager:
    """权重更新管理器 — 支持回滚的权重应用

    职责:
    - 在应用新权重前保存回滚点（引用旧 state_dict）
    - 更新失败时自动恢复到上一版本
    - 跟踪当前权重版本

    用法:
        manager = WeightUpdateManager(model)
        success = manager.update_weights(new_state_dict)
        # 失败时自动回滚，无需外部干预
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: 需要更新权重的模型实例
        """
        self._model = model
        self._previous_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._current_version: int = 0
        self._rollback_count: int = 0

    def update_weights(self, new_state_dict: Dict[str, torch.Tensor]) -> bool:
        """应用新权重，失败时自动回滚

        Args:
            new_state_dict: 新的权重字典

        Returns:
            True 表示更新成功，False 表示更新失败已回滚
        """
        # 保留回滚点（引用，不深拷贝，避免显存翻倍）
        previous_state = {
            k: v.detach() for k, v in self._model.state_dict().items()
        }

        try:
            # 应用新权重
            self._model.load_state_dict(new_state_dict, strict=False)
            # 更新成功，保存当前 state 用于下次回滚
            self._previous_state_dict = previous_state
            self._current_version += 1
            logger.info(
                f"Weight update successful. Version: {self._current_version}"
            )
            return True

        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"CUDA OOM during weight update: {e}. "
                f"Clearing cache and rolling back."
            )
            torch.cuda.empty_cache()
            self._rollback(previous_state)
            return False

        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(
                f"Weight update failed: {type(e).__name__}: {e}. "
                f"Rolling back to previous version."
            )
            self._rollback(previous_state)
            return False

        except Exception as e:
            logger.error(
                f"Unexpected error during weight update: "
                f"{type(e).__name__}: {e}. Rolling back."
            )
            self._rollback(previous_state)
            return False

    def _rollback(self, previous_state: Dict[str, torch.Tensor]):
        """回滚到之前的权重状态"""
        if previous_state is None:
            logger.warning("No previous state available for rollback.")
            return

        try:
            self._model.load_state_dict(previous_state, strict=False)
            self._rollback_count += 1
            logger.info(
                f"Weight rollback successful. "
                f"Total rollbacks: {self._rollback_count}"
            )
        except Exception as rollback_error:
            logger.critical(
                f"Weight rollback FAILED: {type(rollback_error).__name__}: "
                f"{rollback_error}. Model may be in inconsistent state!"
            )

    @property
    def current_version(self) -> int:
        """当前权重版本号"""
        return self._current_version

    @property
    def rollback_count(self) -> int:
        """累计回滚次数"""
        return self._rollback_count
