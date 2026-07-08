"""
自研推理加速模块 (src/inference/)

核心组件:
- KVCacheManager: PagedAttention 内存管理（按需分页分配KV Cache）
- ContinuousBatchingScheduler: 连续批处理调度（动态插入/移除序列）
- PrefixCache: 前缀缓存（RL场景高复用system prompt加速）
- CUDAGraphRunner: CUDA Graph加速decode（消除kernel launch开销）
- InferenceContext: 推理上下文（forward时传递元数据给Attention层）
- PagedAttention: 分页Attention（支持flash-attn加速和PyTorch SDPA fallback）
- ModelRunner: 模型执行器（串联Model+KVCache+Attention+CUDAGraph）
- InferenceEngine: 统一推理引擎（串联以上组件）

设计定位:
  轻量级推理引擎实现，展示对推理系统核心原理的深度理解。
  生产环境使用 vLLM，通过配置切换 backend="custom"/"vllm"。
"""
from .kv_cache import Block, BlockPool, BlockTable, KVCacheManager
from .scheduler import (
    Request,
    RequestState,
    SchedulerOutput,
    ContinuousBatchingScheduler,
)
from .prefix_cache import PrefixCache, CachedBlock
from .cuda_graph import CUDAGraphRunner
from .context import InferenceContext, set_context, get_context, reset_context
from .attention import PagedAttention, store_kv_cache, HAS_FLASH_ATTN
from .model_runner import ModelRunner, ModelRunnerConfig
from .engine import InferenceEngine, InferenceConfig

__all__ = [
    "Block",
    "BlockPool",
    "BlockTable",
    "KVCacheManager",
    "Request",
    "RequestState",
    "SchedulerOutput",
    "ContinuousBatchingScheduler",
    "PrefixCache",
    "CachedBlock",
    # CUDA Graph
    "CUDAGraphRunner",
    # Context
    "InferenceContext",
    "set_context",
    "get_context",
    "reset_context",
    # Attention
    "PagedAttention",
    "store_kv_cache",
    "HAS_FLASH_ATTN",
    # Model Runner
    "ModelRunner",
    "ModelRunnerConfig",
    # Engine
    "InferenceEngine",
    "InferenceConfig",
]
