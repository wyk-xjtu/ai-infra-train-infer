"""分布式训练核心模块：Mini-Megatron (张量并行) + Mini-ZeRO (优化器分片)"""

from .parallel_context import ParallelContext
from .comm import (
    all_reduce,
    all_gather,
    reduce_scatter,
    broadcast,
    copy_to_parallel_region,
    reduce_from_parallel_region,
    gather_from_parallel_region,
    AsyncAllReduce,
    all_reduce_async,
)
from .stream_manager import StreamManager
from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelMLP,
    ParallelAttention,
    ParallelTransformerLayer,
    ParallelTransformerModel,
    RMSNorm,
    load_from_hf_checkpoint,
)
from .zero_optim import ZeROOptimizer
from .zero_optim_v2 import ZeROStage2Optimizer
from .zero_optim_v3 import ZeROStage3Optimizer
from .lora import (
    LoRALayer,
    LoRAColumnParallel,
    LoRARowParallel,
    LinearWithLoRA,
    apply_lora,
    merge_lora_weights,
    get_lora_state_dict,
)

__all__ = [
    # Parallel Context
    "ParallelContext",
    # Communication
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "copy_to_parallel_region",
    "reduce_from_parallel_region",
    "gather_from_parallel_region",
    "AsyncAllReduce",
    "all_reduce_async",
    # Stream Manager
    "StreamManager",
    # Tensor Parallelism
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelMLP",
    "ParallelAttention",
    "ParallelTransformerLayer",
    "ParallelTransformerModel",
    "RMSNorm",
    "load_from_hf_checkpoint",
    # ZeRO
    "ZeROOptimizer",
    "ZeROStage2Optimizer",
    "ZeROStage3Optimizer",
    # LoRA
    "LoRALayer",
    "LoRAColumnParallel",
    "LoRARowParallel",
    "LinearWithLoRA",
    "apply_lora",
    "merge_lora_weights",
    "get_lora_state_dict",
]
