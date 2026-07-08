"""
推理上下文 — 在 model forward 过程中传递 prefill/decode 元数据

仿照 nano-vllm 的 set_context/get_context 机制：
- Attention 层需要知道当前是 prefill 还是 decode
- 需要 slot_mapping 确定 KV 存储位置
- 需要 block_tables 从 cache 读取历史 KV
- 需要 cu_seqlens 做变长序列 attention

设计说明:
- 使用模块级全局变量（与 nano-vllm 一致），而非 thread-local
  因为推理引擎单线程执行 forward，不存在并发 forward 场景
- set_context 在 prepare_prefill/prepare_decode 时调用
- get_context 在 Attention 层 forward 时获取
- reset_context 在 forward 结束后清理
"""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class InferenceContext:
    """推理上下文数据 — Attention 层执行时所需的全部元信息"""
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 通用: token → 物理 slot 映射 (用于写入 KV Cache)
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    # 通用: block_table 用于从 cache 读取历史 KV
    block_tables: Optional[torch.Tensor] = None


# 模块级全局上下文（单线程 forward，无需 thread-local）
_CONTEXT = InferenceContext()


def set_context(
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
):
    """设置当前推理上下文（model forward 前调用）"""
    global _CONTEXT
    _CONTEXT = InferenceContext(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def get_context() -> InferenceContext:
    """获取当前推理上下文（Attention 层 forward 时调用）"""
    return _CONTEXT


def reset_context():
    """重置上下文（forward 结束后调用，释放 tensor 引用）"""
    global _CONTEXT
    _CONTEXT = InferenceContext()
