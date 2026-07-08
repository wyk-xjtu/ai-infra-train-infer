"""
PagedAttention — 支持 Paged KV Cache 的 Attention 实现

性能分层:
1. Linux + CUDA + flash-attn: 使用 flash_attn_varlen_func / flash_attn_with_kvcache（最快）
2. Windows + CUDA（无 flash-attn）: 使用 PyTorch 原生 F.scaled_dot_product_attention（GPU加速，性能可接受）
   flash-attn 不支持 Windows 编译，但 PyTorch SDPA 在 CUDA 上仍有 cuDNN/Math backend 加速

KV Cache 存储:
- 通过 Triton kernel（Linux）或 PyTorch 索引操作（Windows）将新计算的 K/V 写入物理 slot
- slot 由 scheduler 通过 block_table 分配，slot_mapping 指定每个 token 的目标位置

与 nano-vllm 的差异:
- 增加了 HAS_FLASH_ATTN / HAS_TRITON 的 graceful degradation
- SDPA fallback 确保 Windows + GPU 环境也能高效执行
- 保持接口一致，可无缝切换加速后端
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .context import get_context


try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def store_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """将 K/V 存储到 paged cache 的指定 slot

    Args:
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim] (一层的 K cache)
        v_cache: 同上 (一层的 V cache)
        slot_mapping: [num_tokens] 每个 token 的目标物理 slot index
            (物理 slot = block_id * block_size + offset_in_block)
    """
    if HAS_TRITON and key.is_cuda:
        _store_kv_cache_triton(key, value, k_cache, v_cache, slot_mapping)
    else:
        _store_kv_cache_pytorch(key, value, k_cache, v_cache, slot_mapping)


def _store_kv_cache_pytorch(key, value, k_cache, v_cache, slot_mapping):
    """纯 PyTorch fallback 的 KV 存储

    将 cache 视为 [total_slots, num_kv_heads, head_dim] 进行索引。
    """
    # slot_mapping 中 -1 表示跳过（warmup 等场景）
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    total_slots = num_blocks * block_size

    # 展平 cache 为 [total_slots, heads, dim] 方便索引
    k_flat = k_cache.view(total_slots, num_kv_heads, head_dim)
    v_flat = v_cache.view(total_slots, num_kv_heads, head_dim)

    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    k_flat[valid_slots] = key[valid_mask]
    v_flat[valid_slots] = value[valid_mask]


if HAS_TRITON:
    @triton.jit
    def _store_kvcache_kernel(
        key_ptr, key_stride,
        value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        """Triton kernel: 将单个 token 的 KV 写入对应的物理 slot

        每个 program 处理一个 token（所有 head 展平为 D = num_heads * head_dim）
        """
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return

        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)

        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def _store_kv_cache_triton(key, value, k_cache, v_cache, slot_mapping):
        """Triton 加速的 KV 存储"""
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        # 确保内存连续
        key_contig = key.contiguous().view(N, D)
        value_contig = value.contiguous().view(N, D)
        # 展平 cache
        k_flat = k_cache.view(-1, D)
        v_flat = v_cache.view(-1, D)
        _store_kvcache_kernel[(N,)](
            key_contig, key_contig.stride(0),
            value_contig, value_contig.stride(0),
            k_flat, v_flat,
            slot_mapping,
            D=D,
        )


class PagedAttention(nn.Module):
    """Paged Attention 层 — 用于推理引擎的高效 Attention

    与训练时的标准 Attention 区别:
    - 使用 paged KV Cache（不是每次重新计算所有 KV）
    - 支持 Prefill（变长序列）+ Decode（单 token）两种模式
    - 通过 context 获取 slot_mapping / block_tables 等元信息
    - KV Cache 由 ModelRunner 绑定（在 allocate_kv_cache 时设置）
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        # 由 ModelRunner._bind_kv_cache_to_model() 绑定
        self.k_cache: Optional[torch.Tensor] = None  # [num_blocks, block_size, kv_heads, dim]
        self.v_cache: Optional[torch.Tensor] = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: [num_tokens, num_heads, head_dim]
            k: [num_tokens, num_kv_heads, head_dim]
            v: [num_tokens, num_kv_heads, head_dim]

        Returns:
            output: [num_tokens, num_heads, head_dim]
        """
        ctx = get_context()

        if self.k_cache is not None and ctx.slot_mapping is not None:
            store_kv_cache(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)

        if ctx.is_prefill:
            return self._prefill_attention(q, k, v, ctx)
        else:
            return self._decode_attention(q, ctx)

    def _prefill_attention(self, q, k, v, ctx):
        """Prefill: 变长序列 attention（flash-attn 或 SDPA fallback）"""
        if HAS_FLASH_ATTN and q.is_cuda:
            if ctx.block_tables is not None:
                k_src, v_src = self.k_cache, self.v_cache
            else:
                k_src, v_src = k, v
            out = flash_attn_varlen_func(
                q, k_src, v_src,
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=ctx.block_tables,
            )
        else:
            # PyTorch SDPA fallback（Windows + CUDA，无 flash-attn 时使用原生 SDPA）
            out = self._sdpa_prefill_fallback(q, k, v, ctx)
        return out

    def _decode_attention(self, q, ctx):
        """Decode: 从 KV Cache 读取历史 KV，只处理当前 token"""
        if HAS_FLASH_ATTN and q.is_cuda:
            # q shape 需要 [batch, 1, heads, dim] for flash_attn_with_kvcache
            out = flash_attn_with_kvcache(
                q.unsqueeze(1),
                self.k_cache, self.v_cache,
                cache_seqlens=ctx.context_lens,
                block_table=ctx.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
            out = out.squeeze(1)
        else:
            out = self._sdpa_decode_fallback(q, ctx)
        return out

    def _sdpa_prefill_fallback(self, q, k, v, ctx):
        """Prefill fallback: 按序列分段处理，使用 F.scaled_dot_product_attention

        Windows + CUDA 路径: flash-attn 不支持 Windows 编译，但 PyTorch SDPA
        在 CUDA 上仍有 cuDNN/efficient attention backend 加速，性能可接受。
        """
        outputs = []
        cu_q = ctx.cu_seqlens_q
        cu_k = ctx.cu_seqlens_k

        num_seqs = len(cu_q) - 1
        for i in range(num_seqs):
            q_start, q_end = int(cu_q[i]), int(cu_q[i + 1])
            k_start, k_end = int(cu_k[i]), int(cu_k[i + 1])
            qi = q[q_start:q_end]  # [seq_q, heads, dim]
            ki = k[k_start:k_end]  # [seq_k, kv_heads, dim]
            vi = v[k_start:k_end]

            # GQA: 扩展 kv_heads → num_heads
            if self.num_kv_heads < self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                ki = ki.repeat_interleave(repeat, dim=1)
                vi = vi.repeat_interleave(repeat, dim=1)

            # [seq, heads, dim] → [heads, seq, dim] for SDPA
            qi = qi.transpose(0, 1)
            ki = ki.transpose(0, 1)
            vi = vi.transpose(0, 1)

            oi = F.scaled_dot_product_attention(
                qi, ki, vi, is_causal=True, scale=self.scale
            )
            outputs.append(oi.transpose(0, 1))  # → [seq_q, heads, dim]

        return torch.cat(outputs, dim=0)

    def _sdpa_decode_fallback(self, q, ctx):
        """Decode fallback: 从 cache 中 gather 历史 KV 然后用 SDPA 计算

        Windows + CUDA 路径: 无 flash-attn 时从 paged cache 中逐序列 gather KV,
        然后调用 F.scaled_dot_product_attention。比 flash_attn_with_kvcache 慢
        （因为需要 gather），但在 GPU 上仍可正常工作。
        """
        if self.k_cache is None or ctx.block_tables is None or ctx.context_lens is None:
            return q  # 极端 fallback

        batch_size = q.shape[0]
        num_blocks_cache, block_size, num_kv_heads, head_dim = self.k_cache.shape
        outputs = []

        for i in range(batch_size):
            ctx_len = int(ctx.context_lens[i])
            blocks = ctx.block_tables[i]
            # 收集该序列的所有历史 KV
            k_list, v_list = [], []
            for pos in range(ctx_len):
                block_idx = pos // block_size
                offset = pos % block_size
                physical_block = int(blocks[block_idx])
                k_list.append(self.k_cache[physical_block, offset])  # [kv_heads, dim]
                v_list.append(self.v_cache[physical_block, offset])

            ki = torch.stack(k_list, dim=0)  # [ctx_len, kv_heads, dim]
            vi = torch.stack(v_list, dim=0)

            qi = q[i:i+1]  # [1, heads, dim]

            # GQA expand
            if self.num_kv_heads < self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                ki = ki.repeat_interleave(repeat, dim=1)
                vi = vi.repeat_interleave(repeat, dim=1)

            # [seq, heads, dim] → [heads, seq, dim]
            qi = qi.transpose(0, 1)
            ki = ki.transpose(0, 1)
            vi = vi.transpose(0, 1)

            oi = F.scaled_dot_product_attention(
                qi, ki, vi, is_causal=False, scale=self.scale
            )
            outputs.append(oi.transpose(0, 1))  # [1, heads, dim]

        return torch.cat(outputs, dim=0)  # [batch, heads, dim]
