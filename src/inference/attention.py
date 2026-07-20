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
from typing import Optional, List, Tuple

from .context import get_context
from .kv_cache import HAS_FP8, _FP8_DTYPE

# ==================== 可选加速库探测 ====================

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


# ==================== KV Cache 存储 ====================

def store_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    cache_dtype: str = "fp16",
):
    """将 K/V 存储到 paged cache 的指定 slot

    Args:
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim] (一层的 K cache)
        v_cache: 同上 (一层的 V cache)
        slot_mapping: [num_tokens] 每个 token 的目标物理 slot index
            (物理 slot = block_id * block_size + offset_in_block)
        k_scale: FP8 模式下的 K scale tensor [num_blocks, block_size] (展平视图)
        v_scale: FP8 模式下的 V scale tensor [num_blocks, block_size] (展平视图)
        cache_dtype: "fp16" 或 "fp8"
    """
    if cache_dtype == "fp8" and HAS_FP8:
        _store_kv_cache_fp8(key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale)
    elif HAS_TRITON and key.is_cuda:
        _store_kv_cache_triton(key, value, k_cache, v_cache, slot_mapping)
    else:
        _store_kv_cache_pytorch(key, value, k_cache, v_cache, slot_mapping)


def _store_kv_cache_fp8(key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale):
    """FP8 量化存储: per-token 动态量化后写入 cache

    量化策略: per-token absmax 量化
    - 对每个 token 的 KV 向量，计算 abs max
    - scale = max_fp8 / abs_max (max_fp8 = 448.0 for e4m3fn)
    - 量化值 = clamp(value * scale, -448, 448).to(float8_e4m3fn)
    - 反量化: fp8_value.to(fp16) / scale
    """
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    total_slots = num_blocks * block_size
    max_fp8 = 448.0  # torch.float8_e4m3fn 的最大值

    # 展平 cache
    k_flat = k_cache.view(total_slots, num_kv_heads, head_dim)
    v_flat = v_cache.view(total_slots, num_kv_heads, head_dim)
    k_scale_flat = k_scale.view(total_slots)
    v_scale_flat = v_scale.view(total_slots)

    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    k_valid = key[valid_mask]    # [N, kv_heads, head_dim]
    v_valid = value[valid_mask]  # [N, kv_heads, head_dim]

    # Per-token 量化: 计算每个 token 的 absmax（跨 heads 和 dim）
    # k_valid shape: [N, kv_heads, head_dim] -> absmax over last 2 dims
    k_amax = k_valid.float().abs().amax(dim=(-2, -1)).clamp(min=1e-5)  # [N]
    v_amax = v_valid.float().abs().amax(dim=(-2, -1)).clamp(min=1e-5)  # [N]

    # scale = max_fp8 / absmax (用于量化时乘，反量化时除)
    k_s = max_fp8 / k_amax  # [N]
    v_s = max_fp8 / v_amax  # [N]

    # 量化: 乘 scale 然后转 FP8
    k_quantized = (k_valid.float() * k_s[:, None, None]).to(_FP8_DTYPE)
    v_quantized = (v_valid.float() * v_s[:, None, None]).to(_FP8_DTYPE)

    # 写入 cache
    k_flat[valid_slots] = k_quantized
    v_flat[valid_slots] = v_quantized
    k_scale_flat[valid_slots] = k_s
    v_scale_flat[valid_slots] = v_s


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


# ==================== Triton KV Store Kernel ====================

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


# ==================== PagedAttention 模块 ====================

class PagedAttention(nn.Module):
    """Paged Attention 层 — 用于推理引擎的高效 Attention

    与训练时的标准 Attention 区别:
    - 使用 paged KV Cache（不是每次重新计算所有 KV）
    - 支持 Prefill（变长序列）+ Decode（单 token）两种模式
    - 通过 context 获取 slot_mapping / block_tables 等元信息
    - KV Cache 由 ModelRunner 绑定（在 allocate_kv_cache 时设置）
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 cache_dtype: str = "fp16"):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.cache_dtype = cache_dtype
        # 由 ModelRunner._bind_kv_cache_to_model() 绑定
        self.k_cache: Optional[torch.Tensor] = None  # [num_blocks, block_size, kv_heads, dim]
        self.v_cache: Optional[torch.Tensor] = None
        # FP8 模式下的 per-token scale factors
        self.k_scale: Optional[torch.Tensor] = None  # [num_blocks, block_size]
        self.v_scale: Optional[torch.Tensor] = None

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

        # 1. 存储新 K/V 到 cache
        if self.k_cache is not None and ctx.slot_mapping is not None:
            store_kv_cache(
                k, v, self.k_cache, self.v_cache, ctx.slot_mapping,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                cache_dtype=self.cache_dtype,
            )

        # 2. 执行 Attention
        if ctx.is_prefill:
            return self._prefill_attention(q, k, v, ctx)
        else:
            return self._decode_attention(q, ctx)

    def _prefill_attention(self, q, k, v, ctx):
        """Prefill: 变长序列 attention（flash-attn 或 SDPA fallback）"""
        if HAS_FLASH_ATTN and q.is_cuda:
            if ctx.block_tables is not None:
                # Prefix cache 命中: K/V 从 cache 中读取
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
            # 输出 shape [batch, 1, heads, dim] → [batch, heads, dim]
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

        向量化实现: 使用 torch.arange + 整数除法/取模计算 block_indices 和 offsets,
        一次 index_select 完成所有 token 的 gather，避免逐 token 循环。

        FP8 模式: gather 后自动反量化（fp8 -> fp16 / scale）。
        """
        if self.k_cache is None or ctx.block_tables is None or ctx.context_lens is None:
            return q  # 极端 fallback

        batch_size = q.shape[0]
        num_blocks_cache, block_size, num_kv_heads, head_dim = self.k_cache.shape
        outputs = []

        for i in range(batch_size):
            ctx_len = int(ctx.context_lens[i])
            blocks = ctx.block_tables[i]

            # 向量化 gather: 使用 arange + 整数除法/取模计算所有 token 的物理位置
            positions = torch.arange(ctx_len, device=q.device)
            block_indices = positions // block_size  # 每个 token 对应的逻辑 block index
            offsets = positions % block_size          # 每个 token 在 block 内的 offset
            physical_blocks = blocks[block_indices]   # 查找物理 block id
            # 计算展平后的物理 slot index
            flat_indices = physical_blocks.long() * block_size + offsets

            # 展平 cache 为 [total_slots, kv_heads, head_dim] 并一次性 index
            k_flat = self.k_cache.view(-1, num_kv_heads, head_dim)
            v_flat = self.v_cache.view(-1, num_kv_heads, head_dim)
            ki = k_flat[flat_indices]  # [ctx_len, kv_heads, dim]
            vi = v_flat[flat_indices]  # [ctx_len, kv_heads, dim]

            # FP8 反量化: fp8_value.to(compute_dtype) / scale
            if self.cache_dtype == "fp8" and self.k_scale is not None:
                k_scale_flat = self.k_scale.view(-1)  # [total_slots]
                v_scale_flat = self.v_scale.view(-1)
                k_s = k_scale_flat[flat_indices]  # [ctx_len]
                v_s = v_scale_flat[flat_indices]  # [ctx_len]
                # 反量化: 转回 fp16 然后除以 scale
                ki = ki.to(torch.float16) / k_s[:, None, None]
                vi = vi.to(torch.float16) / v_s[:, None, None]

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


# ==================== Partitioned PagedAttention v2 ====================

class PartitionedPagedAttention(PagedAttention):
    """分区 PagedAttention v2：将 decode attention 分为 compute 和 context 两部分并行计算

    当 context_len > partition_size 时，将 KV blocks 划分为多个 partition，
    每个 partition 独立计算局部 attention（局部 softmax + output），
    然后使用 log-sum-exp 技巧精确合并所有 partition 的结果。

    这是 vLLM v2 采用的精确做法（非近似），数学上等价于标准全序列 attention。

    收益：
    - 长序列 decode 时可以并行计算多个 partition（减少 kernel launch 延迟）
    - 降低单次 kernel 的内存占用（每个 partition 只需加载局部 KV）
    - 对 context_len <= partition_size 的短序列自动 fallback 到标准路径
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 cache_dtype: str = "fp16", partition_size: int = 512):
        super().__init__(num_heads, num_kv_heads, head_dim, cache_dtype)
        self.partition_size = partition_size

    def _decode_attention(self, q, ctx):
        """分区 decode attention

        当 context_len <= partition_size 时，走标准路径。
        当 context_len > partition_size 时，分为多个 partition：
          - 每个 partition 独立计算局部 attention
          - 使用 log-sum-exp 技巧精确合并所有 partition 的结果
        """
        # 检查是否有有效 context_lens
        if ctx.context_lens is None:
            return super()._decode_attention(q, ctx)

        # 获取最大 context length
        max_ctx_len = int(ctx.context_lens.max().item()) if ctx.context_lens.numel() > 0 else 0

        if max_ctx_len <= self.partition_size:
            # 短序列直接走标准路径
            return super()._decode_attention(q, ctx)

        # 长序列：使用分区计算
        if HAS_FLASH_ATTN and q.is_cuda:
            # flash_attn_with_kvcache 已经内部优化，直接使用标准路径
            return super()._decode_attention(q, ctx)

        # SDPA fallback 路径：使用分区策略优化长序列
        return self._partitioned_decode_fallback(q, ctx)

    def _partitioned_decode_fallback(self, q, ctx):
        """分区 decode 的 SDPA fallback 实现

        对每个序列的 KV cache 按 partition_size 划分，
        每个 partition 独立计算局部 attention，然后用 LSE 合并。
        """
        if self.k_cache is None or ctx.block_tables is None or ctx.context_lens is None:
            return q  # 极端 fallback

        batch_size = q.shape[0]
        num_blocks_cache, block_size, num_kv_heads, head_dim = self.k_cache.shape
        outputs = []

        for i in range(batch_size):
            ctx_len = int(ctx.context_lens[i])
            blocks = ctx.block_tables[i]

            if ctx_len <= self.partition_size:
                # 短序列走标准非分区路径
                out_i = self._single_sequence_decode(q[i:i+1], blocks, ctx_len, block_size)
                outputs.append(out_i)
            else:
                # 长序列：分区计算
                num_partitions = (ctx_len + self.partition_size - 1) // self.partition_size

                partition_outputs = []
                partition_lse = []  # log-sum-exp for numerically stable merging

                for p in range(num_partitions):
                    start = p * self.partition_size
                    end = min(start + self.partition_size, ctx_len)
                    # 计算该分区的 attention
                    local_out, local_lse = self._compute_partition_attention(
                        q[i:i+1], blocks, start, end, block_size
                    )
                    partition_outputs.append(local_out)
                    partition_lse.append(local_lse)

                # 使用 log-sum-exp 技巧合并所有 partition 的结果
                merged = self._merge_partitions(partition_outputs, partition_lse)
                outputs.append(merged)

        return torch.cat(outputs, dim=0)  # [batch, heads, dim]

    def _single_sequence_decode(self, qi, blocks, ctx_len, block_size):
        """单序列标准 decode（不分区），复用 gather 逻辑"""
        num_kv_heads = self.k_cache.shape[2]
        head_dim = self.k_cache.shape[3]

        positions = torch.arange(ctx_len, device=qi.device)
        block_indices = positions // block_size
        offsets = positions % block_size
        physical_blocks = blocks[block_indices]
        flat_indices = physical_blocks.long() * block_size + offsets

        k_flat = self.k_cache.view(-1, num_kv_heads, head_dim)
        v_flat = self.v_cache.view(-1, num_kv_heads, head_dim)
        ki = k_flat[flat_indices]
        vi = v_flat[flat_indices]

        # FP8 反量化
        if self.cache_dtype == "fp8" and self.k_scale is not None:
            k_scale_flat = self.k_scale.view(-1)
            v_scale_flat = self.v_scale.view(-1)
            k_s = k_scale_flat[flat_indices]
            v_s = v_scale_flat[flat_indices]
            ki = ki.to(torch.float16) / k_s[:, None, None]
            vi = vi.to(torch.float16) / v_s[:, None, None]

        # GQA expand
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            ki = ki.repeat_interleave(repeat, dim=1)
            vi = vi.repeat_interleave(repeat, dim=1)

        # [seq, heads, dim] → [heads, seq, dim]
        qi_t = qi.transpose(0, 1)
        ki_t = ki.transpose(0, 1)
        vi_t = vi.transpose(0, 1)

        oi = F.scaled_dot_product_attention(
            qi_t, ki_t, vi_t, is_causal=False, scale=self.scale
        )
        return oi.transpose(0, 1)  # [1, heads, dim]

    def _compute_partition_attention(
        self, qi: torch.Tensor, blocks: torch.Tensor,
        start: int, end: int, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 [start, end) 范围的 KV 做 attention，返回 (output, log_sum_exp)

        Args:
            qi: [1, num_heads, head_dim] query
            blocks: block_table for this sequence
            start: 起始 token 位置
            end: 结束 token 位置（不含）
            block_size: block 大小

        Returns:
            output: [1, num_heads, head_dim] 该 partition 的局部 attention 输出
            lse: [1, num_heads] 该 partition 的 log-sum-exp 值
        """
        num_kv_heads = self.k_cache.shape[2]
        head_dim = self.k_cache.shape[3]
        partition_len = end - start

        # Gather KV for this partition
        positions = torch.arange(start, end, device=qi.device)
        block_indices = positions // block_size
        offsets = positions % block_size
        physical_blocks = blocks[block_indices]
        flat_indices = physical_blocks.long() * block_size + offsets

        k_flat = self.k_cache.view(-1, num_kv_heads, head_dim)
        v_flat = self.v_cache.view(-1, num_kv_heads, head_dim)
        ki = k_flat[flat_indices]  # [partition_len, kv_heads, dim]
        vi = v_flat[flat_indices]

        # FP8 反量化
        if self.cache_dtype == "fp8" and self.k_scale is not None:
            k_scale_flat = self.k_scale.view(-1)
            v_scale_flat = self.v_scale.view(-1)
            k_s = k_scale_flat[flat_indices]
            v_s = v_scale_flat[flat_indices]
            ki = ki.to(torch.float16) / k_s[:, None, None]
            vi = vi.to(torch.float16) / v_s[:, None, None]

        # GQA expand
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            ki = ki.repeat_interleave(repeat, dim=1)
            vi = vi.repeat_interleave(repeat, dim=1)

        # 手动计算 attention 以获取 LSE
        # qi: [1, num_heads, head_dim], ki: [partition_len, num_heads, head_dim]
        qi_expanded = qi.squeeze(0)  # [num_heads, head_dim]

        # attention scores: [num_heads, partition_len]
        # ki transposed: [num_heads, head_dim, partition_len]
        ki_t = ki.permute(1, 2, 0)  # [num_heads, head_dim, partition_len]
        scores = torch.matmul(qi_expanded.unsqueeze(1), ki_t).squeeze(1) * self.scale
        # scores: [num_heads, partition_len]

        # 计算 log-sum-exp
        lse = torch.logsumexp(scores, dim=-1)  # [num_heads]

        # 计算 softmax 和加权输出
        attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, partition_len]

        # vi: [partition_len, num_heads, head_dim] → [num_heads, partition_len, head_dim]
        vi_t = vi.transpose(0, 1)
        # output: [num_heads, head_dim]
        output = torch.matmul(attn_weights.unsqueeze(1), vi_t).squeeze(1)

        # reshape to [1, num_heads, head_dim] and [1, num_heads]
        return output.unsqueeze(0), lse.unsqueeze(0)

    def _merge_partitions(
        self, partition_outputs: List[torch.Tensor], partition_lse: List[torch.Tensor]
    ) -> torch.Tensor:
        """使用 log-sum-exp 技巧精确合并多个 partition 的 softmax 结果

        数学原理：
        对于 softmax(x) 分块计算，全局结果可以通过局部 LSE 精确恢复：
          global_lse = logsumexp(lse_1, lse_2, ..., lse_n)
          weight_i = exp(lse_i - global_lse)
          global_output = sum(weight_i * local_output_i)

        Args:
            partition_outputs: 每个 partition 的局部 attention 输出 [1, num_heads, head_dim]
            partition_lse: 每个 partition 的 log-sum-exp 值 [1, num_heads]

        Returns:
            merged: [1, num_heads, head_dim] 精确合并后的全局 attention 输出
        """
        # Stack LSE values: [num_partitions, 1, num_heads]
        lse_stack = torch.cat(partition_lse, dim=0)  # [num_partitions, num_heads]

        # Global LSE: [num_heads]
        global_lse = torch.logsumexp(lse_stack, dim=0)  # [num_heads]

        # Compute weights: [num_partitions, num_heads]
        weights = torch.exp(lse_stack - global_lse.unsqueeze(0))  # [num_partitions, num_heads]

        # Weighted sum of outputs
        # partition_outputs: list of [1, num_heads, head_dim]
        out_stack = torch.cat(partition_outputs, dim=0)  # [num_partitions, num_heads, head_dim]

        # weights: [num_partitions, num_heads, 1] for broadcasting
        merged = (out_stack * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        # merged: [1, num_heads, head_dim]

        return merged
