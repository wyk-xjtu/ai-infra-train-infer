"""
Mini-Megatron: 张量并行 (Tensor Parallelism) 实现

核心思想 (Megatron-LM Style):
- ColumnParallelLinear: 权重按列 (output_dim) 切分，每个 rank 计算部分输出
  Y = X @ A^T → Y_i = X @ A_i^T  (A 按行切分，等价于转置后按列切分)
  
- RowParallelLinear: 权重按行 (input_dim) 切分，输出需 AllReduce 聚合
  Y = X @ A^T → Y = Σ(X_i @ A_i^T)  (A 按列切分，X 按最后维切分)

通信模式 (关键技术要点):
- ColumnParallel: 前向 Identity(输入拷贝), 反向 AllReduce(梯度聚合)
- RowParallel: 前向 AllReduce(输出聚合), 反向 Identity(梯度直传)
- 每个 Transformer 层仅 2 次 AllReduce: Attention(o_proj) + MLP(down_proj)

Qwen3 适配:
- SwiGLU MLP: gate_proj + up_proj (Column) → SiLU(gate) * up → down_proj (Row)
- GQA: num_kv_heads < num_heads，每个 rank 分配 num_heads/tp_size 个 Q heads
  和 num_kv_heads/tp_size 个 KV heads
- RMSNorm: 不做切分（每个 rank 保留完整 norm 参数）
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .parallel_context import ParallelContext
from .comm import (
    copy_to_parallel_region,
    reduce_from_parallel_region,
    gather_from_parallel_region,
)

# Flash Attention 条件导入
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None

logger = logging.getLogger(__name__)

# 模块级标志：确保 Flash Attention fallback warning 只打印一次
_flash_attn_warning_shown = False


def _divide(numerator: int, denominator: int) -> int:
    """确保整除，否则报错"""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Qwen3 使用)

    公式: y = x * rsqrt(mean(x^2) + eps) * weight
    相比 LayerNorm 省去了 mean-shift（无 bias），计算更高效。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # [TP-1修复] RMSNorm 权重在各 TP rank 完整复制，标记以便梯度在 TP 组 AllReduce 同步
        self.weight._tp_replicated = True
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class ColumnParallelLinear(nn.Module):
    """列并行线性层 (Megatron Style)

    权重形状: [out_features/tp_size, in_features]  (PyTorch Linear 的 weight 是 [out, in])
    输入: [batch, seq, in_features] — 每个 rank 有完整输入
    输出: [batch, seq, out_features/tp_size] — 每个 rank 有部分输出

    通信行为:
    - 前向: Identity（输入不需通信，每个 rank 都有完整 x）
            但通过 autograd 注册反向 AllReduce
    - 反向: AllReduce 对 x 的梯度（因为各 rank 对相同 x 计算了不同列的梯度）

    可选: gather_output=True 时在 forward 末尾 AllGather 输出
    （通常不需要，后接 RowParallel 时输出保持切分状态）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_context: ParallelContext,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        self.parallel_context = parallel_context
        self.tp_size = parallel_context.tp_size
        self.tp_rank = parallel_context.tp_rank
        self.gather_output = gather_output

        self.out_features_per_rank = _divide(out_features, self.tp_size)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform 初始化（与 PyTorch Linear 默认一致）"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, in_features] — 完整输入

        Returns:
            output: [batch, seq, out_features/tp_size] (或 gather 后 [batch, seq, out_features])
        """
        x = copy_to_parallel_region(x, self.parallel_context.tp_group)

        output = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            output = gather_from_parallel_region(output, self.parallel_context.tp_group)

        return output


class RowParallelLinear(nn.Module):
    """行并行线性层 (Megatron Style)

    权重形状: [out_features, in_features/tp_size]
    输入: [batch, seq, in_features/tp_size] — 每个 rank 有部分输入（来自上游 ColumnParallel）
    输出: [batch, seq, out_features] — AllReduce 后的完整输出

    通信行为:
    - 前向: AllReduce 输出（聚合各 rank 的部分矩阵乘积 y_i = x_i @ W_i^T）
    - 反向: Identity（梯度直接传给各 rank，因为输入本身就是独立分片）

    关键理解:
    Y = X @ W^T = [X_0, X_1, ..., X_{tp-1}] @ [W_0; W_1; ...; W_{tp-1}]^T
      = Σ_i (X_i @ W_i^T)  → 需要 AllReduce 求和
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_context: ParallelContext,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.parallel_context = parallel_context
        self.tp_size = parallel_context.tp_size
        self.tp_rank = parallel_context.tp_rank
        self.input_is_parallel = input_is_parallel

        self.in_features_per_rank = _divide(in_features, self.tp_size)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, in_features/tp_size] — 来自 ColumnParallel 的部分输出

        Returns:
            output: [batch, seq, out_features] — AllReduce 聚合后的完整输出
        """
        output = F.linear(x, self.weight)

        output = reduce_from_parallel_region(output, self.parallel_context.tp_group)

        if self.bias is not None:
            output = output + self.bias

        return output


class ParallelMLP(nn.Module):
    """并行 MLP 层 — SwiGLU 结构 (Qwen3/LLaMA Style)

    结构: 
        gate = SiLU(gate_proj(x))  # gate_proj: ColumnParallel
        up = up_proj(x)            # up_proj: ColumnParallel  
        output = down_proj(gate * up)  # down_proj: RowParallel

    通信量: 仅 1 次 AllReduce（在 down_proj 的 forward 中自动完成）
    
    SwiGLU 相比标准 FFN:
    - 多一个 gate projection（参数量多 50%）
    - 但效果显著优于 ReLU/GELU FFN
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        parallel_context: ParallelContext,
    ):
        super().__init__()
        # gate_proj 和 up_proj 都是 ColumnParallel（输出切分）
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, parallel_context, bias=False
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, parallel_context, bias=False
        )
        # down_proj 是 RowParallel（输入切分，输出 AllReduce）
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, parallel_context, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, hidden_size]
        Returns:
            output: [batch, seq, hidden_size] (经过 AllReduce)
        """
        # gate 和 up 各自得到 [batch, seq, intermediate_size/tp_size]
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        # SwiGLU: 逐元素乘，仍然是 [batch, seq, intermediate_size/tp_size]
        # down_proj 接收切分后的输入，输出经 AllReduce 得到完整 hidden
        return self.down_proj(gate * up)


# ParallelAttention (GQA/MHA)


class ParallelAttention(nn.Module):
    """并行 Attention 层 — 支持 MHA 和 GQA

    GQA (Grouped Query Attention):
    - num_kv_heads < num_heads 时，多个 Q head 共享 KV head
    - 每个 rank: num_heads/tp_size 个 Q heads, num_kv_heads/tp_size 个 KV heads

    结构:
        q, k, v = qkv_proj(x)  # q_proj/k_proj/v_proj 都是 ColumnParallel
        attn_output = attention(q, k, v)
        output = o_proj(attn_output)  # o_proj 是 RowParallel

    通信量: 仅 1 次 AllReduce（在 o_proj 的 forward 中完成）
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        parallel_context: ParallelContext,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        use_flash_attn: bool = False,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.parallel_context = parallel_context
        tp_size = parallel_context.tp_size

        # Attention backend 配置
        self.attention_backend = attention_backend

        # Flash Attention 配置
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN
        if use_flash_attn and not HAS_FLASH_ATTN:
            global _flash_attn_warning_shown
            if not _flash_attn_warning_shown:
                logger.warning(
                    "Flash Attention requested but not available. "
                    "Install with: pip install flash-attn --no-build-isolation. "
                    "Falling back to standard attention."
                )
                _flash_attn_warning_shown = True

        # 验证可整除性
        self.num_heads = _divide(num_heads, tp_size)  # 每个 rank 的 Q heads
        self.num_kv_heads = _divide(num_kv_heads, tp_size)  # 每个 rank 的 KV heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size

        # Q/K/V projection — 全部 ColumnParallel
        # Q: hidden → num_heads * head_dim (按 num_heads 切分)
        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * head_dim, parallel_context, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, parallel_context, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, parallel_context, bias=False
        )

        # QK-Norm (Qwen3): per-head RMSNorm 在 RoPE 之前
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # Output projection — RowParallel
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, parallel_context, bias=False
        )

        # RoPE (Rotary Position Embedding)
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self._init_rope()

        # Scaling factor
        self.scaling = head_dim ** -0.5

    def _init_rope(self):
        """预计算 RoPE 频率"""
        dim = self.head_dim
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rotary_emb(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """应用 RoPE

        Args:
            x: [batch * seq, num_heads, head_dim]
            positions: [batch * seq] 位置索引

        Returns:
            旋转后的 tensor，形状不变
        """
        # 计算 cos/sin
        seq_len = positions.max().item() + 1
        t = positions.float()  # [batch*seq]
        freqs = torch.outer(t, self.inv_freq.to(x.device))  # [batch*seq, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch*seq, dim]
        cos = emb.cos().unsqueeze(1)  # [batch*seq, 1, dim]
        sin = emb.sin().unsqueeze(1)  # [batch*seq, 1, dim]

        # 旋转: [x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, hidden_size]
            positions: [batch, seq] or [batch*seq] 位置编码索引
            attention_mask: [batch, 1, seq, seq] causal mask (可选，Flash Attention 路径忽略)

        Returns:
            output: [batch, seq, hidden_size] (经 o_proj AllReduce)
        """
        batch_size, seq_len, _ = x.shape

        # Q/K/V projection (ColumnParallel，各得到切分后的 heads)
        q = self.q_proj(x)  # [batch, seq, num_heads_per_rank * head_dim]
        k = self.k_proj(x)  # [batch, seq, num_kv_heads_per_rank * head_dim]
        v = self.v_proj(x)  # [batch, seq, num_kv_heads_per_rank * head_dim]

        # Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # QK-Norm (Qwen3): 在 RoPE 前对每个 head 做 RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        if positions is not None:
            pos_flat = positions.reshape(-1)
            q_flat = q.view(-1, self.num_heads, self.head_dim)
            k_flat = k.view(-1, self.num_kv_heads, self.head_dim)
            q_flat = self._apply_rotary_emb(q_flat, pos_flat)
            k_flat = self._apply_rotary_emb(k_flat, pos_flat)
            q = q_flat.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k_flat.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.use_flash_attn:
            # flash_attn_func 期望输入 shape: [batch, seq_len, num_heads, head_dim]
            # 当前 q/k/v 已经是 [batch, seq, num_heads, head_dim]，无需 transpose

            # Flash Attention 要求 fp16/bf16 输入
            input_dtype = q.dtype
            if q.dtype == torch.float32:
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)

            # Flash Attention 2.5+ 原生支持 GQA（q 和 k/v 的 num_heads 可以不同）
            # 无需手动 expand KV heads
            attn_output = flash_attn_func(
                q, k, v,
                softmax_scale=self.scaling,
                causal=True,
            )
            # attn_output: [batch, seq_len, num_heads, head_dim]

            # 转回原始 dtype
            if attn_output.dtype != input_dtype:
                attn_output = attn_output.to(input_dtype)

            # Reshape: [batch, seq, num_heads * head_dim]
            attn_output = attn_output.contiguous().view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
        else:
            # Transpose: [batch, num_heads, seq, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # GQA: 扩展 KV heads 以匹配 Q heads 数量
            if self.num_kv_heads < self.num_heads:
                num_groups = self.num_heads // self.num_kv_heads
                k = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(
                    batch_size, self.num_heads, seq_len, self.head_dim
                )
                v = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(
                    batch_size, self.num_heads, seq_len, self.head_dim
                )

            if self.attention_backend == "sdpa":
                # PyTorch SDPA — memory-efficient（类似 Flash Attention 的 O(n) 显存）
                # q, k, v shape: [batch, num_heads, seq, head_dim]
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=True,
                    scale=self.scaling,
                )
            else:
                # 标准手动 Attention（O(n²) 显存，用于 debug/baseline）
                # Scaled Dot-Product Attention
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

                # Causal mask
                if attention_mask is None:
                    # 自动生成 causal mask
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
                else:
                    attn_weights = attn_weights + attention_mask

                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v)

            # Reshape back: [batch, seq, num_heads * head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection (RowParallel → AllReduce)
        output = self.o_proj(attn_output)
        return output


# ParallelTransformerLayer


class ParallelTransformerLayer(nn.Module):
    """完整的并行 Transformer 层

    结构: RMSNorm → Attention → residual → RMSNorm → MLP → residual

    通信: 每层仅 2 次 AllReduce
    - 1 次在 Attention 的 o_proj (RowParallel)
    - 1 次在 MLP 的 down_proj (RowParallel)

    Overlap优化 (overlap_comm_compute=True):
    - 串行方式: Attn → AllReduce(等待) → MLP → AllReduce(等待)
    - Overlap方式: Attn → AllReduce(异步) + LayerNorm(并行) → 等待 → MLP → AllReduce(异步)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        parallel_context: ParallelContext,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        overlap_comm_compute: bool = False,
        stream_manager=None,
        use_flash_attn: bool = False,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.overlap_comm_compute = overlap_comm_compute
        self.stream_manager = stream_manager

        # 跨层 MLP AllReduce overlap 状态
        # 由上一层(通过model loop)设置，本层开头消费
        self._pending_mlp_handle = None      # AsyncHandle from previous layer's MLP AllReduce
        self._pending_mlp_residual = None    # residual tensor (before MLP add) from previous layer
        self._pending_mlp_output = None      # mlp_output tensor (being AllReduced) from previous layer
        self._pending_mlp_bias = None        # down_proj bias (if any) from previous layer

        # Pre-Attention LayerNorm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # Attention
        self.self_attn = ParallelAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            parallel_context=parallel_context,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            use_flash_attn=use_flash_attn,
            attention_backend=attention_backend,
        )
        # Post-Attention LayerNorm
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        # MLP
        self.mlp = ParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            parallel_context=parallel_context,
        )
        self.parallel_context = parallel_context

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, hidden_size]
            positions: [batch, seq] 位置索引
            attention_mask: 可选 causal mask

        Returns:
            output: [batch, seq, hidden_size]
        """
        if self.overlap_comm_compute and self.stream_manager is not None:
            return self._forward_with_overlap(x, positions, attention_mask)
        return self._forward_standard(x, positions, attention_mask)

    def _forward_standard(self, x, positions, attention_mask):
        """标准前向（无overlap，保持原始行为）"""
        # Pre-Norm + Attention + Residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, attention_mask)
        x = residual + x

        # Pre-Norm + MLP + Residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

    def _forward_with_overlap(self, x, positions, attention_mask):
        """Overlap前向：跨层MLP AllReduce通信与计算重叠

        核心思路:
        - 每层MLP的AllReduce发起后不立即wait，而是推迟到下一层开头
        - 下一层开头先完成上一层的residual add（需wait AllReduce完成），
          然后继续本层计算
        - Overlap窗口 = all_reduce_async()发起 到 下一层handle.wait()之间的时间
          （包括Python函数返回、循环迭代、函数调用的开销，以及GPU上AllReduce
           与CPU执行的并行）

        时序图（跨层）:
        Layer N:  |--Attn(sync)--|--MLP compute--|--async AR start--|-->return
        comm:                                     |=====AllReduce(mlp)=====|
        Layer N+1:|--wait+residual--|--Attn(sync)--|--MLP compute--|--async AR start--|

        数值正确性:
        - 与_forward_standard完全等价，仅改变同步时机
        - AllReduce结果在被读取前一定已完成（wait保证）
        """
        from .comm import all_reduce_async

        if self._pending_mlp_handle is not None:
            # 等待上一层MLP AllReduce完成
            self._pending_mlp_handle.wait()

            # 完成上一层的 residual + mlp_output（AllReduce已完成，结果在mlp_output中）
            x = self._pending_mlp_residual + self._pending_mlp_output
            if self._pending_mlp_bias is not None:
                x = x + self._pending_mlp_bias

            # 清理pending状态
            self._pending_mlp_handle = None
            self._pending_mlp_residual = None
            self._pending_mlp_output = None
            self._pending_mlp_bias = None

        residual = x
        x = self.input_layernorm(x)
        # self_attn内部o_proj调用RowParallelLinear.forward() → 同步AllReduce
        x = self.self_attn(x, positions, attention_mask)
        x = residual + x

        residual = x

        # Post-attention LayerNorm
        normed = self.post_attention_layernorm(x)

        # MLP gate + up projection（ColumnParallel，无通信）
        gate = F.silu(self.mlp.gate_proj(normed))
        up = self.mlp.up_proj(normed)
        mlp_intermediate = gate * up

        # down_proj 的 local matmul only（跳过RowParallelLinear.forward中的同步AllReduce）
        mlp_output = F.linear(mlp_intermediate, self.mlp.down_proj.weight)

        # 发起异步AllReduce（在comm_stream上执行，compute_stream可继续）
        handle = all_reduce_async(
            mlp_output, self.parallel_context.tp_group, self.stream_manager
        )

        # 暂存pending状态，供下一层（或model loop最终处理）消费
        # 不做residual add! 推迟到下一层开头或model forward结尾
        self._pending_mlp_handle = handle
        self._pending_mlp_residual = residual
        self._pending_mlp_output = mlp_output
        self._pending_mlp_bias = self.mlp.down_proj.bias  # 可能为None

        # 返回residual作为占位符
        # 注意：这不是本层的真正输出！真正输出 = residual + mlp_output (after AllReduce)
        # 由下一层的Step 0或model loop最终完成
        return residual


# ParallelTransformerModel (完整模型)


class ParallelTransformerModel(nn.Module):
    """完整的并行 Transformer 模型

    结构: Embedding → N x TransformerLayer → RMSNorm → LM Head

    设计说明:
    - Embedding 和 LM Head 不做 TP 切分（保持简单；大词表场景可选 VocabParallel）
    - 支持从 HuggingFace checkpoint 加载并自动切分权重
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        parallel_context: ParallelContext,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        tie_word_embeddings: bool = False,
        use_flash_attn: bool = False,
        use_gradient_checkpoint: bool = False,
        enable_comm_overlap: bool = False,
        stream_manager=None,
        attention_backend: str = "sdpa",
        pp_context=None,
    ):
        super().__init__()

        # Pipeline Parallel: stage 感知
        # pp_context 为 None 或 pp_size==1 时，构建完整的
        # embed→layers→norm→lm_head，行为与非 PP 版本完全一致（向后兼容）。
        # pp_size>1 时，仅构建本 stage 的组件：
        #   - embed_tokens 仅首 stage 持有；
        #   - transformer 层仅持有 stage_layer_range 指定的全局层号区间；
        #   - norm + lm_head 仅末 stage 持有；
        #   - 其余组件置 None。
        # 设计文档: docs/pipeline_parallel_design.md
        self.pp_context = pp_context
        self._pp_enabled = pp_context is not None and pp_context.pp_size > 1
        if self._pp_enabled:
            self.is_first_stage = pp_context.is_first_stage
            self.is_last_stage = pp_context.is_last_stage
            stage_start, stage_end = pp_context.stage_layer_range
        else:
            self.is_first_stage = True
            self.is_last_stage = True
            stage_start, stage_end = 0, num_layers
        self.stage_start = stage_start
        self.stage_end = stage_end
        self.num_local_layers = stage_end - stage_start

        self.parallel_context = parallel_context
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # 全局总层数（用于 MFU / stage 划分）
        self.use_gradient_checkpoint = use_gradient_checkpoint

        # PP MVP 限制：禁用 weight tying。
        # 原因：首 stage 持有 embed_tokens、末 stage 持有 lm_head，二者位于
        # 不同 rank，无法共享同一 tensor。末 stage 独立持有 lm_head 权重。
        if self._pp_enabled and tie_word_embeddings:
            logger.warning(
                "[PP] tie_word_embeddings=True is not supported under pipeline "
                "parallelism (embed_tokens on first stage, lm_head on last stage "
                "live on different ranks). Disabling weight tying for this MVP; "
                "the last stage holds an independent lm_head weight."
            )
            tie_word_embeddings = False
        self.tie_word_embeddings = tie_word_embeddings

        # Token Embedding（不切分）——仅首 stage 持有
        if self.is_first_stage:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        else:
            self.embed_tokens = None

        # Transformer Layers——仅构建本 stage 的层（数量 = num_local_layers）
        self.layers = nn.ModuleList([
            ParallelTransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                parallel_context=parallel_context,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                use_flash_attn=use_flash_attn,
                attention_backend=attention_backend,
                overlap_comm_compute=enable_comm_overlap,
                stream_manager=stream_manager,
            )
            for _ in range(self.num_local_layers)
        ])

        # Final LayerNorm + LM Head——仅末 stage 持有
        if self.is_last_stage:
            self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            # Weight tying（仅非 PP / 单 stage 时可能生效；PP 下已被禁用）
            if tie_word_embeddings and self.embed_tokens is not None:
                self.lm_head.weight = self.embed_tokens.weight
        else:
            self.norm = None
            self.lm_head = None

        # [TP-2修复] embedding 与 lm_head 未做 TP 切分、在各 rank 完整复制，标记以同步梯度。
        # 置于 weight tying 之后：即使权重共享，标记同一 tensor 也是幂等的。
        if self.embed_tokens is not None:
            self.embed_tokens.weight._tp_replicated = True
        if self.lm_head is not None:
            self.lm_head.weight._tp_replicated = True

    def forward(
        self,
        input,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stage 感知前向。

        Args:
            input:
                - 首 stage（或非 PP）: [batch, seq] token IDs
                - 中间/末 stage: [batch, seq, hidden_size] 上游激活 hidden_states
            positions: [batch, seq] 位置索引（None 时自动生成）
            attention_mask: 可选 causal mask

        Returns:
            - 末 stage（或非 PP）: logits [batch, seq, vocab_size]
            - 首/中间 stage: hidden_states [batch, seq, hidden_size]
        """
        if self.is_first_stage:
            # 首 stage：input 为 token_ids，经 embedding 得到 hidden
            input_ids = input
            batch_size, seq_len = input_ids.shape
            if positions is None:
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            hidden_states = self.embed_tokens(input_ids)
        else:
            # 中间/末 stage：input 已是上游 hidden_states
            hidden_states = input
            batch_size, seq_len, _ = hidden_states.shape
            if positions is None:
                positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        # Transformer Layers
        # 注意: gradient checkpoint与跨层overlap不兼容（checkpoint重计算无法传递pending state）
        # 当两者同时启用时，自动降级为标准模式
        _overlap_active = (not (self.use_gradient_checkpoint and self.training))

        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpoint and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, positions, attention_mask,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, positions, attention_mask)

                # 跨层overlap: 将当前层的pending MLP状态传递给下一层
                if (_overlap_active
                        and layer.overlap_comm_compute and layer.stream_manager is not None
                        and layer._pending_mlp_handle is not None):
                    if i < len(self.layers) - 1:
                        # 传递给下一层
                        next_layer = self.layers[i + 1]
                        next_layer._pending_mlp_handle = layer._pending_mlp_handle
                        next_layer._pending_mlp_residual = layer._pending_mlp_residual
                        next_layer._pending_mlp_output = layer._pending_mlp_output
                        next_layer._pending_mlp_bias = layer._pending_mlp_bias
                        # 清理当前层状态
                        layer._pending_mlp_handle = None
                        layer._pending_mlp_residual = None
                        layer._pending_mlp_output = None
                        layer._pending_mlp_bias = None

        # 处理最后一层的pending MLP AllReduce（没有下一层来消费）
        # 守卫: PP 下极端切分可能出现某 stage 无层（num_local_layers==0）
        if len(self.layers) > 0:
            last_layer = self.layers[-1]
            if last_layer._pending_mlp_handle is not None:
                last_layer._pending_mlp_handle.wait()
                hidden_states = last_layer._pending_mlp_residual + last_layer._pending_mlp_output
                if last_layer._pending_mlp_bias is not None:
                    hidden_states = hidden_states + last_layer._pending_mlp_bias
                # 清理状态
                last_layer._pending_mlp_handle = None
                last_layer._pending_mlp_residual = None
                last_layer._pending_mlp_output = None
                last_layer._pending_mlp_bias = None

        # 末 stage（或非 PP）：norm + lm_head → logits；否则返回 hidden 交给下游 stage
        if self.is_last_stage:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
        return hidden_states


# HuggingFace 权重加载工具


def load_from_hf_checkpoint(
    model: ParallelTransformerModel,
    checkpoint_path: str,
    parallel_context: ParallelContext,
) -> None:
    """从 HuggingFace checkpoint 加载权重并按 TP 切分。

    切分规则:
    - ColumnParallel (q_proj, k_proj, v_proj, gate_proj, up_proj): 按 dim=0 切分
    - RowParallel (o_proj, down_proj): 按 dim=1 切分
    - 其他 (embed, norm, lm_head): 不切分，直接加载

    Pipeline Parallel（stage 感知）:
    - 仅遍历 model.named_parameters()，因此非本 stage 的 embed/norm/lm_head
      （已置 None）会自动跳过，无需额外判断。
    - 本 stage 的 transformer 层在 named_parameters() 中是本地下标 0..N-1，
      需通过 layer_offset = model.stage_start 映射回全局层号，再拼 HF key。

    Args:
        model: 已初始化的 ParallelTransformerModel
        checkpoint_path: HF 格式 checkpoint 目录路径
        parallel_context: 并行上下文
    """
    import glob
    import os
    from safetensors.torch import load_file

    tp_rank = parallel_context.tp_rank
    tp_size = parallel_context.tp_size

    # 加载所有 safetensors 分片
    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
    if not safetensors_files:
        # 尝试 pytorch bin 格式
        state_dict = torch.load(
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
    else:
        state_dict = {}
        for f in safetensors_files:
            state_dict.update(load_file(f, device="cpu"))

    # ColumnParallel 权重名称模式（按 dim=0 切分）
    column_parallel_keys = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
    # RowParallel 权重名称模式（按 dim=1 切分）
    row_parallel_keys = {"o_proj", "down_proj"}

    # PP stage 感知：本地层下标 → 全局层号偏移
    layer_offset = getattr(model, "stage_start", 0)

    for name, param in model.named_parameters():
        # 从 HF key 映射到我们的 key（简化版：假设命名基本对齐）
        # PP 下将本地层下标映射回全局层号
        hf_key = _map_to_hf_key(name, layer_offset=layer_offset)
        if hf_key not in state_dict:
            print(f"[Warning] Key {hf_key} not found in checkpoint, skipping.")
            continue

        loaded_weight = state_dict[hf_key]

        # 判断是否需要切分
        layer_type = _get_layer_type(name)
        if layer_type in column_parallel_keys:
            # 按 dim=0 切分（output_dim）
            shard_size = loaded_weight.shape[0] // tp_size
            loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]
        elif layer_type in row_parallel_keys:
            # 按 dim=1 切分（input_dim），bias 不切分
            if loaded_weight.ndim == 2:
                shard_size = loaded_weight.shape[1] // tp_size
                loaded_weight = loaded_weight[:, tp_rank * shard_size : (tp_rank + 1) * shard_size]
        # else: 不切分

        # 加载到模型
        param.data.copy_(loaded_weight)

    print(f"[Rank {tp_rank}] Checkpoint loaded from {checkpoint_path}")


def _map_to_hf_key(model_key: str, layer_offset: int = 0) -> str:
    """将模型参数名映射到 HF checkpoint 的 key。

    示例映射（layer_offset=0，非 PP）:
        layers.0.self_attn.q_proj.weight → model.layers.0.self_attn.q_proj.weight
        embed_tokens.weight → model.embed_tokens.weight
        norm.weight → model.norm.weight
        lm_head.weight → lm_head.weight

    PP stage 感知（layer_offset>0）:
        本地层下标 local_idx → 全局层号 local_idx + layer_offset。
        例如末 stage stage_start=1 时:
            layers.0.self_attn.q_proj.weight → model.layers.1.self_attn.q_proj.weight
    """
    if model_key.startswith("lm_head"):
        return model_key
    # transformer 层：本地下标 → 全局层号
    if layer_offset and model_key.startswith("layers."):
        parts = model_key.split(".", 2)  # ["layers", "<local_idx>", "<rest>"]
        if len(parts) == 3 and parts[1].isdigit():
            global_idx = int(parts[1]) + layer_offset
            return f"model.layers.{global_idx}.{parts[2]}"
    return "model." + model_key


def _get_layer_type(param_name: str) -> str:
    """从参数名中提取层类型（用于判断切分策略）"""
    parts = param_name.split(".")
    for part in parts:
        if part in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
            return part
    return ""
