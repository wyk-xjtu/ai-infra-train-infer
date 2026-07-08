"""
LoRA (Low-Rank Adaptation) 实现 — 与张量并行兼容

核心思想:
  W' = W + (alpha/rank) * B @ A
  其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d, k)
  只训练 A 和 B，冻结原始权重 W

显存节省: 对于 d=4096, k=4096, rank=16:
  原始参数: 4096*4096 = 16M params
  LoRA参数: 4096*16 + 16*4096 = 128K params (0.8%)

TP 兼容设计:
- 对于 ColumnParallelLinear (权重按 output_dim 切分):
  B 按行切分 (output_dim): B_local ∈ R^{d/tp_size × r}
  A 保持完整: A ∈ R^{r × k}
  → 输出自然是切分的，与原始 ColumnParallel 一致

- 对于 RowParallelLinear (权重按 input_dim 切分):
  A 按列切分 (input_dim): A_local ∈ R^{r × k/tp_size}
  B 保持完整: B ∈ R^{d × r}
  → 各 rank 计算部分结果，需要 AllReduce 聚合

初始化策略:
- A: Kaiming uniform (保证输入方差稳定)
- B: 全零 (确保训练开始时 LoRA 输出为 0，不影响原模型行为)
"""

import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_context import ParallelContext
from .tensor_parallel import ColumnParallelLinear, RowParallelLinear
from .comm import reduce_from_parallel_region


class LoRALayer(nn.Module):
    """LoRA 基础层

    实现 ΔW = (alpha/rank) * B @ A 的低秩近似

    前向计算: lora_output = x @ A^T @ B^T * scaling
    其中 scaling = alpha / rank
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features: 输入维度 (对应原始权重的 k)
            out_features: 输出维度 (对应原始权重的 d)
            rank: LoRA 秩 (r)，越小参数越少但表达能力越弱
            alpha: 缩放因子，控制 LoRA 更新的幅度
            dropout: LoRA 输入 dropout 比例
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 实际缩放因子
        self.in_features = in_features
        self.out_features = out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """初始化: A 用 Kaiming，B 用全零（保证初始 LoRA 输出为 0）"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 LoRA 增量输出

        Args:
            x: [batch, seq, in_features]

        Returns:
            lora_output: [batch, seq, out_features]
            计算过程: x → dropout → x @ A^T → intermediate @ B^T → * scaling
        """
        x = self.lora_dropout(x)
        # 混合精度兼容: 输入可能是 bf16（来自冻结的 base model），
        # 但 LoRA 参数保持 fp32 以保证训练精度
        input_dtype = x.dtype
        lora_input = x
        if x.dtype != self.lora_A.dtype:
            lora_input = x.to(self.lora_A.dtype)
        intermediate = F.linear(lora_input, self.lora_A)
        output = F.linear(intermediate, self.lora_B)
        output = output * self.scaling
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output


# TP-Compatible LoRA Variants


class LoRAColumnParallel(LoRALayer):
    """与 ColumnParallelLinear 兼容的 LoRA

    设计: 
    - B 按行切分 (output_dim): B_local ∈ R^{out_features/tp_size × rank}
    - A 保持完整: A ∈ R^{rank × in_features}

    这样 LoRA 的输出 shape 与 ColumnParallel 一致:
    output_local = x @ A^T @ B_local^T * scaling
    → shape: [batch, seq, out_features/tp_size]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_context: ParallelContext,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        tp_size = parallel_context.tp_size
        tp_rank = parallel_context.tp_rank
        self.parallel_context = parallel_context

        assert out_features % tp_size == 0
        out_features_per_rank = out_features // tp_size

        super().__init__(in_features, out_features_per_rank, rank, alpha, dropout)

        self.lora_A._tp_replicated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [batch, seq, in_features] — 完整输入
        输出: [batch, seq, out_features/tp_size] — 切分输出（与 ColumnParallel 一致）
        """
        x = self.lora_dropout(x)
        # 混合精度兼容: cast 输入到 LoRA 权重 dtype
        input_dtype = x.dtype
        lora_input = x
        if x.dtype != self.lora_A.dtype:
            lora_input = x.to(self.lora_A.dtype)
        intermediate = F.linear(lora_input, self.lora_A)  # [batch, seq, rank]
        output = F.linear(intermediate, self.lora_B)  # [batch, seq, out_features/tp_size]
        output = output * self.scaling
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output


class LoRARowParallel(LoRALayer):
    """与 RowParallelLinear 兼容的 LoRA

    设计:
    - A 按列切分 (input_dim): A_local ∈ R^{rank × in_features/tp_size}
    - B 保持完整: B ∈ R^{out_features × rank}

    各 rank 计算部分中间结果，需要 AllReduce 聚合:
    intermediate_local = x_local @ A_local^T  → [batch, seq, rank]
    intermediate = AllReduce(intermediate_local)  → [batch, seq, rank]
    output = intermediate @ B^T * scaling  → [batch, seq, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_context: ParallelContext,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        tp_size = parallel_context.tp_size
        self.parallel_context = parallel_context

        assert in_features % tp_size == 0
        in_features_per_rank = in_features // tp_size

        super().__init__(in_features_per_rank, out_features, rank, alpha, dropout)

        self.lora_B._tp_replicated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [batch, seq, in_features/tp_size] — 切分输入（来自上游 ColumnParallel）
        输出: [batch, seq, out_features] — AllReduce 后的完整输出
        """
        x = self.lora_dropout(x)
        # 混合精度兼容: cast 输入到 LoRA 权重 dtype
        input_dtype = x.dtype
        lora_input = x
        if x.dtype != self.lora_A.dtype:
            lora_input = x.to(self.lora_A.dtype)
        intermediate = F.linear(lora_input, self.lora_A)  # [batch, seq, rank]
        # AllReduce 聚合中间结果（因为完整 intermediate = Σ(x_i @ A_i^T)）
        intermediate = reduce_from_parallel_region(intermediate, self.parallel_context.tp_group)
        output = F.linear(intermediate, self.lora_B)  # [batch, seq, out_features]
        output = output * self.scaling
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output


class LinearWithLoRA(nn.Module):
    """包装层: 原始 Linear + LoRA 旁路

    forward: output = original_linear(x) + lora(x)
    """

    def __init__(self, original_layer: nn.Module, lora_layer: LoRALayer):
        super().__init__()
        self.original_layer = original_layer
        self.lora = lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.lora(x)


def apply_lora(
    model: nn.Module,
    lora_config: Dict,
    parallel_context: ParallelContext,
) -> nn.Module:
    """给模型的指定层添加 LoRA

    操作:
    1. 冻结所有原始参数 (requires_grad=False)
    2. 根据 target_modules 配置，为匹配的层添加 LoRA 旁路
    3. 只有 LoRA 参数可训练

    Args:
        model: 原始模型
        lora_config: LoRA 配置字典，包含:
            - rank: LoRA 秩
            - alpha: 缩放因子
            - target_modules: 目标模块名列表 (如 ["q_proj", "v_proj"])
            - dropout: dropout 比例
        parallel_context: 并行上下文

    Returns:
        添加了 LoRA 的模型（原始权重已冻结）
    """
    rank = lora_config.get("rank", 16)
    alpha = lora_config.get("alpha", 32.0)
    target_modules: List[str] = lora_config.get("target_modules", [])
    dropout = lora_config.get("dropout", 0.0)

    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        # 检查模块名是否匹配 target_modules
        module_name = name.split(".")[-1]
        if module_name not in target_modules:
            continue

        if isinstance(module, ColumnParallelLinear):
            lora_layer = LoRAColumnParallel(
                in_features=module.in_features,
                out_features=module.out_features,
                parallel_context=parallel_context,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            # 替换模块为包装层
            _replace_module(model, name, LinearWithLoRA(module, lora_layer))

        elif isinstance(module, RowParallelLinear):
            lora_layer = LoRARowParallel(
                in_features=module.in_features,
                out_features=module.out_features,
                parallel_context=parallel_context,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            _replace_module(model, name, LinearWithLoRA(module, lora_layer))

        elif isinstance(module, nn.Linear):
            lora_layer = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            _replace_module(model, name, LinearWithLoRA(module, lora_layer))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA applied: trainable={trainable_params:,} / total={total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model


def _replace_module(model: nn.Module, target_name: str, new_module: nn.Module):
    """替换模型中的指定模块

    支持嵌套名称如 "layers.0.self_attn.q_proj"
    """
    parts = target_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def merge_lora_weights(model: nn.Module) -> dict:
    """合并 LoRA 权重到原始权重，导出用于推理

    合并公式: W_merged = W + (alpha/rank) * B @ A

    合并后模型与标准模型结构一致，无额外推理开销。

    Args:
        model: 包含 LoRA 的模型

    Returns:
        合并后的 state_dict（可直接用 model.load_state_dict() 加载到标准模型）
    """
    merged_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            original = module.original_layer
            lora = module.lora

            if hasattr(original, "weight"):
                weight = original.weight.data.clone()
                # 合并: W_merged = W + scaling * B @ A
                # B @ A: [out, in] — 与 weight 形状一致
                lora_weight = lora.lora_B @ lora.lora_A  # [out, in]
                weight += lora.scaling * lora_weight.to(weight.dtype)
                merged_state_dict[name + ".weight"] = weight

                if hasattr(original, "bias") and original.bias is not None:
                    merged_state_dict[name + ".bias"] = original.bias.data.clone()
        else:
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name
                if full_name not in merged_state_dict:
                    merged_state_dict[full_name] = param.data.clone()

    return merged_state_dict


def get_lora_state_dict(model: nn.Module) -> dict:
    """只导出 LoRA 参数（用于保存 adapter）

    Returns:
        仅包含 LoRA A/B 参数的 state_dict
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state
