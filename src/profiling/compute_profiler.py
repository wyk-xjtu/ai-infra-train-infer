"""
计算性能Profiler

测量：
- 各层前向/反向计算时间
- MFU (Model FLOPs Utilization) 计算
- GPU计算利用率
- 计算-通信比分析

技术要点：
- MFU怎么算？→ actual_flops / theoretical_peak_flops
- Transformer的FLOPs公式？→ 约 6 * num_params * seq_len * batch_size（前向+反向）
  这是因为：每个参数在前向做1次乘加(2 FLOPs)，反向做2次乘加(4 FLOPs)
- 典型MFU是多少？→ 训练时30-50%算正常，推理时更高（60-70%）
- 为什么MFU到不了100%？→ 显存带宽瓶颈、通信开销、kernel launch开销、pipeline bubble
"""

import torch
import torch.cuda as cuda
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class LayerProfile:
    """单层Profile

    技术要点：
    - 为什么要分开记录forward和backward？
      → 反向通常是前向的2倍（需要计算对input和weight的梯度）
      → 知道各层耗时可以定位热点层
    """
    layer_name: str
    forward_ms: float
    backward_ms: float = 0.0
    flops: int = 0
    params: int = 0

    @property
    def total_ms(self) -> float:
        return self.forward_ms + self.backward_ms

    @property
    def tflops(self) -> float:
        """TFLOPs/s (基于前向时间)"""
        if self.forward_ms <= 0 or self.flops <= 0:
            return 0.0
        return (self.flops / 1e12) / (self.forward_ms / 1000)


class ComputeProfiler:
    """计算性能Profiler

    技术要点：
    - 怎么判断一个训练job是计算密集还是显存带宽密集？
      → 计算密集(compute-bound): MFU < 50% 但 显存带宽利用率低
      → 显存密集(memory-bound): batch小/seq短时，GEMM规模不够大
      → 用 arithmetic intensity = FLOPs / memory_bytes 判断
    - Transformer哪些操作是compute-bound？
      → 大矩阵的Linear层（hidden * intermediate的GEMM）
    - 哪些是memory-bound？
      → LayerNorm, Softmax, 小batch的Attention

    用法:
        profiler = ComputeProfiler()
        flops_info = profiler.estimate_flops(batch_size=4, seq_len=2048,
                                             hidden_size=4096, num_layers=32,
                                             intermediate_size=11008, vocab_size=151936)
        mfu = profiler.compute_mfu(actual_time_s=1.5, flops=flops_info['total_flops'])
    """

    def __init__(self, model_config: Optional[dict] = None):
        """
        Args:
            model_config: 模型配置字典（可选，用于自动估算FLOPs）
        """
        self.model_config = model_config or {}
        self.layer_profiles: List[LayerProfile] = []
        self._active_profiles: Dict[str, float] = {}

    def estimate_flops(self, batch_size: int, seq_len: int,
                       hidden_size: int, num_layers: int,
                       intermediate_size: int, vocab_size: int,
                       num_heads: int = 32, head_dim: int = 128,
                       is_training: bool = True) -> dict:
        """估算Transformer的FLOPs

        技术要点（必背公式）：

        Attention FLOPs per layer:
          QKV projection: 3 * 2 * B * S * H^2  (三个linear层)
          Attention score: 2 * B * num_heads * S^2 * head_dim
          Attention @ V: 2 * B * num_heads * S^2 * head_dim
          Output projection: 2 * B * S * H^2

        MLP FLOPs per layer (SwiGLU):
          gate_proj: 2 * B * S * H * I
          up_proj: 2 * B * S * H * I
          down_proj: 2 * B * S * I * H

        近似公式: Total ≈ 6 * N * B * S
          其中 N = 模型参数数（不含embedding）

        为什么是6倍？
          → 前向: 每个参数 2 FLOPs (1次乘法 + 1次加法)
          → 反向对activation: 2 FLOPs
          → 反向对weight: 2 FLOPs
          → 总计 = 2 + 2 + 2 = 6 (训练时)
          → 推理时只有前向 = 2

        Args:
            batch_size: batch大小
            seq_len: 序列长度
            hidden_size: 隐藏维度 (H)
            num_layers: 层数 (L)
            intermediate_size: MLP中间维度 (I)
            vocab_size: 词表大小 (V)
            num_heads: 注意力头数
            head_dim: 每个头的维度
            is_training: 是否训练（训练=前向+反向，推理=仅前向）

        Returns:
            包含各部分FLOPs和总FLOPs的字典
        """
        B, S, H, I, L, V = batch_size, seq_len, hidden_size, intermediate_size, num_layers, vocab_size

        # QKV projections: 3 个 [B*S, H] x [H, H] 矩阵乘
        qkv_flops = 3 * 2 * B * S * H * H
        # Attention scores: [B, heads, S, head_dim] x [B, heads, head_dim, S]
        attn_score_flops = 2 * B * num_heads * S * S * head_dim
        # Attention @ V: [B, heads, S, S] x [B, heads, S, head_dim]
        attn_v_flops = 2 * B * num_heads * S * S * head_dim
        # Output projection: [B*S, H] x [H, H]
        out_proj_flops = 2 * B * S * H * H

        attention_flops_per_layer = qkv_flops + attn_score_flops + attn_v_flops + out_proj_flops

        # gate_proj: [B*S, H] x [H, I]
        gate_flops = 2 * B * S * H * I
        # up_proj: [B*S, H] x [H, I]
        up_flops = 2 * B * S * H * I
        # down_proj: [B*S, I] x [I, H]
        down_flops = 2 * B * S * I * H

        mlp_flops_per_layer = gate_flops + up_flops + down_flops

        per_layer_flops = attention_flops_per_layer + mlp_flops_per_layer
        all_layers_flops = per_layer_flops * L

        # Embedding + LM Head
        embed_flops = 2 * B * S * H  # lookup (近似)
        lm_head_flops = 2 * B * S * H * V  # [B*S, H] x [H, V]

        total_forward_flops = all_layers_flops + embed_flops + lm_head_flops

        # 训练时反向是前向的2倍
        multiplier = 3 if is_training else 1  # 前向1 + 反向2 = 3
        total_flops = total_forward_flops * multiplier

        # 近似公式验证
        # 模型参数数（不含embedding） ≈ L * (4H^2 + 3H*I)
        model_params_approx = L * (4 * H * H + 3 * H * I) + V * H
        approx_flops = (6 if is_training else 2) * model_params_approx * B * S

        return {
            "attention_flops_per_layer": attention_flops_per_layer,
            "mlp_flops_per_layer": mlp_flops_per_layer,
            "per_layer_flops": per_layer_flops,
            "all_layers_forward_flops": all_layers_flops,
            "lm_head_flops": lm_head_flops,
            "total_forward_flops": total_forward_flops,
            "total_flops": total_flops,
            "approximate_formula_flops": approx_flops,
            "model_params_approx": model_params_approx,
            "is_training": is_training,
        }

    def compute_mfu(self, actual_time_s: float, flops: int,
                    device: str = "A100", num_devices: int = 1) -> float:
        """计算MFU (Model FLOPs Utilization)

        MFU 定义: MFU = actual_model_flops / theoretical_peak_flops
        MFU 是单卡性能指标，与设备数量无关。单卡和多卡应报告相同的 MFU 值。

        Megatron-LM 对标说明:
        - 调用方应传入 per-GPU training FLOPs:
          * Full SFT: forward_flops × 3 / (tp × pp)
          * LoRA: forward_flops × 2 / (tp × pp)（frozen W skips dW computation）
        - H20 BF16 peak = 148 TFLOPS
        - TP/PP 归一化由调用方完成，本函数不再除以任何并行度

        技术要点：
        - MFU = 单卡实际 TFLOPS / 单卡峰值 TFLOPS
        - 这是衡量训练效率的金标准指标
        - 典型值：
          - Megatron-LM on A100: 40-55%
          - DeepSpeed on A100: 35-45%
          - 单卡推理: 50-70%
        - MFU低的原因：通信开销、显存带宽瓶颈、bubble、kernel效率

        Args:
            actual_time_s: 实际计算耗时（秒）
            flops: per-GPU training FLOPs（已由调用方完成 ×multiplier / (tp×pp) 归一化）
            device: 设备型号
            num_devices: 设备数量（保留参数兼容性，不参与 MFU 计算）

        Returns:
            MFU (0~1之间)
        """
        peak_tflops = self.get_peak_tflops(device)
        if peak_tflops <= 0 or actual_time_s <= 0:
            return 0.0

        # MFU = 单卡实际FLOPs / (单卡峰值TFLOPS * 时间)
        # flops 已是单卡值（使用 local batch_size 计算），无需除以 num_devices
        peak_flops = peak_tflops * 1e12 * actual_time_s
        mfu = flops / peak_flops
        return min(mfu, 1.0)  # cap at 1.0

    def get_peak_tflops(self, device: str = "A100") -> float:
        """获取设备峰值算力 (TFLOPS, FP16/BF16)

        技术要点：
        - A100的312 TFLOPS怎么来的？→ 108 SM * 256 FP16 ops/clock * 1.41 GHz * 2(FMA)
        - Tensor Core vs CUDA Core？→ Tensor Core做矩阵乘快约8-16倍
        - 为什么实际用不满？→ 显存带宽限制、kernel occupancy、指令流水线

        Args:
            device: 设备型号

        Returns:
            峰值TFLOPS (FP16/BF16 Tensor Core)
        """
        peak_map = {
            "A100": 312.0,      # A100 SXM FP16 Tensor Core
            "A100_PCIE": 312.0,
            "H100": 989.0,      # H100 SXM FP16 Tensor Core
            "H100_SXM": 989.0,
            "H100_PCIE": 756.0,
            "H20": 148.0,       # H20 BF16 Tensor Core
            "4090": 165.0,      # RTX 4090 FP16 Tensor Core
            "3090": 71.0,       # RTX 3090 FP16 Tensor Core
            "4060": 15.0,       # RTX 4060 FP16
            "4080": 97.0,       # RTX 4080 FP16
            "L40": 181.0,       # L40 FP16
        }
        return peak_map.get(device.upper(), 312.0)

    @contextmanager
    def profile_layer(self, layer_name: str, flops: int = 0, params: int = 0):
        """Profile单层的前向计算时间

        使用CUDA Event精确计时。

        Args:
            layer_name: 层名称
            flops: 该层的FLOPs（可选，用于计算单层效率）
            params: 该层的参数量
        """
        if torch.cuda.is_available():
            start_event = cuda.Event(enable_timing=True)
            end_event = cuda.Event(enable_timing=True)
            start_event.record()
            yield
            end_event.record()
            cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            start_t = time.perf_counter()
            yield
            elapsed_ms = (time.perf_counter() - start_t) * 1000

        profile = LayerProfile(
            layer_name=layer_name,
            forward_ms=elapsed_ms,
            flops=flops,
            params=params,
        )
        self.layer_profiles.append(profile)

    def profile_model(self, model, input_ids: torch.Tensor,
                      num_warmup: int = 3, num_runs: int = 10) -> dict:
        """Profile模型各层的前向计算时间

        技术要点：
        - 为什么需要warmup？
          → 首次运行有JIT编译、CUDA context初始化、缓存预热等开销
          → warmup后的时间才是稳态性能
        - 为什么多次运行取平均？
          → GPU调度有波动，多次取平均更准确
          → 排除操作系统调度干扰

        Args:
            model: nn.Module模型
            input_ids: 输入tensor [batch, seq]
            num_warmup: 预热次数
            num_runs: 正式运行次数

        Returns:
            各层和总体的profile结果
        """
        model.eval()

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_ids)
                if torch.cuda.is_available():
                    cuda.synchronize()

        # 正式Profile
        layer_times: Dict[str, List[float]] = {}

        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    start = cuda.Event(enable_timing=True)
                    end = cuda.Event(enable_timing=True)
                    start.record()
                    _ = model(input_ids)
                    end.record()
                    cuda.synchronize()
                    total_ms = start.elapsed_time(end)
                else:
                    t0 = time.perf_counter()
                    _ = model(input_ids)
                    total_ms = (time.perf_counter() - t0) * 1000

                if "total" not in layer_times:
                    layer_times["total"] = []
                layer_times["total"].append(total_ms)

        avg_total_ms = sum(layer_times["total"]) / len(layer_times["total"])

        return {
            "avg_forward_ms": avg_total_ms,
            "num_runs": num_runs,
            "per_run_ms": layer_times["total"],
            "std_ms": (sum((t - avg_total_ms) ** 2 for t in layer_times["total"]) / len(layer_times["total"])) ** 0.5,
        }

    def report(self) -> str:
        """生成计算性能报告

        示例：
        ┌─────────────────────────────────────────┐
        │       计算性能 Profiling Report           │
        ├─────────────────────────────────────────┤
        │ Layer                 │ Time(ms) │ TFLOPS │
        │ transformer.layer_0   │    2.3   │  135.2 │
        │ transformer.layer_1   │    2.4   │  131.0 │
        │ ...                   │          │        │
        ├─────────────────────────────────────────┤
        │ Total Forward: 72.1ms                    │
        │ Estimated MFU: 43.2%                     │
        └─────────────────────────────────────────┘
        """
        if not self.layer_profiles:
            return "No layer profiles recorded."

        lines = []
        lines.append("┌" + "─" * 55 + "┐")
        lines.append("│" + "计算性能 Profiling Report".center(47) + "        │")
        lines.append("├" + "─" * 55 + "┤")
        lines.append(f"│  {'Layer':<30} {'Time(ms)':>9} {'TFLOPS':>8}  │")
        lines.append("│  " + "─" * 30 + " " + "─" * 9 + " " + "─" * 8 + "  │")

        total_ms = 0.0
        for lp in self.layer_profiles:
            total_ms += lp.forward_ms
            tflops_str = f"{lp.tflops:.1f}" if lp.tflops > 0 else "-"
            lines.append(f"│  {lp.layer_name:<30} {lp.forward_ms:>8.2f}  {tflops_str:>7}  │")

        lines.append("├" + "─" * 55 + "┤")
        lines.append(f"│  Total Forward: {total_ms:.1f}ms" + " " * 30 + "│")
        lines.append("└" + "─" * 55 + "┘")

        return "\n".join(lines)

    def reset(self):
        """重置所有记录"""
        self.layer_profiles.clear()
        self._active_profiles.clear()
