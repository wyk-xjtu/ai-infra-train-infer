#!/usr/bin/env python3
"""独立显存分析工具 — 与训练解耦

用法:
    python scripts/profile_memory.py --config configs/sft.yaml
    python scripts/profile_memory.py --config configs/sft.yaml --output report.json

功能:
    加载模型 + LoRA，执行一步 forward/backward profiling，
    输出完整的显存 breakdown 报告（无需运行完整训练流程）
"""
import argparse
import json
import sys
import os
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="GPU Memory Profiling Tool")
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    parser.add_argument("--output", type=str, default=None, help="Output report path (default: outputs/memory_report.json)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to profile on")
    parser.add_argument("--skip-backward", action="store_true", help="Skip backward pass profiling")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_path = config.get("model", {}).get("name_or_path", "./models/Qwen3-8B")
    max_seq_len = config.get("model", {}).get("max_seq_len", 2048)
    precision = config.get("runtime", {}).get("precision", "bf16")
    lora_config = config.get("lora", {})
    lora_rank = lora_config.get("rank", 32)
    lora_alpha = lora_config.get("alpha", 64)
    lora_targets = lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout = lora_config.get("dropout", 0.0)
    tp_size = config.get("runtime", {}).get("tp_size", 1)
    dp_size = config.get("runtime", {}).get("dp_size", 1)
    batch_size = config.get("training", {}).get("batch_size", 1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"GPU Memory Profiling Tool")
    print(f"{'='*60}")
    print(f"  Model: {model_path}")
    print(f"  Precision: {precision}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Seq length: {max_seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    print("\n[1/5] Loading model...")
    from src.distributed.parallel_context import ParallelContext
    from src.distributed.tensor_parallel import ParallelTransformerModel, load_from_hf_checkpoint

    # 创建单进程 ParallelContext（tp_group=None 表示单卡模式）
    parallel_ctx = ParallelContext(tp_size=1, tp_group=None, dp_size=1, dp_group=None)

    # 从 config.json 读取模型配置
    import json as json_module
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        model_config = json_module.load(f)

    model = ParallelTransformerModel(
        vocab_size=model_config.get("vocab_size", 151936),
        hidden_size=model_config.get("hidden_size", 4096),
        num_layers=model_config.get("num_hidden_layers", 36),
        num_heads=model_config.get("num_attention_heads", 32),
        num_kv_heads=model_config.get("num_key_value_heads", 8),
        head_dim=model_config.get("head_dim", 128),
        intermediate_size=model_config.get("intermediate_size", 12288),
        parallel_context=parallel_ctx,
        max_position_embeddings=model_config.get("max_position_embeddings", 40960),
        rms_norm_eps=model_config.get("rms_norm_eps", 1e-6),
        rope_theta=model_config.get("rope_theta", 1000000.0),
        tie_word_embeddings=model_config.get("tie_word_embeddings", False),
        use_flash_attn=False,
    )

    # 加载权重（在 CPU 上完成）
    load_from_hf_checkpoint(model, model_path, parallel_ctx)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    print("[2/5] Applying LoRA...")
    from src.distributed.lora import apply_lora

    lora_cfg = {
        "rank": lora_rank,
        "alpha": lora_alpha,
        "target_modules": lora_targets,
        "dropout": lora_dropout,
    }
    model = apply_lora(model, lora_cfg, parallel_ctx)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA applied: trainable={trainable/1e6:.1f}M / total={total/1e9:.2f}B ({trainable/total*100:.2f}%)")

    if precision != "fp32":
        print(f"[3/5] Casting frozen params to {precision}...")
        cast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        count = 0
        for p in model.parameters():
            if not p.requires_grad:
                p.data = p.data.to(cast_dtype)
                count += 1
        print(f"  Cast {count} frozen tensors to {precision}")
    else:
        print("[3/5] Precision: fp32 (no casting)")

    # 将完整模型移到 GPU
    model = model.to(device)

    print("[4/5] Creating optimizer...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)

    print("[5/5] Running memory profiling...")
    from src.utils.memory_profiler import MemoryReport

    report_config = {
        "model_name": model_path,
        "seq_len": max_seq_len,
        "batch_size": batch_size,
        "parallel_strategy": f"TP={tp_size}, DP={dp_size}",
        "lora_rank": lora_rank,
    }

    # 包装模型以处理混合精度（RoPE 中 cos/sin 为 fp32 会导致 dtype 提升）
    amp_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else None)

    class AmpModelWrapper(torch.nn.Module):
        """Autocast wrapper to handle mixed-precision dtype promotions."""
        def __init__(self, inner_model, amp_dtype):
            super().__init__()
            self.inner_model = inner_model
            self.amp_dtype = amp_dtype

        def forward(self, *args, **kwargs):
            if self.amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    return self.inner_model(*args, **kwargs)
            return self.inner_model(*args, **kwargs)

        def parameters(self, recurse=True):
            return self.inner_model.parameters(recurse=recurse)

        def named_modules(self, *args, **kwargs):
            return self.inner_model.named_modules(*args, **kwargs)

        def named_parameters(self, *args, **kwargs):
            return self.inner_model.named_parameters(*args, **kwargs)

        def train(self, mode=True):
            self.inner_model.train(mode)
            return self

    wrapped_model = AmpModelWrapper(model, amp_dtype)
    reporter = MemoryReport(wrapped_model, optimizer, report_config)

    dummy_input = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
    dummy_labels = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
    dummy_labels[:, :max_seq_len // 2] = -100  # 模拟 prompt mask

    if args.skip_backward:
        report = reporter.generate_report_without_step()
    else:
        report = reporter.profile_step(dummy_input, dummy_labels, train_step_fn=None)

    output_path = args.output or "outputs/memory_report.json"
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # 打印摘要
    print(f"\n{'='*60}")
    print("MEMORY PROFILING REPORT")
    print(f"{'='*60}")

    spec = report.get("model_spec", {})
    print(f"\n[Model Spec]")
    print(f"  Total params: {spec.get('total_params_readable', 'N/A')}")
    print(f"  Trainable: {spec.get('trainable_params_readable', 'N/A')}")
    print(f"  Precision: {spec.get('precision_distribution', {})}")

    static = report.get("static_memory", {})
    print(f"\n[Static Memory]")
    print(f"  Frozen weights (bf16): {static.get('frozen_weights_bf16_gb', 0):.2f} GB")
    print(f"  Frozen weights (fp32): {static.get('frozen_weights_fp32_gb', 0):.3f} GB")
    print(f"  Trainable weights: {static.get('trainable_weights_gb', 0):.3f} GB")
    print(f"  Gradients: {static.get('gradients_gb', 0):.3f} GB")
    print(f"  Optimizer states: {static.get('optimizer_states_gb', 0):.3f} GB")
    print(f"  Static total: {static.get('static_total_gb', 0):.2f} GB")

    peaks = report.get("peak_moments", {})
    if "forward_peak_gb" in peaks:
        print(f"\n[Peak Moments]")
        print(f"  Forward peak: {peaks.get('forward_peak_gb', 0):.2f} GB")
        print(f"  Backward peak: {peaks.get('backward_peak_gb', 0):.2f} GB")
        print(f"  Step peak: {peaks.get('step_peak_gb', 0):.2f} GB")

    overhead = report.get("framework_overhead", {})
    print(f"\n[Framework Overhead]")
    print(f"  PyTorch caching: {overhead.get('pytorch_caching_gb', 0):.2f} GB")
    print(f"  CUDA context: {overhead.get('cuda_context_estimated_gb', 0):.1f} GB")
    if "device_total_memory_gb" in overhead:
        print(f"  Device total memory: {overhead.get('device_total_memory_gb', 0):.1f} GB")

    print(f"\n{'='*60}")
    print(f"Report saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
