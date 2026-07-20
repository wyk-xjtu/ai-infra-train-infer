#!/usr/bin/env python
"""独立推理入口 — 不依赖训练/RL 组件

Usage:
    # 单卡 eager 模式
    python scripts/run_inference.py --model ./models/Qwen3-0.6B --mode eager --prompt "1+1="

    # 单卡 kv_cache 模式
    python scripts/run_inference.py --model ./models/Qwen3-8B --mode kv_cache --max-tokens 256

    # 从文件批量推理
    python scripts/run_inference.py --model ./models/Qwen3-0.6B --input prompts.jsonl --output results.jsonl

    # 交互模式
    python scripts/run_inference.py --model ./models/Qwen3-0.6B --interactive
"""
import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from src.inference.engine import InferenceEngine, InferenceConfig


def load_model_config(model_path: str) -> dict:
    """从模型目录加载 config.json"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="独立推理引擎")
    parser.add_argument("--model", "-m", required=True, help="模型路径")
    parser.add_argument("--mode", default="eager", choices=["eager", "kv_cache", "mock"], help="推理模式")
    parser.add_argument("--prompt", "-p", help="直接传入 prompt")
    parser.add_argument("--input", "-i", help="批量输入文件 (JSONL)")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p 采样")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--num-samples", type=int, default=1, help="每个 prompt 生成样本数")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--config", help="YAML 配置文件路径")
    # KV Cache 参数
    parser.add_argument("--num-blocks", type=int, default=512, help="KV Cache block 数")
    parser.add_argument("--block-size", type=int, default=16, help="每 block token 数")
    parser.add_argument("--enable-prefix-cache", action="store_true", help="启用 prefix caching")
    parser.add_argument("--enable-cuda-graph", action="store_true", help="启用 CUDA Graph 加速 decode")
    parser.add_argument("--enable-async", action="store_true", help="启用异步 prefill-decode 调度")
    parser.add_argument("--enable-partitioned-attn", action="store_true", help="启用分区 PagedAttention v2")
    return parser.parse_args()


def create_inference_config(args, model_config: dict) -> InferenceConfig:
    """根据命令行参数和模型配置创建 InferenceConfig"""
    return InferenceConfig(
        model_path=args.model,
        inference_mode=args.mode,
        num_layers=model_config.get("num_hidden_layers", 32),
        num_heads=model_config.get("num_attention_heads", 32),
        num_kv_heads=model_config.get("num_key_value_heads", model_config.get("num_attention_heads", 32)),
        head_dim=model_config.get("hidden_size", 4096) // model_config.get("num_attention_heads", 32),
        hidden_size=model_config.get("hidden_size", 4096),
        vocab_size=model_config.get("vocab_size", 151936),
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_batched_tokens=max(args.max_tokens * args.batch_size, 2048),
        max_num_sequences=max(args.batch_size * args.num_samples, 256),
        enable_prefix_caching=args.enable_prefix_cache,
        enable_cuda_graph=getattr(args, 'enable_cuda_graph', False),
        enable_async_scheduling=getattr(args, 'enable_async', False),
        enable_partitioned_attention=getattr(args, 'enable_partitioned_attn', False),
        temperature=args.temperature,
        top_p=args.top_p,
    )


def load_prompts_from_file(filepath: str) -> list:
    """从 JSONL 文件加载 prompts"""
    prompts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, str):
                prompts.append(data)
            elif isinstance(data, dict):
                prompts.append(data.get("prompt", data.get("text", "")))
    return prompts


def main():
    args = parse_args()

    print(f"{'=' * 60}")
    print(f"推理引擎独立模式")
    print(f"{'=' * 60}")
    print(f"  模型: {args.model}")
    print(f"  推理模式: {args.mode}")
    print(f"  最大 token: {args.max_tokens}")
    print(f"  温度: {args.temperature}")
    print(f"{'=' * 60}")

    # 1. 加载模型配置
    model_config = load_model_config(args.model)
    print(f"模型配置: {model_config.get('model_type', 'unknown')}, "
          f"{model_config.get('num_hidden_layers', '?')} layers, "
          f"{model_config.get('hidden_size', '?')} hidden")

    # 2. 创建推理配置
    infer_config = create_inference_config(args, model_config)

    # 3. 初始化推理引擎
    print("初始化推理引擎...")
    engine = InferenceEngine(infer_config)
    engine.initialize()
    print("✅ 推理引擎初始化成功")

    # 4. 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"✅ Tokenizer 加载成功 (vocab_size={tokenizer.vocab_size})")

    # 5. 处理输入
    if args.prompt:
        prompts = [args.prompt]
    elif args.input:
        prompts = load_prompts_from_file(args.input)
        print(f"从 {args.input} 加载 {len(prompts)} 条 prompt")
    elif args.interactive:
        _run_interactive(engine, tokenizer, args)
        return
    else:
        print("请指定 --prompt、--input 或 --interactive")
        return

    # 6. 批量推理
    results = []
    for idx, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        print(f"\n[{idx+1}/{len(prompts)}] Prompt ({len(tokens)} tokens): {prompt[:80]}...")

        start_time = time.time()
        outputs = engine.generate(
            [tokens] * args.num_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        elapsed = time.time() - start_time

        for k, output_tokens in enumerate(outputs or []):
            text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            total_tokens = len(output_tokens)
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

            print(f"  Sample {k+1}: {total_tokens} tokens, {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
            print(f"  Output: {text[:200]}")

            results.append({
                "prompt": prompt,
                "sample_idx": k,
                "output": text,
                "num_tokens": total_tokens,
                "time_s": round(elapsed, 3),
                "tokens_per_s": round(tokens_per_sec, 1),
            })

    # 7. 保存输出
    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n✅ 结果保存到 {args.output} ({len(results)} 条)")

    print(f"\n{'='*60}")
    print(f"推理完成: {len(prompts)} prompts, {len(results)} samples")
    print(f"{'='*60}")


def _run_interactive(engine, tokenizer, args):
    """交互式推理模式"""
    print("\n进入交互模式 (输入 'quit' 退出)")
    print("-" * 40)
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        tokens = tokenizer.encode(prompt)
        start = time.time()
        outputs = engine.generate([tokens], max_tokens=args.max_tokens, temperature=args.temperature)
        elapsed = time.time() - start

        if outputs and outputs[0]:
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nResponse ({len(outputs[0])} tokens, {elapsed:.2f}s):")
            print(text)
        else:
            print("(无输出)")


if __name__ == "__main__":
    main()
