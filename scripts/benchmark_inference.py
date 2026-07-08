"""推理引擎性能 Benchmark

测量指标：
1. Prefill 延迟 (ms) - 首 token 生成时间
2. Decode 平均延迟 (ms/token) - 后续每个 token 的生成时间
3. 吞吐量 (tokens/sec) - 端到端生成速率
4. 峰值显存 (GB) - 推理过程中最大 GPU 显存占用
5. 路径对比 - eager vs kv_cache vs mock

使用方式:
  python scripts/benchmark_inference.py --model_path models/Qwen3-0.6B --mode kv_cache
  python scripts/benchmark_inference.py --model_path models/Qwen3-0.6B --mode eager
  python scripts/benchmark_inference.py --model_path models/Qwen3-0.6B --mode all  # 对比所有模式
  python scripts/benchmark_inference.py --mode mock  # 无需模型，测试 scheduler 性能
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))

import torch

from inference.engine import InferenceEngine, InferenceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_inference")


FIXED_PROMPTS = [
    "What is 2+2? Answer with a single number.",
    "Calculate 15 multiplied by 7 and explain.",
    "Solve: if x + 3 = 10, what is x?",
    "What is the square root of 144?",
    "Explain the Pythagorean theorem briefly.",
    "What is 100 divided by 4?",
    "If a triangle has sides 3, 4, 5, is it right-angled?",
    "Calculate the area of a circle with radius 5.",
    "What is 2 to the power of 10?",
    "Sum the first 10 natural numbers.",
    "What is the factorial of 6?",
    "Convert 100 degrees Celsius to Fahrenheit.",
    "What is the greatest common divisor of 24 and 36?",
    "Simplify the fraction 48/64.",
    "What is the derivative of x^3?",
    "Calculate the integral of 2x dx.",
    "What are the prime factors of 84?",
    "Solve: 3x - 7 = 14.",
    "What is log base 2 of 256?",
    "Find the median of: 3, 7, 1, 9, 4.",
]


def generate_prompts_tokens(
    num_prompts: int, tokenizer=None, prompt_length: int = 32
) -> List[List[int]]:
    """生成测试用的 prompt token ids

    如果有 tokenizer，使用固定文本 prompt 编码；
    否则随机生成固定长度的 token ids（用于 mock 模式）。
    """
    if tokenizer is not None:
        prompts_tokens = []
        for i in range(num_prompts):
            text = FIXED_PROMPTS[i % len(FIXED_PROMPTS)]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            prompts_tokens.append(tokens)
        return prompts_tokens
    else:
        import random
        random.seed(42)
        return [
            [random.randint(100, 50000) for _ in range(prompt_length)]
            for _ in range(num_prompts)
        ]


class GPUTimer:
    """使用 torch.cuda.Event 精确计时，避免 CPU-GPU 同步开销干扰"""

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()

    def start(self):
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()

    def stop(self) -> float:
        """返回经过时间（毫秒）"""
        if self.use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        else:
            return (time.perf_counter() - self._cpu_start) * 1000.0


def benchmark_single_mode(
    mode: str,
    model_path: str,
    num_prompts: int,
    max_tokens: int,
    warmup: int,
    num_runs: int,
    batch_size: int,
    dtype: str = "bfloat16",
) -> Dict:
    """对单一推理模式执行 benchmark

    Returns:
        包含性能指标的字典
    """
    use_cuda = torch.cuda.is_available()
    timer = GPUTimer(use_cuda)

    logger.info("=" * 60)
    logger.info("Benchmark mode: %s", mode)
    logger.info("=" * 60)

    config_kwargs = {
        "inference_mode": mode,
        "model_path": model_path if mode != "mock" else "",
        "dtype": dtype,
        "enable_cuda_graph": False,  # benchmark 中禁用 CUDA Graph 以隔离变量
        "enable_prefix_caching": False,
    }

    if mode == "mock":
        config_kwargs["num_blocks"] = 2048
        config_kwargs["block_size"] = 16
    elif mode in ("eager", "kv_cache"):
        # 检查模型路径
        resolved_path = os.path.join(_project_root, model_path) if not os.path.isabs(model_path) else model_path
        if not os.path.exists(resolved_path):
            logger.error(
                "Model path does not exist: %s\n"
                "Please download the model first or use --mode mock for testing.",
                resolved_path,
            )
            return {"error": f"Model not found: {resolved_path}"}
        config_kwargs["model_path"] = resolved_path
        # kv_cache 模式需要更多 blocks
        if mode == "kv_cache":
            config_kwargs["num_blocks"] = 4096

    config = InferenceConfig(**config_kwargs)

    logger.info("Initializing InferenceEngine (mode=%s)...", mode)
    engine = InferenceEngine(config)
    engine.initialize()
    logger.info("Engine initialized successfully.")

    tokenizer = getattr(engine, "_tokenizer", None)

    prompts_tokens = generate_prompts_tokens(num_prompts, tokenizer)
    logger.info(
        "Generated %d prompts (avg length: %.1f tokens)",
        len(prompts_tokens),
        sum(len(p) for p in prompts_tokens) / len(prompts_tokens),
    )

    logger.info("Warmup: %d runs...", warmup)
    for w in range(warmup):
        # 按 batch_size 分批
        batch = prompts_tokens[:batch_size]
        _ = engine.generate(batch, max_tokens=min(max_tokens, 16), temperature=0.0)
        logger.info("  Warmup %d/%d done", w + 1, warmup)

    logger.info("Benchmarking: %d runs x %d prompts (batch_size=%d)...", num_runs, num_prompts, batch_size)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    all_prefill_ms = []
    all_decode_ms_per_token = []
    all_total_ms = []
    all_tokens_generated = []

    for run_idx in range(num_runs):
        run_total_tokens = 0
        run_total_ms = 0.0
        run_prefill_ms_list = []
        run_decode_ms_list = []

        for batch_start in range(0, num_prompts, batch_size):
            batch_end = min(batch_start + batch_size, num_prompts)
            batch = prompts_tokens[batch_start:batch_end]

            # 测量 prefill (第一个 token 的生成时间)
            # 由于 engine.generate 是端到端接口，我们分两步测量：
            timer.start()
            result_prefill = engine.generate(batch, max_tokens=1, temperature=0.0)
            prefill_ms = timer.stop()
            run_prefill_ms_list.append(prefill_ms)

            timer.start()
            results = engine.generate(batch, max_tokens=max_tokens, temperature=0.0)
            total_batch_ms = timer.stop()

            # 统计生成的 token 数
            if results is not None:
                batch_tokens = sum(len(r) for r in results)
            else:
                batch_tokens = 0

            run_total_tokens += batch_tokens
            run_total_ms += total_batch_ms

            # 注意：这里用近似方法，因为完整生成包含 prefill
            decode_tokens = max(batch_tokens - len(batch), 1)
            decode_ms = max(total_batch_ms - prefill_ms, 0.0)
            run_decode_ms_list.append(decode_ms / decode_tokens)

        # 汇总本轮
        avg_prefill_ms = sum(run_prefill_ms_list) / len(run_prefill_ms_list)
        avg_decode_ms = sum(run_decode_ms_list) / len(run_decode_ms_list)

        all_prefill_ms.append(avg_prefill_ms)
        all_decode_ms_per_token.append(avg_decode_ms)
        all_total_ms.append(run_total_ms)
        all_tokens_generated.append(run_total_tokens)

        logger.info(
            "  Run %d/%d: prefill=%.2fms, decode=%.2fms/tok, "
            "tokens=%d, total=%.1fms",
            run_idx + 1, num_runs, avg_prefill_ms, avg_decode_ms,
            run_total_tokens, run_total_ms,
        )

    avg_prefill = sum(all_prefill_ms) / len(all_prefill_ms)
    avg_decode = sum(all_decode_ms_per_token) / len(all_decode_ms_per_token)
    total_time_sec = sum(all_total_ms) / 1000.0
    total_tokens = sum(all_tokens_generated)
    throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0.0

    peak_memory_gb = 0.0
    if use_cuda:
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)

    result = {
        "model": os.path.basename(model_path) if model_path else "mock",
        "mode": mode,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "results": {
            "prefill_latency_ms": round(avg_prefill, 2),
            "decode_latency_ms_per_token": round(avg_decode, 2),
            "throughput_tokens_per_sec": round(throughput, 1),
            "peak_memory_gb": round(peak_memory_gb, 3),
            "total_time_sec": round(total_time_sec, 2),
            "total_tokens_generated": total_tokens,
        },
    }

    logger.info("-" * 60)
    logger.info("Results for mode=%s:", mode)
    logger.info("  Prefill latency:    %.2f ms", avg_prefill)
    logger.info("  Decode latency:     %.2f ms/token", avg_decode)
    logger.info("  Throughput:         %.1f tokens/sec", throughput)
    logger.info("  Peak memory:        %.3f GB", peak_memory_gb)
    logger.info("  Total time:         %.2f sec", total_time_sec)
    logger.info("  Total tokens:       %d", total_tokens)
    logger.info("-" * 60)

    return result


def benchmark_all_modes(
    model_path: str,
    num_prompts: int,
    max_tokens: int,
    warmup: int,
    num_runs: int,
    batch_size: int,
    dtype: str = "bfloat16",
) -> List[Dict]:
    """依次运行所有模式并输出对比表格"""
    modes = ["mock", "eager", "kv_cache"]
    results = []

    for mode in modes:
        # eager/kv_cache 需要模型，如果路径不存在则跳过
        if mode in ("eager", "kv_cache"):
            resolved_path = os.path.join(_project_root, model_path) if not os.path.isabs(model_path) else model_path
            if not os.path.exists(resolved_path):
                logger.warning(
                    "Skipping mode=%s: model not found at %s", mode, resolved_path
                )
                results.append({"mode": mode, "error": "model not found", "skipped": True})
                continue

        result = benchmark_single_mode(
            mode=mode,
            model_path=model_path,
            num_prompts=num_prompts,
            max_tokens=max_tokens,
            warmup=warmup,
            num_runs=num_runs,
            batch_size=batch_size,
            dtype=dtype,
        )
        results.append(result)

    print("\n")
    print("=" * 80)
    print("  INFERENCE BENCHMARK COMPARISON")
    print("=" * 80)
    header = f"{'Mode':<12} {'Prefill(ms)':<14} {'Decode(ms/tok)':<16} {'Throughput(tok/s)':<18} {'Memory(GB)':<12}"
    print(header)
    print("-" * 80)

    valid_results = [r for r in results if "results" in r]
    for r in results:
        if "error" in r:
            print(f"{r['mode']:<12} {'SKIPPED':<14} {r.get('error', '')}")
        else:
            res = r["results"]
            print(
                f"{r['mode']:<12} {res['prefill_latency_ms']:<14.2f} "
                f"{res['decode_latency_ms_per_token']:<16.2f} "
                f"{res['throughput_tokens_per_sec']:<18.1f} "
                f"{res['peak_memory_gb']:<12.3f}"
            )

    eager_result = next((r for r in valid_results if r["mode"] == "eager"), None)
    if eager_result:
        print("\n--- Speedup vs eager ---")
        eager_throughput = eager_result["results"]["throughput_tokens_per_sec"]
        for r in valid_results:
            if r["mode"] != "eager" and eager_throughput > 0:
                speedup = r["results"]["throughput_tokens_per_sec"] / eager_throughput
                print(f"  {r['mode']}: {speedup:.2f}x throughput")

    print("=" * 80)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="推理引擎性能 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_path", type=str, default="models/Qwen3-0.6B",
        help="模型路径（相对项目根目录或绝对路径），默认 models/Qwen3-0.6B",
    )
    parser.add_argument(
        "--mode", type=str, default="mock",
        choices=["eager", "kv_cache", "mock", "all"],
        help="推理模式: eager / kv_cache / mock / all（对比所有模式）",
    )
    parser.add_argument(
        "--num_prompts", type=int, default=10,
        help="测试 prompt 数量（默认 10）",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128,
        help="最大生成 token 数（默认 128）",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup 轮数（默认 2）",
    )
    parser.add_argument(
        "--num_runs", type=int, default=5,
        help="测试轮数（默认 5）",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size（默认 1）",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="结果输出 JSON 路径（可选）",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="模型推理精度（默认 bfloat16）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Inference Benchmark Starting")
    logger.info("  Model path: %s", args.model_path)
    logger.info("  Mode: %s", args.mode)
    logger.info("  Num prompts: %d", args.num_prompts)
    logger.info("  Max tokens: %d", args.max_tokens)
    logger.info("  Warmup: %d", args.warmup)
    logger.info("  Num runs: %d", args.num_runs)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info("  GPU Memory: %.1f GB total, %.1f GB free",
                    total_mem / 1024**3, free_mem / 1024**3)

    if args.mode == "all":
        results = benchmark_all_modes(
            model_path=args.model_path,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            num_runs=args.num_runs,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
    else:
        result = benchmark_single_mode(
            mode=args.mode,
            model_path=args.model_path,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            num_runs=args.num_runs,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        results = [result]

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_data = results[0] if len(results) == 1 else {"comparison": results}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to: %s", args.output_json)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
