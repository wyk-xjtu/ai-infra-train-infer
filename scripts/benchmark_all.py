"""
全量Benchmark脚本

对比两种架构的性能指标:
- RLHF iterations/hour
- 各阶段耗时breakdown
- 峰值显存
- 权重同步延迟

用法:
    # 在4xA100上运行完整benchmark
    python scripts/benchmark_all.py --model Qwen/Qwen3-4B --iterations 20

    # 本地快速测试
    python scripts/benchmark_all.py --model Qwen/Qwen3-0.6B --iterations 5 --quick
"""
import argparse
import asyncio
import json
import time
from datetime import datetime
import sys
sys.path.insert(0, '.')

from src.orchestrator.colocate_orchestrator import ColocateOrchestrator, ColocateConfig
from src.orchestrator.disagg_orchestrator import DisaggOrchestrator, DisaggConfig
from src.utils.logger import setup_logger
from src.utils.memory_profiler import MemoryProfiler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Colocate vs Disaggregated architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model path")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations per benchmark")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="GRPO K value")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip disaggregated benchmark")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000",
                        help="vLLM server URL (for colocate mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: benchmark_TIMESTAMP.json)")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="TP size for colocate mode")
    parser.add_argument("--train-tp", type=int, default=2,
                        help="Training TP for disaggregated mode")
    parser.add_argument("--infer-tp", type=int, default=2,
                        help="Inference TP for disaggregated mode")
    return parser.parse_args()


async def run_colocate_benchmark(args, logger) -> dict:
    """运行Colocate模式benchmark"""
    logger.info("=" * 60)
    logger.info("  BENCHMARK: Colocate Mode")
    logger.info("=" * 60)

    config = ColocateConfig(
        model_path=args.model,
        tp_size=args.tp_size,
        total_iterations=args.iterations,
        batch_size=args.batch_size,
        vllm_url=args.vllm_url,
        num_samples_per_prompt=args.num_samples,
    )

    orchestrator = ColocateOrchestrator(config)
    start_time = time.time()

    try:
        await orchestrator.initialize()
        await orchestrator.run()
        summary = orchestrator.get_summary()
    except Exception as e:
        logger.error(f"Colocate benchmark failed: {e}")
        summary = {"status": "failed", "error": str(e)}
    finally:
        await orchestrator.shutdown()

    elapsed = time.time() - start_time
    summary["benchmark_wall_time_s"] = elapsed
    summary["mode"] = "colocate"

    logger.info(f"Colocate benchmark completed in {elapsed:.1f}s")
    return summary


async def run_disagg_benchmark(args, logger) -> dict:
    """运行Disaggregated模式benchmark"""
    logger.info("=" * 60)
    logger.info("  BENCHMARK: Disaggregated Mode")
    logger.info("=" * 60)

    try:
        import ray
    except ImportError:
        logger.warning("Ray not available, skipping disaggregated benchmark")
        return {"status": "skipped", "reason": "ray not installed"}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    config = DisaggConfig(
        model_path=args.model,
        train_tp_size=args.train_tp,
        infer_tp_size=args.infer_tp,
        total_iterations=args.iterations,
        batch_size=args.batch_size,
        num_samples_per_prompt=args.num_samples,
    )

    orchestrator = DisaggOrchestrator(config)
    start_time = time.time()

    try:
        await orchestrator.initialize()
        await orchestrator.run()
        summary = orchestrator.get_summary()
    except Exception as e:
        logger.error(f"Disaggregated benchmark failed: {e}")
        summary = {"status": "failed", "error": str(e)}
    finally:
        await orchestrator.shutdown()
        ray.shutdown()

    elapsed = time.time() - start_time
    summary["benchmark_wall_time_s"] = elapsed
    summary["mode"] = "disaggregated"

    logger.info(f"Disaggregated benchmark completed in {elapsed:.1f}s")
    return summary


def print_comparison(colocate_results: dict, disagg_results: dict, logger):
    """打印对比表格"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  BENCHMARK COMPARISON")
    logger.info("=" * 70)
    logger.info(f"  {'Metric':<35} {'Colocate':>15} {'Disaggregated':>15}")
    logger.info(f"  {'-'*35} {'-'*15} {'-'*15}")

    def _get(d, *keys, default="N/A"):
        v = d
        for k in keys:
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                return default
        return v

    # 总时间
    col_time = _get(colocate_results, "results", "total_time_s")
    dis_time = _get(disagg_results, "results", "total_time_s")
    col_str = f"{col_time:.1f}s" if isinstance(col_time, (int, float)) else col_time
    dis_str = f"{dis_time:.1f}s" if isinstance(dis_time, (int, float)) else dis_time
    logger.info(f"  {'Total time':<35} {col_str:>15} {dis_str:>15}")

    # 平均迭代时间
    col_avg = _get(colocate_results, "results", "avg_iteration_time_s")
    dis_avg = _get(disagg_results, "timing", "avg_iteration_s")
    col_str = f"{col_avg:.2f}s" if isinstance(col_avg, (int, float)) else col_avg
    dis_str = f"{dis_avg:.2f}s" if isinstance(dis_avg, (int, float)) else dis_avg
    logger.info(f"  {'Avg iteration time':<35} {col_str:>15} {dis_str:>15}")

    # 最终准确率
    col_acc = _get(colocate_results, "results", "final_accuracy")
    dis_acc = _get(disagg_results, "results", "final_accuracy")
    col_str = f"{col_acc:.2%}" if isinstance(col_acc, (int, float)) else col_acc
    dis_str = f"{dis_acc:.2%}" if isinstance(dis_acc, (int, float)) else dis_acc
    logger.info(f"  {'Final accuracy':<35} {col_str:>15} {dis_str:>15}")

    # Overlap ratio (disagg only)
    dis_overlap = _get(disagg_results, "timing", "avg_overlap_ratio")
    dis_str = f"{dis_overlap:.1%}" if isinstance(dis_overlap, (int, float)) else dis_str
    logger.info(f"  {'Avg overlap ratio':<35} {'N/A':>15} {dis_str:>15}")

    # 峰值显存 (colocate only)
    col_mem = _get(colocate_results, "memory", "peak_mb")
    col_str = f"{col_mem:.0f}MB" if isinstance(col_mem, (int, float)) else col_mem
    logger.info(f"  {'Peak GPU memory':<35} {col_str:>15} {'N/A':>15}")

    logger.info("=" * 70)


async def main():
    args = parse_args()
    logger = setup_logger("benchmark")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"benchmark_{timestamp}.json"

    logger.info(f"Benchmark Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Iterations: {args.iterations}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Quick mode: {args.quick}")

    results = {
        "timestamp": timestamp,
        "config": {
            "model": args.model,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
        },
    }

    # 1. 运行Colocate benchmark
    colocate_results = await run_colocate_benchmark(args, logger)
    results["colocate"] = colocate_results

    # 2. 运行Disaggregated benchmark (除非quick模式)
    if not args.quick:
        disagg_results = await run_disagg_benchmark(args, logger)
        results["disaggregated"] = disagg_results
    else:
        disagg_results = {"status": "skipped", "reason": "quick mode"}
        results["disaggregated"] = disagg_results

    # 3. 对比输出
    if colocate_results.get("status") != "failed" and disagg_results.get("status") not in ("failed", "skipped"):
        print_comparison(colocate_results, disagg_results, logger)

    # 4. 保存结果到JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
