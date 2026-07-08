"""
Disaggregated模式运行脚本

用法:
    python scripts/run_disaggregated.py --model Qwen/Qwen3-4B --train-tp 2 --infer-tp 2

    python scripts/run_disaggregated.py --model Qwen/Qwen3-0.6B --train-tp 1 --infer-tp 1 --no-overlap
"""
import argparse
import asyncio
import sys
sys.path.insert(0, '.')

try:
    import ray
except ImportError:
    print("ERROR: Ray is required for Disaggregated mode. Install via: pip install ray")
    sys.exit(1)

from src.orchestrator.disagg_orchestrator import DisaggOrchestrator, DisaggConfig
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Disaggregated Training-Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model path or name")
    parser.add_argument("--train-tp", type=int, default=2,
                        help="Training tensor parallel size")
    parser.add_argument("--infer-tp", type=int, default=2,
                        help="Inference tensor parallel size")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per iteration")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="GRPO K value (samples per prompt)")
    parser.add_argument("--weight-backend", type=str, default="nccl",
                        choices=["nccl", "ray_object_store"],
                        help="Weight synchronization backend")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Disable pipeline overlap (sequential mode)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max generation tokens")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Optional log file path")
    return parser.parse_args()


async def main():
    args = parse_args()
    logger = setup_logger("disaggregated", log_file=args.log_file)

    # 初始化Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info(f"Ray initialized: {ray.cluster_resources()}")

    config = DisaggConfig(
        model_path=args.model,
        train_tp_size=args.train_tp,
        infer_tp_size=args.infer_tp,
        total_iterations=args.iterations,
        batch_size=args.batch_size,
        num_samples_per_prompt=args.num_samples,
        weight_sync_backend=args.weight_backend,
        overlap_training_inference=not args.no_overlap,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        max_tokens=args.max_tokens,
    )

    logger.info(f"Starting Disaggregated mode: model={args.model}")
    logger.info(f"  Train TP: {args.train_tp}, Infer TP: {args.infer_tp}")
    logger.info(f"  Overlap: {not args.no_overlap}, Backend: {args.weight_backend}")
    logger.info(f"  Iterations: {args.iterations}, Batch size: {args.batch_size}")

    orchestrator = DisaggOrchestrator(config)
    await orchestrator.initialize()

    try:
        metrics = await orchestrator.run()

        # 打印总结
        summary = orchestrator.get_summary()
        logger.info("=" * 60)
        logger.info("Run Completed!")
        logger.info(f"  Iterations: {summary['results']['completed_iterations']}")
        logger.info(f"  Final accuracy: {summary['results']['final_accuracy']:.2%}")
        logger.info(f"  Best accuracy: {summary['results']['best_accuracy']:.2%}")
        logger.info(f"  Total time: {summary['results']['total_time_s']:.1f}s")
        logger.info(f"  Avg iter time: {summary['timing']['avg_iteration_s']:.1f}s")
        logger.info(f"  Avg overlap ratio: {summary['timing']['avg_overlap_ratio']:.1%}")
        logger.info("=" * 60)
    finally:
        await orchestrator.shutdown()
        ray.shutdown()
        logger.info("Ray shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
