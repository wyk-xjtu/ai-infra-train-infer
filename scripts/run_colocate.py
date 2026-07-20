"""
Colocate模式运行脚本

用法:
    # 本地3090单卡（0.6B模型调试）
    python scripts/run_colocate.py --model Qwen/Qwen3-0.6B --tp-size 1

    # Windows / 无vLLM环境：使用自研简化推理后端
    python scripts/run_colocate.py --model ./models/Qwen3-0.6B --inference-backend custom --tp-size 1

    # 租卡2xA100（4B模型正式运行）
    python scripts/run_colocate.py --model Qwen/Qwen3-4B --tp-size 2 --iterations 50

vLLM后端前提: 需要先在另一个终端启动vLLM服务:
    bash scripts/start_vllm_server.sh
"""
# 在所有 import 之前清理 CUBLAS_WORKSPACE_CONFIG，防止残留环境变量
# 导致 cuBLASLt 符号加载失败（H20 + PyTorch 2.12 + CUDA 13.0 已知问题）
import os
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

import argparse
import asyncio
import sys
sys.path.insert(0, '.')

from src.orchestrator.colocate_orchestrator import ColocateOrchestrator, ColocateConfig
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Colocate Training-Inference Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config path, e.g. configs/training/sft.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model path or name")
    parser.add_argument("--tp-size", type=int, default=None,
                        help="Tensor parallel size")
    parser.add_argument("--dp-size", type=int, default=None,
                        help="Data parallel size (tp_size * dp_size must equal total GPUs)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size per iteration")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="GRPO K value (samples per prompt)")
    parser.add_argument("--training-mode", type=str, default=None,
                        choices=["grpo", "sft"],
                        help="Training objective: grpo uses rollout rewards, sft uses reference answers")
    parser.add_argument("--inference-backend", type=str, default=None,
                        choices=["vllm", "custom"],
                        help="Inference backend. Use custom on Windows or when vLLM is unavailable")
    parser.add_argument("--vllm-url", type=str, default=None,
                        help="vLLM server URL")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="LoRA rank")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max generation tokens")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a training checkpoint directory")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Optional log file path")
    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(config: dict, path: str, default=None):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _validate_parallel_config(config, logger):
    """校验 3D 并行与 PP 相关配置，不法时抛出清晰错误。

    - pp_size * tp_size * dp_size == world_size（world_size 从 WORLD_SIZE 环境变量读，非 torchrun 为 1）
    - num_micro_batches >= pp_size
    - global_batch % (num_micro_batches * dp_size) == 0
    - pp_size > 1 → zero_stage <= 1（本期 PP 仅支持 ZeRO<=1）
    """
    pp = config.pp_size
    tp = config.tp_size
    dp = config.dp_size
    nmb = config.num_micro_batches
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if pp * tp * dp != world_size:
        raise ValueError(
            f"并行度乘积与 world_size 不匹配：pp_size({pp}) * tp_size({tp}) * dp_size({dp}) "
            f"= {pp * tp * dp} != WORLD_SIZE({world_size})。请用 torchrun --nproc_per_node={pp * tp * dp} 启动。"
        )
    if nmb < pp:
        raise ValueError(
            f"num_micro_batches({nmb}) 必须 >= pp_size({pp})（否则 1F1B 无法填满流水线）。"
        )
    global_batch = config.batch_size
    if global_batch % (nmb * dp) != 0:
        raise ValueError(
            f"global_batch({global_batch}) 必须能被 num_micro_batches*dp_size = {nmb}*{dp} = {nmb * dp} 整除"
            f"（否则无法均匀切分 per-DP micro-batch）。"
        )
    if pp > 1 and config.zero_stage > 1:
        raise ValueError(
            f"pp_size({pp}) > 1 本期仅支持 zero_stage <= 1，当前 zero_stage={config.zero_stage}。"
            f"请将 training.zero_stage 设为 0 或 1。"
        )
    if pp > 1 and config.training_mode != "sft":
        raise ValueError(
            f"pp_size({pp}) > 1 本期仅支持 training_mode='sft'，当前 training_mode='{config.training_mode}'。"
        )

    if pp > 1:
        bubble = (pp - 1) / nmb
        logger.info(
            "Pipeline Parallel config OK: pp=%d, tp=%d, dp=%d, world_size=%d, "
            "num_micro_batches=%d, est_bubble=(pp-1)/nmb=%.3f",
            pp, tp, dp, world_size, nmb, bubble,
        )
    else:
        logger.info(
            "Parallel config OK: pp=%d, tp=%d, dp=%d, world_size=%d, num_micro_batches=%d",
            pp, tp, dp, world_size, nmb,
        )


async def main():
    args = parse_args()
    logger = setup_logger("colocate", log_file=args.log_file)
    cfg = load_yaml_config(args.config) if args.config else {}

    # 确定性模式（从 YAML runtime.deterministic 读取，默认 False）
    # H20 + PyTorch 2.12 + CUDA 13.0 环境下，CUBLAS_WORKSPACE_CONFIG 与 cuBLASLt 存在硬性冲突，
    # 确定性模式在多卡(DP>1)场景必须禁用。
    import torch, os
    deterministic = get_nested(cfg, "runtime.deterministic", False)

    # ============================================================
    # [关键修复] PyTorch 2.12 新增 cuDNN SDPA 后端，其内部调用 cuBLASLt。
    # H20 + CUDA 13.0 环境中 cuBLASLt 存在符号加载 bug：
    #   "Invalid handle. Cannot load symbol cublasLtGetVersion"
    # 必须全局禁用 cudnn_sdp 后端，否则任何 F.scaled_dot_product_attention
    # 调用都可能触发 SIGABRT。此设置与 deterministic 无关。
    # ============================================================
    torch.backends.cuda.enable_cudnn_sdp(False)
    logger.info("Disabled cudnn_sdp backend (cuBLASLt incompatible on H20/CUDA 13.0)")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        logger.info("Deterministic mode enabled")
        logger.warning("CUBLAS_WORKSPACE_CONFIG set — may conflict with cuBLASLt on H20/CUDA 13.0")
    else:
        # 显式禁用确定性算法，防止残留状态
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # 清理可能导致 SDPA/cuBLASLt 崩溃的 CUBLAS_WORKSPACE_CONFIG
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        logger.info("Deterministic mode disabled, CUBLAS_WORKSPACE_CONFIG cleared")

    training_mode = args.training_mode or get_nested(cfg, "training.mode", "grpo")
    inference_backend = args.inference_backend or get_nested(cfg, "runtime.inference_backend", "vllm")
    if training_mode == "sft":
        max_tokens_default = get_nested(cfg, "inference_preview.max_tokens", 1)
        temperature_default = get_nested(cfg, "inference_preview.temperature", 0.7)
        top_p_default = get_nested(cfg, "inference_preview.top_p", 0.95)
    else:
        max_tokens_default = get_nested(cfg, "rollout.max_tokens", get_nested(cfg, "inference_preview.max_tokens", 512))
        temperature_default = get_nested(cfg, "rollout.temperature", get_nested(cfg, "inference_preview.temperature", 0.7))
        top_p_default = get_nested(cfg, "rollout.top_p", get_nested(cfg, "inference_preview.top_p", 0.95))

    config = ColocateConfig(
        model_path=args.model or get_nested(cfg, "model.name_or_path", "Qwen/Qwen3-0.6B"),
        tp_size=args.tp_size if args.tp_size is not None else get_nested(cfg, "runtime.tp_size", 1),
        pp_size=get_nested(cfg, "distributed.pp_size", 1),
        num_micro_batches=get_nested(cfg, "distributed.num_micro_batches", 1),
        dp_size=args.dp_size if args.dp_size is not None else get_nested(cfg, "runtime.dp_size", 1),
        max_seq_len=get_nested(cfg, "model.max_seq_len", 2048),
        use_flash_attn=get_nested(cfg, "model.use_flash_attn", False),
        attention_backend=get_nested(cfg, "model.attention_backend", "sdpa"),
        use_gradient_checkpoint=get_nested(cfg, "model.use_gradient_checkpoint", False),
        training_mode=training_mode,
        output_dir=get_nested(cfg, "experiment.output_dir", "outputs"),
        checkpoint_save_dir=get_nested(cfg, "checkpoint.save_dir", "checkpoints"),
        checkpoint_save_interval=get_nested(cfg, "checkpoint.save_interval", 0),
        checkpoint_save_enabled=get_nested(cfg, "checkpoint.save_enabled", True),
        resume_from=args.resume_from or get_nested(cfg, "checkpoint.resume_from", None),
        total_iterations=args.iterations if args.iterations is not None else get_nested(cfg, "training.total_iterations", 10),
        batch_size=args.batch_size if args.batch_size is not None else get_nested(cfg, "training.batch_size", 4),
        train_steps_per_iteration=get_nested(cfg, "training.train_steps_per_iteration", 1),
        gradient_accumulation_steps=get_nested(cfg, "training.gradient_accumulation_steps", 4),
        inference_backend=inference_backend,
        vllm_url=args.vllm_url or get_nested(cfg, "vllm.url", "http://localhost:8000"),
        num_samples_per_prompt=args.num_samples if args.num_samples is not None else get_nested(cfg, "rollout.num_samples_per_prompt", 4),
        max_tokens=args.max_tokens if args.max_tokens is not None else max_tokens_default,
        temperature=temperature_default,
        top_p=top_p_default,
        lora_rank=args.lora_rank if args.lora_rank is not None else get_nested(cfg, "lora.rank", 16),
        lora_alpha=get_nested(cfg, "lora.alpha", 32.0),
        lora_target_modules=get_nested(
            cfg,
            "lora.target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        learning_rate=args.lr if args.lr is not None else get_nested(cfg, "training.learning_rate", 1e-5),
        weight_decay=get_nested(cfg, "training.weight_decay", 0.01),
        max_grad_norm=get_nested(cfg, "training.max_grad_norm", 1.0),
        amp_dtype=get_nested(cfg, "runtime.precision", "bf16"),
        grpo_beta=get_nested(cfg, "training.grpo_beta", 0.1),
        grpo_clip_eps=get_nested(cfg, "training.grpo_clip_eps", 0.2),
        lr_scheduler_type=get_nested(cfg, "training.lr_scheduler", "cosine"),
        warmup_steps=get_nested(cfg, "training.warmup_steps", 100),
        min_lr_ratio=get_nested(cfg, "training.min_lr_ratio", 0.1),
        max_data_samples=get_nested(cfg, "data.max_samples", 200),
        dataset_name=get_nested(cfg, "data.dataset_name", "openai/gsm8k"),
        dataset_config=get_nested(cfg, "data.dataset_config", "main"),
        dataset_split=get_nested(cfg, "data.split", "train"),
        prompt_template=get_nested(cfg, "data.prompt_template", "qwen3_math"),
        prompt_field=get_nested(cfg, "data.prompt_field", "question"),
        input_field=get_nested(cfg, "data.input_field", None),
        response_field=get_nested(cfg, "data.response_field", "answer"),
        answer_field=get_nested(cfg, "data.answer_field", "answer"),
        eval_max_tokens=get_nested(cfg, "eval.max_tokens", 128),
        eval_num_prompts=get_nested(cfg, "eval.num_prompts", 0),
        enable_comm_overlap=get_nested(cfg, "runtime.enable_comm_overlap", False),
        zero_stage=get_nested(cfg, "training.zero_stage", 1),
        inference_preview_enabled=get_nested(cfg, "inference_preview.enabled", True),
        enable_profiling=get_nested(cfg, "profiling.enabled", False),
        profile_interval=get_nested(cfg, "profiling.interval", 10),
        # IcePop 训推对齐检测
        icepop_enabled=get_nested(cfg, "icepop.enabled", False),
        icepop_divergence_threshold=get_nested(cfg, "icepop.divergence_threshold", 0.5),
        icepop_max_mask_ratio=get_nested(cfg, "icepop.max_mask_ratio", 0.5),
        # C3PO++ 动态 Rollout 分割调度
        c3po_plus_enabled=get_nested(cfg, "c3po_plus.enabled", False),
        c3po_plus_token_budget=get_nested(cfg, "c3po_plus.token_budget", 1024),
        c3po_plus_target_batch_tokens=get_nested(cfg, "c3po_plus.target_batch_tokens", 4096),
        c3po_plus_packing_strategy=get_nested(cfg, "c3po_plus.packing_strategy", "ffd"),
        # SwiftSync 增量权重同步
        swift_sync_enabled=get_nested(cfg, "swift_sync.enabled", False),
        swift_sync_fallback_full_every=get_nested(cfg, "swift_sync.fallback_full_every", 10),
        swift_sync_double_buffer=get_nested(cfg, "swift_sync.double_buffer", True),
        # 异步 Pipeline
        async_pipeline_enabled=get_nested(cfg, "async_pipeline.enabled", False),
        async_pipeline_max_staleness=get_nested(cfg, "async_pipeline.max_staleness", 2),
        async_pipeline_queue_size=get_nested(cfg, "async_pipeline.queue_size", 2),
    )

    if args.config:
        logger.info(f"Loaded config: {args.config}")

    # ============================================================
    # 并行配置校验（pp/tp/dp、num_micro_batches、global_batch、PP+ZeRO 限制）
    # ============================================================
    _validate_parallel_config(config, logger)

    logger.info(f"Starting Colocate mode: model={config.model_path}, tp={config.tp_size}, dp={config.dp_size}, iters={config.total_iterations}")
    logger.info(f"  Training mode: {config.training_mode}")
    logger.info(f"  Inference backend: {config.inference_backend}")
    if config.inference_backend == "vllm":
        logger.info(f"  vLLM URL: {config.vllm_url}")
    logger.info(f"  Dataset: {config.dataset_name} ({config.dataset_split})")
    logger.info(f"  Batch size: {config.batch_size}, Samples/prompt: {config.num_samples_per_prompt}")
    logger.info(f"  LoRA rank: {config.lora_rank}, LR: {config.learning_rate}")
    logger.info(f"  Checkpoint: save_enabled={config.checkpoint_save_enabled}, interval={config.checkpoint_save_interval}")
    logger.info(f"  Gradient checkpointing: use_gradient_checkpoint={config.use_gradient_checkpoint}")

    orchestrator = ColocateOrchestrator(config)
    await orchestrator.initialize()

    try:
        metrics = await orchestrator.run()

        # 打印总结
        summary = orchestrator.get_summary()
        logger.info("=" * 60)
        logger.info("Run Completed!")
        if summary.get("status") == "no_data":
            logger.warning("  No successful iterations completed. Check earlier error logs.")
        else:
            logger.info(f"  Iterations: {summary['results']['completed_iterations']}")
            logger.info(f"  Final accuracy: {summary['results']['final_accuracy']:.2%}")
            logger.info(f"  Best accuracy: {summary['results']['best_accuracy']:.2%}")
            logger.info(f"  Total time: {summary['results']['total_time_s']:.1f}s")
            logger.info(f"  Avg iter time: {summary['results']['avg_iteration_time_s']:.1f}s")
            logger.info(f"  Peak memory: {summary['memory']['peak_mb']:.1f} MB")
        logger.info("=" * 60)
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
