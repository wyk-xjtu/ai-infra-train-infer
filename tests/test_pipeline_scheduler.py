"""
测试 Pipeline stage 感知模型 (T4) + 1F1B 调度器 (T5)

覆盖:
1. stage 模型 (mode=stage, 2 卡 pp=2):
   - 首/末 stage 组件持有正确（embed 仅首、norm+lm_head 仅末、layers 仅本 stage）
   - forward 输出形状正确（首/中返回 hidden [b,s,h]、末返回 logits [b,s,vocab]）
2. 调度器数值等价 (mode=equiv, 2 卡 pp=2):
   - 参考：单进程 full model（pp=1 语义）逐 micro-batch 前后向累积，得 ref_loss / ref_grad
   - pp=2：各 stage 从 full model 复制对应权重，跑 run_1f1b_schedule
   - 断言：末 stage 累积 loss 与 ref_loss 接近等价（<1%）
   - 断言：反向后各 stage 参数 p.grad 非 None（梯度确实回传）
   - 断言：各 stage 参数 grad 与 full model 对应参数 grad 接近（强化等价证据）

运行方式:
    source /root/aikesi/miniconda3/bin/activate infer
    torchrun --nproc_per_node=2 tests/test_pipeline_scheduler.py stage
    torchrun --nproc_per_node=2 tests/test_pipeline_scheduler.py equiv
"""
import os
import sys

import torch

sys.path.insert(0, ".")

from src.distributed.parallel_context import ParallelContext
from src.distributed.pipeline_parallel import PipelineParallelContext
from src.distributed.tensor_parallel import ParallelTransformerModel
from src.distributed.pipeline_scheduler import PipelineScheduler, default_loss_fn

import torch.distributed as dist


# ------------------------------------------------------------------
# toy config（小到可在单卡快速跑）
# ------------------------------------------------------------------
CFG = dict(
    vocab_size=128,
    hidden_size=64,
    num_layers=4,
    num_heads=4,
    num_kv_heads=2,
    head_dim=16,
    intermediate_size=128,
    max_position_embeddings=64,
)
MBS = 2
SEQ = 8
NUM_MICRO = 4
SEED = 1234


def _build_model(parallel_context, pp_context, device):
    torch.manual_seed(SEED)  # 确定性初始化：各 rank 的 full model 权重一致
    model = ParallelTransformerModel(
        parallel_context=parallel_context,
        pp_context=pp_context,
        **CFG,
    )
    return model.to(device)


def _gen_batches(device):
    """确定性生成 NUM_MICRO 个 (input_ids, labels)，所有 rank 一致。"""
    g = torch.Generator().manual_seed(999)
    batches = []
    for _ in range(NUM_MICRO):
        ids = torch.randint(0, CFG["vocab_size"], (MBS, SEQ), generator=g)
        labels = torch.randint(0, CFG["vocab_size"], (MBS, SEQ), generator=g)
        batches.append((ids.to(device), labels.to(device)))
    return batches


def _local_to_global_key(key, offset):
    if key.startswith("layers."):
        parts = key.split(".", 2)
        return f"layers.{int(parts[1]) + offset}.{parts[2]}"
    return key


# ------------------------------------------------------------------
# mode = stage
# ------------------------------------------------------------------
def test_stage_model():
    ctx = ParallelContext.init_distributed(tp_size=1, dp_size=1, pp_size=2, backend="nccl")
    rank = ctx.rank
    device = torch.device(f"cuda:{ctx.local_rank}")
    ppc = PipelineParallelContext(ctx, num_layers=CFG["num_layers"])

    model = _build_model(ctx, ppc, device)

    start, end = ppc.stage_layer_range
    n_local = end - start
    assert len(model.layers) == n_local, f"[rank {rank}] layers {len(model.layers)} != {n_local}"

    if ppc.is_first_stage:
        assert model.embed_tokens is not None, "first stage must hold embed_tokens"
        assert model.norm is None and model.lm_head is None, "first stage must NOT hold norm/lm_head"
        # forward: tokens -> hidden
        ids = torch.randint(0, CFG["vocab_size"], (MBS, SEQ), device=device)
        out = model(ids)
        assert out.shape == (MBS, SEQ, CFG["hidden_size"]), f"first stage out {out.shape}"
        print(f"  [rank {rank}] first stage: embed✓ layers={n_local} hidden_out={tuple(out.shape)} OK")
    else:
        assert model.embed_tokens is None, "last stage must NOT hold embed_tokens"
        assert model.norm is not None and model.lm_head is not None, "last stage must hold norm+lm_head"
        # forward: hidden -> logits
        hidden = torch.randn(MBS, SEQ, CFG["hidden_size"], device=device)
        out = model(hidden)
        assert out.shape == (MBS, SEQ, CFG["vocab_size"]), f"last stage out {out.shape}"
        print(f"  [rank {rank}] last stage: norm+lm_head✓ layers={n_local} logits_out={tuple(out.shape)} OK")

    dist.barrier()
    if rank == 0:
        print("  ✓ test_stage_model passed")


# ------------------------------------------------------------------
# mode = equiv
# ------------------------------------------------------------------
def test_equivalence():
    ctx = ParallelContext.init_distributed(tp_size=1, dp_size=1, pp_size=2, backend="nccl")
    rank = ctx.rank
    device = torch.device(f"cuda:{ctx.local_rank}")
    ppc = PipelineParallelContext(ctx, num_layers=CFG["num_layers"])

    batches = _gen_batches(device)

    # ---- 参考：单进程 full model（pp=1 语义），逐 micro-batch 前后向累积 ----
    ref_ctx = ParallelContext(tp_size=1, tp_group=None)  # 无分布式，comm no-op
    full_model = _build_model(ref_ctx, None, device)  # pp_context=None → 完整模型
    full_model.zero_grad(set_to_none=True)
    ref_loss = torch.zeros((), device=device)
    for ids, labels in batches:
        logits = full_model(ids)
        l = default_loss_fn(logits, labels) / NUM_MICRO
        ref_loss = ref_loss + l.detach()
        l.backward()

    # ---- pp=2 stage model：从 full model 复制对应权重 ----
    stage_model = _build_model(ctx, ppc, device)
    offset = stage_model.stage_start
    full_sd = full_model.state_dict()
    stage_sd = stage_model.state_dict()
    new_sd = {k: full_sd[_local_to_global_key(k, offset)] for k in stage_sd}
    stage_model.load_state_dict(new_sd)
    stage_model.zero_grad(set_to_none=True)
    stage_model.train()

    scheduler = PipelineScheduler(ppc, num_micro_batches=NUM_MICRO, loss_fn=default_loss_fn)
    sched_loss = scheduler.run_1f1b_schedule(stage_model, iter(batches))

    # ---- 断言 1：末 stage loss 等价（<1%）----
    if ppc.is_last_stage:
        assert sched_loss is not None, "last stage must return accumulated loss"
        ref = ref_loss.item()
        got = sched_loss.item()
        rel = abs(got - ref) / (abs(ref) + 1e-12)
        assert rel < 0.01, f"[rank {rank}] loss mismatch: pp2={got:.6f} ref={ref:.6f} rel={rel:.3%}"
        print(f"  [rank {rank}] LAST stage loss: pp2={got:.6f} ref={ref:.6f} rel_err={rel:.3e} (<1%) OK")
    else:
        assert sched_loss is None, "non-last stage must return None"
        print(f"  [rank {rank}] FIRST stage loss=None (as expected)")

    # ---- 断言 2：p.grad 非 None（梯度确实回传）----
    none_grads = [n for n, p in stage_model.named_parameters() if p.requires_grad and p.grad is None]
    assert not none_grads, f"[rank {rank}] params with grad=None: {none_grads}"
    n_params = sum(1 for _ in stage_model.parameters())
    print(f"  [rank {rank}] all {n_params} params have grad (梯度回传✓)")

    # ---- 断言 3：grad 与 full model 对应参数接近（强化等价）----
    full_named = dict(full_model.named_parameters())
    max_grad_diff = 0.0
    for n, p in stage_model.named_parameters():
        gk = _local_to_global_key(n, offset)
        ref_g = full_named[gk].grad
        diff = (p.grad - ref_g).abs().max().item()
        max_grad_diff = max(max_grad_diff, diff)
    assert max_grad_diff < 1e-3, f"[rank {rank}] grad diff too large: {max_grad_diff}"
    print(f"  [rank {rank}] max grad diff vs full model = {max_grad_diff:.3e} (<1e-3) OK")

    dist.barrier()
    if rank == 0:
        print("  ✓ test_equivalence passed (loss 等价 + 梯度回传 + grad 匹配)")


# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stage"
    is_rank0 = int(os.environ.get("RANK", "0")) == 0

    if mode == "stage":
        if is_rank0:
            print("\nRunning stage-model test (pp=2, 2 卡)")
            print("-" * 50)
        test_stage_model()
        if dist.get_rank() == 0:
            print("-" * 50)
            print("\n✅ stage-model test passed!")
        dist.destroy_process_group()

    elif mode == "equiv":
        if is_rank0:
            print("\nRunning 1F1B equivalence test (pp=2 vs pp=1 reference)")
            print("-" * 50)
        test_equivalence()
        if dist.get_rank() == 0:
            print("-" * 50)
            print("\n✅ equivalence test passed!")
        dist.destroy_process_group()

    else:
        raise SystemExit(f"unknown mode: {mode}")
