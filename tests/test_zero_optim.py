"""
测试Mini-ZeRO优化器

本文件覆盖 ZeRO 优化器的两类语义（与 X-2 修复对齐）:

A. 真正的 DP 场景 (tp_size=1, dp_size>1)
   ZeRO 在 dp_group 上分片是正确的，验证:
   1. 各 rank 的参数分片大小为 padded_total / dp_size
   2. optimizer.world_size == dp_size、optimizer.group is not None
   3. 优化步骤后参数与标准 Adam 数值一致（各 rank 输入相同 → 梯度平均后等价）
   4. 多步更新后 AllGather，各 DP rank 参数一致
   5. 显存节省描述正确

B. 退化场景 (dp_size<=1，例如 tp_size=2,dp_size=1 或单进程)  [X-2 修复]
   dp_size<=1 时 ZeRO 不再回退 tp_group，而是退化为本地非分片优化器，验证:
   1. optimizer.group is None
   2. optimizer.world_size == 1
   3. shard_size == padded_size == total_params（不分片）
   4. step() 能正常本地更新，且与标准 Adam 数值一致

运行:
    torchrun --nproc_per_node=2 tests/test_zero_optim.py
    python tests/test_zero_optim.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
sys.path.insert(0, '.')

from src.distributed.parallel_context import ParallelContext
from src.distributed.zero_optim import ZeROOptimizer


def _is_main() -> bool:
    """是否为主进程（用于收敛打印）"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def test_dp_shard_size(ctx: ParallelContext):
    """DP 场景下各 rank 的参数分片大小为 padded_total / dp_size"""
    assert ctx.dp_size > 1, "test_dp_shard_size 需要 dp_size>1 的多进程环境"

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.Linear(128, 64, bias=False),
    )

    total_params = sum(p.numel() for p in model.parameters())

    optimizer = ZeROOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        parallel_context=ctx,
        lr=1e-3,
    )

    assert optimizer.group is not None, "DP 场景下通信组不应为 None"
    assert optimizer.world_size == ctx.dp_size, (
        f"ZeRO world_size({optimizer.world_size}) 应等于 dp_size({ctx.dp_size})"
    )

    padded_total = optimizer._pad_to_divisible(total_params, ctx.dp_size)
    expected_shard = padded_total // ctx.dp_size
    assert optimizer.shard_size == expected_shard, (
        f"Shard size mismatch: {optimizer.shard_size} != {expected_shard}"
    )
    assert optimizer.padded_size == padded_total, (
        f"padded_size mismatch: {optimizer.padded_size} != {padded_total}"
    )

    if _is_main():
        print(f"  ✓ test_dp_shard_size passed "
              f"(total={total_params}, dp_size={ctx.dp_size}, shard={optimizer.shard_size})")


def test_dp_step_consistency(ctx: ParallelContext):
    """DP 场景下优化步骤后参数与标准 Adam 一致

    各 DP rank 输入相同 → 梯度相同，ReduceScatter(SUM)/world_size = 平均 = 单卡梯度，
    因此 ZeRO 更新应等价于标准 AdamW。
    """
    assert ctx.dp_size > 1, "test_dp_step_consistency 需要 dp_size>1 的多进程环境"

    torch.manual_seed(42)
    model_zero = nn.Linear(32, 32, bias=False)

    torch.manual_seed(42)
    model_ref = nn.Linear(32, 32, bias=False)

    zero_optim = ZeROOptimizer(
        model_zero.parameters(),
        optimizer_class=torch.optim.AdamW,
        parallel_context=ctx,
        lr=1e-3,
        weight_decay=0.0,
    )

    ref_optim = torch.optim.AdamW(model_ref.parameters(), lr=1e-3, weight_decay=0.0)

    torch.manual_seed(100)
    x = torch.randn(4, 32)

    out_ref = model_ref(x)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    ref_optim.step()
    ref_optim.zero_grad()

    out_zero = model_zero(x)
    loss_zero = out_zero.sum()
    loss_zero.backward()
    zero_optim.step()
    zero_optim.zero_grad()

    for p_ref, p_zero in zip(model_ref.parameters(), model_zero.parameters()):
        max_diff = (p_ref.data - p_zero.data).abs().max().item()
        assert max_diff < 1e-5, f"Parameter mismatch after step: max_diff={max_diff}"

    if _is_main():
        print(f"  ✓ test_dp_step_consistency passed")


def test_dp_memory_savings(ctx: ParallelContext):
    """DP 场景下显存节省描述正确（分片后有真实节省）"""
    assert ctx.dp_size > 1, "test_dp_memory_savings 需要 dp_size>1 的多进程环境"

    torch.manual_seed(42)
    model = nn.Linear(256, 256, bias=False)

    optimizer = ZeROOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        parallel_context=ctx,
        lr=1e-3,
    )

    x = torch.randn(2, 256)
    out = model(x)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    state_mem = optimizer.state_memory_usage
    assert state_mem > 0, "Optimizer state memory should be > 0 after step"

    savings_str = optimizer.total_optimizer_state_savings
    assert "Savings" in savings_str, f"Savings string should contain 'Savings': {savings_str}"

    if _is_main():
        print(f"  ✓ test_dp_memory_savings passed")
        print(f"    {savings_str}")


def test_dp_multiple_steps(ctx: ParallelContext):
    """DP 场景下多步更新后参数依然正确，且各 DP rank 参数一致"""
    assert ctx.dp_size > 1, "test_dp_multiple_steps 需要 dp_size>1 的多进程环境"

    torch.manual_seed(42)
    model = nn.Linear(64, 64, bias=False)

    optimizer = ZeROOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        parallel_context=ctx,
        lr=1e-3,
        weight_decay=0.0,
    )

    initial_params = model.weight.data.clone()

    for step in range(5):
        torch.manual_seed(step)
        x = torch.randn(4, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    param_diff = (model.weight.data - initial_params).abs().max().item()
    assert param_diff > 1e-6, f"Parameters should have changed after 5 steps, diff={param_diff}"

    all_params = [torch.empty_like(model.weight.data) for _ in range(ctx.dp_size)]
    dist.all_gather(all_params, model.weight.data.contiguous(), group=ctx.dp_group)

    for i in range(1, len(all_params)):
        max_diff = (all_params[0] - all_params[i]).abs().max().item()
        assert max_diff < 1e-6, f"DP rank 0 and rank {i} parameters diverged: {max_diff}"

    if _is_main():
        print(f"  ✓ test_dp_multiple_steps passed (param change={param_diff:.4e})")


def test_degenerate_no_sharding():
    """dp_size<=1 时 ZeRO 退化为本地非分片优化器

    [X-2 修复] 例如 tp_size=2,dp_size=1（纯 TP 场景）或单进程时:
    - group is None（不再回退 tp_group，避免对 TP 切分参数做语义错误的 ReduceScatter）
    - world_size == 1
    - shard_size == padded_size == total_params（不分片）
    - step() 正常本地更新，与标准 Adam 一致
    """
    degenerate_ctx = ParallelContext(
        tp_size=2, tp_group=None, dp_size=1, dp_group=None,
    )

    torch.manual_seed(42)
    model_zero = nn.Linear(32, 32, bias=False)

    torch.manual_seed(42)
    model_ref = nn.Linear(32, 32, bias=False)

    total_params = sum(p.numel() for p in model_zero.parameters())

    zero_optim = ZeROOptimizer(
        model_zero.parameters(),
        optimizer_class=torch.optim.AdamW,
        parallel_context=degenerate_ctx,
        lr=1e-3,
        weight_decay=0.0,
    )

    assert zero_optim.group is None, (
        f"dp_size<=1 时通信组应为 None（不回退 tp_group），实际为 {zero_optim.group}"
    )
    assert zero_optim.world_size == 1, (
        f"dp_size<=1 时 world_size 应为 1，实际为 {zero_optim.world_size}"
    )
    assert zero_optim.shard_size == zero_optim.padded_size, (
        f"退化场景不应分片：shard_size({zero_optim.shard_size}) 应等于 "
        f"padded_size({zero_optim.padded_size})"
    )
    assert zero_optim.shard_size == total_params, (
        f"退化场景 shard_size({zero_optim.shard_size}) 应等于 total_params({total_params})"
    )

    ref_optim = torch.optim.AdamW(model_ref.parameters(), lr=1e-3, weight_decay=0.0)

    torch.manual_seed(100)
    x = torch.randn(4, 32)

    out_ref = model_ref(x)
    out_ref.sum().backward()
    ref_optim.step()
    ref_optim.zero_grad()

    out_zero = model_zero(x)
    out_zero.sum().backward()
    zero_optim.step()
    zero_optim.zero_grad()

    for p_ref, p_zero in zip(model_ref.parameters(), model_zero.parameters()):
        max_diff = (p_ref.data - p_zero.data).abs().max().item()
        assert max_diff < 1e-5, f"退化场景本地更新与标准 Adam 不一致: max_diff={max_diff}"

    if _is_main():
        print(f"  ✓ test_degenerate_no_sharding passed "
              f"(group=None, world_size=1, shard_size={zero_optim.shard_size}=total)")


if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        dist.init_process_group(backend="gloo")  # gloo 支持 CPU，方便测试
        ctx = ParallelContext.init_distributed(
            tp_size=1, dp_size=dist.get_world_size(), backend="gloo"
        )

        if _is_main():
            print(f"\nRunning ZeRO tests with world_size={ctx.world_size} "
                  f"(tp_size={ctx.tp_size}, dp_size={ctx.dp_size})")
            print("-" * 50)
            print("[A] 真正的 DP 场景：ZeRO 在 dp_group 上分片")

        test_dp_shard_size(ctx)
        test_dp_step_consistency(ctx)
        test_dp_memory_savings(ctx)
        test_dp_multiple_steps(ctx)

        if _is_main():
            print("[B] 退化场景：dp_size<=1 不分片")
        test_degenerate_no_sharding()

        if _is_main():
            print("-" * 50)
            print("\n✅ All ZeRO optimizer tests passed!")

        dist.destroy_process_group()
    else:
        ctx = ParallelContext.init_distributed(
            tp_size=1, dp_size=1, backend="gloo", init_single_process=True
        )

        print(f"\nRunning ZeRO tests in single process (dp_size=1)")
        print("-" * 50)
        print("[B] 退化场景：dp_size<=1 不分片")
        print("  (DP 分片用例需 torchrun --nproc_per_node>=2，单进程下跳过)")

        test_degenerate_no_sharding()

        print("-" * 50)
        print("\n✅ ZeRO degenerate-semantics test passed (single process)!")

        if dist.is_initialized():
            dist.destroy_process_group()
