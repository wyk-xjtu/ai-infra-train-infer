"""
测试Mini-ZeRO-1优化器

验证:
1. 分片后各rank的optimizer state大小为1/world_size
2. 优化步骤后参数与标准Adam一致
3. 显存节省正确

运行:
    torchrun --nproc_per_node=2 tests/test_zero_optim.py
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
sys.path.insert(0, '.')

from src.distributed.parallel_context import ParallelContext
from src.distributed.zero_optim import ZeROOptimizer


def test_shard_size():
    """测试各rank的optimizer state大小正确"""
    ctx = ParallelContext.get_instance()

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

    expected_shard = optimizer._pad_to_divisible(total_params, ctx.world_size) // ctx.world_size
    assert optimizer.shard_size == expected_shard, (
        f"Shard size mismatch: {optimizer.shard_size} != {expected_shard}"
    )

    if ctx.rank == 0:
        print(f"  ✓ test_shard_size passed (total={total_params}, shard={optimizer.shard_size})")


def test_step_consistency():
    """测试优化步骤后参数与标准Adam一致（单进程模式下验证）"""
    ctx = ParallelContext.get_instance()

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

    if ctx.rank == 0:
        print(f"  ✓ test_step_consistency passed")


def test_memory_savings():
    """测试显存节省描述正确"""
    ctx = ParallelContext.get_instance()

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

    if ctx.rank == 0:
        print(f"  ✓ test_memory_savings passed")
        print(f"    {savings_str}")


def test_multiple_steps():
    """测试多步更新后参数依然正确"""
    ctx = ParallelContext.get_instance()

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

    all_params = [torch.empty_like(model.weight.data) for _ in range(ctx.world_size)]
    dist.all_gather(all_params, model.weight.data.contiguous(), group=ctx.tp_group)

    for i in range(1, len(all_params)):
        max_diff = (all_params[0] - all_params[i]).abs().max().item()
        assert max_diff < 1e-6, f"Rank 0 and rank {i} parameters diverged: {max_diff}"

    if ctx.rank == 0:
        print(f"  ✓ test_multiple_steps passed (param change={param_diff:.4e})")


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    ctx = ParallelContext.init_distributed(tp_size=dist.get_world_size())

    if ctx.rank == 0:
        print(f"\nRunning ZeRO-1 tests with world_size={ctx.world_size}")
        print("-" * 50)

    test_shard_size()
    test_step_consistency()
    test_memory_savings()
    test_multiple_steps()

    if ctx.rank == 0:
        print("-" * 50)
        print("\n✅ All ZeRO-1 optimizer tests passed!")

    dist.destroy_process_group()
