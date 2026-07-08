"""
测试 TrainEngine._clip_grad_norm 的 3D 全局梯度范数归约正确性。

背景:
    旧实现用 `norm_group = dp_group if dp>1 else tp_group` 做"二选一"归约，
    当 dp>1 且 tp>1 时会漏掉 TP 维度，导致全局范数低估、裁剪偏弱。
    修复后对存在的每个并行组 (dp/tp/pp) 各自独立 AllReduce 平方范数和。

验证方法:
    - 每个 rank 构造一组已知、rank 间互不相同的梯度；
    - 调用 _clip_grad_norm 得到"引擎计算的全局范数"（max_grad_norm 设为极大值以跳过裁剪）；
    - 参考值 = 对【全部 rank】的本地梯度平方和 AllReduce(SUM over WORLD) 再开方；
    - 断言二者一致 => 证明 dp 与 tp（以及 pp）维度都被计入。
      若仍是旧的"二选一"实现，引擎值只会覆盖 2 个 rank（部分和），与 4 rank 参考值不符。

运行方式:
    torchrun --nproc_per_node=4 tests/test_grad_norm_3d.py
    torchrun --nproc_per_node=8 tests/test_grad_norm_3d.py
"""
import sys
import math

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, '.')

from src.distributed.parallel_context import ParallelContext
from src.engines.train_engine import TrainEngine


class _FakeConfig:
    """仅提供 _clip_grad_norm 需要的字段。"""
    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = max_grad_norm


class _FakeEngine:
    """最小化桩对象：复用真实 TrainEngine._clip_grad_norm，仅注入其访问的属性。"""
    def __init__(self, model: nn.Module, ctx: ParallelContext, max_grad_norm: float):
        self.model = model
        self.parallel_ctx = ctx
        self.config = _FakeConfig(max_grad_norm)
        self.metrics_collector = None


def _build_model_with_known_grads(ctx: ParallelContext) -> nn.Module:
    """构造一个含 2 个参数的小模型，并写入 rank 相关的已知梯度。

    梯度值依赖 rank，保证各 rank 本地范数不同，从而"是否计入某 rank"可被检测。
    不设置 `_tp_replicated`，跳过 TP replicated 参数的 SUM 分支，聚焦范数归约本身。
    """
    torch.manual_seed(1234 + ctx.rank)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.Linear(16, 4, bias=False),
    )
    for p in model.parameters():
        p.requires_grad_(True)
        g = torch.full_like(p.data, fill_value=float(ctx.rank + 1) * 0.1)
        g = g + torch.randn_like(p.data) * 0.01
        p.grad = g
    return model


def _reference_global_norm(model: nn.Module) -> float:
    """参考实现：对全部 rank（WORLD 默认组）的本地梯度平方和求 SUM 再开方。"""
    local_sq = sum(
        p.grad.data.float().norm(2).item() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    t = torch.tensor(local_sq, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)  # WORLD 默认组，覆盖全部 rank
    return math.sqrt(t.item())


def test_3d_grad_norm(ctx: ParallelContext):
    """核心断言：引擎全局范数 == 全部 rank 参考范数（证明 dp+tp[+pp] 均被计入）。"""
    model = _build_model_with_known_grads(ctx)

    ref_model_grads = [p.grad.clone() for p in model.parameters()]
    ref_norm = _reference_global_norm(model)

    for p, g in zip(model.parameters(), ref_model_grads):
        p.grad = g.clone()

    engine = _FakeEngine(model, ctx, max_grad_norm=1e30)
    engine_norm = TrainEngine._clip_grad_norm(engine)

    rel_err = abs(engine_norm - ref_norm) / (ref_norm + 1e-12)
    assert rel_err < 1e-6, (
        f"[rank {ctx.rank}] 3D grad-norm 不一致: engine={engine_norm:.6f} "
        f"ref(all-rank)={ref_norm:.6f} rel_err={rel_err:.2e}"
    )

    if ctx.rank == 0:
        print(
            f"  ✓ test_3d_grad_norm passed "
            f"(tp{ctx.tp_size}xdp{ctx.dp_size}xpp{ctx.pp_size}, "
            f"engine={engine_norm:.6f}, ref={ref_norm:.6f}, rel_err={rel_err:.2e})"
        )


def test_single_dim_unchanged(ctx: ParallelContext):
    """行为不变性佐证：单维度场景下，引擎全局范数依然等于该维度全部 rank 之和开方。

    对 tp-only 或 dp-only（这里通过整体 WORLD 参考验证）：由于本 run 是多维，
    该用例主要复用上面的参考验证逻辑做一次一致性再确认，不额外归约维度。
    """
    model = _build_model_with_known_grads(ctx)
    ref_norm = _reference_global_norm(model)
    engine = _FakeEngine(model, ctx, max_grad_norm=1e30)
    engine_norm = TrainEngine._clip_grad_norm(engine)
    assert abs(engine_norm - ref_norm) / (ref_norm + 1e-12) < 1e-6
    if ctx.rank == 0:
        print(f"  ✓ test_single_dim_unchanged passed (consistency re-check)")


def _resolve_layout(world_size: int):
    """按 world_size 选择并行布局。"""
    if world_size == 4:
        return dict(tp_size=2, dp_size=2, pp_size=1)
    if world_size == 8:
        return dict(tp_size=2, dp_size=2, pp_size=2)
    if world_size == 2:
        return dict(tp_size=2, dp_size=1, pp_size=1)
    return dict(tp_size=world_size, dp_size=1, pp_size=1)


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    layout = _resolve_layout(world_size)
    ctx = ParallelContext.init_distributed(backend="gloo", **layout)

    if ctx.rank == 0:
        print(f"\nRunning 3D grad-norm tests with world_size={world_size}, layout={layout}")
        print("-" * 60)

    test_3d_grad_norm(ctx)
    test_single_dim_unchanged(ctx)

    dist.barrier()
    if ctx.rank == 0:
        print("-" * 60)
        print("\n✅ All 3D grad-norm tests passed!")

    dist.destroy_process_group()
