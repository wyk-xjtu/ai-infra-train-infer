"""
测试 Pipeline Parallel 基础设施（进程组布局 + P2P + 层切分）

覆盖:
1. split_layers_to_stages: 纯函数层切分（无需分布式）
2. 进程组布局: pp2tp2dp2 下 TP/PP/DP 组成员与 prev/next 正确；
   pp_size=1 (tp2dp4) 回归：TP/DP 组与旧布局一致
3. P2P: 2-rank 收发一个 tensor，数值一致、无死锁

运行方式:
    source /root/aikesi/miniconda3/bin/activate infer

    # 1. 层切分（纯函数，单进程）
    python tests/test_pipeline_parallel.py split

    # 2. 进程组布局（8 卡）
    torchrun --nproc_per_node=8 tests/test_pipeline_parallel.py pg 2 2 2   # pp tp dp
    torchrun --nproc_per_node=8 tests/test_pipeline_parallel.py pg 1 2 4   # 回归 pp_size=1

    # 3. P2P（2 卡）
    torchrun --nproc_per_node=2 tests/test_pipeline_parallel.py p2p
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, ".")

from src.distributed.parallel_context import ParallelContext
from src.distributed.pipeline_parallel import (
    split_layers_to_stages,
    PipelineParallelContext,
)
from src.distributed import comm


# ============================================================
# 1. split_layers_to_stages（纯函数）
# ============================================================


def test_split_layers_to_stages():
    # (32,4) = 4×8
    assert split_layers_to_stages(32, 4) == [(0, 8), (8, 16), (16, 24), (24, 32)]
    # (33,4) = [9,8,8,8]
    assert split_layers_to_stages(33, 4) == [(0, 9), (9, 17), (17, 25), (25, 33)]
    # (6,4) = [2,2,1,1]
    assert split_layers_to_stages(6, 4) == [(0, 2), (2, 4), (4, 5), (5, 6)]
    # pp_size=1 → 单 stage 全部层
    assert split_layers_to_stages(28, 1) == [(0, 28)]
    # 层数 < pp_size：前几个 stage 各 1 层，其余 0 层
    assert split_layers_to_stages(2, 4) == [(0, 1), (1, 2), (2, 2), (2, 2)]
    # 整除
    assert split_layers_to_stages(8, 2) == [(0, 4), (4, 8)]

    # 每个 case 的层数总和守恒
    for num_layers, pp in [(32, 4), (33, 4), (6, 4), (28, 1), (2, 4), (100, 7)]:
        stages = split_layers_to_stages(num_layers, pp)
        assert len(stages) == pp
        assert stages[0][0] == 0
        assert stages[-1][1] == num_layers
        total = sum(e - s for s, e in stages)
        assert total == num_layers
        # 相邻 stage 连续、无重叠
        for i in range(1, pp):
            assert stages[i][0] == stages[i - 1][1]

    print("  ✓ test_split_layers_to_stages passed")


def test_pipeline_context_pure():
    """无分布式环境下 PipelineParallelContext 退化行为（pp_size=1）。"""
    ctx = ParallelContext(tp_size=1, tp_group=None)  # 单卡默认 pp_size=1
    ppc = PipelineParallelContext(ctx, num_layers=28)
    assert ppc.pp_size == 1
    assert ppc.pp_rank == 0
    assert ppc.is_first_stage and ppc.is_last_stage
    assert ppc.stage_layer_range == (0, 28)
    assert ppc.num_layers_this_stage == 28
    assert ppc.prev_rank is None and ppc.next_rank is None
    print("  ✓ test_pipeline_context_pure passed")


# ============================================================
# 2. 进程组布局
# ============================================================

# 期望布局（全局 rank 成员），来自 design doc / 任务规范
EXPECTED = {
    (2, 2, 2): {  # (pp, tp, dp)
        "tp": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "pp": [[0, 2], [1, 3], [4, 6], [5, 7]],
        "dp": [[0, 4], [1, 5], [2, 6], [3, 7]],
    },
    (1, 2, 4): {  # 回归：pp_size=1，与旧布局一致
        "tp": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "pp": None,  # 无 PP 组
        "dp": [[0, 2, 4, 6], [1, 3, 5, 7]],
    },
}


def _find_group(groups, rank):
    for g in groups:
        if rank in g:
            return g
    raise AssertionError(f"rank {rank} not found in {groups}")


def test_process_group_layout(pp_size, tp_size, dp_size):
    ctx = ParallelContext.init_distributed(
        tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, backend="nccl"
    )
    rank = ctx.rank
    expected = EXPECTED[(pp_size, tp_size, dp_size)]

    # --- TP 组 ---
    tp_members = dist.get_process_group_ranks(ctx.tp_group)
    exp_tp = _find_group(expected["tp"], rank)
    assert sorted(tp_members) == sorted(exp_tp), (
        f"[rank {rank}] TP members {tp_members} != expected {exp_tp}"
    )

    # --- DP 组 ---
    dp_members = dist.get_process_group_ranks(ctx.dp_group)
    exp_dp = _find_group(expected["dp"], rank)
    assert sorted(dp_members) == sorted(exp_dp), (
        f"[rank {rank}] DP members {dp_members} != expected {exp_dp}"
    )

    # --- PP 组 ---
    if expected["pp"] is None:
        assert ctx.pp_group is None, f"[rank {rank}] expected no PP group"
        assert ctx.pp_size == 1
        assert ctx.prev_rank is None and ctx.next_rank is None
    else:
        pp_members = dist.get_process_group_ranks(ctx.pp_group)
        exp_pp = _find_group(expected["pp"], rank)
        assert sorted(pp_members) == sorted(exp_pp), (
            f"[rank {rank}] PP members {pp_members} != expected {exp_pp}"
        )
        # prev/next 校验：= global_rank ∓ tp_size，边界 None
        pp_rank = ctx.pp_rank
        exp_prev = None if pp_rank == 0 else rank - tp_size
        exp_next = None if pp_rank == pp_size - 1 else rank + tp_size
        assert ctx.prev_rank == exp_prev, (
            f"[rank {rank}] prev_rank {ctx.prev_rank} != {exp_prev}"
        )
        assert ctx.next_rank == exp_next, (
            f"[rank {rank}] next_rank {ctx.next_rank} != {exp_next}"
        )

    # 每个 rank 单独打印其结果
    dist.barrier()
    for r in range(ctx.world_size):
        if r == rank:
            print(
                f"  [rank {rank}] TP={tp_members} DP={dp_members} "
                f"PP={None if ctx.pp_group is None else dist.get_process_group_ranks(ctx.pp_group)} "
                f"prev={ctx.prev_rank} next={ctx.next_rank}  OK"
            )
        dist.barrier()

    if rank == 0:
        print(f"  ✓ test_process_group_layout(pp={pp_size},tp={tp_size},dp={dp_size}) passed")


# ============================================================
# 3. P2P（2-rank）
# ============================================================


def test_p2p_2rank():
    """pp2tp1dp1: rank0 -> rank1 前向 send/recv，rank1 -> rank0 反向 send/recv。"""
    ctx = ParallelContext.init_distributed(
        tp_size=1, dp_size=1, pp_size=2, backend="nccl"
    )
    rank = ctx.rank
    device = torch.device(f"cuda:{ctx.local_rank}")
    shape = (2, 4, 8)
    dtype = torch.float32

    # ---- 前向：rank0 send_forward -> rank1 recv_forward ----
    if ctx.pp_rank == 0:  # rank 0 (first stage)
        act = torch.arange(2 * 4 * 8, dtype=dtype, device=device).reshape(shape)
        comm.send_forward(act, ctx.next_rank, ctx.pp_group)
        # 反向：接收来自 rank1 的 grad
        grad = comm.recv_backward(shape, dtype, device, ctx.next_rank, ctx.pp_group)
        expected_grad = torch.arange(2 * 4 * 8, dtype=dtype, device=device).reshape(shape) * 2.0
        max_diff = (grad - expected_grad).abs().max().item()
        assert max_diff < 1e-6, f"recv_backward mismatch: {max_diff}"
        print(f"  [rank {rank}] send_forward + recv_backward OK (grad_diff={max_diff:.2e})")
    else:  # rank 1
        act = comm.recv_forward(shape, dtype, device, ctx.prev_rank, ctx.pp_group)
        expected_act = torch.arange(2 * 4 * 8, dtype=dtype, device=device).reshape(shape)
        max_diff = (act - expected_act).abs().max().item()
        assert max_diff < 1e-6, f"recv_forward mismatch: {max_diff}"
        # 反向：把 act*2 作为 grad 发回 rank0
        comm.send_backward(act * 2.0, ctx.prev_rank, ctx.pp_group)
        print(f"  [rank {rank}] recv_forward + send_backward OK (act_diff={max_diff:.2e})")

    dist.barrier()

    # ---- 组合原语校验：send_forward_recv_backward / send_backward_recv_forward ----
    if ctx.pp_rank == 0:  # rank 0
        out = torch.full(shape, 3.0, dtype=dtype, device=device)
        recv_grad = comm.send_forward_recv_backward(
            out, shape, dtype, device, ctx.next_rank, ctx.pp_group
        )
        # rank1 会回发 out+1 = 4.0
        assert recv_grad is not None
        assert torch.allclose(recv_grad, torch.full(shape, 4.0, device=device)), \
            "send_forward_recv_backward mismatch"
        print(f"  [rank {rank}] send_forward_recv_backward OK")
    else:  # rank 1
        recv_act = comm.recv_forward(shape, dtype, device, ctx.prev_rank, ctx.pp_group)
        # 收到 3.0，回发 recv_act+1 = 4.0
        comm.send_backward(recv_act + 1.0, ctx.prev_rank, ctx.pp_group)
        print(f"  [rank {rank}] recv_forward + send_backward (combined peer) OK")

    dist.barrier()
    if rank == 0:
        print("  ✓ test_p2p_2rank passed")


# ============================================================
# Entry
# ============================================================


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "split"

    if mode == "split":
        print("\nRunning split_layers_to_stages tests (single process)")
        print("-" * 50)
        test_split_layers_to_stages()
        test_pipeline_context_pure()
        print("-" * 50)
        print("\n✅ split tests passed!")

    elif mode == "pg":
        pp_size = int(sys.argv[2])
        tp_size = int(sys.argv[3])
        dp_size = int(sys.argv[4])
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"\nRunning process-group layout test pp={pp_size} tp={tp_size} dp={dp_size}")
            print("-" * 50)
        test_process_group_layout(pp_size, tp_size, dp_size)
        if dist.get_rank() == 0:
            print("-" * 50)
            print("\n✅ process-group layout test passed!")
        dist.destroy_process_group()

    elif mode == "p2p":
        if int(os.environ.get("RANK", "0")) == 0:
            print("\nRunning P2P 2-rank test")
            print("-" * 50)
        test_p2p_2rank()
        if dist.get_rank() == 0:
            print("-" * 50)
            print("\n✅ P2P test passed!")
        dist.destroy_process_group()

    else:
        raise SystemExit(f"unknown mode: {mode}")
