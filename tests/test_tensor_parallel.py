"""
测试Mini-Megatron张量并行正确性

验证方法:
1. 创建标准nn.Linear和ColumnParallelLinear/RowParallelLinear
2. 给相同输入，对比输出是否一致（误差<1e-5）
3. 验证反向传播梯度正确性

运行方式:
    # 单卡模拟2进程TP
    torchrun --nproc_per_node=2 tests/test_tensor_parallel.py
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
sys.path.insert(0, '.')

from src.distributed.parallel_context import ParallelContext
from src.distributed.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear, ParallelMLP
)
from src.distributed.comm import all_gather_last_dim


def test_column_parallel():
    """测试ColumnParallelLinear输出与标准Linear一致"""
    ctx = ParallelContext.get_instance()
    tp_rank = ctx.tp_rank
    tp_size = ctx.tp_size

    torch.manual_seed(42)

    in_features = 64
    out_features = 128
    batch_size = 2
    seq_len = 4

    # 1. 创建标准Linear（所有rank用相同随机种子）
    torch.manual_seed(100)
    ref_linear = nn.Linear(in_features, out_features, bias=True)

    # 2. 创建ColumnParallelLinear
    col_linear = ColumnParallelLinear(
        in_features, out_features, ctx, bias=True, gather_output=False
    )

    # 3. 将参考权重的对应切片加载到ColumnParallelLinear
    shard_size = out_features // tp_size
    col_linear.weight.data.copy_(
        ref_linear.weight.data[tp_rank * shard_size: (tp_rank + 1) * shard_size]
    )
    col_linear.bias.data.copy_(
        ref_linear.bias.data[tp_rank * shard_size: (tp_rank + 1) * shard_size]
    )

    # 4. 构造相同输入（所有rank一致）
    torch.manual_seed(200)
    x = torch.randn(batch_size, seq_len, in_features)

    # 5. 前向传播
    ref_output = ref_linear(x)  # [batch, seq, out_features]
    col_output = col_linear(x)  # [batch, seq, out_features/tp_size]

    # 6. AllGather输出并对比
    gathered = all_gather_last_dim(col_output, ctx.tp_group)

    # 对比
    max_diff = (ref_output - gathered).abs().max().item()
    assert max_diff < 1e-5, f"ColumnParallel forward mismatch: max_diff={max_diff}"

    if tp_rank == 0:
        print(f"  ✓ test_column_parallel passed (max_diff={max_diff:.2e})")


def test_row_parallel():
    """测试RowParallelLinear输出与标准Linear一致"""
    ctx = ParallelContext.get_instance()
    tp_rank = ctx.tp_rank
    tp_size = ctx.tp_size

    in_features = 128
    out_features = 64
    batch_size = 2
    seq_len = 4

    # 1. 创建标准Linear
    torch.manual_seed(300)
    ref_linear = nn.Linear(in_features, out_features, bias=True)

    # 2. 创建RowParallelLinear
    row_linear = RowParallelLinear(
        in_features, out_features, ctx, bias=True, input_is_parallel=True
    )

    # 3. 加载对应切片的权重
    shard_size = in_features // tp_size
    row_linear.weight.data.copy_(
        ref_linear.weight.data[:, tp_rank * shard_size: (tp_rank + 1) * shard_size]
    )
    # bias 不切分，完整加载
    row_linear.bias.data.copy_(ref_linear.bias.data)

    # 4. 构造输入：RowParallel期望输入已经按last dim切分
    torch.manual_seed(400)
    x_full = torch.randn(batch_size, seq_len, in_features)
    x_shard = x_full[..., tp_rank * shard_size: (tp_rank + 1) * shard_size].contiguous()

    # 5. 前向传播
    ref_output = ref_linear(x_full)  # [batch, seq, out_features]
    row_output = row_linear(x_shard)  # [batch, seq, out_features] (AllReduce后)

    # 对比
    max_diff = (ref_output - row_output).abs().max().item()
    assert max_diff < 1e-5, f"RowParallel forward mismatch: max_diff={max_diff}"

    if tp_rank == 0:
        print(f"  ✓ test_row_parallel passed (max_diff={max_diff:.2e})")


def test_parallel_mlp():
    """测试ParallelMLP (SwiGLU)输出正确性

    验证: ParallelMLP的输出 == 手动实现的SwiGLU（无TP）
    """
    ctx = ParallelContext.get_instance()
    tp_rank = ctx.tp_rank
    tp_size = ctx.tp_size

    hidden_size = 64
    intermediate_size = 128  # 必须能被tp_size整除
    batch_size = 2
    seq_len = 4

    # 1. 创建ParallelMLP
    torch.manual_seed(500)
    mlp = ParallelMLP(hidden_size, intermediate_size, ctx)

    # 2. 创建参考实现（用标准Linear模拟SwiGLU）
    torch.manual_seed(500)
    # 重建相同的初始化（每个rank需要收集完整权重做参考）
    ref_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
    ref_up = nn.Linear(hidden_size, intermediate_size, bias=False)
    ref_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    # 收集并行MLP的完整权重构建参考
    # gate_proj weight: 每个rank有 [intermediate/tp, hidden]
    gate_weights_list = [torch.empty_like(mlp.gate_proj.weight.data) for _ in range(tp_size)]
    up_weights_list = [torch.empty_like(mlp.up_proj.weight.data) for _ in range(tp_size)]
    down_weights_list = [torch.empty_like(mlp.down_proj.weight.data) for _ in range(tp_size)]

    dist.all_gather(gate_weights_list, mlp.gate_proj.weight.data.contiguous(), group=ctx.tp_group)
    dist.all_gather(up_weights_list, mlp.up_proj.weight.data.contiguous(), group=ctx.tp_group)
    dist.all_gather(down_weights_list, mlp.down_proj.weight.data.contiguous(), group=ctx.tp_group)

    # ColumnParallel按dim=0拼接
    ref_gate.weight.data.copy_(torch.cat(gate_weights_list, dim=0))
    ref_up.weight.data.copy_(torch.cat(up_weights_list, dim=0))
    # RowParallel按dim=1拼接
    ref_down.weight.data.copy_(torch.cat(down_weights_list, dim=1))

    # 3. 相同输入
    torch.manual_seed(600)
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 4. 前向传播
    mlp_output = mlp(x)

    # 参考实现
    import torch.nn.functional as F
    gate_out = F.silu(ref_gate(x))
    up_out = ref_up(x)
    ref_output = ref_down(gate_out * up_out)

    # 对比
    max_diff = (ref_output - mlp_output).abs().max().item()
    assert max_diff < 1e-4, f"ParallelMLP forward mismatch: max_diff={max_diff}"

    if tp_rank == 0:
        print(f"  ✓ test_parallel_mlp passed (max_diff={max_diff:.2e})")


def test_backward_gradients():
    """测试反向传播梯度正确性

    验证: ColumnParallel + RowParallel组合后的梯度与标准Linear一致
    """
    ctx = ParallelContext.get_instance()
    tp_rank = ctx.tp_rank
    tp_size = ctx.tp_size

    in_features = 64
    hidden_features = 128
    out_features = 64
    batch_size = 2
    seq_len = 4

    # 标准两层MLP: Linear1(64->128) -> ReLU -> Linear2(128->64)
    torch.manual_seed(700)
    ref_l1 = nn.Linear(in_features, hidden_features, bias=False)
    ref_l2 = nn.Linear(hidden_features, out_features, bias=False)

    # 并行版本: ColumnParallel(64->128) -> ReLU -> RowParallel(128->64)
    col = ColumnParallelLinear(in_features, hidden_features, ctx, bias=False, gather_output=False)
    row = RowParallelLinear(hidden_features, out_features, ctx, bias=False, input_is_parallel=True)

    # 加载对应切片权重
    shard_out = hidden_features // tp_size
    col.weight.data.copy_(ref_l1.weight.data[tp_rank * shard_out: (tp_rank + 1) * shard_out])
    shard_in = hidden_features // tp_size
    row.weight.data.copy_(ref_l2.weight.data[:, tp_rank * shard_in: (tp_rank + 1) * shard_in])

    # 相同输入
    torch.manual_seed(800)
    x_ref = torch.randn(batch_size, seq_len, in_features, requires_grad=True)
    x_par = x_ref.clone().detach().requires_grad_(True)

    # 前向 + 反向（参考）
    h_ref = torch.relu(ref_l1(x_ref))
    out_ref = ref_l2(h_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # 前向 + 反向（并行）
    h_par = torch.relu(col(x_par))
    out_par = row(h_par)
    loss_par = out_par.sum()
    loss_par.backward()

    # 验证输出一致
    max_diff_output = (out_ref - out_par).abs().max().item()
    assert max_diff_output < 1e-5, f"Backward output mismatch: {max_diff_output}"

    # 验证输入梯度一致
    max_diff_grad = (x_ref.grad - x_par.grad).abs().max().item()
    assert max_diff_grad < 1e-5, f"Backward grad mismatch: {max_diff_grad}"

    if tp_rank == 0:
        print(f"  ✓ test_backward_gradients passed (output_diff={max_diff_output:.2e}, grad_diff={max_diff_grad:.2e})")


if __name__ == "__main__":
    # 初始化分布式环境
    dist.init_process_group(backend="gloo")  # gloo支持CPU，方便测试
    ctx = ParallelContext.init_distributed(tp_size=dist.get_world_size())

    if ctx.rank == 0:
        print(f"\nRunning TP tests with world_size={ctx.world_size}")
        print("-" * 50)

    test_column_parallel()
    test_row_parallel()
    test_parallel_mlp()
    test_backward_gradients()

    if ctx.rank == 0:
        print("-" * 50)
        print("\n✅ All tensor parallel tests passed!")

    dist.destroy_process_group()
