"""
流水线并行 1F1B 调度器 (Pipeline Scheduler, One-Forward-One-Backward)

职责:
- 在给定 PipelineParallelContext 与 micro-batch 迭代器下，按 1F1B（PipeDream-Flush）
  调度前向/反向，跨 stage 传递 activation / gradient，完成一次梯度累积。
- 仅累积梯度到 p.grad，不做 optimizer.step（step 由上层 train engine 负责）。

1F1B 调度（照 docs/pipeline_parallel_design.md:87-89）:
- warmup   = pp_size - 1 - pp_rank     （只前向，填充流水线）
- steady   = num_micro_batches - warmup（交替 1F1B）
- cooldown = warmup                    （只反向，排空流水线）

关键设计决策（务必阅读）:
1. **通信集中在 run_1f1b_schedule**：跨 stage 的 P2P（send/recv）统一由调度主循环
   编排，而非分散在 forward_step/backward_step 内。原因是 1F1B 稳态下相邻 stage
   会「同时向对端发送前向 activation 与反向 grad」，若两侧都各自先做阻塞 send，
   会造成双向死锁。因此稳态使用组合原语 send_forward_recv_backward /
   send_backward_recv_forward（内部 batch_isend_irecv 打包收发），这是 Megatron
   PipeDream-Flush 的标准做法。forward_step/backward_step 只负责「计算 + 缓存 +
   求梯度」，语义清晰且可单测。
2. **recv 到的 activation 必须 requires_grad_(True)**：否则 1F1B 反向时
   input_tensor.grad 为 None，无法把梯度回传给上一个 stage。首 stage 的输入是
   token_ids（无梯度），故首 stage 不向上游发送梯度。
3. **loss 缩放 1/num_micro_batches**：每个 micro-batch 的 loss 除以 num_micro，
   累加后等价于对各 micro-batch loss 求平均；梯度同样按 1/num_micro 缩放后累加，
   与单卡「大 batch 前后向一次」在数学上一致。
4. **seq_len 握手**：在 schedule 开始时做一次极小的形状握手——首 stage 依据 token
   形状构造 int64 张量 [mbs, seq, hidden]，沿流水线单向前传（每个非末 stage 收到后
   再转发）。单向传播不会死锁，且只需一次，开销可忽略。之后所有 micro-batch 的
   activation/grad 均使用该固定形状。

参考: docs/pipeline_parallel_design.md, Megatron-LM schedules.py (forward_backward_pipelining_without_interleaving)
"""

import logging
from collections import deque
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F

from .comm import (
    send_forward,
    recv_forward,
    send_backward,
    recv_backward,
    send_forward_recv_backward,
    send_backward_recv_forward,
)

logger = logging.getLogger(__name__)


def default_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """默认损失：对 [*, vocab] 的 logits 与 [*] 的 labels 做交叉熵（token 均值）。

    注意：不做 next-token shift；只要 PP 与非 PP 路径使用同一 loss_fn，
    数值等价性即成立。上层可通过 loss_fn 传入自定义（含 shift/ignore_index）。
    """
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab).float(),
        labels.reshape(-1).long(),
    )


class PipelineScheduler:
    """1F1B 流水线调度器。

    对外接口（供 T6 train engine 集成）:
        scheduler = PipelineScheduler(pp_context, num_micro_batches, loss_fn=None)
        total_loss = scheduler.run_1f1b_schedule(model, micro_batch_iterator)
        # 之后由上层做 optimizer.step()

    micro_batch_iterator: 可迭代对象，产出 num_micro_batches 个 (input_ids, labels)。
        - 所有 pp rank 应产出「相同」的 micro-batch 序列（同一 DP 组内数据一致）。
        - 首 stage 使用 input_ids 作为模型输入；末 stage 使用 labels 计算 loss；
          中间 stage 两者都不用（其输入来自上游 activation）。
    """

    def __init__(
        self,
        pp_context,
        num_micro_batches: int,
        loss_fn: Optional[Callable] = None,
    ):
        assert num_micro_batches >= 1, "num_micro_batches must be >= 1"
        self.pp = pp_context
        self.num_micro_batches = num_micro_batches
        self.loss_fn = loss_fn if loss_fn is not None else default_loss_fn

        # 1F1B FIFO：缓存已前向、待反向的 (input_tensor, output_tensor)
        self.input_tensors: deque = deque()
        self.output_tensors: deque = deque()
        # 末 stage：缓存已缩放的 loss，backward 时逐个 .backward()
        self._pending_losses: deque = deque()
        # 末 stage：累积 loss（detach，用于日志/返回）
        self._accumulated_loss: Optional[torch.Tensor] = None

    # 计算步骤（不含跨 stage 通信；通信由 run_1f1b_schedule 编排）

    def forward_step(
        self,
        model,
        input_tensor: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """单个 micro-batch 前向：跑 model、缓存 (input, output)、末 stage 算缩放 loss。

        Args:
            model: stage 感知的 ParallelTransformerModel
            input_tensor:
                - 首 stage: token_ids [mbs, seq]
                - 中间/末 stage: 上游 activation [mbs, seq, hidden]（**须已 requires_grad_(True)**）
            labels: 末 stage 计算 loss 用的 [mbs, seq]；其他 stage 传 None

        Returns:
            output_tensor：
                - 末 stage: logits（同时内部已算好缩放 loss 供 backward）
                - 其他 stage: hidden_states（用于 send_forward 给下游）
        """
        output_tensor = model(input_tensor)

        if self.pp.is_last_stage:
            assert labels is not None, "last stage requires labels to compute loss"
            loss = self.loss_fn(output_tensor, labels) / self.num_micro_batches
            self._pending_losses.append(loss)
            if self._accumulated_loss is None:
                self._accumulated_loss = loss.detach().clone()
            else:
                self._accumulated_loss = self._accumulated_loss + loss.detach()

        self.input_tensors.append(input_tensor)
        self.output_tensors.append(output_tensor)
        return output_tensor

    def backward_step(self, grad_output: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """单个 micro-batch 反向：从 FIFO 取最旧的 (input, output)，回传梯度。

        Args:
            grad_output: 来自下游 stage 的 output 梯度；末 stage 传 None（用缩放 loss.backward）

        Returns:
            input_tensor.grad —— 需 send_backward 给上游的梯度；
            首 stage（输入为 token_ids，无梯度）返回 None。
        """
        input_tensor = self.input_tensors.popleft()
        output_tensor = self.output_tensors.popleft()

        if self.pp.is_last_stage:
            # 末 stage：loss 已缩放，直接 backward
            loss = self._pending_losses.popleft()
            loss.backward()
        else:
            # 非末 stage：用下游回传的 grad_output 驱动本 stage 反向
            torch.autograd.backward(tensors=output_tensor, grad_tensors=grad_output)

        # 取本 stage 输入的梯度（供回传给上游）；首 stage 的 token 无梯度 → None
        if (
            isinstance(input_tensor, torch.Tensor)
            and input_tensor.grad is not None
        ):
            return input_tensor.grad
        return None

    # 主调度

    def run_1f1b_schedule(self, model, micro_batch_iterator: Iterable) -> Optional[torch.Tensor]:
        """执行一次 1F1B 调度（warmup → steady → cooldown）。

        Returns:
            末 stage：累积 loss（标量 Tensor）；非末 stage：None。
            梯度已累积到各参数 p.grad，不在此处 optimizer.step。
        """
        pp = self.pp
        batches = list(micro_batch_iterator)
        assert len(batches) == self.num_micro_batches, (
            f"iterator yielded {len(batches)} micro-batches, "
            f"expected {self.num_micro_batches}"
        )

        param = next(model.parameters())
        device, dtype = param.device, param.dtype

        # 重置状态
        self.input_tensors.clear()
        self.output_tensors.clear()
        self._pending_losses.clear()
        self._accumulated_loss = None

        if pp.pp_size == 1:
            for input_ids, labels in batches:
                self.forward_step(
                    model, input_ids.to(device), labels=labels.to(device)
                )
                self.backward_step(None)
            return self._accumulated_loss

        act_shape = self._resolve_act_shape(model, batches, device)

        num_warmup = min(pp.pp_size - 1 - pp.pp_rank, self.num_micro_batches)
        num_remaining = self.num_micro_batches - num_warmup

        mb_idx = 0  # 前向 micro-batch 指针

        for _ in range(num_warmup):
            input_tensor = self._recv_activation(batches, mb_idx, act_shape, dtype, device)
            labels = self._labels_for(batches, mb_idx, device)
            output_tensor = self.forward_step(model, input_tensor, labels=labels)
            if not pp.is_last_stage:
                send_forward(output_tensor, pp.next_rank, pp.pp_group)
            mb_idx += 1

        input_tensor = None
        if num_remaining > 0:
            input_tensor = self._recv_activation(batches, mb_idx, act_shape, dtype, device)

        for i in range(num_remaining):
            last_iter = i == num_remaining - 1
            labels = self._labels_for(batches, mb_idx, device)
            output_tensor = self.forward_step(model, input_tensor, labels=labels)
            mb_idx += 1

            # 前向 activation 发下游 + 反向 grad 收回（组合，避免死锁）
            if pp.is_last_stage:
                grad_output = None
            else:
                grad_output = send_forward_recv_backward(
                    output_tensor, act_shape, dtype, device, pp.next_rank, pp.pp_group
                )

            # 反向最旧的 micro-batch
            input_grad = self.backward_step(grad_output)

            if last_iter:
                # 稳态最后一个反向的 grad 直接发上游；cooldown 排空剩余
                if not pp.is_first_stage:
                    send_backward(input_grad, pp.prev_rank, pp.pp_group)
                input_tensor = None
            else:
                # 反向 grad 发上游 + 前向 activation 收回（组合，避免死锁）
                if pp.is_first_stage:
                    # 首 stage：无上游可发 grad，直接取下一个 micro-batch 的 token
                    input_tensor = self._recv_activation(
                        batches, mb_idx, act_shape, dtype, device
                    )
                else:
                    input_tensor = send_backward_recv_forward(
                        input_grad, act_shape, dtype, device, pp.prev_rank, pp.pp_group
                    )
                    if input_tensor is not None:
                        input_tensor.requires_grad_(True)

        for _ in range(num_warmup):
            if pp.is_last_stage:
                grad_output = None
            else:
                grad_output = recv_backward(
                    act_shape, dtype, device, pp.next_rank, pp.pp_group
                )
            input_grad = self.backward_step(grad_output)
            if not pp.is_first_stage:
                send_backward(input_grad, pp.prev_rank, pp.pp_group)

        return self._accumulated_loss if pp.is_last_stage else None

    # 内部工具

    def _resolve_act_shape(self, model, batches, device):
        """一次性形状握手：返回本 stage 需要接收/发送的 activation 形状。

        首 stage 依据 token 形状构造 [mbs, seq, hidden]，沿流水线单向前传；
        非首 stage 接收后（若非末 stage）继续前传。单向传播不会死锁。
        """
        pp = self.pp
        hidden = model.hidden_size
        if pp.is_first_stage:
            input_ids = batches[0][0]
            mbs, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
            shape_t = torch.tensor([mbs, seq, hidden], dtype=torch.int64, device=device)
            send_forward(shape_t, pp.next_rank, pp.pp_group)
            return (mbs, seq, hidden)
        else:
            shape_t = recv_forward((3,), torch.int64, device, pp.prev_rank, pp.pp_group)
            shape = tuple(int(x) for x in shape_t.tolist())
            if not pp.is_last_stage:
                send_forward(shape_t, pp.next_rank, pp.pp_group)
            return shape

    def _recv_activation(self, batches, idx, act_shape, dtype, device):
        """取本 stage 第 idx 个前向输入。

        首 stage：从 batches 取 token_ids（无梯度）；
        其他 stage：recv_forward 上游 activation，并置 requires_grad_(True)。
        """
        pp = self.pp
        if pp.is_first_stage:
            return batches[idx][0].to(device)
        tensor = recv_forward(act_shape, dtype, device, pp.prev_rank, pp.pp_group)
        tensor.requires_grad_(True)
        return tensor

    def _labels_for(self, batches, idx, device):
        """末 stage 取第 idx 个 micro-batch 的 labels；其他 stage 返回 None。"""
        if self.pp.is_last_stage:
            return batches[idx][1].to(device)
        return None
