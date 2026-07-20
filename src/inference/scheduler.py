"""
Continuous Batching 调度器

===================== 核心思想 =====================

静态Batching的问题:
  一个batch中所有序列必须等最长的序列完成才能处理下一个batch。
  假设batch中4个序列分别需要生成 10, 50, 200, 500 tokens:
  - 短序列(10 tokens)完成后空等490步 → GPU严重浪费

Continuous Batching的解法:
  序列一旦完成生成(EOS/达到max_tokens)，立即移出batch。
  空出的位置立刻被等待队列中的新请求填入。
  → GPU始终在处理尽可能满的batch，吞吐量提升 2-3x

===================== 调度策略 =====================

两个阶段的计算特征:
  - Prefill: 处理整个prompt（计算密集 compute-bound，适合大 batch 并行）
  - Decode: 每个序列只生成1个token（访存密集 memory-bound，延迟敏感）

Token Budget:
  每步最多处理 max_num_batched_tokens 个token:
  - Prefill请求消耗 prompt_length 个token
  - Decode请求每个消耗 1 个token
  - 调度器在budget内最大化吞吐

Chunked Prefill:
  当prompt很长(>chunk_size)时，分多步处理:
  - 避免单个长prompt阻塞整个batch（类似协程的yield）
  - 允许decode请求与prefill共享同一步的token budget
"""
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RequestState(Enum):
    """请求生命周期状态"""
    WAITING = "waiting"        # 在等待队列中，尚未开始Prefill
    RUNNING = "running"        # Prefill完成，正在Decode生成
    FINISHED = "finished"      # 生成完毕（EOS或达到max_tokens）
    PREEMPTED = "preempted"    # 被抢占，KV Cache已释放，需重新Prefill


@dataclass
class Request:
    """推理请求
    
    完整生命周期: WAITING → (Prefill) → RUNNING → (Decode循环) → FINISHED
    异常路径: RUNNING → PREEMPTED → WAITING → 重新开始
    """
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int = 512
    temperature: float = 0.7

    # 运行时状态（由调度器管理）
    state: RequestState = RequestState.WAITING
    output_tokens: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0   # 已计算的prompt tokens数（用于chunked prefill）
    arrival_time: float = field(default_factory=time.time)

    @property
    def num_prompt_tokens(self) -> int:
        """Prompt总长度"""
        return len(self.prompt_tokens)

    @property
    def num_generated_tokens(self) -> int:
        """已生成的output tokens数"""
        return len(self.output_tokens)

    @property
    def total_tokens(self) -> int:
        """序列总长度（prompt + generated）"""
        return self.num_prompt_tokens + self.num_generated_tokens

    @property
    def is_prefill_complete(self) -> bool:
        """Prefill是否已全部完成"""
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        """还剩多少prompt token需要计算"""
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    @property
    def is_finished(self) -> bool:
        """是否完成生成"""
        # 条件1: 达到最大生成长度
        if self.num_generated_tokens >= self.max_tokens:
            return True
        # 条件2: 生成了EOS token (假设EOS=2)
        if self.output_tokens and self.output_tokens[-1] == 2:
            return True
        return False


@dataclass
class SchedulerOutput:
    """调度器输出 — 告诉执行器这一步应该处理什么
    
    执行器根据此输出:
    1. 对prefill_requests做KV计算（可能是chunked的一部分）
    2. 对decode_requests做单步token生成
    3. 对preempted_requests释放KV Cache
    """
    prefill_requests: List[Request] = field(default_factory=list)
    decode_requests: List[Request] = field(default_factory=list)
    preempted_requests: List[Request] = field(default_factory=list)
    chunk_size: int = 512  # 用于计算num_prefill_tokens

    @property
    def num_prefill_tokens(self) -> int:
        """本步Prefill消耗的token数（考虑chunked prefill）"""
        total = 0
        for r in self.prefill_requests:
            remaining = r.num_prompt_tokens - r.num_computed_tokens
            total += min(remaining, self.chunk_size)
        return total

    @property
    def num_decode_tokens(self) -> int:
        """本步Decode消耗的token数（每个running序列1个）"""
        return len(self.decode_requests)

    @property
    def total_tokens(self) -> int:
        """本步总token消耗"""
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_empty(self) -> bool:
        """本步是否为空（无请求可调度）"""
        return not self.prefill_requests and not self.decode_requests


class ContinuousBatchingScheduler:
    """Continuous Batching调度器
    
    核心数据结构:
    - waiting_queue: FIFO队列，新请求进入此队列等待Prefill
    - running_queue: 已完成Prefill的序列，每步做1次Decode
    
    调度策略:
    1. Decode优先: 先为所有running序列预留1 token/seq的budget
    2. 剩余budget分给waiting队列做Prefill
    3. 若显存不足running都放不下，抢占(LIFO)释放空间
    """

    def __init__(self, max_num_batched_tokens: int = 2048,
                 max_num_sequences: int = 256,
                 max_prefill_tokens: int = 1024,
                 enable_chunked_prefill: bool = True,
                 chunk_size: int = 512,
                 enable_adaptive_chunking: bool = True):
        """
        Args:
            max_num_batched_tokens: 每步最大token数（硬上限，防OOM）
            max_num_sequences: 最大并发序列数
            max_prefill_tokens: 单步最大prefill token数（避免prefill独占）
            enable_chunked_prefill: 是否启用分块prefill
            chunk_size: 每次prefill处理的最大token数/请求
            enable_adaptive_chunking: 是否启用自适应chunk_size调整
        """
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.max_prefill_tokens = max_prefill_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.chunk_size = chunk_size
        self.enable_adaptive_chunking = enable_adaptive_chunking

        # 两级队列
        self.waiting_queue: List[Request] = []
        self.running_queue: List[Request] = []

        # 统计
        self._total_scheduled = 0
        self._total_preempted = 0
        self._total_finished = 0

    def add_request(self, request: Request):
        """添加新请求到等待队列（FIFO入队）"""
        request.state = RequestState.WAITING
        self.waiting_queue.append(request)

    def schedule(self) -> SchedulerOutput:
        """核心调度逻辑
        
        一步调度流程:
        ┌─────────────────────────────────────────────┐
        │ 1. 计算decode预留budget = len(running) × 1  │
        │ 2. 检查是否超总budget，超了则抢占           │
        │ 3. 剩余budget = total - decode预留           │
        │ 4. 用剩余budget调度waiting中的prefill请求   │
        │ 5. 输出 SchedulerOutput                     │
        └─────────────────────────────────────────────┘
        """
        output = SchedulerOutput(chunk_size=self.chunk_size)

        # === Step 1: 处理running队列（Decode） ===
        # 每个running序列需要1个token的budget
        decode_budget_needed = len(self.running_queue)

        # === Step 2: 如果decode都装不下，需要抢占 ===
        while decode_budget_needed > self.max_num_batched_tokens:
            preempted = self._preempt_one()
            if preempted is None:
                break  # 无法再抢占
            output.preempted_requests.append(preempted)
            decode_budget_needed -= 1

        # 检查并发序列数限制
        while len(self.running_queue) > self.max_num_sequences:
            preempted = self._preempt_one()
            if preempted is None:
                break
            output.preempted_requests.append(preempted)

        # 所有running序列参与decode
        output.decode_requests = list(self.running_queue)

        # === Step 3: 计算剩余budget ===
        remaining_budget = self.max_num_batched_tokens - len(self.running_queue)
        prefill_budget = min(remaining_budget, self.max_prefill_tokens)
        available_seq_slots = self.max_num_sequences - len(self.running_queue)

        # === Step 4: 调度Prefill请求 ===
        if prefill_budget > 0 and available_seq_slots > 0 and self.waiting_queue:
            scheduled_prefill = self._schedule_prefill(prefill_budget, available_seq_slots)
            output.prefill_requests = scheduled_prefill

        self._total_scheduled += 1
        return output

    def _get_adaptive_chunk_size(self) -> int:
        """根据 decode 队列压力动态调整 chunk_size

        策略:
        - decode 队列积压时减小 chunk_size，让出预算给 decode
        - 无 decode 请求时加大 chunk_size 加速 prefill
        - 其他情况使用默认 chunk_size
        """
        if not self.enable_adaptive_chunking:
            return self.chunk_size
        # decode 队列积压时减小 chunk_size，让出预算给 decode
        if len(self.running_queue) > 3:
            return max(self.chunk_size // 2, 128)
        # 无 decode 请求时加大 chunk_size 加速 prefill
        if len(self.running_queue) == 0:
            return min(self.chunk_size * 2, 2048)
        return self.chunk_size

    def _schedule_prefill(self, budget: int, max_new_seqs: int) -> List[Request]:
        """尝试调度Prefill请求
        
        FCFS策略：按到达顺序逐个尝试调度。
        Chunked Prefill: 每个请求最多消耗chunk_size个token。
        支持自适应chunk_size: 根据decode队列压力动态调整。
        
        Args:
            budget: 可用token预算
            max_new_seqs: 最多可接纳的新序列数
            
        Returns:
            本步参与prefill的请求列表
        """
        scheduled: List[Request] = []
        remaining = budget
        new_seqs = 0

        # 使用自适应 chunk_size
        effective_chunk_size = self._get_adaptive_chunk_size()

        to_remove = []
        for i, request in enumerate(self.waiting_queue):
            if new_seqs >= max_new_seqs:
                break

            # 计算此请求本步消耗的token数
            tokens_remaining = request.remaining_prefill_tokens
            if self.enable_chunked_prefill:
                tokens_this_step = min(tokens_remaining, effective_chunk_size)
            else:
                tokens_this_step = tokens_remaining

            # 检查budget是否足够
            if tokens_this_step > remaining:
                # 第一个请求允许超budget（避免饥饿）
                if not scheduled:
                    tokens_this_step = remaining
                else:
                    break

            remaining -= tokens_this_step
            scheduled.append(request)
            new_seqs += 1
            
            # 判断此请求prefill是否本步就能完成
            will_complete = (request.num_computed_tokens + tokens_this_step >= 
                          request.num_prompt_tokens)
            if will_complete:
                to_remove.append(i)

        # 移动已完成prefill的请求到running队列
        for idx in reversed(to_remove):
            request = self.waiting_queue.pop(idx)
            request.state = RequestState.RUNNING
            self.running_queue.append(request)

        return scheduled

    def _preempt_one(self) -> Optional[Request]:
        """抢占一个running序列（LIFO: 最后到达的最先被抢占）
        
        理由: 最后到达的序列生成的token最少，抢占它损失最小。
        被抢占的序列回到waiting队列头部（下次优先重新调度）。
        """
        if not self.running_queue:
            return None

        # LIFO: 找最晚到达的（不修改原queue顺序）
        victim_idx = max(range(len(self.running_queue)),
                         key=lambda i: self.running_queue[i].arrival_time)
        victim = self.running_queue.pop(victim_idx)

        victim.state = RequestState.PREEMPTED
        # 重置计算状态（需要重新prefill）
        victim.num_computed_tokens = 0
        victim.output_tokens.clear()
        victim.state = RequestState.WAITING
        # 插入waiting队列头部（优先重新调度）
        self.waiting_queue.insert(0, victim)

        self._total_preempted += 1
        return victim

    def finish_request(self, request_id: str):
        """标记请求完成，从running队列移除
        
        由执行器在检测到EOS/max_tokens后调用。
        """
        for i, req in enumerate(self.running_queue):
            if req.request_id == request_id:
                req.state = RequestState.FINISHED
                self.running_queue.pop(i)
                self._total_finished += 1
                return

    def update_request_progress(self, request_id: str, num_new_tokens: int):
        """更新请求的计算进度（chunked prefill场景）"""
        for req in self.waiting_queue + self.running_queue:
            if req.request_id == request_id:
                req.num_computed_tokens += num_new_tokens
                return

    def has_pending_requests(self) -> bool:
        """是否还有待处理的请求"""
        return bool(self.waiting_queue) or bool(self.running_queue)

    def get_stats(self) -> dict:
        """调度统计"""
        return {
            "waiting_count": len(self.waiting_queue),
            "running_count": len(self.running_queue),
            "total_scheduled_steps": self._total_scheduled,
            "total_preempted": self._total_preempted,
            "total_finished": self._total_finished,
            "config": {
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_num_sequences": self.max_num_sequences,
                "chunked_prefill": self.enable_chunked_prefill,
                "chunk_size": self.chunk_size,
            },
        }
