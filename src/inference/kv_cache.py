"""
KV Cache Block 管理器 — 实现 PagedAttention 的核心内存管理

===================== 核心思想（PagedAttention） =====================

传统方案的问题:
  为每个序列预分配 max_seq_len 的连续KV显存。假设 max_seq_len=2048 但平均只用 200 tokens，
  那浪费率高达 90%。更严重的是连续分配导致显存碎片化，难以支撑大 batch。

PagedAttention 的解法:
  借鉴操作系统虚拟内存分页（Paging）思想：
  1. 将KV Cache切分为固定大小的Block（如16 tokens/block）
  2. 序列按需分配block——一个block写满才分配下一个
  3. 物理block可以不连续，通过block_table维护 逻辑位置→物理位置 的映射
  4. Attention kernel在计算时根据block_table索引物理位置

显存节省对比:
  - 传统: batch_size × max_seq_len × 2(K,V) × num_layers × num_kv_heads × head_dim × dtype_bytes
  - PagedAttention: 只分配实际使用的blocks → 利用率从 ~50% 提升到 ~95%
"""
import torch
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Block:
    """物理KV Cache Block
    
    每个block可存储 block_size 个 token 的 K 和 V 向量。
    ref_count 支持多序列共享同一个物理block（前缀共享场景）。
    """
    block_id: int
    ref_count: int = 0       # 引用计数 >0 表示正在使用，支持 CoW 前缀共享
    token_count: int = 0     # 当前block中已存储的token数（<= block_size）
    is_full: bool = False    # token_count == block_size 时为 True
    hash_value: Optional[int] = None  # 用于 prefix caching 的哈希值

    def reset(self):
        """重置block状态（回收到池中前调用）"""
        self.ref_count = 0
        self.token_count = 0
        self.is_full = False
        self.hash_value = None


class BlockPool:
    """物理Block池 — 预分配所有GPU显存给block池，运行时按需分配/回收
    
    设计理念：
    - 系统启动时一次性分配全部KV Cache显存（避免运行时碎片）
    - 维护 free_blocks 队列，分配和释放均为 O(1)
    - 通过 num_free_blocks 判断能否接纳新序列
    
    显存布局：
    kv_cache tensor shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    其中 [0] 是 K cache, [1] 是 V cache
    """

    def __init__(self, num_blocks: int, block_size: int,
                 num_layers: int, num_kv_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.float16, device: str = "cuda"):
        """
        预分配KV Cache显存：
        总显存 = num_blocks × block_size × 2(K,V) × num_layers × num_kv_heads × head_dim × dtype_bytes
        
        示例（Qwen3-4B参数）:
        1024 blocks × 16 tokens × 2 × 32 layers × 8 heads × 128 dim × 2 bytes ≈ 16GB
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # 创建所有物理block对象
        self._blocks: List[Block] = [Block(block_id=i) for i in range(num_blocks)]
        
        self._free_queue: deque = deque(range(num_blocks))
        
        # 已使用的block id集合（用于快速判断）
        self._used_set: Set[int] = set()
        
        # 预分配GPU显存（教学实现中用cpu模拟，避免无GPU环境报错）
        # 生产环境应为: torch.empty(..., device=device, dtype=dtype)
        self._kv_cache: Optional[torch.Tensor] = None
        self._allocate_gpu_memory()

    def _allocate_gpu_memory(self):
        """预分配KV Cache的GPU显存
        
        实际shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        - 维度0: K=0, V=1
        - 维度1: transformer层编号
        - 维度2: 物理block编号（通过block_table索引）
        - 维度3: block内token位置
        - 维度4: KV head编号
        - 维度5: head维度
        """
        try:
            self._kv_cache = torch.zeros(
                2, self.num_layers, self.num_blocks,
                self.block_size, self.num_kv_heads, self.head_dim,
                dtype=self.dtype,
                device=self.device if torch.cuda.is_available() else "cpu",
            )
        except RuntimeError:
            # 显存不足时fallback到CPU（教学环境）
            self._kv_cache = torch.zeros(
                2, self.num_layers, self.num_blocks,
                self.block_size, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device="cpu",
            )

    def allocate(self) -> Optional[Block]:
        """分配一个空闲block
        
        Returns:
            Block对象，若池已满则返回None
        """
        if not self._free_queue:
            return None
        
        block_id = self._free_queue.popleft()
        block = self._blocks[block_id]
        block.reset()
        block.ref_count = 1  # 首次分配引用计数为1
        self._used_set.add(block_id)
        return block

    def free(self, block: Block):
        """释放一个block（引用计数减1，为0时真正回收）
        
        引用计数语义：
        - ref_count > 1: 多个序列共享此block（前缀共享），仅减计数
        - ref_count == 1: 最后一个引用者释放，真正回收
        """
        if block.ref_count <= 0:
            import warnings
            warnings.warn(
                f"Double-free detected for block {block.block_id} "
                f"(ref_count={block.ref_count}). Ignoring."
            )
            return
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.reset()
            self._used_set.discard(block.block_id)
            self._free_queue.append(block.block_id)

    def get_block(self, block_id: int) -> Block:
        """通过ID获取Block对象"""
        return self._blocks[block_id]

    @property
    def num_free_blocks(self) -> int:
        """当前空闲block数"""
        return len(self._free_queue)

    @property
    def num_used_blocks(self) -> int:
        """当前已用block数"""
        return len(self._used_set)

    @property
    def utilization(self) -> float:
        """显存利用率 = 已用blocks / 总blocks"""
        if self.num_blocks == 0:
            return 0.0
        return self.num_used_blocks / self.num_blocks

    @property
    def kv_cache_tensor(self) -> Optional[torch.Tensor]:
        """获取底层KV Cache张量（供Attention kernel使用）
        
        Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        """
        return self._kv_cache

    def get_flattened_kv_cache(self) -> Optional[torch.Tensor]:
        """获取展平视图的 KV Cache（供 ModelRunner 绑定到 Attention 层）
        
        返回 shape: [2, num_layers, total_slots, num_kv_heads, head_dim]
        其中 total_slots = num_blocks * block_size
        
        这个视图允许通过 slot_mapping 直接索引，无需分别计算 block_id 和 offset。
        注意: 返回的是 view，不是 copy，与原始 tensor 共享内存。
        """
        if self._kv_cache is None:
            return None
        # [2, layers, blocks, block_size, heads, dim] → [2, layers, blocks*block_size, heads, dim]
        shape = self._kv_cache.shape
        return self._kv_cache.view(
            shape[0], shape[1],
            shape[2] * shape[3],  # total_slots
            shape[4], shape[5],
        )


class BlockTable:
    """逻辑→物理Block映射表（每个序列维护一个）
    
    序列的token在逻辑上是连续的 [0, 1, 2, ..., seq_len-1]，
    但物理上可能分散在不同的block中。BlockTable记录这个映射：
    
    逻辑block 0 → 物理block 42
    逻辑block 1 → 物理block 7
    逻辑block 2 → 物理block 128
    ...
    
    Attention kernel通过 block_table[logical_idx] 找到物理位置。
    """

    def __init__(self, block_size: int):
        self._blocks: List[Block] = []  # 有序的物理block列表
        self._block_size = block_size

    def append_block(self, block: Block):
        """追加一个物理block到映射表末尾"""
        self._blocks.append(block)

    def get_physical_block_ids(self) -> List[int]:
        """返回物理block id列表
        
        这个列表直接作为attention kernel的block_tables参数，
        kernel通过 block_tables[seq_idx][logical_block_idx] 索引物理位置。
        """
        return [b.block_id for b in self._blocks]

    @property
    def last_block(self) -> Optional[Block]:
        """获取最后一个block（用于检查是否需要分配新block）"""
        return self._blocks[-1] if self._blocks else None

    @property
    def num_blocks(self) -> int:
        """逻辑block总数"""
        return len(self._blocks)

    @property
    def num_tokens(self) -> int:
        """当前已存储的总token数"""
        if not self._blocks:
            return 0
        # 前面的blocks都是满的，只有最后一个可能未满
        full_blocks = len(self._blocks) - 1
        last_tokens = self._blocks[-1].token_count if self._blocks else 0
        return full_blocks * self._block_size + last_tokens

    @property
    def blocks(self) -> List[Block]:
        """获取所有物理block引用"""
        return self._blocks


class KVCacheManager:
    """KV Cache管理器 — 统筹block分配和回收
    
    职责：
    1. 为新序列分配初始blocks（prefill阶段）
    2. 序列生成新token时按需追加block（decode阶段）
    3. 序列完成时回收所有blocks
    4. 容量查询（调度器决策依据）
    
    与Scheduler的协作：
    - Scheduler调用 can_allocate() 判断能否接纳新请求
    - Scheduler调度Prefill时调用 allocate_for_sequence()
    - 每步Decode后调用 append_token() 更新block状态
    - 请求完成/抢占时调用 free_sequence() 回收资源
    """

    def __init__(self, block_pool: BlockPool, block_size: int = 16):
        self._pool = block_pool
        self._block_size = block_size
        self._seq_tables: Dict[str, BlockTable] = {}

    def allocate_for_sequence(self, seq_id: int, num_initial_tokens: int = 0) -> BlockTable:
        """为新序列分配block table
        
        如果有初始tokens（prefill），计算需要多少个block并一次分配。
        
        Args:
            seq_id: 序列唯一标识
            num_initial_tokens: 初始token数（prompt长度）
            
        Returns:
            分配好的BlockTable
            
        Raises:
            RuntimeError: 显存不足无法分配
        """
        table = BlockTable(self._block_size)
        
        if num_initial_tokens > 0:
            # 计算需要多少个block
            num_blocks_needed = (num_initial_tokens + self._block_size - 1) // self._block_size
            
            for i in range(num_blocks_needed):
                block = self._pool.allocate()
                if block is None:
                    # 分配失败，回滚已分配的blocks
                    for b in table.blocks:
                        self._pool.free(b)
                    raise RuntimeError(
                        f"OOM: Cannot allocate {num_blocks_needed} blocks for seq {seq_id}. "
                        f"Free: {self._pool.num_free_blocks}"
                    )
                
                # 计算此block中的token数
                tokens_in_block = min(
                    self._block_size,
                    num_initial_tokens - i * self._block_size
                )
                block.token_count = tokens_in_block
                block.is_full = (tokens_in_block == self._block_size)
                table.append_block(block)
        
        self._seq_tables[seq_id] = table
        return table

    def append_token(self, seq_id: int) -> bool:
        """序列生成一个新token
        
        检查当前最后一个block是否已满:
        - 未满: token_count++ 即可
        - 已满: 需要分配新block
        
        Args:
            seq_id: 序列标识
            
        Returns:
            True=成功，False=显存不足
        """
        table = self._seq_tables.get(seq_id)
        if table is None:
            return False
        
        last_block = table.last_block
        
        if last_block is None:
            block = self._pool.allocate()
            if block is None:
                return False
            block.token_count = 1
            block.is_full = (self._block_size == 1)
            table.append_block(block)
            return True
        
        if not last_block.is_full:
            last_block.token_count += 1
            last_block.is_full = (last_block.token_count == self._block_size)
            return True
        
        new_block = self._pool.allocate()
        if new_block is None:
            return False  # OOM
        new_block.token_count = 1
        new_block.is_full = (self._block_size == 1)
        table.append_block(new_block)
        return True

    def free_sequence(self, seq_id: int):
        """释放序列的所有blocks
        
        序列结束（生成完毕/被抢占）时调用，归还所有物理blocks到池中。
        """
        table = self._seq_tables.pop(seq_id, None)
        if table is None:
            return
        
        for block in table.blocks:
            self._pool.free(block)

    def can_allocate(self, num_tokens: int) -> bool:
        """检查是否能容纳指定token数的新序列
        
        调度器在决定是否接纳新请求前调用此方法。
        保留1个block的余量（防止decode阶段OOM）。
        """
        num_blocks_needed = (num_tokens + self._block_size - 1) // self._block_size
        # 预留1个block用于decode扩展
        return self._pool.num_free_blocks >= num_blocks_needed + 1

    def get_block_table(self, seq_id: int) -> List[int]:
        """获取序列的block table（物理block id列表）
        
        用于构造attention kernel的输入。
        """
        table = self._seq_tables.get(seq_id)
        if table is None:
            return []
        return table.get_physical_block_ids()

    def get_seq_token_count(self, seq_id: int) -> int:
        """获取序列当前的token总数"""
        table = self._seq_tables.get(seq_id)
        if table is None:
            return 0
        return table.num_tokens

    def get_stats(self) -> dict:
        """获取KV Cache使用统计
        
        用于监控和调试，包括:
        - 总block数、已用数、空闲数
        - 利用率
        - 活跃序列数
        """
        return {
            "total_blocks": self._pool.num_blocks,
            "used_blocks": self._pool.num_used_blocks,
            "free_blocks": self._pool.num_free_blocks,
            "utilization": f"{self._pool.utilization:.1%}",
            "active_sequences": len(self._seq_tables),
            "block_size": self._block_size,
            "memory_bytes": (
                self._pool.num_blocks * self._block_size * 2 *
                self._pool.num_layers * self._pool.num_kv_heads *
                self._pool.head_dim * torch.tensor([], dtype=self._pool.dtype).element_size()
            ),
        }
