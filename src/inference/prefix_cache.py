"""
Prefix Caching（前缀缓存） — RL 场景的关键优化

===================== 核心思想 =====================

RL 训练中的问题:
  GRPO/PPO 每一步需要对同一batch的prompt生成多个回复（K=4或更多）:
  - Prompt A → Response A1, A2, A3, A4
  - Prompt B → Response B1, B2, B3, B4
  
  所有 Response Ax 共享完全相同的 Prompt A 的 KV Cache。
  如果每次都重新计算，浪费巨大。

Prefix Cache 的解法:
  缓存已计算好的 KV Cache blocks:
  1. 首次计算 Prompt A 的 KV，将 blocks 放入 cache
  2. 后续 A2, A3, A4 直接复用已缓存的 blocks（ref_count++）
  3. 只需计算各自不同的 response 部分

实现原理:
  - 对每个 block 内的 token_ids 计算哈希（xxhash64）
  - 使用链式哈希确保只有完全相同的前缀才匹配:
    hash(block_i) = xxhash64(prev_block_hash || token_ids_in_block)
  - 新请求逐 block 匹配: 匹配 → 复用; 不匹配 → 从此处开始 prefill

RL 场景收益量化:
  设 system_prompt = 500 tokens, batch = 4 prompts × K=4 samples = 16 次推理
  - 无 Prefix Cache: 16 × 500 = 8000 prefill tokens
  - 有 Prefix Cache: 4 × 500 = 2000 tokens (仅每个prompt首次计算)
  - 加速比: 4x; 若 system_prompt 相同则仅 500 tokens → 16x
"""
import time
from typing import Dict, List, Optional, Tuple

try:
    import xxhash
    _XXHASH_AVAILABLE = True
except ImportError:
    import hashlib
    _XXHASH_AVAILABLE = False


class CachedBlock:
    """缓存的KV Block元数据"""

    __slots__ = ['hash_value', 'block_id', 'ref_count', 'last_access_time', 'token_ids']

    def __init__(self, hash_value: int, block_id: int, token_ids: Tuple[int, ...]):
        self.hash_value = hash_value
        self.block_id = block_id          # 对应的物理 block id
        self.ref_count = 0                # 当前引用此 cached block 的序列数
        self.last_access_time = time.time()
        self.token_ids = token_ids        # 用于校验（可选的防冲突措施）


class PrefixCache:
    """前缀缓存管理器
    
    对外接口:
    - compute_block_hash(): 计算 block 的链式哈希
    - match_prefix(): 给定 token_ids, 匹配缓存中的最长前缀
    - insert(): 将新计算的 blocks 插入缓存
    - evict_lru(): LRU 淘汰
    - invalidate_all(): 权重更新后清空缓存
    """

    def __init__(self, block_size: int = 16, max_cached_blocks: int = 1024):
        self._cache: Dict[int, CachedBlock] = {}  # hash_value → CachedBlock
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        
        # 统计指标
        self.hits = 0       # 命中的 block 总数
        self.misses = 0     # 未命中的 block 总数
        self._evictions = 0 # 淘汰次数

    def compute_block_hash(self, token_ids: List[int], prev_hash: int = 0) -> int:
        """计算block的链式哈希值
        
        链式哈希确保前缀匹配的正确性:
          hash(block_0) = H(token_ids_0)
          hash(block_1) = H(hash(block_0) || token_ids_1)
          hash(block_i) = H(hash(block_{i-1}) || token_ids_i)
        
        这保证了只有从头开始完全相同的前缀序列才能匹配。
        如果中间某个 block 不同，后续所有 hash 都不同。
        
        Args:
            token_ids: 当前 block 中的 token ids (长度 <= block_size)
            prev_hash: 前一个 block 的哈希值（首个 block 传 0）
            
        Returns:
            当前 block 的 64-bit 哈希值
        """
        if _XXHASH_AVAILABLE:
            h = xxhash.xxh64()
            # 链式: 将前一个block的hash作为seed的一部分
            h.update(prev_hash.to_bytes(8, byteorder='little', signed=False))
            # 当前block的内容
            for tid in token_ids:
                h.update(tid.to_bytes(4, byteorder='little', signed=False))
            return h.intdigest()
        else:
            data = prev_hash.to_bytes(8, byteorder='little', signed=False)
            for tid in token_ids:
                data += tid.to_bytes(4, byteorder='little', signed=False)
            return int(hashlib.sha256(data).hexdigest()[:16], 16)

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """匹配前缀，返回匹配的token数和对应的物理block ids
        
        逐 block 计算 hash 并查找 cache:
        - 匹配成功: 记录 block_id, 继续下一个 block
        - 匹配失败: 停止（前缀匹配的本质——一旦断裂就无法继续）
        
        Args:
            token_ids: 完整的 prompt token ids
            
        Returns:
            (matched_tokens, physical_block_ids):
            - matched_tokens: 缓存命中的 token 数（从头开始连续的）
            - physical_block_ids: 命中的物理 block id 列表
        """
        matched_tokens = 0
        matched_block_ids: List[int] = []
        prev_hash = 0

        # 按 block_size 切分 token_ids，逐 block 匹配
        num_full_blocks = len(token_ids) // self.block_size
        
        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]
            
            # 计算此 block 的哈希
            block_hash = self.compute_block_hash(block_tokens, prev_hash)
            
            cached = self._cache.get(block_hash)
            if cached is not None and tuple(block_tokens) == cached.token_ids:
                cached.last_access_time = time.time()
                cached.ref_count += 1
                matched_tokens += self.block_size
                matched_block_ids.append(cached.block_id)
                prev_hash = block_hash
                self.hits += 1
            else:
                self.misses += 1
                break
        else:
            # 所有full blocks都命中了，检查是否还有残余不足一个block的部分（不缓存）
            pass

        return matched_tokens, matched_block_ids

    def insert(self, token_ids: List[int], block_ids: List[int]):
        """将新计算的blocks插入缓存
        
        在 prefill 完成后调用:将计算好的 blocks 注册到缓存中。
        后续相同前缀的请求可以复用这些 blocks。
        
        Args:
            token_ids: 完整的 token ids（用于计算各 block 的 hash）
            block_ids: 对应的物理 block id 列表
        """
        prev_hash = 0
        num_full_blocks = len(token_ids) // self.block_size
        num_blocks_to_cache = min(num_full_blocks, len(block_ids))
        
        for block_idx in range(num_blocks_to_cache):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]
            
            block_hash = self.compute_block_hash(block_tokens, prev_hash)
            
            if len(self._cache) >= self.max_cached_blocks and block_hash not in self._cache:
                self.evict_lru(num_blocks=1)
            
            # 插入或更新缓存条目
            if block_hash not in self._cache:
                self._cache[block_hash] = CachedBlock(
                    hash_value=block_hash,
                    block_id=block_ids[block_idx],
                    token_ids=tuple(block_tokens),
                )
            else:
                # 已存在则更新访问时间
                self._cache[block_hash].last_access_time = time.time()
            
            prev_hash = block_hash

    def evict_lru(self, num_blocks: int = 1):
        """LRU淘汰: 移除最久未访问且未被引用的blocks
        
        淘汰优先级:
        1. ref_count == 0 的 blocks (无人引用) 中 last_access_time 最小的
        2. 如果所有 blocks 都有引用, 淘汰 last_access_time 最小的
        """
        evicted = 0
        while evicted < num_blocks and self._cache:
            # 优先淘汰无引用的
            unreferenced = [
                (h, cb) for h, cb in self._cache.items() if cb.ref_count <= 0
            ]
            
            if unreferenced:
                unreferenced.sort(key=lambda x: x[1].last_access_time)
                victim_hash = unreferenced[0][0]
            else:
                all_entries = list(self._cache.items())
                all_entries.sort(key=lambda x: x[1].last_access_time)
                victim_hash = all_entries[0][0]
            
            del self._cache[victim_hash]
            evicted += 1
            self._evictions += 1

    def release_block(self, block_hash: int):
        """减少某个cached block的引用计数（序列释放时调用）"""
        cached = self._cache.get(block_hash)
        if cached is not None:
            cached.ref_count = max(0, cached.ref_count - 1)

    def invalidate_all(self):
        """清空所有缓存
        
        权重更新后必须调用！
        权重变了 → 所有已缓存的 KV 值不再正确 → 必须全部丢弃。
        """
        self._cache.clear()
        # 保留统计数据不清零，方便观察历史表现

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def size(self) -> int:
        """当前缓存的block数"""
        return len(self._cache)

    def get_stats(self) -> dict:
        """缓存统计"""
        return {
            "cached_blocks": len(self._cache),
            "max_cached_blocks": self.max_cached_blocks,
            "utilization": f"{len(self._cache) / self.max_cached_blocks:.1%}" if self.max_cached_blocks > 0 else "N/A",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "evictions": self._evictions,
            "block_size": self.block_size,
        }
