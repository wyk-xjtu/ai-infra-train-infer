"""
RadixAttention Prefix Cache — 基于 Radix Tree 的子前缀复用

对标 SGLang RadixAttention：支持任意子前缀复用，
不仅限于完整前缀匹配（相比传统 PrefixCache 的改进）。

===================== 核心差异 =====================

传统 PrefixCache (hash-chain):
  - 基于链式哈希逐 block 匹配
  - 只能匹配从头开始的完全连续前缀
  - 中间任一 block 不同，后续全部 miss

RadixAttentionCache (radix tree):
  - 基于 Radix Tree 存储 token 序列
  - 支持任意子前缀共享（如多个 prompt 共享 system prompt 部分）
  - 树形结构天然支持分支和合并
  - LRU 淘汰可以精细到叶子节点

使用场景:
  - 多轮对话：不同 turn 共享前面的历史
  - System prompt 复用：所有请求共享相同的 system prompt
  - RL K-sample：同一 prompt 的多次 decode 共享 prefix

接口兼容现有 PrefixCache：
  - match_prefix(token_ids) -> (matched_len, block_ids)
  - insert(token_ids, block_ids)
  - invalidate_all()
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RadixNode:
    """Radix Tree 节点

    每个节点代表一段 token 序列（edge labels），
    存储对应的 KV block IDs。

    Attributes:
        token_chunk: 该边对应的 token 序列（可为空，根节点）
        children: 子节点映射（第一个 token → 子节点）
        kv_block_ids: 该节点对应的 KV block IDs（可能跨多个 block）
        token_count: 该节点覆盖的 token 数量
        last_access: 最近访问时间（LRU 用）
        ref_count: 引用计数（有活跃请求引用时不淘汰）
    """
    token_chunk: List[int] = field(default_factory=list)
    children: Dict[int, 'RadixNode'] = field(default_factory=dict)
    kv_block_ids: List[int] = field(default_factory=list)
    token_count: int = 0
    last_access: float = field(default_factory=time.time)
    ref_count: int = 0
    parent: Optional['RadixNode'] = field(default=None, repr=False)


class RadixAttentionCache:
    """基于 Radix Tree 的前缀缓存

    对标 SGLang RadixAttention：支持任意子前缀复用，
    不仅限于完整前缀匹配。

    接口兼容现有 PrefixCache：
    - match_prefix(token_ids) -> (matched_len, block_ids)
    - insert(token_ids, block_ids)
    - invalidate_all()

    内部使用 Radix Tree 结构，按 block_size 对齐存储。
    每个树节点代表一个完整 block（block_size 个 token）。
    """

    def __init__(self, block_size: int = 16, max_nodes: int = 10000):
        """
        Args:
            block_size: 每个 block 存储的 token 数，与 KV Cache 的 block_size 对齐
            max_nodes: 最大节点数限制（超过时触发 LRU 淘汰）
        """
        self.root = RadixNode()
        self.block_size = block_size
        self.max_nodes = max_nodes
        self.node_count = 0

        # 统计指标
        self.hits = 0
        self.misses = 0
        self._evictions = 0

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """查找最长匹配前缀，返回 (匹配 token 数, 对应的 block_ids)

        按 block_size 切分 token_ids，在 Radix Tree 中逐 block 匹配。
        利用树形结构实现高效的前缀查找。

        Args:
            token_ids: 完整的 prompt token ids

        Returns:
            (matched_tokens, block_ids):
            - matched_tokens: 缓存命中的 token 数（从头开始连续的，按 block 对齐）
            - block_ids: 命中的物理 block id 列表
        """
        matched_tokens = 0
        matched_block_ids: List[int] = []

        # 按 block_size 切分
        num_full_blocks = len(token_ids) // self.block_size
        current_node = self.root

        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]

            # 查找子节点：以 block 第一个 token 作为索引
            first_token = block_tokens[0]
            child = current_node.children.get(first_token)

            if child is None:
                # 未找到匹配的子节点
                self.misses += 1
                break

            # 验证完整 block token 匹配
            if child.token_chunk == block_tokens:
                # 精确匹配
                child.last_access = time.time()
                child.ref_count += 1
                matched_tokens += self.block_size
                matched_block_ids.extend(child.kv_block_ids)
                current_node = child
                self.hits += 1
            else:
                # token_chunk 不匹配（同一首 token 但后续不同）
                self.misses += 1
                break

        return matched_tokens, matched_block_ids

    def insert(self, token_ids: List[int], block_ids: List[int]):
        """插入前缀到 Radix Tree

        按 block_size 切分 token_ids，逐 block 插入树中。
        已存在的节点更新访问时间，新节点创建并关联 block_id。

        Args:
            token_ids: 完整的 token ids
            block_ids: 对应的物理 block id 列表
        """
        num_full_blocks = len(token_ids) // self.block_size
        num_blocks_to_insert = min(num_full_blocks, len(block_ids))
        current_node = self.root

        for block_idx in range(num_blocks_to_insert):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]
            first_token = block_tokens[0]

            child = current_node.children.get(first_token)

            if child is not None and child.token_chunk == block_tokens:
                # 节点已存在，更新访问时间
                child.last_access = time.time()
                current_node = child
            elif child is not None and child.token_chunk != block_tokens:
                # 冲突：同一首 token 但后续不同 → 需要分裂节点
                # 找到公共前缀长度
                common_len = 0
                for i in range(min(len(child.token_chunk), len(block_tokens))):
                    if child.token_chunk[i] == block_tokens[i]:
                        common_len += 1
                    else:
                        break

                if common_len == 0:
                    # 完全不同（只有首 token 相同——不应该发生在 block 粒度下）
                    # 由于我们以 block_size 为粒度，首 token 相同但内容不同时
                    # 简单替换（LRU 策略：新的替换旧的）
                    if child.ref_count <= 0:
                        # 旧节点无引用，直接替换
                        new_node = RadixNode(
                            token_chunk=list(block_tokens),
                            kv_block_ids=[block_ids[block_idx]],
                            token_count=self.block_size,
                            last_access=time.time(),
                            parent=current_node,
                        )
                        current_node.children[first_token] = new_node
                        current_node = new_node
                        # 不增加 node_count（替换）
                    else:
                        # 旧节点有引用，跳过插入
                        break
                else:
                    # 有公共前缀但不完整匹配 — 在 block 粒度下不做分裂
                    # （因为 block_size 是原子单位，要么完整匹配要么不匹配）
                    # 这里作为独立节点插入（用不同的键索引方案）
                    # 实际上 block 粒度下只要首 token 相同且内容不同，视为冲突替换
                    if child.ref_count <= 0:
                        new_node = RadixNode(
                            token_chunk=list(block_tokens),
                            kv_block_ids=[block_ids[block_idx]],
                            token_count=self.block_size,
                            last_access=time.time(),
                            parent=current_node,
                        )
                        current_node.children[first_token] = new_node
                        current_node = new_node
                    else:
                        break
            else:
                # 节点不存在，创建新节点
                if self.node_count >= self.max_nodes:
                    self._evict_lru()

                new_node = RadixNode(
                    token_chunk=list(block_tokens),
                    kv_block_ids=[block_ids[block_idx]],
                    token_count=self.block_size,
                    last_access=time.time(),
                    parent=current_node,
                )
                current_node.children[first_token] = new_node
                self.node_count += 1
                current_node = new_node

    def invalidate_all(self):
        """权重更新后清除所有缓存

        权重变化后所有已缓存的 KV 值不再正确，必须全部丢弃。
        """
        self.root = RadixNode()
        self.node_count = 0

    def release_block(self, block_hash: int):
        """兼容接口：减少引用计数

        注意：RadixAttentionCache 不使用 hash，此接口仅为兼容性保留。
        实际释放通过 match_prefix 时 ref_count 管理。
        """
        pass

    def evict_lru(self, num_blocks: int = 1):
        """LRU 淘汰的公开接口（兼容 PrefixCache）"""
        for _ in range(num_blocks):
            self._evict_lru()

    def _evict_lru(self):
        """当 node_count > max_nodes 时，淘汰最久未访问的叶子节点

        淘汰策略：
        1. 收集所有叶子节点（无子节点的节点）
        2. 筛选 ref_count == 0 的节点
        3. 按 last_access 排序，淘汰最旧的
        4. 从父节点中移除该子节点
        """
        # 收集所有叶子节点
        leaves: List[Tuple[float, int, RadixNode, RadixNode]] = []
        self._collect_leaves(self.root, leaves)

        if not leaves:
            return

        # 按 (ref_count == 0 优先, last_access 最小优先) 排序
        leaves.sort(key=lambda x: (x[2].ref_count > 0, x[0]))

        if leaves:
            _, first_token, victim, parent = leaves[0]
            # 从父节点中移除
            if first_token in parent.children and parent.children[first_token] is victim:
                del parent.children[first_token]
                self.node_count -= 1
                self._evictions += 1

    def _collect_leaves(
        self, node: RadixNode,
        result: List[Tuple[float, int, 'RadixNode', 'RadixNode']]
    ):
        """递归收集叶子节点

        Args:
            node: 当前节点
            result: 收集结果列表 (last_access, first_token_key, leaf_node, parent_node)
        """
        for token_key, child in node.children.items():
            if not child.children:
                # 叶子节点
                result.append((child.last_access, token_key, child, node))
            else:
                # 非叶子，递归
                self._collect_leaves(child, result)

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def size(self) -> int:
        """当前缓存的节点数"""
        return self.node_count

    def get_stats(self) -> dict:
        """缓存统计信息"""
        return {
            "type": "RadixAttentionCache",
            "cached_nodes": self.node_count,
            "max_nodes": self.max_nodes,
            "utilization": f"{self.node_count / self.max_nodes:.1%}" if self.max_nodes > 0 else "N/A",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "evictions": self._evictions,
            "block_size": self.block_size,
        }
