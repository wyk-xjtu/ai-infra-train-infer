"""
C3PO++ 动态 Rollout 分割调度模块

参考 Ring-1T 论文 (arxiv.org/abs/2510.18855) 中的 C3PO++ 技术:
- 将长 rollout 按 token 预算分割成短 chunk
- 使用 First Fit Decreasing (FFD) 贪心装箱算法将 chunk 打包成均衡的 sub-batch
- 每个 sub-batch 独立调用 grpo_step，grpo_step 签名和逻辑不变

通过 C3POConfig.enabled 开关控制，disabled 时代码路径完全不变。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..utils.logger import get_logger

logger = get_logger("engines.c3po_plus")


@dataclass
class C3POConfig:
    """C3PO++ 配置"""
    enabled: bool = False
    token_budget: int = 1024          # 单个 response chunk 的最大 token 数
    target_batch_tokens: int = 4096   # 目标 batch 总 token 数（每个 sub-batch 的总 token 容量）
    packing_strategy: str = "ffd"     # "ffd" (First Fit Decreasing) 贪心打包


@dataclass
class ResponseChunk:
    """分割后的 response chunk"""
    prompt_tokens: List[int]
    response_tokens: List[int]        # 当前 chunk 的 tokens
    response_prefix: List[int]        # 该 chunk 之前的所有 response tokens
    reward: float                     # 继承自父 response 的 reward
    source_idx: Tuple[int, int]       # (batch_idx, k_idx) 来源追踪
    advantage: float = 0.0            # 预计算的 advantage（由外部传入）


class C3POPlusScheduler:
    """C3PO++ 动态 Rollout 分割调度器

    核心流程:
    1. 将超过 token_budget 的 response 分割成多个 chunk
    2. 使用 FFD 贪心算法将 chunk 打包成均衡的 sub-batch
    3. 返回可直接传入 grpo_step 的 sub-batch 参数字典列表
    """

    def __init__(self, config: C3POConfig):
        self.config = config
        logger.info(
            f"C3POPlusScheduler initialized: enabled={config.enabled}, "
            f"token_budget={config.token_budget}, "
            f"target_batch_tokens={config.target_batch_tokens}, "
            f"packing_strategy={config.packing_strategy}"
        )

    def split_and_pack(
        self,
        prompts_tokens: List[List[int]],
        responses_group: List[List[List[int]]],  # [batch, K, seq_len]
        rewards_group: List[List[float]],         # [batch, K]
        advantages_group: List[List[float]] = None,  # [batch, K] 预计算的 advantage
    ) -> List[Dict]:
        """分割 + 打包，返回多个 sub-batch 参数字典列表

        每个字典格式：
        {
            "prompts_tokens": List[List[int]],
            "responses_group": List[List[List[int]]],
            "rewards_group": List[List[float]],
            "precomputed_advantages": List[List[float]],  # 预计算的 advantage
        }
        可直接传入 grpo_step(**sub_batch)

        Args:
            prompts_tokens: [batch_size] 每个 prompt 的 token 列表
            responses_group: [batch_size, K, seq_len] 每个 prompt 对应 K 个 response
            rewards_group: [batch_size, K] 每个 response 的 reward
            advantages_group: [batch_size, K] 预计算的 advantage（可选）

        Returns:
            sub-batch 参数字典列表
        """
        # Step 1: 分割所有 response 为 chunk
        all_chunks = self._split_all_responses(
            prompts_tokens, responses_group, rewards_group, advantages_group
        )

        if not all_chunks:
            logger.warning("C3PO++: No chunks generated, returning empty list")
            return []

        # Step 2: FFD 贪心打包 chunk 到均衡 bin
        bins = self._ffd_pack(all_chunks)

        # Step 3: 将每个 bin 转换为 grpo_step 所需的参数格式
        sub_batches = self._bins_to_sub_batches(bins)

        logger.info(
            f"C3PO++ split_and_pack: {len(all_chunks)} chunks -> "
            f"{len(sub_batches)} sub-batches"
        )

        return sub_batches

    def _split_response(self, response_tokens: List[int], budget: int) -> List[List[int]]:
        """将超长 response 按 budget 分割为多个 chunk

        Args:
            response_tokens: 原始 response token 序列
            budget: 每个 chunk 的最大 token 数

        Returns:
            分割后的 chunk 列表
        """
        if len(response_tokens) <= budget:
            return [response_tokens]
        chunks = []
        for i in range(0, len(response_tokens), budget):
            chunks.append(response_tokens[i:i + budget])
        return chunks

    def _split_all_responses(
        self,
        prompts_tokens: List[List[int]],
        responses_group: List[List[List[int]]],
        rewards_group: List[List[float]],
        advantages_group: List[List[float]] = None,
    ) -> List[ResponseChunk]:
        """对所有 response 执行分割，生成 ResponseChunk 列表

        Args:
            prompts_tokens: [batch_size] prompt token 列表
            responses_group: [batch_size, K, seq_len]
            rewards_group: [batch_size, K]
            advantages_group: [batch_size, K] 预计算的 advantage（可选）

        Returns:
            所有分割后的 ResponseChunk 列表
        """
        all_chunks: List[ResponseChunk] = []
        budget = self.config.token_budget

        for batch_idx, (prompt_tokens, response_group, reward_group) in enumerate(
            zip(prompts_tokens, responses_group, rewards_group)
        ):
            adv_group = advantages_group[batch_idx] if advantages_group is not None else None
            for k_idx, (response_tokens, reward) in enumerate(
                zip(response_group, reward_group)
            ):
                advantage = adv_group[k_idx] if adv_group is not None else 0.0
                # 分割 response，为每个 chunk 记录前缀
                for chunk_start in range(0, len(response_tokens), budget):
                    chunk_end = min(chunk_start + budget, len(response_tokens))
                    all_chunks.append(ResponseChunk(
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens[chunk_start:chunk_end],
                        response_prefix=response_tokens[:chunk_start],
                        reward=reward,
                        source_idx=(batch_idx, k_idx),
                        advantage=advantage,
                    ))

        return all_chunks

    def _ffd_pack(self, chunks: List[ResponseChunk]) -> List[List[ResponseChunk]]:
        """First Fit Decreasing 贪心装箱算法

        1. 对所有 chunk 按 (prompt_len + chunk_len) 降序排列
        2. 逐个放入第一个能容纳的 bin（bin 内总 token 数 <= target_batch_tokens）
        3. 如果没有合适的 bin，创建新 bin

        Args:
            chunks: 所有待打包的 ResponseChunk

        Returns:
            分好的 bin 列表，每个 bin 是一组 ResponseChunk
        """
        target = self.config.target_batch_tokens

        # 按 (prompt_len + prefix_len + chunk_len) 降序排列
        sorted_chunks = sorted(
            chunks,
            key=lambda c: len(c.prompt_tokens) + len(c.response_prefix) + len(c.response_tokens),
            reverse=True,
        )

        bins: List[List[ResponseChunk]] = []
        bin_sizes: List[int] = []  # 每个 bin 当前的总 token 数

        for chunk in sorted_chunks:
            chunk_size = len(chunk.prompt_tokens) + len(chunk.response_prefix) + len(chunk.response_tokens)

            # 单个 chunk 超过 target 时告警
            if chunk_size > target:
                logger.warning(
                    "C3PO++ FFD: single chunk_size=%d exceeds target_batch_tokens=%d; "
                    "creating dedicated bin (capacity constraint violated).",
                    chunk_size, target,
                )
                bins.append([chunk])
                bin_sizes.append(chunk_size)
                continue

            # 正常 FFD 逻辑：寻找第一个能容纳此 chunk 的 bin
            placed = False
            for bin_idx, current_size in enumerate(bin_sizes):
                if current_size + chunk_size <= target:
                    bins[bin_idx].append(chunk)
                    bin_sizes[bin_idx] += chunk_size
                    placed = True
                    break

            # 没有合适的 bin，创建新 bin
            if not placed:
                bins.append([chunk])
                bin_sizes.append(chunk_size)

        return bins

    def _bins_to_sub_batches(self, bins: List[List[ResponseChunk]]) -> List[Dict]:
        """将每个 bin 转换为 grpo_step 所需的参数格式

        sub-batch 格式：每个 sub-batch 中，一个 prompt 对应一组 responses。
        来自同一个 prompt 的 chunk 归入同一个 prompt 下。

        重要：grpo_step 假设每个 prompt 拥有相同数量的 responses (K)。
        因此，需要在每个 bin 内按 response 数量再分组，保证同一 sub-batch
        中所有 prompt 的 responses 数量一致。

        完整输入 = prompt + response_prefix（作为 prompt 的一部分传给 grpo_step）
        responses_group 中只放 current_chunk（这是需要计算 loss 的部分）

        Args:
            bins: FFD 打包后的 bin 列表

        Returns:
            sub-batch 参数字典列表
        """
        sub_batches: List[Dict] = []

        for bin_chunks in bins:
            # 按 prompt_tokens (使用 id 作为 key) 分组
            # 由于同一个 prompt 的 tokens 列表是同一个对象引用，
            # 使用 id(prompt_tokens) 作为 prompt 标识来分组
            prompt_groups: Dict[int, Dict] = {}

            for chunk in bin_chunks:
                # 使用 prompt 在原始 batch 中的 index 作为 key
                # 这样同一 prompt 下的不同 chunk 会被归到一起
                prompt_key = id(chunk.prompt_tokens)

                if prompt_key not in prompt_groups:
                    prompt_groups[prompt_key] = {
                        "prompt_tokens": chunk.prompt_tokens,
                        "responses": [],
                        "rewards": [],
                        "advantages": [],
                        "prefixes": [],
                    }

                prompt_groups[prompt_key]["responses"].append(chunk.response_tokens)
                prompt_groups[prompt_key]["rewards"].append(chunk.reward)
                prompt_groups[prompt_key]["advantages"].append(chunk.advantage)
                prompt_groups[prompt_key]["prefixes"].append(chunk.response_prefix)

            # 按 response 数量 (K) 再分组，确保同一 sub-batch 内所有
            # prompt 有相同数量的 responses，满足 grpo_step 的假设
            k_groups: Dict[int, List[Dict]] = {}
            for group in prompt_groups.values():
                k = len(group["responses"])
                if k not in k_groups:
                    k_groups[k] = []
                k_groups[k].append(group)

            # 每个 K 值生成一个独立的 sub-batch
            for k_value, groups in k_groups.items():
                prompts_tokens: List[List[int]] = []
                responses_group: List[List[List[int]]] = []
                rewards_group: List[List[float]] = []
                advantages_list: List[List[float]] = []

                for group in groups:
                    # 完整输入 = prompt + response_prefix（每个 chunk 的 prefix 可能不同）
                    # 由于同一 prompt 下不同 chunk 有不同 prefix，需要逐个处理
                    # 每个 chunk 作为独立的 "prompt"（含prefix）+ response
                    for i, (resp, prefix, reward, adv) in enumerate(zip(
                        group["responses"], group["prefixes"],
                        group["rewards"], group["advantages"]
                    )):
                        full_prompt = group["prompt_tokens"] + prefix
                        prompts_tokens.append(full_prompt)
                        responses_group.append([resp])
                        rewards_group.append([reward])
                        advantages_list.append([adv])

                sub_batches.append({
                    "prompts_tokens": prompts_tokens,
                    "responses_group": responses_group,
                    "rewards_group": rewards_group,
                    "precomputed_advantages": advantages_list,
                })

        return sub_batches
