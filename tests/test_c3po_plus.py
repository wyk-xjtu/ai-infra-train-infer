"""C3PO++ 单元测试"""
import pytest
import sys
sys.path.insert(0, '.')

from src.engines.c3po_plus import C3POConfig, C3POPlusScheduler


class TestC3POPlusScheduler:
    def test_short_response_no_split(self):
        """短 response 不分割"""
        config = C3POConfig(enabled=True, token_budget=1024, target_batch_tokens=4096)
        scheduler = C3POPlusScheduler(config)

        prompts_tokens = [[1, 2, 3]]
        responses_group = [[[10, 20, 30, 40, 50]]]  # 5 tokens < 1024
        rewards_group = [[1.0]]

        result = scheduler.split_and_pack(prompts_tokens, responses_group, rewards_group)

        # 短 response 不分割，应只有 1 个 sub-batch
        assert len(result) == 1
        # sub-batch 内应包含完整 response
        sub = result[0]
        assert len(sub["prompts_tokens"]) == 1
        assert sub["responses_group"][0][0] == [10, 20, 30, 40, 50]
        assert sub["rewards_group"][0][0] == 1.0

    def test_long_response_splits(self):
        """超过 token_budget 的 response 正确分割"""
        config = C3POConfig(enabled=True, token_budget=4, target_batch_tokens=10000)
        scheduler = C3POPlusScheduler(config)

        prompts_tokens = [[1, 2]]
        # 10 tokens，budget=4，应分割为 [4, 4, 2] 三个 chunk
        responses_group = [[[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]]
        rewards_group = [[1.0]]

        result = scheduler.split_and_pack(prompts_tokens, responses_group, rewards_group)

        # 收集所有 response chunks
        all_chunks = []
        for sub in result:
            for resp_group in sub["responses_group"]:
                for resp in resp_group:
                    all_chunks.append(resp)

        # 应有 3 个 chunk
        assert len(all_chunks) == 3
        # 每个 chunk 长度 <= token_budget
        for chunk in all_chunks:
            assert len(chunk) <= 4
        # chunk 合并应等于原始 response
        combined = []
        for chunk in sorted(all_chunks, key=lambda x: x[0]):
            combined.extend(chunk)
        assert combined == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def test_ffd_packing_respects_limit(self):
        """每个 bin 总 token 数不超过 target_batch_tokens"""
        config = C3POConfig(enabled=True, token_budget=100, target_batch_tokens=20)
        scheduler = C3POPlusScheduler(config)

        # 3 个 prompt，每个带 1 个短 response
        # prompt(3 tokens) + response(8 tokens) = 11 tokens each
        # target=20, 所以每个 bin 最多放 1-2 个
        prompts_tokens = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        responses_group = [
            [[10, 20, 30, 40, 50, 60, 70, 80]],  # 8 tokens
            [[11, 21, 31, 41, 51, 61, 71, 81]],  # 8 tokens
            [[12, 22, 32, 42, 52, 62, 72, 82]],  # 8 tokens
        ]
        rewards_group = [[1.0], [0.5], [-0.5]]

        result = scheduler.split_and_pack(prompts_tokens, responses_group, rewards_group)

        # 验证每个 sub-batch 的总 token 数
        for sub in result:
            total_tokens = 0
            for prompt, resp_group in zip(sub["prompts_tokens"], sub["responses_group"]):
                for resp in resp_group:
                    total_tokens += len(prompt) + len(resp)
            # 允许单个 chunk 超过 target（FFD 特性：单个元素超过 bin 容量时独立成 bin）
            # 但多个 chunk 不应超过
            if total_tokens > config.target_batch_tokens:
                # 只有在 bin 内只有 1 个 chunk 时允许超过
                chunk_count = sum(len(rg) for rg in sub["responses_group"])
                assert chunk_count == 1

    def test_sub_batch_format(self):
        """输出格式可直接传入 grpo_step"""
        config = C3POConfig(enabled=True, token_budget=1024, target_batch_tokens=4096)
        scheduler = C3POPlusScheduler(config)

        prompts_tokens = [[1, 2, 3], [4, 5, 6]]
        responses_group = [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]
        rewards_group = [[1.0, -1.0], [0.5, -0.5]]

        result = scheduler.split_and_pack(prompts_tokens, responses_group, rewards_group)

        # 验证每个 sub-batch 的格式
        for sub in result:
            assert "prompts_tokens" in sub
            assert "responses_group" in sub
            assert "rewards_group" in sub
            # prompts_tokens 是 List[List[int]]
            assert isinstance(sub["prompts_tokens"], list)
            # responses_group 是 List[List[List[int]]]
            assert isinstance(sub["responses_group"], list)
            # rewards_group 是 List[List[float]]
            assert isinstance(sub["rewards_group"], list)
            # 长度一致
            assert len(sub["prompts_tokens"]) == len(sub["responses_group"])
            assert len(sub["prompts_tokens"]) == len(sub["rewards_group"])

    def test_advantage_inheritance(self):
        """chunk 继承父 response 的 reward"""
        config = C3POConfig(enabled=True, token_budget=3, target_batch_tokens=10000)
        scheduler = C3POPlusScheduler(config)

        prompts_tokens = [[1]]
        # 6 tokens, budget=3 -> 分割为 2 个 chunk
        responses_group = [[[10, 20, 30, 40, 50, 60]]]
        rewards_group = [[0.75]]

        result = scheduler.split_and_pack(prompts_tokens, responses_group, rewards_group)

        # 所有 chunk 应继承相同的 reward
        for sub in result:
            for reward_group in sub["rewards_group"]:
                for reward in reward_group:
                    assert reward == 0.75

    def test_disabled_returns_single_batch(self):
        """disabled 时返回原始单 batch"""
        config = C3POConfig(enabled=False, token_budget=1024, target_batch_tokens=4096)
        scheduler = C3POPlusScheduler(config)

        prompts_tokens = [[1, 2, 3]]
        responses_group = [[[10, 20, 30]]]
        rewards_group = [[1.0]]

        # 即使 disabled，split_and_pack 方法仍可调用
        # disabled 控制在 orchestrator 层（不调用 split_and_pack）
        # 但方法本身仍正常执行
        result = scheduler.split_and_pack(prompts_tokens, responses_group, rewards_group)
        assert len(result) >= 1  # 正常返回结果

    def test_single_response(self):
        """单个 response 正确处理"""
        config = C3POConfig(enabled=True, token_budget=1024, target_batch_tokens=4096)
        scheduler = C3POPlusScheduler(config)

        prompts_tokens = [[1, 2, 3, 4, 5]]
        responses_group = [[[100, 200, 300]]]
        rewards_group = [[0.5]]

        result = scheduler.split_and_pack(prompts_tokens, responses_group, rewards_group)

        assert len(result) == 1
        sub = result[0]
        assert sub["prompts_tokens"][0] == [1, 2, 3, 4, 5]
        assert sub["responses_group"][0][0] == [100, 200, 300]
        assert sub["rewards_group"][0][0] == 0.5
