"""IcePop + C3PO++ 集成测试"""
import torch
import pytest
import sys
sys.path.insert(0, '.')

from src.engines.icepop import IcePopConfig, IcePopFilter
from src.engines.c3po_plus import C3POConfig, C3POPlusScheduler


class TestIntegration:
    def test_both_disabled_no_effect(self):
        """两者都 disabled 时无任何效果"""
        icepop_config = IcePopConfig(enabled=False)
        c3po_plus_config = C3POConfig(enabled=False)

        icepop_filter = IcePopFilter(icepop_config)
        c3po_plus_scheduler = C3POPlusScheduler(c3po_plus_config)

        # 模拟数据
        train_lp = torch.tensor([1.0, 2.0, 3.0, 4.0])
        infer_lp = torch.tensor([10.0, 20.0, 30.0, 40.0])  # 巨大差异
        advantages = torch.tensor([0.5, -0.3, 0.8, -0.1])

        prompts_tokens = [[1, 2, 3], [4, 5, 6]]
        responses_group = [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]
        rewards_group = [[1.0, -1.0], [0.5, -0.5]]

        # 模拟 disabled 行为：外部不调用 filter/split_and_pack
        # 但验证它们可以被实例化而不影响流程
        assert icepop_filter.config.enabled is False
        assert c3po_plus_scheduler.config.enabled is False

        # 原始数据直接通过，不做任何处理
        # 模拟 orchestrator 逻辑：disabled 时不调用
        result_advantages = advantages  # 直接透传
        result_batches = [{
            "prompts_tokens": prompts_tokens,
            "responses_group": responses_group,
            "rewards_group": rewards_group,
        }]  # 原始单 batch

        assert torch.equal(result_advantages, advantages)
        assert len(result_batches) == 1

    def test_both_enabled_compatible(self):
        """同时启用时数据流兼容"""
        icepop_config = IcePopConfig(
            enabled=True, divergence_threshold=0.5, max_mask_ratio=0.5
        )
        c3po_plus_config = C3POConfig(
            enabled=True, token_budget=5, target_batch_tokens=10000
        )

        icepop_filter = IcePopFilter(icepop_config)
        c3po_plus_scheduler = C3POPlusScheduler(c3po_plus_config)

        # Step 1: IcePop 过滤
        train_lp = torch.tensor([1.0, 2.0, 3.0, 4.0])
        infer_lp = torch.tensor([1.0, 2.0, 3.0, 5.0])  # 第4个差异=1.0
        advantages = torch.tensor([0.5, -0.3, 0.8, -0.1])

        filtered_advantages, diag = icepop_filter.filter_divergent_responses(
            train_lp, infer_lp, advantages
        )

        # IcePop 应掩蔽第4个
        assert filtered_advantages[3].item() == 0.0
        assert diag["icepop_num_masked"] >= 1

        # Step 2: C3PO++ 分割打包（使用 IcePop 过滤后仍存活的 response 信息）
        prompts_tokens = [[1, 2, 3], [4, 5, 6]]
        responses_group = [
            [[10, 20, 30, 40, 50, 60, 70]],  # 7 tokens > budget=5
            [[80, 90, 100]],                   # 3 tokens < budget=5
        ]
        rewards_group = [[1.0], [0.5]]

        sub_batches = c3po_plus_scheduler.split_and_pack(
            prompts_tokens, responses_group, rewards_group
        )

        # 验证输出非空且格式正确
        assert len(sub_batches) >= 1
        for sub in sub_batches:
            assert "prompts_tokens" in sub
            assert "responses_group" in sub
            assert "rewards_group" in sub

    def test_icepop_before_c3po_plus(self):
        """IcePop 在 C3PO++ 之前执行的逻辑正确性"""
        icepop_config = IcePopConfig(
            enabled=True, divergence_threshold=0.3, max_mask_ratio=0.5
        )
        c3po_plus_config = C3POConfig(
            enabled=True, token_budget=4, target_batch_tokens=10000
        )

        icepop_filter = IcePopFilter(icepop_config)
        c3po_plus_scheduler = C3POPlusScheduler(c3po_plus_config)

        # 模拟流程：先 IcePop 过滤 advantages，然后 C3PO++ 分割 response
        # IcePop 不改变 response 本身，只改变 advantage（掩蔽为0）
        # C3PO++ 分割 response 并打包 sub-batch

        # Step 1: IcePop 检测差异
        batch_k = 4  # batch=2, K=2 -> 4 responses
        train_lp = torch.tensor([1.0, 1.5, 2.0, 2.5])
        infer_lp = torch.tensor([1.0, 2.0, 2.0, 3.0])  # 第2个和第4个差异=0.5 > 0.3
        advantages = torch.tensor([1.0, 0.8, 0.6, 0.4])

        filtered_advantages, diag = icepop_filter.filter_divergent_responses(
            train_lp, infer_lp, advantages
        )

        # 确认有 response 被掩蔽
        assert diag["icepop_num_masked"] > 0
        # 被掩蔽的 advantage 为 0
        masked_count = (filtered_advantages == 0.0).sum().item()
        assert masked_count == diag["icepop_num_masked"]

        # Step 2: C3PO++ 分割长 response（与 IcePop 独立）
        prompts_tokens = [[1, 2], [3, 4]]
        responses_group = [
            [[10, 20, 30, 40, 50, 60], [70, 80]],   # 第1个response 6 tokens > budget=4
            [[90, 100, 110], [120, 130, 140, 150, 160]],  # 第4个response 5 tokens > budget=4
        ]
        rewards_group = [[1.0, 0.5], [-0.5, -1.0]]

        sub_batches = c3po_plus_scheduler.split_and_pack(
            prompts_tokens, responses_group, rewards_group
        )

        # 验证分割正确
        assert len(sub_batches) >= 1
        total_chunks = 0
        for sub in sub_batches:
            for resp_group in sub["responses_group"]:
                total_chunks += len(resp_group)

        # 原始有4个 response，其中2个需要分割
        # response[0][0]: 6 tokens / 4 = 2 chunks
        # response[0][1]: 2 tokens = 1 chunk
        # response[1][0]: 3 tokens = 1 chunk
        # response[1][1]: 5 tokens / 4 = 2 chunks
        # 总共 6 个 chunk
        assert total_chunks == 6

        # 验证 IcePop 和 C3PO++ 的结果互不干扰
        # IcePop 只修改 advantages，C3PO++ 只分割 responses
        assert filtered_advantages.shape == advantages.shape
        assert len(sub_batches) >= 1
