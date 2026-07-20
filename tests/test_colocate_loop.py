"""
测试Colocate完整循环（集成测试）

验证状态机转换正确性（使用mock组件）:
INIT → INFERRING → REWARDING → TRAINING → SLEEPING → SYNCING → INFERRING → ...

运行:
    python tests/test_colocate_loop.py
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
sys.path.insert(0, '.')

from src.orchestrator.colocate_orchestrator import (
    ColocateOrchestrator,
    ColocateConfig,
    OrchestratorState,
    IterationMetrics,
)


class MockVLLMClient:
    """Mock VLLMClient，模拟所有HTTP API"""

    def __init__(self):
        self.sleep_calls = 0
        self.wake_calls = 0
        self.state = "awake"

    async def sleep(self, level=1):
        self.sleep_calls += 1
        self.state = "sleeping"
        return True

    async def wake_up(self, tags=None):
        self.wake_calls += 1
        self.state = "awake"
        return True

    async def init_weight_transfer(self, init_info=None):
        return True

    async def start_weight_update(self, is_checkpoint_format=True):
        return True

    async def update_weights(self, update_info, weight_version=None):
        return True

    async def finish_weight_update(self):
        return True

    async def close(self):
        pass

    def _get_client(self):
        """Mock HTTP client for inference calls"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "choices": [
                {"text": "Step 1: Calculate 5+3=8\n#### 8"},
                {"text": "Step 1: 5+3=8\nThe answer is 8"},
                {"text": "5+3=8\n#### 8"},
                {"text": "答案是: 8"},
            ]
        })
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        return mock_client


@dataclass
class MockTrainMetrics:
    loss: float = 0.5
    policy_loss: float = 0.3
    kl_divergence: float = 0.01
    grad_norm: float = 1.0
    learning_rate: float = 1e-5
    step: int = 1


def test_state_transitions():
    """测试状态机转换正确性"""
    async def _test():
        config = ColocateConfig(
            model_path="test-model",
            tp_size=1,
            total_iterations=2,
            batch_size=2,
        )

        orchestrator = ColocateOrchestrator(config)

        # 验证初始状态
        assert orchestrator.state == OrchestratorState.INIT

        # 模拟状态转换
        await orchestrator._transition(OrchestratorState.INFERRING)
        assert orchestrator.state == OrchestratorState.INFERRING

        await orchestrator._transition(OrchestratorState.REWARDING)
        assert orchestrator.state == OrchestratorState.REWARDING

        await orchestrator._transition(OrchestratorState.TRAINING)
        assert orchestrator.state == OrchestratorState.TRAINING

        await orchestrator._transition(OrchestratorState.SLEEPING)
        assert orchestrator.state == OrchestratorState.SLEEPING

        await orchestrator._transition(OrchestratorState.SYNCING)
        assert orchestrator.state == OrchestratorState.SYNCING

        await orchestrator._transition(OrchestratorState.DONE)
        assert orchestrator.state == OrchestratorState.DONE

    asyncio.run(_test())
    print("  ✓ test_state_transitions passed")


def test_error_recovery():
    """测试错误恢复"""
    async def _test():
        config = ColocateConfig(
            model_path="test-model",
            tp_size=1,
            total_iterations=1,
            max_retries=2,
            retry_delay=0.01,
        )

        orchestrator = ColocateOrchestrator(config)
        orchestrator._vllm_client = MockVLLMClient()

        # 模拟同步阶段错误
        error = RuntimeError("Sync failed")
        await orchestrator._handle_error(error, "syncing")

        # 验证状态转为ERROR
        assert orchestrator.state == OrchestratorState.ERROR
        # 验证恢复尝试（wake_up被调用）
        assert orchestrator._vllm_client.wake_calls > 0

    asyncio.run(_test())
    print("  ✓ test_error_recovery passed")


def test_full_iteration_mock():
    """测试完整的一轮循环（全mock）"""
    async def _test():
        config = ColocateConfig(
            model_path="test-model",
            tp_size=1,
            total_iterations=2,
            batch_size=2,
            num_samples_per_prompt=4,
        )

        orchestrator = ColocateOrchestrator(config)

        # Mock所有组件
        mock_client = MockVLLMClient()
        orchestrator._vllm_client = mock_client

        # Mock _do_inference
        async def mock_inference(batch):
            return [
                ["Step 1: 5+3=8\n#### 8"] * config.num_samples_per_prompt
                for _ in batch.prompts
            ]
        orchestrator._do_inference = mock_inference

        # Mock _do_training
        async def mock_training(batch, responses, rewards):
            return MockTrainMetrics(step=orchestrator.step_count if hasattr(orchestrator, 'step_count') else 1)
        orchestrator._do_training = mock_training

        # Mock _do_weight_sync
        async def mock_sync():
            pass
        orchestrator._do_weight_sync = mock_sync

        # Mock reward function
        from src.reward.gsm8k_reward import GSM8KRewardFunction
        orchestrator._reward_fn = GSM8KRewardFunction()

        # Mock data pipeline
        from src.data.data_pipeline import GRPOBatch
        mock_batches = [
            GRPOBatch(
                prompts=["What is 5+3?", "What is 2+2?"],
                prompt_tokens=[[1, 2, 3], [4, 5, 6]],
                ground_truths=["8", "4"],
            )
        ]

        # Mock data pipeline's get_batches
        class MockDataPipeline:
            def get_batches(self, batch_size):
                return mock_batches
            def __len__(self):
                return 2
        orchestrator._data_pipeline = MockDataPipeline()

        # 标记初始化完成
        await orchestrator._transition(OrchestratorState.INIT)

        # 运行循环
        metrics = await orchestrator.run()

        # 验证
        assert len(metrics) == 2, f"Expected 2 iterations, got {len(metrics)}"
        assert orchestrator.state == OrchestratorState.DONE
        assert mock_client.sleep_calls == 2
        assert mock_client.wake_calls == 2

        # 验证metrics结构
        for m in metrics:
            assert isinstance(m, IterationMetrics)
            assert m.total_time > 0
            assert "inferring" in m.state_durations
            assert "rewarding" in m.state_durations
            assert "training" in m.state_durations

    asyncio.run(_test())
    print("  ✓ test_full_iteration_mock passed")


def test_get_summary_empty():
    """测试空状态下的summary"""
    config = ColocateConfig(model_path="test-model")
    orchestrator = ColocateOrchestrator(config)

    summary = orchestrator.get_summary()
    assert summary == {"status": "no_data"}
    print("  ✓ test_get_summary_empty passed")


def test_get_summary_with_data():
    """测试有数据时的summary结构"""
    config = ColocateConfig(model_path="test-model", total_iterations=5)
    orchestrator = ColocateOrchestrator(config)

    # 手动添加一些metrics
    for i in range(3):
        m = IterationMetrics(
            iteration=i,
            state_durations={"inferring": 1.0, "training": 2.0, "syncing": 0.5},
            avg_reward=0.5 + i * 0.1,
            accuracy=0.3 + i * 0.1,
            total_time=3.5,
        )
        orchestrator.metrics_history.append(m)

    summary = orchestrator.get_summary()

    assert "config" in summary
    assert "results" in summary
    assert summary["results"]["completed_iterations"] == 3
    assert summary["results"]["final_accuracy"] == 0.5
    assert "stage_avg_durations" in summary
    assert "inferring" in summary["stage_avg_durations"]

    print("  ✓ test_get_summary_with_data passed")


if __name__ == "__main__":
    print("\nRunning Colocate Loop tests")
    print("-" * 50)

    test_state_transitions()
    test_error_recovery()
    test_full_iteration_mock()
    test_get_summary_empty()
    test_get_summary_with_data()

    print("-" * 50)
    print("\n✅ All Colocate loop tests passed!")
