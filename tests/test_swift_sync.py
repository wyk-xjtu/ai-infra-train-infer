"""SwiftSync 增量权重同步单元测试"""
import asyncio
import threading

import pytest
import torch

from src.transfer.swift_sync import (
    DeltaComputer,
    DeltaVersion,
    DoubleBufferedLoRA,
    SwiftSyncConfig,
    SwiftSyncTransfer,
)


# ============================================================
# TestDeltaComputer
# ============================================================


class TestDeltaComputer:
    def test_compute_delta_correct(self):
        """验证 delta = current - prev"""
        config = SwiftSyncConfig(enabled=True)
        computer = DeltaComputer(config)
        prev = {"lora_A": torch.zeros(8, 64), "lora_B": torch.zeros(64, 8)}
        current = {"lora_A": torch.ones(8, 64), "lora_B": torch.ones(64, 8) * 2}
        delta = computer.compute_delta(current, prev)
        assert torch.allclose(delta["lora_A"], torch.ones(8, 64))
        assert torch.allclose(delta["lora_B"], torch.ones(64, 8) * 2)

    def test_compute_delta_zero_change(self):
        """权重未变化时 delta 全为 0"""
        config = SwiftSyncConfig(enabled=True)
        computer = DeltaComputer(config)
        state = {"lora_A": torch.randn(8, 64), "lora_B": torch.randn(64, 8)}
        delta = computer.compute_delta(state, state)
        assert torch.allclose(delta["lora_A"], torch.zeros(8, 64))
        assert torch.allclose(delta["lora_B"], torch.zeros(64, 8))

    def test_should_full_sync_periodic(self):
        """每 fallback_full_every 步触发全量同步"""
        config = SwiftSyncConfig(enabled=True, fallback_full_every=5)
        computer = DeltaComputer(config)
        assert not computer.should_full_sync(0)  # step 0 不触发
        assert not computer.should_full_sync(3)
        assert computer.should_full_sync(5)
        assert computer.should_full_sync(10)
        assert not computer.should_full_sync(7)

    def test_should_full_sync_step_zero(self):
        """step=0 不触发全量同步"""
        config = SwiftSyncConfig(enabled=True, fallback_full_every=1)
        computer = DeltaComputer(config)
        # 即使 fallback_full_every=1，step=0 也不触发
        assert not computer.should_full_sync(0)

    def test_compute_delta_new_param(self):
        """prev 中不存在的参数，delta 等于当前参数本身"""
        config = SwiftSyncConfig(enabled=True)
        computer = DeltaComputer(config)
        prev = {"lora_A": torch.zeros(4, 4)}
        current = {"lora_A": torch.ones(4, 4), "lora_B": torch.ones(4, 4) * 3}
        delta = computer.compute_delta(current, prev)
        assert torch.allclose(delta["lora_A"], torch.ones(4, 4))
        assert torch.allclose(delta["lora_B"], torch.ones(4, 4) * 3)

    def test_compute_delta_bytes(self):
        """验证 delta 字节数计算"""
        delta = {"A": torch.zeros(4, 4)}  # float32: 4*4*4 = 64 bytes
        assert DeltaComputer.compute_delta_bytes(delta) == 64

    def test_step_count_increments(self):
        """compute_delta 每调用一次内部步数递增"""
        config = SwiftSyncConfig(enabled=True)
        computer = DeltaComputer(config)
        state = {"A": torch.ones(2, 2)}
        computer.compute_delta(state, state)
        computer.compute_delta(state, state)
        assert computer._step_count == 2


# ============================================================
# TestDoubleBufferedLoRA
# ============================================================


class TestDoubleBufferedLoRA:
    def test_initial_state(self):
        """初始化后 active 和 shadow 相同"""
        state = {"A": torch.ones(4, 4), "B": torch.zeros(4, 4)}
        db = DoubleBufferedLoRA(state)
        active = db.get_active_state()
        assert torch.allclose(active["A"], torch.ones(4, 4))
        assert torch.allclose(active["B"], torch.zeros(4, 4))

    def test_apply_delta_and_swap(self):
        """delta 应用到 shadow 后 swap，active 应该反映更新"""
        state = {"A": torch.zeros(4, 4)}
        db = DoubleBufferedLoRA(state)
        delta = {"A": torch.ones(4, 4)}
        db.apply_delta_to_shadow(delta)
        db.swap_buffers()
        active = db.get_active_state()
        assert torch.allclose(active["A"], torch.ones(4, 4))

    def test_swap_preserves_old_active_as_shadow(self):
        """swap 后旧 active 变为新 shadow"""
        state = {"A": torch.zeros(4, 4)}
        db = DoubleBufferedLoRA(state)
        # shadow 先加 delta
        db.apply_delta_to_shadow({"A": torch.ones(4, 4) * 3})
        # swap: 旧 active(zeros) 变为新 shadow，旧 shadow(3s) 变为新 active
        db.swap_buffers()
        # 再次向新 shadow(原来的 active=zeros) 加 delta
        db.apply_delta_to_shadow({"A": torch.ones(4, 4) * 5})
        db.swap_buffers()
        # 新 active 应该是 zeros + 5 = 5
        active = db.get_active_state()
        assert torch.allclose(active["A"], torch.ones(4, 4) * 5)

    def test_multiple_deltas_accumulate(self):
        """连续多次 apply_delta 累积效果"""
        state = {"A": torch.zeros(4, 4)}
        db = DoubleBufferedLoRA(state)
        db.apply_delta_to_shadow({"A": torch.ones(4, 4)})
        db.apply_delta_to_shadow({"A": torch.ones(4, 4)})
        db.swap_buffers()
        active = db.get_active_state()
        assert torch.allclose(active["A"], torch.ones(4, 4) * 2)

    def test_version_increments(self):
        """每次 swap 版本号递增"""
        state = {"A": torch.ones(2, 2)}
        db = DoubleBufferedLoRA(state)
        v0 = db.version
        db.apply_delta_to_shadow({"A": torch.ones(2, 2)})
        db.swap_buffers()
        assert db.version == v0 + 1
        db.apply_delta_to_shadow({"A": torch.ones(2, 2)})
        db.swap_buffers()
        assert db.version == v0 + 2

    def test_set_full_state(self):
        """全量设置 shadow 后 swap"""
        state = {"A": torch.zeros(3, 3)}
        db = DoubleBufferedLoRA(state)
        new_state = {"A": torch.ones(3, 3) * 5}
        db.set_full_state(new_state)
        db.swap_buffers()
        assert torch.allclose(db.get_active_state()["A"], torch.ones(3, 3) * 5)

    def test_thread_safety_swap(self):
        """多线程并发 swap 不崩溃"""
        state = {"A": torch.randn(10, 10)}
        db = DoubleBufferedLoRA(state)
        errors = []

        def worker():
            try:
                for _ in range(100):
                    db.apply_delta_to_shadow({"A": torch.randn(10, 10) * 0.01})
                    db.swap_buffers()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_initial_version_zero(self):
        """初始版本号为 0"""
        state = {"A": torch.ones(2, 2)}
        db = DoubleBufferedLoRA(state)
        assert db.version == 0

    def test_set_full_state_adds_new_key(self):
        """set_full_state 可以添加 shadow 中不存在的 key"""
        state = {"A": torch.zeros(2, 2)}
        db = DoubleBufferedLoRA(state)
        db.set_full_state({"A": torch.ones(2, 2), "B": torch.ones(3, 3) * 7})
        db.swap_buffers()
        active = db.get_active_state()
        assert torch.allclose(active["A"], torch.ones(2, 2))
        assert torch.allclose(active["B"], torch.ones(3, 3) * 7)


# ============================================================
# TestSwiftSyncTransfer
# ============================================================


class _MockInferWorker:
    """模拟推理 worker，用于测试 SwiftSyncTransfer"""

    def __init__(self):
        self.updates = []

    def update_weights(self, state, mode="delta"):
        self.updates.append({"state": state, "mode": mode})


class _MockAsyncInferWorker:
    """异步推理 worker mock"""

    def __init__(self):
        self.updates = []

    async def update_weights(self, state, mode="delta"):
        self.updates.append({"state": state, "mode": mode})


class TestSwiftSyncTransfer:
    def test_disabled_config(self):
        """disabled 时不执行任何操作"""
        config = SwiftSyncConfig(enabled=False)
        transfer = SwiftSyncTransfer(config)
        assert not transfer.config.enabled

    def test_version_increments_on_sync(self):
        """每次 sync 后版本号递增"""
        config = SwiftSyncConfig(enabled=True, async_transfer=False)
        transfer = SwiftSyncTransfer(config)
        initial = {"A": torch.zeros(4, 4)}
        transfer.initialize(initial)

        worker = _MockInferWorker()
        delta = {"A": torch.ones(4, 4)}

        v0 = transfer.version
        asyncio.run(transfer.sync_delta(delta, worker))
        assert transfer.version == v0 + 1
        asyncio.run(transfer.sync_delta(delta, worker))
        assert transfer.version == v0 + 2

    def test_initialize_sets_state(self):
        """initialize 后 double_buffer 和 prev_snapshot 被正确设置"""
        config = SwiftSyncConfig(enabled=True, enable_double_buffer=True)
        transfer = SwiftSyncTransfer(config)
        initial = {"A": torch.ones(4, 4)}
        transfer.initialize(initial)

        assert transfer.initialized
        assert transfer.double_buffer is not None
        assert transfer.prev_snapshot is not None
        assert torch.allclose(transfer.prev_snapshot["A"], torch.ones(4, 4))

    def test_sync_delta_updates_stats(self):
        """sync_delta 更新统计信息"""
        config = SwiftSyncConfig(enabled=True, async_transfer=False)
        transfer = SwiftSyncTransfer(config)
        transfer.initialize({"A": torch.zeros(4, 4)})

        worker = _MockInferWorker()
        delta = {"A": torch.ones(4, 4)}
        asyncio.run(transfer.sync_delta(delta, worker))

        assert transfer.stats.total_delta_syncs == 1
        assert transfer.stats.last_sync_type == "delta"
        assert transfer.stats.total_delta_bytes > 0

    def test_sync_full_updates_snapshot(self):
        """sync_full 后 prev_snapshot 被更新"""
        config = SwiftSyncConfig(enabled=True, async_transfer=False)
        transfer = SwiftSyncTransfer(config)
        transfer.initialize({"A": torch.zeros(4, 4)})

        worker = _MockInferWorker()
        full_state = {"A": torch.ones(4, 4) * 9}
        asyncio.run(transfer.sync_full(full_state, worker))

        assert torch.allclose(transfer.prev_snapshot["A"], torch.ones(4, 4) * 9)
        assert transfer.stats.total_full_syncs == 1
        assert transfer.stats.last_sync_type == "full"

    def test_sync_delta_async_worker(self):
        """异步 worker 也能正常 sync"""
        config = SwiftSyncConfig(enabled=True, async_transfer=True)
        transfer = SwiftSyncTransfer(config)
        transfer.initialize({"A": torch.zeros(4, 4)})

        worker = _MockAsyncInferWorker()
        delta = {"A": torch.ones(4, 4)}
        asyncio.run(transfer.sync_delta(delta, worker))

        assert len(worker.updates) == 1
        assert worker.updates[0]["mode"] == "delta"
        assert transfer.version == 1

    def test_get_stats_summary(self):
        """get_stats_summary 返回正确结构"""
        config = SwiftSyncConfig(enabled=True, async_transfer=False)
        transfer = SwiftSyncTransfer(config)
        transfer.initialize({"A": torch.zeros(2, 2)})

        worker = _MockInferWorker()
        asyncio.run(transfer.sync_delta({"A": torch.ones(2, 2)}, worker))

        summary = transfer.get_stats_summary()
        assert summary["version"] == 1
        assert summary["initialized"] is True
        assert summary["total_delta_syncs"] == 1
        assert summary["config"]["enabled"] is True

    def test_no_double_buffer(self):
        """disable double buffer 时不崩溃"""
        config = SwiftSyncConfig(
            enabled=True, enable_double_buffer=False, async_transfer=False
        )
        transfer = SwiftSyncTransfer(config)
        transfer.initialize({"A": torch.zeros(4, 4)})

        assert transfer.double_buffer is None
        worker = _MockInferWorker()
        asyncio.run(transfer.sync_delta({"A": torch.ones(4, 4)}, worker))
        assert transfer.version == 1

    def test_worker_without_update_weights(self):
        """worker 无 update_weights 方法时应 raise 且不更新版本"""
        config = SwiftSyncConfig(enabled=True, async_transfer=False)
        transfer = SwiftSyncTransfer(config)
        transfer.initialize({"A": torch.zeros(4, 4)})

        class DummyWorker:
            pass

        worker = DummyWorker()
        # 应当抛出 AttributeError，不执行 swap 和版本递增
        import pytest
        with pytest.raises(AttributeError):
            asyncio.run(transfer.sync_delta({"A": torch.ones(4, 4)}, worker))
        assert transfer.version == 0
