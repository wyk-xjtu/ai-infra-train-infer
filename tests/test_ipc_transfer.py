"""
测试CUDA IPC权重传输

验证:
1. IPC handle创建和序列化
2. VLLMClient HTTP调用（mock模式）
3. 分块传输逻辑

运行:
    python tests/test_ipc_transfer.py
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch
sys.path.insert(0, '.')

from src.transfer.ipc_transfer import (
    IPCHandle,
    IPCWeightTransfer,
    VLLMClient,
    TransferStats,
)


def test_transfer_stats():
    """测试TransferStats基本功能"""
    stats = TransferStats()
    stats.start()

    import time
    time.sleep(0.01)

    stats.total_bytes = 1024 * 1024 * 100  # 100MB
    stats.transfer_time = 0.5
    stats.finish()

    assert stats.total_time > 0, "Total time should be > 0"
    assert stats.bandwidth_gbps > 0, "Bandwidth should be > 0"
    print(f"  ✓ test_transfer_stats passed (bandwidth={stats.bandwidth_gbps:.2f} GB/s)")


def test_chunk_handles():
    """测试分块逻辑"""
    mock_client = MagicMock()
    transfer = IPCWeightTransfer(vllm_client=mock_client, chunk_size=4)

    import torch
    handles = [
        IPCHandle(
            name=f"layer.{i}.weight",
            handle=b"fake_handle",
            shape=(128, 64),
            dtype=torch.float32,
            device_id=0,
        )
        for i in range(10)
    ]

    chunks = transfer._chunk_handles(handles)

    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert len(chunks[0]) == 4, f"First chunk should have 4 handles"
    assert len(chunks[1]) == 4, f"Second chunk should have 4 handles"
    assert len(chunks[2]) == 2, f"Third chunk should have 2 handles"

    print(f"  ✓ test_chunk_handles passed (10 handles -> {len(chunks)} chunks)")


def test_vllm_client_health_check():
    """测试VLLMClient health_check（mock HTTP）"""
    async def _test():
        client = VLLMClient(base_url="http://localhost:9999", timeout=2)

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_response)
        client._client = mock_httpx_client

        result = await client.health_check()
        assert result is True, "Health check should return True"

        mock_httpx_client.get.assert_called_once_with("/health")
        await client.close()

    asyncio.run(_test())
    print("  ✓ test_vllm_client_health_check passed")


def test_vllm_client_sleep_wake():
    """测试VLLMClient sleep/wake_up（mock HTTP）"""
    async def _test():
        client = VLLMClient(base_url="http://localhost:9999", timeout=2)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "ok"})

        mock_httpx_client = AsyncMock()
        mock_httpx_client.post = AsyncMock(return_value=mock_response)
        mock_httpx_client.get = AsyncMock(return_value=mock_response)
        client._client = mock_httpx_client

        result = await client.sleep(level=1)
        assert result is True

        result = await client.wake_up(tags=["weights", "kv_cache"])
        assert result is True

        await client.close()

    asyncio.run(_test())
    print("  ✓ test_vllm_client_sleep_wake passed")


def test_vllm_client_weight_update_protocol():
    """测试完整的权重更新协议流程（mock）"""
    async def _test():
        client = VLLMClient(base_url="http://localhost:9999", timeout=2)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "ok"})

        mock_httpx_client = AsyncMock()
        mock_httpx_client.post = AsyncMock(return_value=mock_response)
        client._client = mock_httpx_client

        await client.init_weight_transfer()
        await client.start_weight_update(is_checkpoint_format=True)
        await client.update_weights({"names": ["test"], "shapes": [[64, 64]]})
        await client.finish_weight_update()

        assert mock_httpx_client.post.call_count == 4
        await client.close()

    asyncio.run(_test())
    print("  ✓ test_vllm_client_weight_update_protocol passed")


def test_ipc_handle_serialization():
    """测试IPCHandle数据结构正确"""
    import torch
    handle = IPCHandle(
        name="model.layers.0.self_attn.q_proj.weight",
        handle=b"\x00" * 64,
        shape=(4096, 4096),
        dtype=torch.float16,
        device_id=0,
    )

    assert handle.name == "model.layers.0.self_attn.q_proj.weight"
    assert handle.shape == (4096, 4096)
    assert handle.dtype == torch.float16
    assert handle.device_id == 0
    assert len(handle.handle) == 64

    print("  ✓ test_ipc_handle_serialization passed")


if __name__ == "__main__":
    print("\nRunning IPC Transfer tests")
    print("-" * 50)

    test_transfer_stats()
    test_chunk_handles()
    test_vllm_client_health_check()
    test_vllm_client_sleep_wake()
    test_vllm_client_weight_update_protocol()
    test_ipc_handle_serialization()

    print("-" * 50)
    print("\n✅ All IPC transfer tests passed!")
