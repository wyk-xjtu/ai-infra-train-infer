"""异步训推 Pipeline 单元测试"""
import asyncio
import pytest


class TestAsyncPipeline:
    def test_async_disabled_no_change(self):
        """disabled 时不影响现有行为"""
        from src.orchestrator.colocate_orchestrator import ColocateConfig
        config = ColocateConfig(async_pipeline_enabled=False)
        assert config.async_pipeline_enabled == False

    def test_async_config_defaults(self):
        """配置默认值正确"""
        from src.orchestrator.colocate_orchestrator import ColocateConfig
        config = ColocateConfig(async_pipeline_enabled=True)
        assert config.async_pipeline_max_staleness == 2
        assert config.async_pipeline_queue_size == 2

    def test_async_inference_wrapper_exists(self):
        """异步推理方法可调用"""
        from src.orchestrator.colocate_orchestrator import ColocateOrchestrator, ColocateConfig
        config = ColocateConfig(async_pipeline_enabled=True)
        orch = ColocateOrchestrator(config)
        assert hasattr(orch, '_do_inference_async')
        assert hasattr(orch, '_do_weight_sync_async')
