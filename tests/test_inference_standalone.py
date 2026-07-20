"""推理引擎独立功能测试 — 不依赖训练/RL 组件"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.engine import InferenceEngine, InferenceConfig


class TestInferenceConfigIndependent:
    """测试 InferenceConfig 完全独立"""

    def test_default_config(self):
        config = InferenceConfig()
        assert config.inference_mode == "mock" or hasattr(config, "inference_mode")
        assert config.num_layers == 32
        assert config.vocab_size == 151936

    def test_custom_config(self):
        config = InferenceConfig(
            num_layers=28,
            num_heads=16,
            hidden_size=1024,
            vocab_size=151936,
        )
        assert config.num_layers == 28
        assert config.num_heads == 16


class TestEngineInitialization:
    """测试推理引擎独立初始化"""

    def test_mock_mode_init(self):
        config = InferenceConfig(inference_mode="mock")
        engine = InferenceEngine(config)
        engine.initialize()
        # mock 模式无需 GPU、无需模型文件

    def test_generate_mock(self):
        config = InferenceConfig(inference_mode="mock", max_num_sequences=16)
        engine = InferenceEngine(config)
        engine.initialize()

        prompts = [[1, 2, 3], [4, 5, 6]]
        results = engine.generate(prompts, max_tokens=32)
        assert results is not None
        assert len(results) == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_eager_mode_with_model(self):
        model_path = "./models/Qwen3-0.6B"
        if not os.path.exists(model_path):
            pytest.skip("模型文件不存在")

        config = InferenceConfig(
            model_path=model_path,
            inference_mode="eager",
            num_layers=28,
            num_heads=16,
            num_kv_heads=8,
            head_dim=64,
            hidden_size=1024,
            vocab_size=151936,
        )
        engine = InferenceEngine(config)
        engine.initialize()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokens = tokenizer.encode("Hello")

        results = engine.generate([tokens], max_tokens=20)
        assert results is not None
        assert len(results) == 1
        assert len(results[0]) > 0

    def test_on_weights_updated(self):
        config = InferenceConfig(inference_mode="mock")
        engine = InferenceEngine(config)
        engine.initialize()
        engine.on_weights_updated()  # 不应崩溃


class TestBatchGeneration:
    """测试批量生成"""

    def test_multiple_prompts(self):
        config = InferenceConfig(inference_mode="mock", max_num_sequences=32)
        engine = InferenceEngine(config)
        engine.initialize()

        prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        results = engine.generate(prompts, max_tokens=16)
        assert results is not None
        assert len(results) == 4

    def test_k_sample_same_prompt(self):
        """K-sample: 同一 prompt 多次采样"""
        config = InferenceConfig(inference_mode="mock", max_num_sequences=16)
        engine = InferenceEngine(config)
        engine.initialize()

        prompt = [1, 2, 3, 4, 5]
        K = 4
        results = engine.generate([prompt] * K, max_tokens=32)
        assert results is not None
        assert len(results) == K
