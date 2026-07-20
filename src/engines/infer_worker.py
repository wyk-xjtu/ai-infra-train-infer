"""
推理Worker - Ray Actor封装

将推理引擎封装为Ray Actor，支持：
- 批量生成（每个prompt生成多个回复）
- 权重热更新
- 健康检查与状态查询
- 双backend: vLLM（默认）或自研InferenceEngine（backend="custom"）
"""
import logging
import time
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False


def _create_infer_worker_class():
    """动态创建Ray Actor类（避免import时ray未初始化）"""
    if not _RAY_AVAILABLE:
        raise ImportError("Ray is required for InferWorker. Install via: pip install ray")

    @ray.remote(num_gpus=2)
    class InferWorker:
        """Ray Actor: 推理Worker

        支持双backend:
        - "vllm": 使用vLLM离线推理引擎（默认，生产环境）
        - "custom": 使用自研InferenceEngine（展示核心原理理解）

        封装为可远程调用的Actor，
        用于Disaggregated架构中独立GPU组上执行推理任务。
        """

        def __init__(
            self,
            model_path: str,
            tp_size: int = 2,
            gpu_memory_utilization: float = 0.85,
            max_model_len: int = 4096,
            backend: str = "vllm",
        ):
            """
            Args:
                model_path: 模型路径或HuggingFace模型名
                tp_size: 张量并行度
                gpu_memory_utilization: GPU显存利用率
                max_model_len: 最大模型序列长度
                backend: 推理后端 "vllm" 或 "custom"
            """
            self.model_path = model_path
            self.tp_size = tp_size
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_model_len = max_model_len
            self.backend = backend
            self._engine = None
            self._custom_engine = None  # 自研引擎实例
            self._tokenizer = None  # tokenizer实例（用于custom backend）
            self._double_buffer = None  # SwiftSync DoubleBufferedLoRA instance
            self._initialized = False
            self._total_generate_time = 0.0
            self._total_requests = 0
            self._weight_version = 0

        def initialize(self):
            """启动推理引擎

            根据 backend 配置选择:
            - "vllm": 使用vLLM的LLM类进行离线批量推理
            - "custom": 使用自研InferenceEngine
            """
            if self.backend == "custom":
                from ..inference.engine import InferenceEngine, InferenceConfig

                config = InferenceConfig(
                    max_num_batched_tokens=2048,
                    max_num_sequences=256,
                    enable_prefix_caching=True,
                    enable_cuda_graph=True,
                )
                self._custom_engine = InferenceEngine(config)
                self._custom_engine.initialize()
            else:
                from vllm import LLM

                self._engine = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.tp_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    trust_remote_code=True,
                )
            self._initialized = True
            return True

        def generate(
            self,
            prompts: List[str],
            num_samples: int = 4,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> List[List[str]]:
            """批量生成，每个prompt生成num_samples个回复

            Args:
                prompts: prompt文本列表
                num_samples: 每个prompt生成的回复数量
                max_tokens: 最大生成token数
                temperature: 采样温度

            Returns:
                [num_prompts, num_samples] 的回复文本列表
            """
            assert self._initialized, "InferWorker not initialized. Call initialize() first."

            t0 = time.time()

            if self.backend == "custom":
                results = self._generate_custom(
                    prompts, num_samples, max_tokens, temperature
                )
            else:
                results = self._generate_vllm(
                    prompts, num_samples, max_tokens, temperature
                )

            self._total_generate_time += time.time() - t0
            self._total_requests += len(prompts)

            return results

        def _generate_vllm(
            self,
            prompts: List[str],
            num_samples: int,
            max_tokens: int,
            temperature: float,
        ) -> List[List[str]]:
            """使用vLLM后端生成"""
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                n=num_samples,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            )

            outputs = self._engine.generate(prompts, sampling_params)

            results: List[List[str]] = []
            for output in outputs:
                completions = [o.text for o in output.outputs]
                results.append(completions)
            return results

        def _generate_custom(
            self,
            prompts: List[str],
            num_samples: int,
            max_tokens: int,
            temperature: float,
        ) -> List[List[str]]:
            """使用自研InferenceEngine后端生成
        
            使用transformers AutoTokenizer进行encode/decode。
            每个prompt重复num_samples次以模拟多回复采样。
            通过将同一prompt的K个副本堆叠为一次batch调用来提升吞吐。
            """
            # 懒加载tokenizer
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
        
            results: List[List[str]] = []
            for prompt in prompts:
                prompt_tokens = self._tokenizer.encode(prompt)
                # 将同一prompt的K个副本堆叠，一次batch调用
                batch_input = [prompt_tokens] * num_samples
                output_tokens_list = self._custom_engine.generate(
                    batch_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if output_tokens_list is None:
                    logger.error(
                        "Inference generation failed (OOM), returning empty results for this prompt"
                    )
                    results.append([""] * num_samples)
                    continue
                # 按原始prompt索引重组结果
                samples = []
                for i in range(num_samples):
                    if i < len(output_tokens_list) and output_tokens_list[i]:
                        text = self._tokenizer.decode(output_tokens_list[i], skip_special_tokens=True)
                    else:
                        text = ""
                    samples.append(text)
                results.append(samples)
            return results

        def update_weights(self, state_dict: Dict[str, torch.Tensor], mode: str = "full") -> bool:
            """热更新模型权重

            根据backend和mode选择不同的权重更新策略:
            - mode="full": 全量更新（原有逻辑）
            - mode="delta": 增量更新（SwiftSync）

            Args:
                state_dict: 权重字典（full模式为完整state_dict，delta模式为增量）
                mode: "full" 全量更新 | "delta" 增量更新（SwiftSync）

            Returns:
                bool: 更新是否成功
            """
            assert self._initialized, "InferWorker not initialized."

            try:
                if mode == "delta":
                    from ..transfer.swift_sync import DoubleBufferedLoRA
                    if self._double_buffer is None:
                        # 首次：用 delta 作为初始 LoRA 状态初始化双缓冲
                        self._double_buffer = DoubleBufferedLoRA(state_dict)
                        logger.info("SwiftSync: DoubleBufferedLoRA initialized with %d params", len(state_dict))
                    else:
                        self._double_buffer.apply_delta_to_shadow(state_dict)
                        version = self._double_buffer.swap_buffers()
                        logger.debug("SwiftSync: Buffer swapped to version %d", version)
                    # 通知推理引擎（仅清 prefix cache，不 abort running requests）
                    if self._custom_engine:
                        self._custom_engine.on_weights_updated()
                    self._weight_version += 1
                    return True
                elif self.backend == "custom":
                    # 自研引擎: 通知权重已更新，清除所有缓存
                    self._custom_engine.on_weights_updated()
                    self._weight_version += 1
                    return True
                else:
                    # vLLM: 直接替换模型参数
                    model = self._engine.llm_engine.model_executor.driver_worker.model_runner.model
                    model_state = model.state_dict()

                    updated = 0
                    for name, param in state_dict.items():
                        if name in model_state:
                            model_state[name].copy_(param.to(model_state[name].device))
                            updated += 1

                    self._weight_version += 1
                    return True
            except Exception as e:
                print(f"[InferWorker] Weight update failed: {e}")
                return False

        def health_check(self) -> bool:
            """健康检查

            Returns:
                bool: 引擎是否可用
            """
            if not self._initialized:
                return False
            if self.backend == "custom":
                return self._custom_engine is not None
            else:
                return self._engine is not None

        def get_status(self) -> dict:
            """获取当前状态

            Returns:
                包含引擎状态、生成统计等信息的字典
            """
            status = {
                "initialized": self._initialized,
                "backend": self.backend,
                "model_path": self.model_path,
                "tp_size": self.tp_size,
                "weight_version": self._weight_version,
                "total_requests": self._total_requests,
                "total_generate_time_s": self._total_generate_time,
            }

            if self.backend == "custom" and self._custom_engine is not None:
                status["engine_stats"] = self._custom_engine.get_stats()

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                status["gpu"] = {
                    "device_id": device,
                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
                }

            return status

    return InferWorker


# 延迟创建，避免import时ray未初始化报错
InferWorker = None


def get_infer_worker_class():
    """获取InferWorker Ray Actor类

    Returns:
        Ray remote class: InferWorker
    """
    global InferWorker
    if InferWorker is None:
        InferWorker = _create_infer_worker_class()
    return InferWorker
