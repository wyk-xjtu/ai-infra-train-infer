"""
自研推理引擎 — 串联 KV Cache + Scheduler + Prefix Cache + CUDA Graph

===================== 架构总览 =====================

┌──────────────────────────────────────────────────────────┐
│                    InferenceEngine                         │
├──────────────────────────────────────────────────────────┤
│  add_request(prompt_tokens)                               │
│       ↓                                                   │
│  ContinuousBatchingScheduler.schedule()                   │
│       ↓                                                   │
│  ┌─ Prefill路径 ──────────────────────────────────────┐  │
│  │  PrefixCache.match_prefix() → 跳过已缓存的blocks  │  │
│  │  KVCacheManager.allocate() → 分配新blocks          │  │
│  │  model_forward(prefill) → 计算KV                   │  │
│  │  PrefixCache.insert() → 缓存新计算的blocks         │  │
│  └────────────────────────────────────────────────────┘  │
│       ↓                                                   │
│  ┌─ Decode路径 ───────────────────────────────────────┐  │
│  │  CUDAGraphRunner.replay() → 加速的单步decode       │  │
│  │  KVCacheManager.append_token() → 更新block状态     │  │
│  └────────────────────────────────────────────────────┘  │
│       ↓                                                   │
│  Sampling → output tokens                                 │
│       ↓                                                   │
│  finish_request() / continue decode                       │
└──────────────────────────────────────────────────────────┘

说明:
- 这是一个教学/展示级推理引擎，展示各组件如何协作
- 不包含实际的 Attention kernel 实现（需 Flash Attention / Triton）
- 生产环境仍使用 vLLM，但此实现证明了对核心原理的深度理解
- engine.on_weights_updated() 供 orchestrator 在 RL 训练后调用

与项目的集成点:
- colocate_orchestrator: 可选 backend="custom" 时创建此 engine
- disagg_orchestrator: InferWorker 内部可选用此 engine
- ipc_transfer: 权重更新后调用 engine.on_weights_updated()
"""
import logging
import time
import random
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

from .kv_cache import KVCacheManager, BlockPool
from .scheduler import ContinuousBatchingScheduler, Request, SchedulerOutput, RequestState
from .prefix_cache import PrefixCache
from .cuda_graph import CUDAGraphRunner
from .model_runner import ModelRunner, ModelRunnerConfig


@dataclass
class InferenceConfig:
    """推理引擎配置"""
    num_layers: int = 32
    num_heads: int = 32          # Query head 数
    num_kv_heads: int = 8       # GQA/MQA 的 KV head 数
    head_dim: int = 128
    hidden_size: int = 4096
    vocab_size: int = 151936    # Qwen3 词表大小

    num_blocks: int = 1024      # 总物理 block 数
    block_size: int = 16        # 每个 block 存储的 token 数

    max_num_batched_tokens: int = 2048
    max_num_sequences: int = 256
    max_prefill_tokens: int = 1024
    enable_chunked_prefill: bool = True
    chunk_size: int = 512

    # Prefix Cache 默认关闭；GSM8K prompt 复用度低，hash 查找开销通常大于收益。
    enable_prefix_caching: bool = False
    enable_cuda_graph: bool = True
    max_cached_blocks: int = 512

    temperature: float = 0.7
    top_p: float = 0.9

    #   - "kv_cache": ModelRunner + PagedAttention + 真实模型（O(1) per-token decode）
    #   - "eager": HF 模型原生 forward（O(n²) 每步重计，作为正确性 baseline）
    #   - "mock": scheduler + mock 采样（无真实推理，用于功能测试）
    inference_mode: str = "mock"

    use_real_model: bool = False

    model_path: str = ""
    dtype: str = "bfloat16"

    @classmethod
    def auto_from_device(cls, num_layers=32, num_kv_heads=8, head_dim=128,
                         block_size=16, gpu_memory_fraction=0.3, **kwargs):
        """根据GPU显存自动配置KV Cache参数

        查询可用显存，按比例计算可容纳的 KV Cache block 数，避免手工配置导致 OOM。

        Args:
            num_layers: transformer层数
            num_kv_heads: KV head数
            head_dim: head维度
            block_size: 每个block存储的token数
            gpu_memory_fraction: KV Cache最多占用显存的比例（默认30%）
            **kwargs: 其他InferenceConfig参数

        Returns:
            自动配置的InferenceConfig实例
        """
        import logging
        _logger = logging.getLogger("inference.engine.auto_config")

        if not torch.cuda.is_available():
            _logger.info("CUDA not available, using minimal config (64 blocks, no CUDA Graph)")
            return cls(num_blocks=64, block_size=block_size,
                      num_layers=num_layers, num_kv_heads=num_kv_heads,
                      head_dim=head_dim, enable_cuda_graph=False, **kwargs)

        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
        except (RuntimeError, AssertionError) as e:
            _logger.warning("Failed to query CUDA memory: %s, using fallback config", e)
            return cls(num_blocks=64, block_size=block_size,
                      num_layers=num_layers, num_kv_heads=num_kv_heads,
                      head_dim=head_dim, enable_cuda_graph=False, **kwargs)
        available = int(free_mem * gpu_memory_fraction)

        # 每个block的字节数 = 2(KV) * layers * block_size * heads * dim * dtype_bytes(fp16=2)
        bytes_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim * 2
        max_blocks = max(32, available // bytes_per_block)

        # 小显存禁用CUDA Graph（预热消耗大量显存）
        enable_graph = total_mem > 12 * 1024**3  # >12GB才启用

        result = cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=min(max_blocks, 2048),  # 上限2048
            block_size=block_size,
            enable_cuda_graph=enable_graph,
            **kwargs,
        )
        _logger.info(
            "Auto-config: free_mem=%.1fMB, available_for_kv=%.1fMB, "
            "bytes_per_block=%d, num_blocks=%d, enable_cuda_graph=%s",
            free_mem / 1e6, available / 1e6,
            bytes_per_block, result.num_blocks, enable_graph,
        )
        return result


class InferenceEngine:
    """自研推理引擎 — 核心组件协调器
    
    职责:
    1. 组件生命周期管理（初始化、清理）
    2. 请求管理（add/finish）
    3. 单步推理循环（schedule → compute → sample → update）
    4. 权重更新后的缓存清理
    """

    def __init__(self, config: InferenceConfig, model=None):
        """
        Args:
            config: 推理引擎配置
            model: 可选的模型对象（None时使用模拟模型）
                   传入模型且 config.use_real_model=True 时使用真实 forward+sample
        """
        self.config = config
        self._model = model

        # HF 模型和 tokenizer（use_real_model=True 时从 model_path 加载）
        self._hf_model = None
        self._tokenizer = None

        self._block_pool: Optional[BlockPool] = None
        self._kv_manager: Optional[KVCacheManager] = None
        self._scheduler: Optional[ContinuousBatchingScheduler] = None
        self._prefix_cache: Optional[PrefixCache] = None
        self._cuda_graph: Optional[CUDAGraphRunner] = None
        self._model_runner: Optional[ModelRunner] = None

        self._requests: Dict[str, Request] = {}  # request_id → Request
        self._finished_outputs: Dict[str, List[int]] = {}  # 完成的请求输出

        self._step_count = 0
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        self._initialized = False

    def initialize(self):
        """初始化所有组件
        
        初始化顺序:
        1. BlockPool: 预分配 KV Cache 显存
        2. KVCacheManager: block 分配/回收逻辑
        3. ContinuousBatchingScheduler: 请求调度
        4. PrefixCache: 前缀缓存
        5. CUDAGraphRunner: CUDA Graph 加速
        6. 真实模型加载（use_real_model=True 或 inference_mode in ["eager","kv_cache"] 时）
        """
        cfg = self.config

        # 支持旧的 use_real_model 开关，也支持新的 inference_mode 路由。
        need_hf_model = (
            (cfg.use_real_model and self._model is None and cfg.model_path) or
            (cfg.inference_mode in ("eager", "kv_cache") and cfg.model_path and self._hf_model is None)
        )
        if need_hf_model:
            self._load_hf_model(cfg.model_path)

        # eager 路径不依赖 KV Cache 和 scheduler。
        if cfg.inference_mode == "eager" or (self._hf_model is not None and cfg.inference_mode != "kv_cache"):
            self._initialized = True
            return


        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        cache_dtype = dtype_map.get(cfg.dtype, torch.float16)

        self._block_pool = BlockPool(
            num_blocks=cfg.num_blocks,
            block_size=cfg.block_size,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            dtype=cache_dtype,
            device="cuda",
        )

        self._kv_manager = KVCacheManager(
            block_pool=self._block_pool,
            block_size=cfg.block_size,
        )

        if cfg.inference_mode == "kv_cache" and self._hf_model is not None:
            runner_config = ModelRunnerConfig(
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
                block_size=cfg.block_size,
                max_num_blocks=cfg.num_blocks,
                dtype=torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                enable_cuda_graph=False,  # KV cache 路径暂不启用 CUDA Graph
                max_cuda_graph_batch_size=cfg.max_num_sequences,
            )
            self._model_runner = ModelRunner(None, runner_config)
            self._model_runner.bind_hf_model(self._hf_model)
            if self._block_pool.kv_cache_tensor is not None:
                self._model_runner.bind_kv_cache(self._block_pool.kv_cache_tensor)
            self._initialized = True
            return

        self._scheduler = ContinuousBatchingScheduler(
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_sequences=cfg.max_num_sequences,
            max_prefill_tokens=cfg.max_prefill_tokens,
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            chunk_size=cfg.chunk_size,
        )

        if cfg.enable_prefix_caching:
            self._prefix_cache = PrefixCache(
                block_size=cfg.block_size,
                max_cached_blocks=cfg.max_cached_blocks,
            )

        if cfg.enable_cuda_graph:
            self._cuda_graph = CUDAGraphRunner(max_batch_size=cfg.max_num_sequences)

        if self._model is not None and cfg.use_real_model:
            runner_config = ModelRunnerConfig(
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
                block_size=cfg.block_size,
                max_num_blocks=cfg.num_blocks,
                dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                enable_cuda_graph=cfg.enable_cuda_graph,
                max_cuda_graph_batch_size=cfg.max_num_sequences,
            )
            self._model_runner = ModelRunner(self._model, runner_config)
            if self._block_pool.kv_cache_tensor is not None:
                self._model_runner.bind_kv_cache(self._block_pool.kv_cache_tensor)

        self._initialized = True

    def _load_hf_model(self, model_path: str):
        """从 HuggingFace checkpoint 加载推理模型（MVP: 使用 transformers 原生模型）

        加载后模型存储在 self._hf_model，tokenizer 存储在 self._tokenizer。
        generate() 方法检测到 _hf_model 存在时走真实推理路径。
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading HF model from %s for real inference...", model_path)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            trust_remote_code=True,
            # (PyTorch 2.12 + CUDA 13.0 + H20 上 SDPA 触发 cublasLtGetVersion 崩溃)
            attn_implementation="eager",
        )
        if torch.cuda.is_available():
            self._hf_model = self._hf_model.to("cuda")
        self._hf_model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        hf_cfg = self._hf_model.config
        self.config.vocab_size = hf_cfg.vocab_size
        self.config.num_layers = hf_cfg.num_hidden_layers
        self.config.num_heads = hf_cfg.num_attention_heads
        self.config.num_kv_heads = getattr(hf_cfg, 'num_key_value_heads', hf_cfg.num_attention_heads)
        self.config.hidden_size = hf_cfg.hidden_size
        self.config.head_dim = getattr(hf_cfg, 'head_dim', hf_cfg.hidden_size // hf_cfg.num_attention_heads)

        logger.info(
            "HF model loaded: %s, layers=%d, heads=%d, kv_heads=%d, hidden=%d, vocab=%d",
            model_path, self.config.num_layers, self.config.num_heads,
            self.config.num_kv_heads, self.config.hidden_size, self.config.vocab_size,
        )

    def add_request(self, request_id: str, prompt_tokens: List[int],
                    max_tokens: int = 512, **sampling_params):
        """添加推理请求
        
        Args:
            request_id: 唯一请求标识
            prompt_tokens: 编码后的 prompt token ids
            max_tokens: 最大生成 token 数
            **sampling_params: 采样参数 (temperature, top_p等)
        """
        assert self._initialized, "Engine not initialized. Call initialize() first."

        request = Request(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=sampling_params.get("temperature", self.config.temperature),
        )
        self._requests[request_id] = request
        self._scheduler.add_request(request)

    def step(self) -> Dict[str, List[int]]:
        """执行一步推理
        
        完整单步流程:
        1. 调度: Scheduler 决定本步处理哪些请求
        2. Prefill: 对新请求计算 KV Cache (利用 Prefix Cache 加速)
        3. Decode: 对 running 请求生成下一个 token (利用 CUDA Graph 加速)
        4. 采样: 从 logits 采样 token
        5. 更新状态: append token, 检查结束条件, 释放资源
        
        Returns:
            {request_id: output_tokens} — 本步完成生成的请求及其输出
        """
        assert self._initialized, "Engine not initialized."

        schedule_output: SchedulerOutput = self._scheduler.schedule()
        
        if schedule_output.is_empty:
            return {}

        newly_finished: Dict[str, List[int]] = {}

        if schedule_output.prefill_requests:
            self._execute_prefill(schedule_output.prefill_requests)

        if schedule_output.decode_requests:
            self._execute_decode(schedule_output.decode_requests)

        for req in list(self._scheduler.running_queue):
            if req.is_finished:
                newly_finished[req.request_id] = list(req.output_tokens)
                self._scheduler.finish_request(req.request_id)
                self._kv_manager.free_sequence(req.request_id)
                self._requests.pop(req.request_id, None)

        for req in schedule_output.preempted_requests:
            self._kv_manager.free_sequence(req.request_id)

        self._step_count += 1
        self._finished_outputs.update(newly_finished)
        return newly_finished

    def _execute_prefill(self, requests: List[Request]):
        """执行 Prefill（可能使用 Prefix Cache 跳过部分计算）
        
        流程:
        1. 检查 Prefix Cache 是否有匹配（节省重复计算）
        2. 为序列分配 KV Cache blocks
        3. 执行模型 forward 计算 KV（真实模型或 mock）
        4. 将新计算的 blocks 注册到 Prefix Cache
        5. 更新请求状态
        
        请求级异常隔离: 单个请求的失败不影响其他请求继续处理。
        """
        sequences_for_runner = []  # 收集需要真实执行的序列信息

        for request in requests:
            try:
                seq_id = request.request_id
                prompt = request.prompt_tokens

                matched_tokens = 0
                cached_block_ids: List[int] = []
                if self._prefix_cache is not None:
                    matched_tokens, cached_block_ids = self._prefix_cache.match_prefix(prompt)

                total_tokens = request.num_prompt_tokens
                try:
                    block_table = self._kv_manager.allocate_for_sequence(seq_id, total_tokens)
                except RuntimeError:
                    # OOM: 无法分配，保持在 waiting 队列等下一步
                    continue

                tokens_to_compute = total_tokens - matched_tokens
                self._total_prefill_tokens += tokens_to_compute

                if self._model_runner is not None:
                    sequences_for_runner.append({
                        'token_ids': prompt,
                        'block_table': block_table.get_physical_block_ids(),
                        'num_cached_tokens': matched_tokens,
                        'num_scheduled_tokens': tokens_to_compute,
                        'request': request,
                        'block_table_obj': block_table,
                    })

                if self._prefix_cache is not None and block_table is not None:
                    self._prefix_cache.insert(
                        token_ids=prompt,
                        block_ids=block_table.get_physical_block_ids(),
                    )

                request.num_computed_tokens = total_tokens

            except Exception as e:
                logger.error(
                    f"Prefill failed for request {request.request_id}: "
                    f"{type(e).__name__}: {e}. Skipping this request."
                )
                # 清理该请求已分配的资源
                try:
                    self._kv_manager.free_sequence(request.request_id)
                except Exception:
                    pass
                continue

        # 使用 ModelRunner 执行真实 prefill（如果可用）
        if self._model_runner is not None and sequences_for_runner:
            input_ids, positions = self._model_runner.prepare_prefill(sequences_for_runner)
            logits = self._model_runner.run(input_ids, positions, is_prefill=True)
            # 取每个序列最后一个 token 的 logits
            cu_seqlens = [0]
            for seq_info in sequences_for_runner:
                cu_seqlens.append(cu_seqlens[-1] + seq_info['num_scheduled_tokens'])
            for i, seq_info in enumerate(sequences_for_runner):
                last_idx = cu_seqlens[i + 1] - 1
                seq_logits = logits[last_idx:last_idx + 1]
                sampled = self._model_runner.sample(
                    seq_logits,
                    temperature=seq_info['request'].temperature,
                    top_p=self.config.top_p,
                )
                seq_info['request'].output_tokens.append(sampled[0])

    def _execute_decode(self, requests: List[Request]):
        """执行 Decode（使用 CUDA Graph 加速或 mock 采样）
        
        流程:
        1. 准备输入 (input_ids = 每个序列的最后一个 token)
        2. 尝试使用 CUDA Graph 加速（batch_size 匹配时）
        3. 执行 forward（或 graph replay）
        4. 采样 next token
        5. 更新 KV Cache (append_token)
        
        请求级异常隔离: 单个请求的失败不影响其他请求继续处理。
        """
        batch_size = len(requests)
        self._total_decode_tokens += batch_size
        failed_requests: List[str] = []  # 收集失败的请求ID，循环后统一清理

        if self._model_runner is not None:
            sequences_for_runner = []
            for request in requests:
                seq_id = request.request_id
                # 构造序列完整 token_ids（prompt + 已生成）
                all_tokens = list(request.prompt_tokens) + list(request.output_tokens)
                block_table = self._kv_manager.get_block_table(seq_id)
                # 获取最后一个 block 的 token 数
                table_obj = self._kv_manager._seq_tables.get(seq_id)
                last_block_tokens = table_obj.last_block.token_count if table_obj and table_obj.last_block else 1
                sequences_for_runner.append({
                    'token_ids': all_tokens,
                    'block_table': block_table,
                    'last_block_num_tokens': last_block_tokens,
                })

            input_ids, positions = self._model_runner.prepare_decode(sequences_for_runner)
            logits = self._model_runner.run(input_ids, positions, is_prefill=False)

            sampled_tokens = self._model_runner.sample(
                logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            for i, request in enumerate(requests):
                try:
                    next_token = sampled_tokens[i] if i < len(sampled_tokens) else 2
                    request.output_tokens.append(next_token)
                    seq_id = request.request_id
                    success = self._kv_manager.append_token(seq_id)
                    if not success:
                        logger.warning(
                            f"KV Cache append failed for request {request.request_id}, "
                            "marking for preemption."
                        )
                except Exception as e:
                    logger.error(
                        f"Decode post-processing failed for request {request.request_id}: "
                        f"{type(e).__name__}: {e}. Aborting this request."
                    )
                    try:
                        self._kv_manager.free_sequence(request.request_id)
                    except Exception:
                        pass
                    failed_requests.append(request.request_id)
        else:
            use_cuda_graph = (
                self._cuda_graph is not None and
                self._cuda_graph.is_captured(batch_size)
            )

            for request in requests:
                try:
                    next_token = self._mock_sample(request)
                    request.output_tokens.append(next_token)

                    seq_id = request.request_id
                    success = self._kv_manager.append_token(seq_id)
                    if not success:
                        # KV Cache OOM — 标记为需要抢占（下一步调度器处理）
                        logger.warning(
                            f"KV Cache append failed for request {request.request_id}, "
                            "marking for preemption."
                        )
                except Exception as e:
                    logger.error(
                        f"Decode failed for request {request.request_id}: "
                        f"{type(e).__name__}: {e}. Aborting this request."
                    )
                    try:
                        self._kv_manager.free_sequence(request.request_id)
                    except Exception:
                        pass
                    failed_requests.append(request.request_id)

        # 清理失败的请求：从 scheduler running_queue 和 requests 字典中移除
        for req_id in failed_requests:
            self._scheduler.running_queue = [
                r for r in self._scheduler.running_queue if r.request_id != req_id
            ]
            self._requests.pop(req_id, None)

    def _mock_sample(self, request: Request) -> int:
        """模拟采样（无真实模型时的placeholder）
        
        实际实现:
        1. 从model获取logits
        2. temperature scaling: logits /= temperature  
        3. top-p filtering
        4. torch.multinomial 采样
        
        生产环境这里应该调用真实的 Sampler 类。
        """
        if request.num_generated_tokens >= request.max_tokens - 1:
            return 2  # EOS
        if random.random() < 0.02:  # 2%概率自然结束
            return 2
        return random.randint(3, self.config.vocab_size - 1)

    def generate(self, prompts_tokens: List[List[int]],
                 max_tokens: int = 512,
                 temperature: float = 0.7) -> Optional[List[List[int]]]:
        """批量生成接口 — 运行完整生成直到所有请求完成
        
        这是面向外部的高层接口（类似 vLLM 的 LLM.generate()）
        
        路由逻辑:
        - inference_mode == "eager" → _generate_real() 路径A（O(n²) baseline）
        - inference_mode == "kv_cache" → _generate_kv_cache() 路径C（O(1) decode）
        - inference_mode == "mock" → scheduler + mock 采样 路径B
        - 兼容旧配置: use_real_model=True 且有 _hf_model → _generate_real()
        
        Args:
            prompts_tokens: 多个 prompt 的 token ids
            max_tokens: 最大生成长度
            temperature: 采样温度
            
        Returns:
            每个 prompt 对应的生成结果 token ids，CUDA OOM 时返回 None
        """
        if self.config.inference_mode == "eager":
            return self._generate_real(prompts_tokens, max_tokens, temperature)
        elif self.config.inference_mode == "kv_cache":
            return self._generate_kv_cache(prompts_tokens, max_tokens, temperature)

        if hasattr(self, '_hf_model') and self._hf_model is not None:
            return self._generate_real(prompts_tokens, max_tokens, temperature)

        try:
            request_ids = []
            for i, tokens in enumerate(prompts_tokens):
                req_id = f"req_{time.time_ns()}_{i}"
                self.add_request(req_id, tokens, max_tokens=max_tokens,
                              temperature=temperature)
                request_ids.append(req_id)

            all_outputs: Dict[str, List[int]] = {}
            max_steps = max_tokens + max(len(t) for t in prompts_tokens) + 10

            for _ in range(max_steps):
                finished = self.step()
                all_outputs.update(finished)

                # 检查是否所有请求都完成了
                if not self._scheduler.has_pending_requests():
                    break

            # 按请求顺序返回结果
            results = []
            for req_id in request_ids:
                results.append(all_outputs.get(req_id, []))
            return results

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA OOM during inference generation. "
                "Clearing KV cache and returning None."
            )
            torch.cuda.empty_cache()
            if self._kv_manager is not None:
                for seq_id in list(self._kv_manager._seq_tables.keys()):
                    self._kv_manager.free_sequence(seq_id)
            if self._prefix_cache is not None:
                self._prefix_cache.invalidate_all()
            # 清除 CUDA Graph（可能已损坏）
            if self._cuda_graph is not None:
                self._cuda_graph.invalidate()
            # 清除所有请求状态
            if self._scheduler is not None:
                self._scheduler.running_queue.clear()
                self._scheduler.waiting_queue.clear()
            self._requests.clear()
            return None

        except Exception as e:
            logger.error(f"Inference generation failed: {type(e).__name__}: {e}")
            raise

    @torch.inference_mode()
    def _generate_real(
        self,
        prompts_tokens: List[List[int]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Optional[List[List[int]]]:
        """真实模型推理路径（MVP: 无 KV Cache，每步重新计算全部 token）

        原理:
        - 每一步将完整的 input_ids（prompt + 已生成 tokens）送入模型
        - 取最后一个位置的 logits 做采样
        - 拼接 next_token 到 input_ids，重复直到 EOS 或达到 max_tokens

        正确性: 等价于带 KV Cache 的版本（KV Cache 只是避免重复计算的优化）
        性能: O(n²) — 每步都重新计算所有 token 的 attention，后续可通过 KV Cache 优化

        Args:
            prompts_tokens: 多个 prompt 的 token ids
            max_tokens: 最大生成长度
            temperature: 采样温度（0 = greedy）

        Returns:
            每个 prompt 生成的 output token ids（不含 prompt 部分）
        """
        try:
            model = self._hf_model
            device = next(model.parameters()).device

            eos_token_id = getattr(model.config, 'eos_token_id', 151645)
            if isinstance(eos_token_id, list):
                eos_token_ids = set(eos_token_id)
            else:
                eos_token_ids = {eos_token_id}

            results: List[List[int]] = []

            for prompt_tokens in prompts_tokens:
                input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
                prompt_len = input_ids.shape[1]
                generated_tokens: List[int] = []

                for step in range(max_tokens):
                    outputs = model(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

                    next_token_id = self._sample_from_logits(
                        next_token_logits, temperature=temperature, top_p=self.config.top_p
                    )

                    generated_tokens.append(next_token_id)

                    if next_token_id in eos_token_ids:
                        break

                    next_token_tensor = torch.tensor(
                        [[next_token_id]], dtype=torch.long, device=device
                    )
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

                results.append(generated_tokens)

            return results

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during real model generation. Clearing cache.")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.error(f"Real model generation failed: {type(e).__name__}: {e}")
            raise

    def _sample_from_logits(
        self, logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9
    ) -> int:
        """从 logits 采样单个 token id

        Args:
            logits: [1, vocab_size] 或 [vocab_size]
            temperature: 采样温度（0 = greedy decoding）
            top_p: nucleus sampling 阈值

        Returns:
            采样得到的 token id
        """
        if logits.dim() == 2:
            logits = logits[0]  # [vocab_size]

        if temperature <= 0:
            return int(torch.argmax(logits).item())

        scaled_logits = logits.float() / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)

            # 移除累积概率超过 top_p 的 token
            mask = cumulative > top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            sorted_logits[mask] = float('-inf')

            scaled_logits = torch.zeros_like(scaled_logits).scatter_(
                0, sorted_indices, sorted_logits
            )

        probs = torch.softmax(scaled_logits, dim=-1)
        return int(torch.multinomial(probs.unsqueeze(0), num_samples=1).item())

    @torch.inference_mode()
    def _generate_kv_cache(
        self,
        prompts_tokens: List[List[int]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Optional[List[List[int]]]:
        """KV Cache 推理路径 — ModelRunner + PagedAttention + 真实模型

        原理:
        - Prefill: 完整序列 forward（use_cache=True），KV 存入 paged block pool
        - Decode: 每步仅 forward 1 个新 token，从 paged cache 加载历史 KV
        - 通过 paged KV cache 实现 O(1) per-token decode（仅计算单 token 的 MLP/Attention）

        与 eager 路径的对比:
        - eager: 每步 forward 完整序列（O(n²) 总计算量）
        - kv_cache: prefill O(n) + decode O(1) per step = O(n) 总计算量

        Args:
            prompts_tokens: 多个 prompt 的 token ids
            max_tokens: 最大生成长度
            temperature: 采样温度（0 = greedy）

        Returns:
            每个 prompt 生成的 output token ids（不含 prompt）
        """
        try:
            assert self._model_runner is not None, "ModelRunner not initialized for kv_cache mode"
            assert self._kv_manager is not None, "KVCacheManager not initialized"
            assert self._hf_model is not None, "HF model not loaded"

            # 使用 KV cache 所在设备（确保所有 tensor 在同一设备上）
            if self._block_pool.kv_cache_tensor is not None:
                device = self._block_pool.kv_cache_tensor.device
            else:
                device = next(self._hf_model.parameters()).device

            eos_token_id = getattr(self._hf_model.config, 'eos_token_id', 151645)
            if isinstance(eos_token_id, list):
                eos_token_ids = set(eos_token_id)
            else:
                eos_token_ids = {eos_token_id}

            results: List[List[int]] = []

            for prompt_idx, prompt_tokens in enumerate(prompts_tokens):
                seq_id = f"kv_seq_{time.time_ns()}_{prompt_idx}"
                num_prompt_tokens = len(prompt_tokens)

                try:
                    block_table = self._kv_manager.allocate_for_sequence(
                        seq_id, num_prompt_tokens
                    )
                except RuntimeError as e:
                    logger.error(f"Cannot allocate blocks for prompt {prompt_idx}: {e}")
                    results.append([])
                    continue

                input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
                positions = torch.arange(num_prompt_tokens, dtype=torch.long, device=device)

                # 计算 slot_mapping: 每个 token 对应的物理 slot
                block_ids = block_table.get_physical_block_ids()
                block_size = self.config.block_size
                slot_mapping = self._compute_slot_mapping(
                    block_ids, 0, num_prompt_tokens, block_size, device
                )

                from .context import set_context, reset_context
                set_context(
                    is_prefill=True,
                    cu_seqlens_q=torch.tensor([0, num_prompt_tokens], dtype=torch.int32, device=device),
                    cu_seqlens_k=torch.tensor([0, num_prompt_tokens], dtype=torch.int32, device=device),
                    max_seqlen_q=num_prompt_tokens,
                    max_seqlen_k=num_prompt_tokens,
                    slot_mapping=slot_mapping,
                    block_tables=None,
                )

                logits = self._model_runner.run(input_ids, positions, is_prefill=True)
                last_logits = logits[-1:]  # [1, vocab]
                next_token_id = self._sample_from_logits(
                    last_logits, temperature=temperature, top_p=self.config.top_p
                )

                generated_tokens: List[int] = [next_token_id]

                current_len = num_prompt_tokens
                for step in range(max_tokens - 1):
                    if next_token_id in eos_token_ids:
                        break

                    # 为新 token 分配 slot（可能分配新 block）
                    success = self._kv_manager.append_token(seq_id)
                    if not success:
                        logger.warning(f"KV cache full for seq {prompt_idx}, stopping early")
                        break
                    current_len += 1

                    # 新 token 的 slot_mapping
                    block_ids = block_table.get_physical_block_ids()
                    token_pos = current_len - 1
                    new_block_idx = token_pos // block_size
                    new_offset = token_pos % block_size
                    new_slot = block_ids[new_block_idx] * block_size + new_offset
                    decode_slot_mapping = torch.tensor(
                        [new_slot], dtype=torch.int32, device=device
                    )

                    # block_tables tensor: [1, num_blocks]
                    block_tables_tensor = torch.tensor(
                        [block_ids], dtype=torch.int32, device=device
                    )

                    # context_lens: 包含当前 token 的完整长度
                    context_lens = torch.tensor(
                        [current_len], dtype=torch.int32, device=device
                    )

                    set_context(
                        is_prefill=False,
                        slot_mapping=decode_slot_mapping,
                        context_lens=context_lens,
                        block_tables=block_tables_tensor,
                    )

                    decode_input = torch.tensor(
                        [next_token_id], dtype=torch.long, device=device
                    )
                    decode_pos = torch.tensor(
                        [token_pos], dtype=torch.long, device=device
                    )
                    logits = self._model_runner.run(
                        decode_input, decode_pos, is_prefill=False
                    )

                    next_token_id = self._sample_from_logits(
                        logits, temperature=temperature, top_p=self.config.top_p
                    )
                    generated_tokens.append(next_token_id)

                self._kv_manager.free_sequence(seq_id)
                results.append(generated_tokens)

            return results

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during KV cache generation. Clearing cache.")
            torch.cuda.empty_cache()
            if self._kv_manager is not None:
                for seq_id in list(self._kv_manager._seq_tables.keys()):
                    self._kv_manager.free_sequence(seq_id)
            return None
        except Exception as e:
            logger.error(f"KV cache generation failed: {type(e).__name__}: {e}")
            raise

    def _compute_slot_mapping(
        self,
        block_ids: List[int],
        start_pos: int,
        num_tokens: int,
        block_size: int,
        device: str,
    ) -> torch.Tensor:
        """计算 token 位置到物理 slot 的映射

        Args:
            block_ids: 物理 block id 列表
            start_pos: 起始 token 位置
            num_tokens: token 数量
            block_size: 每个 block 的大小
            device: 目标设备

        Returns:
            slot_mapping: [num_tokens] 每个 token 的物理 slot index
        """
        slots = []
        for i in range(num_tokens):
            pos = start_pos + i
            block_idx = pos // block_size
            offset = pos % block_size
            physical_slot = block_ids[block_idx] * block_size + offset
            slots.append(physical_slot)
        return torch.tensor(slots, dtype=torch.int32, device=device)

    def on_weights_updated(self):
        """权重更新后的清理工作
        
        RL训练每一步都会更新模型权重，更新后需要:
        1. 清除 Prefix Cache — KV 值由旧权重计算，不再正确
        2. 清除 CUDA Graph — Graph 中记录的是旧权重的 kernel 调用
        3. 正在运行的序列 — 策略可选:
           a) abort: 终止所有running序列（简单但浪费）
           b) continue: 继续decode（结果可能不一致但影响小）
           → 本实现选择 abort，因为 RL 场景中每轮都是新 prompt
        
        Orchestrator 在 weight_sync 完成后调用此方法。
        """
        if self._prefix_cache is not None:
            self._prefix_cache.invalidate_all()

        if self._cuda_graph is not None:
            self._cuda_graph.invalidate()

        if self._model_runner is not None:
            self._model_runner.invalidate_cuda_graphs()

        if self._scheduler is not None:
            for req in list(self._scheduler.running_queue):
                self._kv_manager.free_sequence(req.request_id)
            self._scheduler.running_queue.clear()

            # 也清除 waiting 队列中的请求
            for req in list(self._scheduler.waiting_queue):
                # waiting 的请求还没分配 KV Cache，直接清除
                pass
            self._scheduler.waiting_queue.clear()

        self._requests.clear()

    def has_pending(self) -> bool:
        """是否还有待处理的请求"""
        if self._scheduler is None:
            return False
        return self._scheduler.has_pending_requests()

    def get_stats(self) -> dict:
        """获取引擎综合统计"""
        stats = {
            "initialized": self._initialized,
            "step_count": self._step_count,
            "total_prefill_tokens": self._total_prefill_tokens,
            "total_decode_tokens": self._total_decode_tokens,
            "active_requests": len(self._requests),
            "finished_requests": len(self._finished_outputs),
        }

        if self._kv_manager is not None:
            stats["kv_cache"] = self._kv_manager.get_stats()
        if self._scheduler is not None:
            stats["scheduler"] = self._scheduler.get_stats()
        if self._prefix_cache is not None:
            stats["prefix_cache"] = self._prefix_cache.get_stats()
        if self._cuda_graph is not None:
            stats["cuda_graph"] = self._cuda_graph.get_stats()

        return stats
