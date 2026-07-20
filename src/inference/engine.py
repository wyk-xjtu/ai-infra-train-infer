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
from .radix_cache import RadixAttentionCache
from .cuda_graph import CUDAGraphRunner
from .model_runner import ModelRunner, ModelRunnerConfig
from .speculative import SpeculativeConfig, SpeculativeDecoder


@dataclass
class InferenceConfig:
    """推理引擎配置"""
    # 模型架构参数
    num_layers: int = 32
    num_heads: int = 32          # Query head 数
    num_kv_heads: int = 8       # GQA/MQA 的 KV head 数
    head_dim: int = 128
    hidden_size: int = 4096
    vocab_size: int = 151936    # Qwen3 词表大小

    # KV Cache 配置
    num_blocks: int = 1024      # 总物理 block 数
    block_size: int = 16        # 每个 block 存储的 token 数
    kv_cache_dtype: str = "fp16"  # KV Cache 存储精度: "fp16"(默认) 或 "fp8"(显存减半)

    # 调度器配置
    max_num_batched_tokens: int = 2048
    max_num_sequences: int = 256
    max_prefill_tokens: int = 1024
    enable_chunked_prefill: bool = True
    chunk_size: int = 512

    # 优化开关
    # [P1-3根因修复] Prefix Cache默认关闭。
    # GSM8K等数据集prompt各不相同，命中率极低，hash查找开销大于收益。
    # 仅在对话/system_prompt复用度高的场景手动启用。
    enable_prefix_caching: bool = False
    enable_cuda_graph: bool = False
    max_cached_blocks: int = 512

    # [P1优化] K-sample Prefix 复用
    # 同一 batch 内多个相同 prompt（K-sample）场景，
    # 第一个 prompt prefill 后注册 KV cache，后续相同 prompt 复用。
    enable_prefix_reuse: bool = True

    # [P1优化] 异步 Prefill-Decode 流调度
    # 使用独立 CUDA Stream 实现 prefill 和 decode 的流水线并行
    enable_async_scheduling: bool = False

    # [P2优化] 分区 PagedAttention v2
    # 长序列 decode 时将 KV 分区并行计算，每个 partition 独立计算局部 attention
    # 然后使用 log-sum-exp 技巧精确合并。context_len <= partition_size 时走标准路径。
    enable_partitioned_attention: bool = False
    attention_partition_size: int = 512

    # [P2优化] RadixAttention Prefix Cache
    # 基于 Radix Tree 的子前缀复用，对标 SGLang RadixAttention。
    # 支持任意子前缀共享，不仅限于完整前缀匹配。
    use_radix_cache: bool = False
    radix_cache_max_nodes: int = 10000

    # 采样参数（默认值，可被请求级别覆盖）
    temperature: float = 0.7
    top_p: float = 0.9

    # 投机解码配置（Speculative Decoding）
    speculative_enabled: bool = False
    speculative_draft_model_path: str = "./models/Qwen3-0.6B"
    speculative_num_tokens: int = 5

    # 推理模式: "kv_cache" | "eager" | "mock"
    #   - "kv_cache": ModelRunner + PagedAttention + 真实模型（O(1) per-token decode）
    #   - "eager": HF 模型原生 forward（O(n²) 每步重计，作为正确性 baseline）
    #   - "mock": scheduler + mock 采样（无真实推理，用于功能测试）
    inference_mode: str = "mock"

    # 真实模型推理开关（False 时使用 mock 采样，保持向后兼容）
    use_real_model: bool = False

    # 模型路径（use_real_model=True 时，若未传入 model 对象则从此路径加载）
    model_path: str = ""
    # 模型推理精度
    dtype: str = "bfloat16"

    @classmethod
    def auto_from_device(cls, num_layers=32, num_kv_heads=8, head_dim=128,
                         block_size=16, gpu_memory_fraction=0.3, **kwargs):
        """根据GPU显存自动配置KV Cache参数

        [P1-2解决思路] 查询可用显存，按比例分配给KV Cache，
        计算可容纳的最大block数。避免手工配置导致OOM。

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

        # 核心组件（initialize时创建）
        self._block_pool: Optional[BlockPool] = None
        self._kv_manager: Optional[KVCacheManager] = None
        self._scheduler: Optional[ContinuousBatchingScheduler] = None
        self._prefix_cache: Optional[PrefixCache] = None
        self._cuda_graph: Optional[CUDAGraphRunner] = None
        self._model_runner: Optional[ModelRunner] = None

        # 投机解码器
        self._speculative_decoder: Optional[SpeculativeDecoder] = None
        if config.speculative_enabled:
            spec_config = SpeculativeConfig(
                enabled=True,
                draft_model_path=config.speculative_draft_model_path,
                num_speculative_tokens=config.speculative_num_tokens,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._speculative_decoder = SpeculativeDecoder(spec_config, device)

        # 请求跟踪
        self._requests: Dict[str, Request] = {}  # request_id → Request
        self._finished_outputs: Dict[str, List[int]] = {}  # 完成的请求输出

        # [P1优化] 异步 Prefill-Decode 流调度的 CUDA Stream
        self._prefill_stream: Optional[torch.cuda.Stream] = None
        self._decode_stream: Optional[torch.cuda.Stream] = None
        self._prefill_done_event: Optional[torch.cuda.Event] = None
        self._enable_async_scheduling = config.enable_async_scheduling

        # 统计
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

        # 0. 真实模型加载
        # 兼容两种触发方式:
        #   - 旧方式: use_real_model=True
        #   - 新方式: inference_mode in ["eager", "kv_cache"]
        need_hf_model = (
            (cfg.use_real_model and self._model is None and cfg.model_path) or
            (cfg.inference_mode in ("eager", "kv_cache") and cfg.model_path and self._hf_model is None)
        )
        if need_hf_model:
            self._load_hf_model(cfg.model_path)

        # 当使用 eager 模式时（MVP），跳过 KV Cache/Scheduler 等组件初始化
        # 因为 _generate_real() 不依赖这些组件
        if cfg.inference_mode == "eager" or (self._hf_model is not None and cfg.inference_mode != "kv_cache"):
            self._initialized = True
            return

        # kv_cache 模式: 需要初始化 BlockPool + KVCacheManager + ModelRunner
        # mock 模式: 需要初始化全部组件

        # 解析 dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        cache_dtype = dtype_map.get(cfg.dtype, torch.float16)

        # 1. Block Pool — 预分配物理显存
        self._block_pool = BlockPool(
            num_blocks=cfg.num_blocks,
            block_size=cfg.block_size,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            dtype=cache_dtype,
            device="cuda",
            cache_dtype=cfg.kv_cache_dtype,
        )

        # 2. KV Cache Manager
        self._kv_manager = KVCacheManager(
            block_pool=self._block_pool,
            block_size=cfg.block_size,
        )

        # kv_cache 模式: 初始化 ModelRunner 并绑定 HF 模型
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
                enable_cuda_graph=cfg.enable_cuda_graph,
                max_cuda_graph_batch_size=cfg.max_num_sequences,
            )
            self._model_runner = ModelRunner(None, runner_config)
            # 绑定 HF 模型到 ModelRunner
            self._model_runner.bind_hf_model(self._hf_model)
            # 绑定 KV Cache 张量
            if self._block_pool.kv_cache_tensor is not None:
                self._model_runner.bind_kv_cache(
                    self._block_pool.kv_cache_tensor,
                    k_scale=self._block_pool.k_scale,
                    v_scale=self._block_pool.v_scale,
                    cache_dtype=self._block_pool.cache_dtype,
                )

            # CUDA Graph Runner (kv_cache 模式)
            if cfg.enable_cuda_graph and torch.cuda.is_available():
                self._cuda_graph = CUDAGraphRunner(
                    max_batch_size=cfg.max_num_sequences,
                    supported_batch_sizes=[1, 2, 4, 8],
                )
                # 预录制常见 batch_size（模型加载后立即执行）
                self._pre_capture_cuda_graphs(device="cuda")

            self._initialized = True
            return

        # 3. Scheduler (mock 模式需要)
        self._scheduler = ContinuousBatchingScheduler(
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_sequences=cfg.max_num_sequences,
            max_prefill_tokens=cfg.max_prefill_tokens,
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            chunk_size=cfg.chunk_size,
        )

        # 4. Prefix Cache (可选)
        if cfg.enable_prefix_caching:
            if cfg.use_radix_cache:
                # [P2优化] 使用 RadixAttention Prefix Cache
                self._prefix_cache = RadixAttentionCache(
                    block_size=cfg.block_size,
                    max_nodes=cfg.radix_cache_max_nodes,
                )
            else:
                self._prefix_cache = PrefixCache(
                    block_size=cfg.block_size,
                    max_cached_blocks=cfg.max_cached_blocks,
                )

        # 5. CUDA Graph Runner (可选)
        if cfg.enable_cuda_graph:
            self._cuda_graph = CUDAGraphRunner(max_batch_size=cfg.max_num_sequences)

        # 6. ModelRunner（当传入真实模型且开启 use_real_model 时）
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
            # 将 BlockPool 预分配的 KV Cache 绑定给 ModelRunner
            if self._block_pool.kv_cache_tensor is not None:
                self._model_runner.bind_kv_cache(
                    self._block_pool.kv_cache_tensor,
                    k_scale=self._block_pool.k_scale,
                    v_scale=self._block_pool.v_scale,
                    cache_dtype=self._block_pool.cache_dtype,
                )

        # [P1优化] 初始化异步调度的 CUDA Streams
        if self.config.enable_async_scheduling and torch.cuda.is_available():
            self._prefill_stream = torch.cuda.Stream()
            self._decode_stream = torch.cuda.Stream()
            self._prefill_done_event = torch.cuda.Event()

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

        # 尝试按优先级加载 attention 实现，自动降级
        attn_implementations = ["flash_attention_2", "sdpa", "eager"]
        model = None
        for impl in attn_implementations:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch_dtype,
                    trust_remote_code=True,
                    attn_implementation=impl,
                )
                logger.info(f"HF model loaded with attn_implementation='{impl}'")
                break
            except (ImportError, RuntimeError, ValueError) as e:
                logger.debug(f"attn_implementation='{impl}' failed: {e}, trying next...")
                continue

        if model is None:
            raise RuntimeError("All attention implementations failed to load")

        self._hf_model = model
        # 移动到 GPU
        if torch.cuda.is_available():
            self._hf_model = self._hf_model.to("cuda")
        self._hf_model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # 更新 config 中的模型参数（从实际加载的模型获取）
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

        # === Step 1: 调度 ===
        schedule_output: SchedulerOutput = self._scheduler.schedule()
        
        if schedule_output.is_empty:
            return {}

        newly_finished: Dict[str, List[int]] = {}

        # === Step 2 & 3: Prefill + Decode（支持异步流调度） ===
        if self._enable_async_scheduling and self._prefill_stream is not None:
            # 异步路径：prefill 和 decode 使用独立 CUDA Stream 实现流水线并行
            if schedule_output.prefill_requests:
                with torch.cuda.stream(self._prefill_stream):
                    self._execute_prefill(schedule_output.prefill_requests)
                self._prefill_done_event.record(self._prefill_stream)

            if schedule_output.decode_requests:
                self._decode_stream.wait_event(self._prefill_done_event)
                with torch.cuda.stream(self._decode_stream):
                    self._execute_decode(schedule_output.decode_requests)

            # 同步确保结果可用
            torch.cuda.current_stream().wait_stream(self._decode_stream)
            if schedule_output.prefill_requests and not schedule_output.decode_requests:
                torch.cuda.current_stream().wait_stream(self._prefill_stream)
        else:
            # 原有同步路径
            if schedule_output.prefill_requests:
                self._execute_prefill(schedule_output.prefill_requests)

            if schedule_output.decode_requests:
                self._execute_decode(schedule_output.decode_requests)

        # === Step 4: 检查完成的请求 ===
        for req in list(self._scheduler.running_queue):
            if req.is_finished:
                newly_finished[req.request_id] = list(req.output_tokens)
                self._scheduler.finish_request(req.request_id)
                self._kv_manager.free_sequence(req.request_id)
                self._requests.pop(req.request_id, None)

        # === Step 5: 处理被抢占的请求 ===
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

                # 1. 尝试匹配 Prefix Cache
                matched_tokens = 0
                cached_block_ids: List[int] = []
                if self._prefix_cache is not None:
                    matched_tokens, cached_block_ids = self._prefix_cache.match_prefix(prompt)

                # 2. 分配 KV Cache blocks
                total_tokens = request.num_prompt_tokens
                try:
                    block_table = self._kv_manager.allocate_for_sequence(seq_id, total_tokens)
                except RuntimeError:
                    # OOM: 无法分配，保持在 waiting 队列等下一步
                    continue

                # 3. 计算需要处理的 token 数
                tokens_to_compute = total_tokens - matched_tokens
                self._total_prefill_tokens += tokens_to_compute

                # 收集 ModelRunner 所需的序列信息
                if self._model_runner is not None:
                    sequences_for_runner.append({
                        'token_ids': prompt,
                        'block_table': block_table.get_physical_block_ids(),
                        'num_cached_tokens': matched_tokens,
                        'num_scheduled_tokens': tokens_to_compute,
                        'request': request,
                        'block_table_obj': block_table,
                    })

                # 4. 注册到 Prefix Cache
                if self._prefix_cache is not None and block_table is not None:
                    self._prefix_cache.insert(
                        token_ids=prompt,
                        block_ids=block_table.get_physical_block_ids(),
                    )

                # 5. 更新请求状态
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
            # Prefill 阶段对每个序列采样第一个 decode token
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
            # === 真实模型执行路径 ===
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

            # 采样 + 更新状态
            sampled_tokens = self._model_runner.sample(
                logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            for i, request in enumerate(requests):
                try:
                    next_token = sampled_tokens[i] if i < len(sampled_tokens) else 2
                    request.output_tokens.append(next_token)
                    # 更新 KV Cache: 新 token 需要存入 block
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
                    # 释放 KV Cache blocks
                    try:
                        self._kv_manager.free_sequence(request.request_id)
                    except Exception:
                        pass
                    failed_requests.append(request.request_id)
        else:
            # === Mock 采样路径（无模型时的 fallback） ===
            # 尝试 CUDA Graph（实际环境中）
            use_cuda_graph = (
                self._cuda_graph is not None and
                self._cuda_graph.is_captured(batch_size)
            )

            for request in requests:
                try:
                    next_token = self._mock_sample(request)
                    request.output_tokens.append(next_token)

                    # 更新 KV Cache: 新 token 需要存入 block
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
                    # 释放 KV Cache blocks
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
        1. 从 model获取logits
        2. temperature scaling: logits /= temperature  
        3. top-p filtering
        4. torch.multinomial 采样

        生产环境这里应该调用真实的 Sampler 类。
        """
        # 模拟: 随机生成token，有小概率生成EOS(2)以模拟终止
        if request.num_generated_tokens >= request.max_tokens - 1:
            return 2  # EOS
        if random.random() < 0.02:  # 2%概率自然结束
            return 2
        return random.randint(3, self.config.vocab_size - 1)
    
    @staticmethod
    def _has_duplicate_prompts(prompts_tokens: List[List[int]]) -> bool:
        """检查 batch 内是否存在重复的 prompt（K-sample 场景检测）
    
        Args:
            prompts_tokens: 多个 prompt 的 token ids
    
        Returns:
            True 如果存在至少两个相同的 prompt
        """
        if len(prompts_tokens) <= 1:
            return False
        seen = set()
        for tokens in prompts_tokens:
            key = tuple(tokens)
            if key in seen:
                return True
            seen.add(key)
        return False

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
        # 路由逻辑（新 inference_mode 优先）
        if self.config.inference_mode == "eager":
            return self._generate_real(prompts_tokens, max_tokens, temperature)
        elif self.config.inference_mode == "kv_cache":
            return self._generate_kv_cache(prompts_tokens, max_tokens, temperature)

        # 兼容旧配置: use_real_model + _hf_model → eager 路径
        if hasattr(self, '_hf_model') and self._hf_model is not None:
            return self._generate_real(prompts_tokens, max_tokens, temperature)

        # mock 路径: scheduler + mock 采样
        try:
            # [P1优化] K-sample Prefix 复用: 在 mock 路径中，若启用 prefix_reuse
            # 且全局 prefix_cache 未启用，临时创建一个用于 batch 内复用
            if (self.config.enable_prefix_reuse
                    and self._prefix_cache is None
                    and self._has_duplicate_prompts(prompts_tokens)):
                self._prefix_cache = PrefixCache(
                    block_size=self.config.block_size,
                    max_cached_blocks=self.config.max_cached_blocks,
                )
                _temp_prefix_cache = True
            else:
                _temp_prefix_cache = False

            # 添加所有请求
            request_ids = []
            for i, tokens in enumerate(prompts_tokens):
                req_id = f"req_{time.time_ns()}_{i}"
                self.add_request(req_id, tokens, max_tokens=max_tokens,
                              temperature=temperature)
                request_ids.append(req_id)

            # 循环 step 直到所有请求完成
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

            # [P1优化] 清理临时 prefix cache
            if _temp_prefix_cache:
                self._prefix_cache = None

            return results

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA OOM during inference generation. "
                "Clearing KV cache and returning None."
            )
            # 清理 GPU 缓存
            torch.cuda.empty_cache()
            # 清理 KV Cache Manager — 释放所有序列的 blocks
            if self._kv_manager is not None:
                for seq_id in list(self._kv_manager._seq_tables.keys()):
                    self._kv_manager.free_sequence(seq_id)
            # 清理 Prefix Cache（KV 值已失效）
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

            # 获取 EOS token id
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
                    # Forward: 完整序列送入模型
                    outputs = model(input_ids)
                    # outputs.logits shape: [1, seq_len, vocab_size]
                    next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

                    # 采样
                    next_token_id = self._sample_from_logits(
                        next_token_logits, temperature=temperature, top_p=self.config.top_p
                    )

                    generated_tokens.append(next_token_id)

                    # 检查 EOS
                    if next_token_id in eos_token_ids:
                        break

                    # 拼接 next token 到 input_ids
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
            # Greedy
            return int(torch.argmax(logits).item())

        # Temperature scaling
        scaled_logits = logits.float() / temperature

        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)

            # 移除累积概率超过 top_p 的 token
            mask = cumulative > top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            sorted_logits[mask] = float('-inf')

            # 还原到原始顺序
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
        - Decode: 收集所有 active 序列的当前 token，组成 batch tensor，
                  一次 forward 产出所有序列的 next-token logits（Batched Decode）
        - 通过 paged KV cache 实现 O(1) per-token decode（仅计算单 token 的 MLP/Attention）

        与 eager 路径的对比:
        - eager: 每步 forward 完整序列（O(n²) 总计算量）
        - kv_cache: prefill O(n) + decode O(1) per step = O(n) 总计算量

        Batched Decode 优势:
        - 多序列共享一次 model forward，显著提升 GPU 利用率
        - 不同序列的 KV block 互相独立（通过 block_tables 索引）
        - 不同序列可能有不同的 context_lens，attention 中正确 mask

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

            # 获取 EOS token id
            eos_token_id = getattr(self._hf_model.config, 'eos_token_id', 151645)
            if isinstance(eos_token_id, list):
                eos_token_ids = set(eos_token_id)
            else:
                eos_token_ids = {eos_token_id}

            block_size = self.config.block_size
            num_seqs = len(prompts_tokens)

            # === 每个序列的状态跟踪 ===
            @dataclass
            class SeqState:
                seq_id: str
                prompt_idx: int
                block_table: object  # KVCacheManager 返回的 block table
                generated_tokens: List[int] = field(default_factory=list)
                current_len: int = 0  # 当前已处理的总 token 数（prompt + generated）
                finished: bool = False

            seq_states: List[SeqState] = []

            # === Phase 1: Prefill — 逐序列执行（支持 K-sample Prefix 复用） ===
            from .context import set_context, reset_context

            # [P1优化] K-sample Prefix 复用:
            # 同一 batch 内多个相同 prompt，第一个 prefill 后注册 cache，后续复用
            enable_prefix_reuse = self.config.enable_prefix_reuse
            # 临时 prefix cache（用于 batch 内复用，即使全局 prefix_cache 未启用）
            batch_prefix_cache = None
            if enable_prefix_reuse:
                from .prefix_cache import PrefixCache
                batch_prefix_cache = PrefixCache(
                    block_size=block_size,
                    max_cached_blocks=self.config.max_cached_blocks,
                )

            for prompt_idx, prompt_tokens in enumerate(prompts_tokens):
                seq_id = f"kv_seq_{time.time_ns()}_{prompt_idx}"
                num_prompt_tokens = len(prompt_tokens)

                # [P1优化] K-sample Prefix 复用: 尝试匹配 batch 内已缓存的前缀
                matched_tokens = 0
                cached_block_ids: List[int] = []
                if batch_prefix_cache is not None:
                    matched_tokens, cached_block_ids = batch_prefix_cache.match_prefix(prompt_tokens)
                    # [FIX] 临时禁用 prefix 跳过逻辑：当前实现获取 cached_block_ids 后
                    # 仍分配全新 block，但跳过 prefill 会导致从空 block 读取垃圾 KV 数据。
                    # TODO: 实现 block 共享/复制后可重新启用。
                    matched_tokens = 0

                # 分配 KV Cache blocks
                try:
                    block_table = self._kv_manager.allocate_for_sequence(
                        seq_id, num_prompt_tokens
                    )
                except RuntimeError as e:
                    logger.error(f"Cannot allocate blocks for prompt {prompt_idx}: {e}")
                    # 标记为已完成（空结果）
                    seq_states.append(SeqState(
                        seq_id=seq_id, prompt_idx=prompt_idx,
                        block_table=None, finished=True,
                    ))
                    continue

                # 如果前缀完全命中（所有 full blocks 都匹配），可以跳过大部分 prefill
                if matched_tokens > 0 and matched_tokens >= (num_prompt_tokens // block_size) * block_size:
                    # 完全复用: 只需处理尾部不足一个 block 的 token
                    remaining_tokens = num_prompt_tokens - matched_tokens
                    if remaining_tokens > 0:
                        # 对尾部 token 做轻量 prefill
                        tail_tokens = prompt_tokens[matched_tokens:]
                        input_ids = torch.tensor(tail_tokens, dtype=torch.long, device=device)
                        positions = torch.arange(
                            matched_tokens, num_prompt_tokens, dtype=torch.long, device=device
                        )
                        block_ids = block_table.get_physical_block_ids()
                        slot_mapping = self._compute_slot_mapping(
                            block_ids, matched_tokens, remaining_tokens, block_size, device
                        )
                        set_context(
                            is_prefill=True,
                            cu_seqlens_q=torch.tensor([0, remaining_tokens], dtype=torch.int32, device=device),
                            cu_seqlens_k=torch.tensor([0, num_prompt_tokens], dtype=torch.int32, device=device),
                            max_seqlen_q=remaining_tokens,
                            max_seqlen_k=num_prompt_tokens,
                            slot_mapping=slot_mapping,
                            block_tables=None,
                        )
                        logits = self._model_runner.run(input_ids, positions, is_prefill=True)
                        last_logits = logits[-1:]
                    else:
                        # 全部命中，无需 prefill，但仍需获取 last token logits
                        # 做一次单 token forward 获取 logits
                        last_token = prompt_tokens[-1]
                        input_ids = torch.tensor([last_token], dtype=torch.long, device=device)
                        positions = torch.tensor([num_prompt_tokens - 1], dtype=torch.long, device=device)
                        block_ids = block_table.get_physical_block_ids()
                        block_tables_tensor = torch.tensor([block_ids], dtype=torch.int32, device=device)
                        ctx_lens = torch.tensor([num_prompt_tokens], dtype=torch.int32, device=device)
                        last_slot_idx = (num_prompt_tokens - 1) // block_size
                        last_slot_offset = (num_prompt_tokens - 1) % block_size
                        decode_slot = block_ids[last_slot_idx] * block_size + last_slot_offset
                        slot_mapping = torch.tensor([decode_slot], dtype=torch.int32, device=device)
                        set_context(
                            is_prefill=False,
                            slot_mapping=slot_mapping,
                            context_lens=ctx_lens,
                            block_tables=block_tables_tensor,
                        )
                        logits = self._model_runner.run(input_ids, positions, is_prefill=False)
                        last_logits = logits[-1:]

                    next_token_id = self._sample_from_logits(
                        last_logits, temperature=temperature, top_p=self.config.top_p
                    )

                    state = SeqState(
                        seq_id=seq_id,
                        prompt_idx=prompt_idx,
                        block_table=block_table,
                        generated_tokens=[next_token_id],
                        current_len=num_prompt_tokens,
                        finished=(next_token_id in eos_token_ids),
                    )
                    seq_states.append(state)
                    continue

                # 正常 Prefill forward（无命中或部分命中）
                input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
                positions = torch.arange(num_prompt_tokens, dtype=torch.long, device=device)

                block_ids = block_table.get_physical_block_ids()
                slot_mapping = self._compute_slot_mapping(
                    block_ids, 0, num_prompt_tokens, block_size, device
                )

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
                # 取最后一个 token 的 logits 做采样
                last_logits = logits[-1:]  # [1, vocab]
                next_token_id = self._sample_from_logits(
                    last_logits, temperature=temperature, top_p=self.config.top_p
                )

                # [P1优化] Prefill 完成后立即注册到 batch prefix cache，供后续相同 prompt 复用
                if batch_prefix_cache is not None:
                    batch_prefix_cache.insert(
                        token_ids=prompt_tokens,
                        block_ids=block_ids,
                    )

                state = SeqState(
                    seq_id=seq_id,
                    prompt_idx=prompt_idx,
                    block_table=block_table,
                    generated_tokens=[next_token_id],
                    current_len=num_prompt_tokens,
                    finished=(next_token_id in eos_token_ids),
                )
                seq_states.append(state)

            # 投机解码器初始化（延迟加载 draft model）
            spec_decoder = self._speculative_decoder
            if spec_decoder is not None and spec_decoder.config.enabled and not spec_decoder.is_loaded:
                spec_decoder.load_draft_model()

            # === Phase 2: Batched Decode 循环 ===
            for step in range(max_tokens - 1):
                # 收集所有未完成序列
                active_states = [s for s in seq_states if not s.finished]
                if not active_states:
                    break

                # 为每个 active 序列的新 token 分配 KV slot
                still_active = []
                for state in active_states:
                    success = self._kv_manager.append_token(state.seq_id)
                    if not success:
                        logger.warning(
                            f"KV cache full for seq {state.prompt_idx}, stopping early"
                        )
                        state.finished = True
                    else:
                        state.current_len += 1
                        still_active.append(state)

                if not still_active:
                    break

                # === 投机解码分支（单序列时可启用） ===
                use_speculative = (
                    spec_decoder is not None
                    and spec_decoder.config.enabled
                    and spec_decoder.is_loaded
                    and spec_decoder.acceptance_rate > 0.5
                    and len(still_active) == 1
                )

                if use_speculative:
                    state = still_active[0]
                    # 构造当前完整序列
                    all_tokens = list(prompts_tokens[state.prompt_idx]) + state.generated_tokens
                    input_ids_for_spec = torch.tensor(
                        [all_tokens], dtype=torch.long, device=device
                    )

                    # 1. Draft model 生成 K 个候选 token
                    draft_tokens, draft_logits = spec_decoder.speculate(input_ids_for_spec)
                    K = draft_tokens.shape[0]

                    # 2. 构造 target model 输入：原始序列 + K 个候选
                    #    Target model 一次 forward 得到 K+1 个 logits
                    spec_input_ids = torch.cat([
                        input_ids_for_spec,
                        draft_tokens.unsqueeze(0),  # [1, K]
                    ], dim=1)  # [1, seq_len + K]

                    # Target model forward (eager 模式，不走 KV cache 以保持简单)
                    target_outputs = self._hf_model(spec_input_ids)
                    # logits: [1, seq_len+K, vocab]
                    # 我们需要从 seq_len-1 开始的 K+1 个 logits
                    seq_len = len(all_tokens)
                    target_logits = target_outputs.logits[:, seq_len - 1:seq_len + K, :]  # [1, K+1, vocab]

                    # 3. 验证
                    accepted_tokens, num_accepted = spec_decoder.verify(
                        input_ids_for_spec, draft_tokens, target_logits
                    )

                    # 4. 追加所有接受的 token
                    for token_id in accepted_tokens.tolist():
                        state.generated_tokens.append(token_id)
                        if token_id in eos_token_ids or len(state.generated_tokens) >= max_tokens:
                            state.finished = True
                            break
                        # 为后续 token 分配 KV slot（除了第一个已经 append 过的）
                        if token_id != accepted_tokens[0].item():
                            extra_success = self._kv_manager.append_token(state.seq_id)
                            if not extra_success:
                                state.finished = True
                                break
                            state.current_len += 1

                    continue

                # === 单序列 fallback: 只有 1 个 active 序列时走原有单序列路径 ===
                if len(still_active) == 1:
                    state = still_active[0]
                    last_token = state.generated_tokens[-1]
                    token_pos = state.current_len - 1
                    block_ids = state.block_table.get_physical_block_ids()

                    new_block_idx = token_pos // block_size
                    new_offset = token_pos % block_size
                    new_slot = block_ids[new_block_idx] * block_size + new_offset
                    decode_slot_mapping = torch.tensor(
                        [new_slot], dtype=torch.int32, device=device
                    )
                    block_tables_tensor = torch.tensor(
                        [block_ids], dtype=torch.int32, device=device
                    )
                    ctx_lens = torch.tensor(
                        [state.current_len], dtype=torch.int32, device=device
                    )

                    set_context(
                        is_prefill=False,
                        slot_mapping=decode_slot_mapping,
                        context_lens=ctx_lens,
                        block_tables=block_tables_tensor,
                    )

                    decode_input = torch.tensor(
                        [last_token], dtype=torch.long, device=device
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
                    state.generated_tokens.append(next_token_id)
                    if next_token_id in eos_token_ids or len(state.generated_tokens) >= max_tokens:
                        state.finished = True
                    continue

                # === Batched decode: 多序列共享一次 forward ===
                batch_size = len(still_active)

                # 收集 batch tensors
                input_ids_list = []
                positions_list = []
                block_tables_list = []
                context_lens_list = []

                for state in still_active:
                    last_token = state.generated_tokens[-1]
                    token_pos = state.current_len - 1
                    block_ids = state.block_table.get_physical_block_ids()

                    input_ids_list.append(last_token)
                    positions_list.append(token_pos)
                    block_tables_list.append(block_ids)
                    context_lens_list.append(state.current_len)

                # 构造 batch tensors
                batch_input_ids = torch.tensor(
                    input_ids_list, dtype=torch.long, device=device
                )
                batch_positions = torch.tensor(
                    positions_list, dtype=torch.long, device=device
                )
                batch_context_lens = torch.tensor(
                    context_lens_list, dtype=torch.int32, device=device
                )

                # block_tables: [batch_size, max_blocks] — 需要 padding 到统一长度
                max_blocks = max(len(bt) for bt in block_tables_list)
                batch_block_tables = torch.zeros(
                    batch_size, max_blocks, dtype=torch.int32, device=device
                )
                for i, bt in enumerate(block_tables_list):
                    for j, block_id in enumerate(bt):
                        batch_block_tables[i, j] = block_id

                # === CUDA Graph capture/replay 加速 ===
                logits = self._try_cuda_graph_decode(
                    batch_size=batch_size,
                    input_ids=batch_input_ids,
                    positions=batch_positions,
                    block_tables=batch_block_tables,
                    context_lens=batch_context_lens,
                    max_blocks=max_blocks,
                    device=device,
                )
                # logits: [batch_size, vocab_size]

                # 逐序列采样 next_token
                for i, state in enumerate(still_active):
                    seq_logits = logits[i:i + 1]  # [1, vocab_size]
                    next_token_id = self._sample_from_logits(
                        seq_logits, temperature=temperature, top_p=self.config.top_p
                    )
                    state.generated_tokens.append(next_token_id)
                    # 检查终止条件
                    if next_token_id in eos_token_ids or len(state.generated_tokens) >= max_tokens:
                        state.finished = True

            # === 收集结果并释放资源 ===
            results: List[List[int]] = []
            # 按 prompt_idx 排序输出
            seq_states.sort(key=lambda s: s.prompt_idx)
            for state in seq_states:
                results.append(state.generated_tokens)
                # 释放 KV Cache blocks
                if state.block_table is not None:
                    self._kv_manager.free_sequence(state.seq_id)

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

    def _try_cuda_graph_decode(
        self,
        batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_blocks: int,
        device,
    ) -> torch.Tensor:
        """Decode 阶段尝试使用 CUDA Graph 加速，失败时 fallback 到 eager 执行

        CUDA Graph 适用条件:
        - enable_cuda_graph 配置开启
        - batch_size 在支持范围内
        - 非 HF 模型路径（HF 模型的 Python 级 KV 加载不兼容 Graph）
          或已成功录制过

        Args:
            batch_size: 当前 batch 大小
            input_ids: [batch_size] token ids
            positions: [batch_size] position ids
            block_tables: [batch_size, max_blocks] block table
            context_lens: [batch_size] context lengths
            max_blocks: 最大 block 数
            device: 设备

        Returns:
            logits: [batch_size, vocab_size]
        """
        # 检查是否可以使用 CUDA Graph
        use_cuda_graph = (
            self._cuda_graph is not None
            and self.config.enable_cuda_graph
            and torch.cuda.is_available()
            and self._cuda_graph._get_padded_batch_size(batch_size) is not None
        )

        if use_cuda_graph:
            padded_bs = self._cuda_graph._get_padded_batch_size(batch_size)

            # 尝试录制（首次遇到未录制的 batch_size）
            if not self._cuda_graph.is_captured(batch_size):
                try:
                    self._capture_decode_graph(
                        batch_size=padded_bs,
                        max_blocks=max(max_blocks, self.config.num_blocks // 4),
                        device=device,
                    )
                except Exception as e:
                    logger.warning(
                        f"CUDA Graph capture failed for batch_size={padded_bs}: "
                        f"{type(e).__name__}: {e}. Falling back to eager execution."
                    )

            # 尝试回放
            if self._cuda_graph.is_captured(batch_size):
                try:
                    graph_logits = self._cuda_graph.replay(
                        batch_size=batch_size,
                        input_ids=input_ids,
                        positions=positions,
                        block_tables=block_tables,
                        context_lens=context_lens,
                    )
                    if graph_logits is not None:
                        return graph_logits
                except Exception as e:
                    logger.warning(
                        f"CUDA Graph replay failed: {type(e).__name__}: {e}. "
                        "Falling back to eager execution."
                    )

        # Fallback: eager 执行
        return self._model_runner.run_batch_decode(
            input_ids=input_ids,
            positions=positions,
            block_tables=block_tables,
            context_lens=context_lens,
        )

    def _capture_decode_graph(
        self,
        batch_size: int,
        max_blocks: int,
        device,
    ):
        """Decode 阶段 CUDA Graph 录制

        将 model_runner.run_batch_decode 包装为 CUDA Graph 可录制的 forward 函数。
        录制期间执行一次完整 forward，之后 replay 时复用录制的 kernel 序列。

        Args:
            batch_size: 要录制的 batch_size
            max_blocks: block_tables 的第二维大小
            device: 设备
        """
        vocab_size = self.config.vocab_size

        def model_forward_fn(input_ids, positions, **extra_buffers):
            """CUDA Graph 录制用的 forward 函数"""
            block_tables = extra_buffers.get('block_tables')
            context_lens = extra_buffers.get('context_lens')
            return self._model_runner.run_batch_decode(
                input_ids=input_ids,
                positions=positions,
                block_tables=block_tables,
                context_lens=context_lens,
            )

        self._cuda_graph.capture(
            model_forward_fn=model_forward_fn,
            batch_size=batch_size,
            input_ids_shape=(batch_size,),
            hidden_size=vocab_size,  # 输出维度是 vocab_size
            device=device,
            dtype=torch.float32,  # logits 为 float32
            num_warmup=3,
            block_tables=(batch_size, max_blocks),
            context_lens=(batch_size,),
        )
        logger.info(f"CUDA Graph captured for decode batch_size={batch_size}")

    def _pre_capture_cuda_graphs(self, device: str = "cuda"):
        """Pre-capture CUDA Graphs for common batch sizes [1, 2, 4, 8]

        在模型加载完成后调用，避免首次使用时的录制延迟。
        如果录制失败（如 CUDA 兼容性问题），静默跳过，runtime 时会 fallback 到 eager。
        """
        if self._cuda_graph is None:
            return

        # 预分配的 block_tables 第二维使用保守值
        max_blocks_for_graph = min(self.config.num_blocks // 4, 64)

        for bs in [1, 2, 4, 8]:
            try:
                self._capture_decode_graph(
                    batch_size=bs,
                    max_blocks=max_blocks_for_graph,
                    device=device,
                )
            except Exception as e:
                logger.debug(
                    f"Pre-capture failed for batch_size={bs}: "
                    f"{type(e).__name__}: {e}. Will retry at runtime or fallback."
                )
                break  # 如果一个失败，后续的可能也会失败

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
        # 1. 清除 Prefix Cache
        if self._prefix_cache is not None:
            self._prefix_cache.invalidate_all()

        # 2. 清除 CUDA Graph
        if self._cuda_graph is not None:
            self._cuda_graph.invalidate()

        # 2.5 清除 ModelRunner 的 CUDA Graph
        if self._model_runner is not None:
            self._model_runner.invalidate_cuda_graphs()

        # 3. Abort 所有 running 序列（释放 KV Cache）
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
