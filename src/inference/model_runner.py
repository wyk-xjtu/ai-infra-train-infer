"""
模型执行器 — 串联 Model + KV Cache + Attention + CUDA Graph

核心职责:
1. 管理 KV Cache GPU 显存（通过 BlockPool 预分配）
2. 准备 Prefill/Decode 的输入数据（slot_mapping, block_tables 等）
3. 设置推理上下文 → 执行模型前向 → 采样
4. 管理 CUDA Graph 的录制和回放

与 nano-vllm ModelRunner 的区别:
- 复用项目已有的 BlockPool 和 KVCacheManager（不自己管理 block 分配）
- 不处理 Tensor Parallel 通信（由外层 ParallelTransformerModel 处理）
- 支持 graceful degradation（无 flash-attn/triton 时退回慢速路径）
- 无模型时仍可 fallback 到 mock 采样

集成方式:
- InferenceEngine 持有 ModelRunner 实例
- InferenceEngine._execute_prefill/decode 委托给 ModelRunner
"""
import logging
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .context import set_context, get_context, reset_context
from .attention import PagedAttention, store_kv_cache

logger = logging.getLogger(__name__)


@dataclass
class ModelRunnerConfig:
    """ModelRunner 配置 — 从 InferenceConfig 中提取模型相关参数"""
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096
    vocab_size: int = 151936
    block_size: int = 16
    max_num_blocks: int = 1024
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    # CUDA Graph 配置
    enable_cuda_graph: bool = True
    max_cuda_graph_batch_size: int = 64


class ModelRunner:
    """模型执行器 — 推理引擎的计算核心

    使用流程:
    1. 构造: ModelRunner(model, config)
    2. 绑定 KV Cache: bind_kv_cache(kv_cache_tensor)
    3. Prefill: prepare_prefill(sequences) → run(input_ids, positions) → sample(logits)
    4. Decode: prepare_decode(sequences) → run(input_ids, positions) → sample(logits)
    """

    def __init__(self, model: Optional[torch.nn.Module], config: ModelRunnerConfig):
        """
        Args:
            model: 模型对象（支持 model(input_ids, positions=positions) → logits）
                   如果为 None，则 run() 返回随机 logits（mock 模式）
            config: 模型和执行配置
        """
        self.model = model
        self.config = config
        self.block_size = config.block_size
        self._kv_cache: Optional[torch.Tensor] = None

        self._hf_model: Optional[torch.nn.Module] = None

        self._cuda_graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_output_buffers: Dict[int, torch.Tensor] = {}
        self._graph_pool: Optional[object] = None
        self._graph_vars: Optional[Dict[str, torch.Tensor]] = None
        self._graph_batch_sizes: List[int] = []

    def bind_kv_cache(self, kv_cache_tensor: torch.Tensor):
        """绑定预分配的 KV Cache tensor 到模型的 Attention 层

        Args:
            kv_cache_tensor: shape [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                            其中 [0]=K, [1]=V
        """
        self._kv_cache = kv_cache_tensor
        if self.model is not None:
            self._bind_kv_cache_to_model()

    def _bind_kv_cache_to_model(self):
        """将 KV Cache 分层绑定到模型中每个 PagedAttention 模块"""
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, PagedAttention):
                module.k_cache = self._kv_cache[0, layer_id]
                module.v_cache = self._kv_cache[1, layer_id]
                layer_id += 1

    def bind_hf_model(self, model: torch.nn.Module):
        """绑定 HuggingFace 模型用于 KV Cache 推理路径

        Args:
            model: HF AutoModelForCausalLM 实例（已加载权重、已移到目标设备）
        """
        self._hf_model = model
        logger.info("HF model bound to ModelRunner for KV Cache inference")

    @torch.inference_mode()
    def run(self, input_ids: torch.Tensor, positions: torch.Tensor,
            is_prefill: bool = True) -> torch.Tensor:
        """执行模型前向，返回 logits

        Args:
            input_ids: [num_tokens] token ids
            positions: [num_tokens] position ids
            is_prefill: 当前是否为 prefill 阶段

        Returns:
            logits: [num_tokens_or_batch, vocab_size]
        """
        if self._hf_model is not None:
            return self._run_hf_model(input_ids, positions, is_prefill)

        if self.model is None:
            num_tokens = input_ids.shape[0]
            logits = torch.randn(num_tokens, self.config.vocab_size,
                                 device=input_ids.device, dtype=torch.float32)
            reset_context()
            return logits

        # 判断是否使用 CUDA Graph（仅 decode 阶段、小 batch）
        use_graph = (
            not is_prefill
            and not self.config.enable_cuda_graph is False
            and input_ids.size(0) in self._cuda_graphs
        )

        if use_graph:
            logits = self._run_with_cuda_graph(input_ids, positions)
        else:
            logits = self.model(input_ids, positions=positions)

        reset_context()
        return logits

    def _run_hf_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                      is_prefill: bool) -> torch.Tensor:
        """使用 HF 模型执行 forward，通过 Paged KV Cache 管理历史 KV

        Prefill: 全序列 forward → KV 存入 paged cache → 返回 logits
        Decode: 从 paged cache 加载历史 KV → 单 token forward → 新 KV 存入 cache

        Args:
            input_ids: [num_tokens] for prefill, [batch_size] for decode
            positions: [num_tokens] for prefill, [batch_size] for decode
            is_prefill: True=prefill, False=decode

        Returns:
            logits: [num_tokens, vocab_size] for prefill, [batch, vocab_size] for decode
        """
        ctx = get_context()

        if is_prefill:
            logits = self._hf_prefill_forward(input_ids, positions, ctx)
        else:
            logits = self._hf_decode_forward(input_ids, positions, ctx)

        reset_context()
        return logits

    def _hf_prefill_forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                            ctx) -> torch.Tensor:
        """HF 模型 Prefill: 完整序列 forward，KV 存入 paged cache

        处理流程:
        1. 构造 position_ids 和 attention_mask
        2. forward 计算（use_cache=True 获取 KV）
        3. 将 past_key_values 通过 slot_mapping 写入 BlockPool 的 paged cache
        4. 返回所有 token 位置的 logits（供外层取最后一个做采样）
        """
        model = self._hf_model
        device = input_ids.device

        if input_ids.dim() == 1:
            input_ids_2d = input_ids.unsqueeze(0)
            positions_2d = positions.unsqueeze(0)
        else:
            input_ids_2d = input_ids
            positions_2d = positions

        seq_len = input_ids_2d.shape[1]
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

        outputs = model(
            input_ids=input_ids_2d,
            position_ids=positions_2d,
            attention_mask=attention_mask,
            use_cache=True,
        )

        # 将 KV 写入 paged cache
        if ctx.slot_mapping is not None and self._kv_cache is not None:
            self._store_hf_kv_to_paged_cache(outputs.past_key_values, ctx.slot_mapping)

        logits = outputs.logits.squeeze(0)
        return logits

    def _hf_decode_forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                           ctx) -> torch.Tensor:
        """HF 模型 Decode: 从 paged cache 加载历史 KV，单 token forward

        处理流程:
        1. 从 paged cache 按 block_table 加载历史 KV
        2. 构造 HF 兼容的 past_key_values
        3. 单 token forward
        4. 新生成的 KV 写入 paged cache 的对应 slot
        5. 返回 logits
        """
        model = self._hf_model
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if input_ids.dim() == 1:
            input_ids_2d = input_ids.unsqueeze(1)
            positions_2d = positions.unsqueeze(1)
        else:
            input_ids_2d = input_ids
            positions_2d = positions

        # 从 paged cache 加载历史 KV → 构造 past_key_values
        past_key_values = self._load_kv_from_paged_cache(
            ctx.block_tables, ctx.context_lens
        )

        # attention_mask 需要覆盖完整序列长度 (past + current)
        # context_lens 包含了当前 token（因为 engine 在 append_token 后才 forward）
        # 实际 past 长度 = context_lens - 1（当前 token 不在 past 中）
        if ctx.context_lens is not None:
            max_past_len = int(ctx.context_lens.max().item()) - 1
            attention_mask = torch.ones(
                batch_size, max_past_len + 1, dtype=torch.long, device=device
            )
        else:
            attention_mask = None

        outputs = model(
            input_ids=input_ids_2d,
            position_ids=positions_2d,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # 将新 token 的 KV 写入 paged cache
        if ctx.slot_mapping is not None and self._kv_cache is not None:
            self._store_hf_decode_kv_to_cache(outputs.past_key_values, ctx)

        logits = outputs.logits.squeeze(1)
        return logits

    def _store_hf_kv_to_paged_cache(self, past_key_values, slot_mapping: torch.Tensor):
        """将 HF 模型 prefill 输出的 past_key_values 存入 paged KV cache

        past_key_values 格式:
        - DynamicCache: past_key_values.key_cache[layer] = [batch, heads, seq, dim]
        - Legacy tuple: past_key_values[layer] = (K, V), K=[batch, heads, seq, dim]

        paged cache 格式:
        - self._kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        - store via slot_mapping: each token → physical slot
        """
        num_layers = self.config.num_layers
        cache_dtype = self._kv_cache.dtype

        for layer_idx in range(num_layers):
            k, v = self._extract_layer_kv(past_key_values, layer_idx)
            # → [seq_len, num_kv_heads, head_dim] for store_kv_cache
            k = k.squeeze(0).transpose(0, 1).contiguous().to(cache_dtype)
            v = v.squeeze(0).transpose(0, 1).contiguous().to(cache_dtype)

            # 使用现有的 store_kv_cache 函数写入 paged cache
            # k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            k_cache = self._kv_cache[0, layer_idx]
            v_cache = self._kv_cache[1, layer_idx]
            store_kv_cache(k, v, k_cache, v_cache, slot_mapping)

    def _store_hf_decode_kv_to_cache(self, past_key_values, ctx):
        """将 decode 步骤新生成的 KV (最后一个 token) 存入 paged cache

        decode forward 输出的 past_key_values 包含 past + new，
        只需存储最后一个 token（新生成的）的 KV。
        """
        num_layers = self.config.num_layers
        slot_mapping = ctx.slot_mapping  # [batch_size] — 每个序列新 token 的 slot
        cache_dtype = self._kv_cache.dtype

        for layer_idx in range(num_layers):
            k, v = self._extract_layer_kv(past_key_values, layer_idx)
            # k: [batch, heads, past_len+1, dim] → 取最后一个 token
            k_new = k[:, :, -1:, :]  # [batch, heads, 1, dim]
            v_new = v[:, :, -1:, :]

            # → [batch, num_kv_heads, head_dim] → reshape for store
            k_new = k_new.squeeze(2).transpose(0, 1)  # [heads, batch, dim]
            k_new = k_new.transpose(0, 1).contiguous().to(cache_dtype)  # [batch, heads, dim]
            v_new = v_new.squeeze(2).transpose(0, 1)
            v_new = v_new.transpose(0, 1).contiguous().to(cache_dtype)

            k_cache = self._kv_cache[0, layer_idx]
            v_cache = self._kv_cache[1, layer_idx]
            store_kv_cache(k_new, v_new, k_cache, v_cache, slot_mapping)

    def _load_kv_from_paged_cache(self, block_tables: torch.Tensor,
                                   context_lens: torch.Tensor):
        """从 paged KV cache 加载历史 KV，构造 HF 兼容的 past_key_values

        Args:
            block_tables: [batch, max_blocks] 物理 block id 映射
            context_lens: [batch] 每个序列的上下文长度（含当前 token）

        Returns:
            past_key_values: 可直接传入 HF model 的 past_key_values
                            格式为 tuple of (K, V) per layer
                            K/V shape: [batch, num_kv_heads, past_len, head_dim]
        """
        if self._kv_cache is None or block_tables is None or context_lens is None:
            return None

        batch_size = block_tables.shape[0]
        num_layers = self.config.num_layers
        num_kv_heads = self.config.num_kv_heads
        head_dim = self.config.head_dim
        block_size = self.block_size
        device = block_tables.device

        # past_len = context_lens - 1 (不含当前正在 forward 的 token)
        past_lens = context_lens - 1
        max_past_len = int(past_lens.max().item())

        if max_past_len <= 0:
            return None

        # 预分配输出 tensors
        all_keys = []
        all_values = []

        for layer_idx in range(num_layers):
            # k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            k_cache = self._kv_cache[0, layer_idx]
            v_cache = self._kv_cache[1, layer_idx]

            # 展平为 [total_slots, num_kv_heads, head_dim]
            num_blocks_total = k_cache.shape[0]
            k_flat = k_cache.view(num_blocks_total * block_size, num_kv_heads, head_dim)
            v_flat = v_cache.view(num_blocks_total * block_size, num_kv_heads, head_dim)

            # 每个 (batch_idx, token_pos) → physical_slot
            slot_indices = torch.zeros(
                batch_size, max_past_len, dtype=torch.long, device=device
            )

            for b in range(batch_size):
                past_len_b = int(past_lens[b].item())
                for pos in range(past_len_b):
                    block_idx = pos // block_size
                    offset = pos % block_size
                    physical_block = int(block_tables[b, block_idx].item())
                    slot_indices[b, pos] = physical_block * block_size + offset

            # Gather: [batch, max_past_len, num_kv_heads, head_dim]
            # 使用 index_select + reshape（比逐元素 gather 更高效）
            slot_flat = slot_indices.view(-1)  # [batch * max_past_len]
            k_gathered = k_flat[slot_flat].view(
                batch_size, max_past_len, num_kv_heads, head_dim
            )
            v_gathered = v_flat[slot_flat].view(
                batch_size, max_past_len, num_kv_heads, head_dim
            )

            for b in range(batch_size):
                past_len_b = int(past_lens[b].item())
                if past_len_b < max_past_len:
                    k_gathered[b, past_len_b:] = 0
                    v_gathered[b, past_len_b:] = 0

            k_layer = k_gathered.transpose(1, 2).contiguous()
            v_layer = v_gathered.transpose(1, 2).contiguous()

            all_keys.append(k_layer)
            all_values.append(v_layer)

        try:
            from transformers import DynamicCache
            cache = DynamicCache()
            for layer_idx in range(num_layers):
                cache.update(all_keys[layer_idx], all_values[layer_idx], layer_idx)
            return cache
        except ImportError:
            return tuple(
                (all_keys[i], all_values[i]) for i in range(num_layers)
            )

    def _extract_layer_kv(self, past_key_values, layer_idx: int):
        """从 HF past_key_values 中提取指定层的 K, V

        兼容多种 transformers 版本:
        - transformers 5.x DynamicCache: cache.layers[idx].keys / .values
        - transformers 4.x DynamicCache: cache.key_cache[idx] / cache.value_cache[idx]
        - Legacy tuple 格式: past_key_values[layer] = (K, V)

        Returns:
            (k, v): 每个 shape [batch, num_kv_heads, seq_len, head_dim]
        """
        if hasattr(past_key_values, 'layers'):
            layer = past_key_values.layers[layer_idx]
            return layer.keys, layer.values

        if hasattr(past_key_values, 'key_cache'):
            k = past_key_values.key_cache[layer_idx]
            v = past_key_values.value_cache[layer_idx]
            return k, v

        layer_kv = past_key_values[layer_idx]
        if isinstance(layer_kv, (tuple, list)):
            return layer_kv[0], layer_kv[1]

        raise ValueError(f"Unsupported past_key_values format: {type(past_key_values)}")

    def prepare_prefill(self, sequences: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备 Prefill 输入并设置推理上下文

        Args:
            sequences: 序列信息列表，每项包含:
                - 'token_ids': List[int] — prompt token ids
                - 'block_table': List[int] — 物理 block id 列表
                - 'num_cached_tokens': int — prefix cache 已缓存的 token 数
                - 'num_scheduled_tokens': int — 本次要计算的 token 数

        Returns:
            (input_ids, positions) tensors

        Side effect:
            设置全局 InferenceContext（Attention 层会读取）
        """
        input_ids_list = []
        positions_list = []
        slot_mapping_list = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        has_prefix_cache = False

        for seq in sequences:
            start = seq['num_cached_tokens']
            num_tokens = seq['num_scheduled_tokens']
            end = start + num_tokens

            # 收集要处理的 token ids 和 positions
            input_ids_list.extend(seq['token_ids'][start:end])
            positions_list.extend(range(start, end))

            # 计算 slot_mapping: 每个 token 对应的物理 slot 位置
            block_table = seq.get('block_table', [])
            if block_table:
                start_block = start // self.block_size
                end_block = (end + self.block_size - 1) // self.block_size
                for i in range(start_block, end_block):
                    slot_start = block_table[i] * self.block_size
                    if i == start_block:
                        slot_start += start % self.block_size
                    if i != end_block - 1:
                        slot_end = block_table[i] * self.block_size + self.block_size
                    else:
                        slot_end = block_table[i] * self.block_size + end - i * self.block_size
                    slot_mapping_list.extend(range(slot_start, slot_end))

            cu_seqlens_q.append(cu_seqlens_q[-1] + num_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)  # K 包含所有 token（含 cached）
            max_seqlen_q = max(max_seqlen_q, num_tokens)
            max_seqlen_k = max(max_seqlen_k, end)

            if start > 0:
                has_prefix_cache = True

        device = self._get_device()
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        positions = torch.tensor(positions_list, dtype=torch.long, device=device)

        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int32, device=device) if slot_mapping_list else None
        block_tables = self._prepare_block_tables(sequences, device) if has_prefix_cache else None

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )

        return input_ids, positions

    def prepare_decode(self, sequences: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备 Decode 输入（每个序列只处理最后 1 个 token）

        Args:
            sequences: 序列信息列表，每项包含:
                - 'token_ids': List[int] — 全部 token ids（含已生成的）
                - 'block_table': List[int] — 物理 block id 列表
                - 'last_block_num_tokens': int — 最后一个 block 中的 token 数

        Returns:
            (input_ids, positions) tensors
        """
        input_ids_list = []
        positions_list = []
        slot_mapping_list = []
        context_lens_list = []

        for seq in sequences:
            last_token = seq['token_ids'][-1]
            position = len(seq['token_ids']) - 1
            context_len = len(seq['token_ids'])

            block_table = seq['block_table']
            last_block_tokens = seq.get('last_block_num_tokens', None)
            if last_block_tokens is None:
                last_block_tokens = (position % self.block_size) + 1
            slot = block_table[-1] * self.block_size + last_block_tokens - 1

            input_ids_list.append(last_token)
            positions_list.append(position)
            slot_mapping_list.append(slot)
            context_lens_list.append(context_len)

        device = self._get_device()
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        positions = torch.tensor(positions_list, dtype=torch.long, device=device)

        set_context(
            is_prefill=False,
            slot_mapping=torch.tensor(slot_mapping_list, dtype=torch.int32, device=device),
            context_lens=torch.tensor(context_lens_list, dtype=torch.int32, device=device),
            block_tables=self._prepare_block_tables(sequences, device),
        )

        return input_ids, positions

    def sample(self, logits: torch.Tensor, temperature: float = 0.7,
               top_p: float = 0.9) -> List[int]:
        """从 logits 采样 token ids

        Args:
            logits: [batch_size, vocab_size]
            temperature: 采样温度（0 表示 greedy）
            top_p: nucleus sampling 阈值

        Returns:
            采样得到的 token id 列表
        """
        # 只取最后一个 token 位置的 logits（如果是 prefill 输出多个 token）
        if logits.dim() == 2 and logits.size(0) > 1:
            # 对于 prefill，通常只需要最后一个 token 的 logits 用于采样
            # 但如果是 batch decode，每个位置都需要采样
            pass

        if temperature <= 0:
            return torch.argmax(logits, dim=-1).tolist()

        scaled_logits = logits.float() / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)

            # 移除累积概率超过 top_p 的 tokens
            mask = cumulative > top_p
            # 右移 mask，保留第一个超过 top_p 的 token
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = float('-inf')

            scaled_logits = torch.zeros_like(scaled_logits).scatter_(
                -1, sorted_indices, sorted_logits
            )

        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()


    def capture_cuda_graphs(self, batch_sizes: Optional[List[int]] = None):
        """为指定 batch sizes 录制 CUDA Graph（仅 decode 阶段使用）

        Args:
            batch_sizes: 要录制的 batch size 列表
                        默认 [1, 2, 4, 8, 16, 32, 64]
        """
        if not torch.cuda.is_available() or self.model is None:
            return

        if batch_sizes is None:
            max_bs = min(self.config.max_cuda_graph_batch_size, 64)
            batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        self._graph_batch_sizes = sorted(batch_sizes)
        max_bs = max(batch_sizes)

        device = self.config.device
        # 预分配共享 buffer
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=device)
        positions = torch.zeros(max_bs, dtype=torch.long, device=device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        max_blocks_per_seq = self.config.max_num_blocks
        block_tables = torch.zeros(max_bs, max_blocks_per_seq, dtype=torch.int32, device=device)

        for bs in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs],
                       context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            _ = self.model(input_ids[:bs], positions=positions[:bs])
            with torch.cuda.graph(graph, pool=self._graph_pool):
                output_buffer = self.model(input_ids[:bs], positions=positions[:bs])
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            self._cuda_graphs[bs] = graph
            self._graph_output_buffers[bs] = output_buffer
            torch.cuda.synchronize()
            reset_context()

        self._graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
        }

    def _run_with_cuda_graph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """使用 CUDA Graph 回放执行 decode forward"""
        bs = input_ids.size(0)
        graph_bs = next((x for x in self._graph_batch_sizes if x >= bs), None)
        if graph_bs is None or graph_bs not in self._cuda_graphs:
            return self.model(input_ids, positions=positions)

        ctx = get_context()
        gv = self._graph_vars

        # In-place 拷贝输入到 graph buffer
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        if ctx.slot_mapping is not None:
            gv["slot_mapping"][:bs] = ctx.slot_mapping
        gv["context_lens"].zero_()
        if ctx.context_lens is not None:
            gv["context_lens"][:bs] = ctx.context_lens
        if ctx.block_tables is not None:
            bt = ctx.block_tables
            gv["block_tables"][:bs, :bt.size(1)] = bt

        self._cuda_graphs[graph_bs].replay()
        # 从 graph 录制时的 output buffer 中提取 logits
        logits = self._extract_logits_from_graph_buffer(graph_bs, bs)
        return logits

    def _extract_logits_from_graph_buffer(self, graph_bs: int, bs: int) -> torch.Tensor:
        """从 CUDA Graph 录制时的 output buffer 中提取有效的 logits

        Args:
            graph_bs: graph 录制时的 batch size（buffer 的完整大小）
            bs: 实际请求的 batch size（<=graph_bs）

        Returns:
            logits: [bs, vocab_size] — 仅返回实际有效的部分
        """
        output_buffer = self._graph_output_buffers[graph_bs]
        return output_buffer[:bs]

    def invalidate_cuda_graphs(self):
        """清除所有 CUDA Graph（权重更新后调用）"""
        self._cuda_graphs.clear()
        self._graph_output_buffers.clear()
        self._graph_pool = None
        self._graph_vars = None
        self._graph_batch_sizes = []


    def _prepare_block_tables(self, sequences: List[dict], device: str) -> torch.Tensor:
        """构造 block_tables tensor [num_seqs, max_blocks]"""
        block_tables_list = [seq.get('block_table', []) for seq in sequences]
        if not block_tables_list or all(len(bt) == 0 for bt in block_tables_list):
            return None
        max_blocks = max(len(bt) for bt in block_tables_list)
        tables = torch.full(
            (len(sequences), max_blocks), -1,
            dtype=torch.int32, device=device,
        )
        for i, bt in enumerate(block_tables_list):
            for j, block_id in enumerate(bt):
                tables[i, j] = block_id
        return tables

    def _get_device(self) -> str:
        """获取当前设备"""
        if torch.cuda.is_available():
            return self.config.device
        return "cpu"
