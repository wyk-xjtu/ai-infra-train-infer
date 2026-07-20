"""
CUDA Graph 加速 Decode 阶段

===================== 核心思想 =====================

Decode 阶段的性能瓶颈:
  每次 forward 只生成 1 个 token，计算量极小（一个 token 的矩阵乘法），
  但每次需要依次 launch 几十个 CUDA kernel:
  - Embedding lookup, RoPE, QKV projection, Attention, FFN, LayerNorm, LM head...
  - 每个 kernel launch 有 ~5-15μs 的 CPU→GPU 调度开销
  - 32层 Transformer ≈ 100+ kernels × 10μs ≈ 1ms 纯 launch 开销
  - 而实际 GPU 计算可能只需 0.5ms → launch 开销占 2/3!

CUDA Graph 的解法:
  1. 录制 (Capture): 执行一次 forward，将所有 kernel 调用序列记录为一个 Graph
  2. 回放 (Replay): 之后每次 forward 只需一次 API 调用就重放整个 kernel 序列
  → 将 100+ 次 kernel launch 变为 1 次 graph replay → launch 开销从 1ms 降到 ~10μs
  → Decode 延迟降低 10-30%

限制:
  - 录制时的 tensor shape 必须固定（batch_size, seq_len 不能变）
  - 不能包含动态控制流（if/for 分支依赖 tensor 值）
  - Graph 内不能分配新显存（所有 tensor 必须提前分配好）
  - 需要为不同 batch_size 分别录制
"""
import torch
from typing import Dict, List, Optional, Callable, Tuple


class CUDAGraphRunner:
    """CUDA Graph 执行器
    
    为不同 batch_size 预录制 compute graph，decode 时选择匹配的 graph 回放。
    
    使用流程:
    1. 初始化时调用 capture() 为各个 batch_size 录制
    2. 每步 decode 调用 replay() 回放
    3. 权重更新后调用 invalidate() 清除
    """

    def __init__(self, max_batch_size: int = 32,
                 supported_batch_sizes: Optional[List[int]] = None):
        """
        Args:
            max_batch_size: 最大支持的 batch_size
            supported_batch_sizes: 预录制的 batch_size 列表
                默认 [1, 2, 4, 8, 16, 32] — 覆盖常见大小，padding 浪费最多 ~50%
        """
        if supported_batch_sizes is None:
            supported_batch_sizes = [bs for bs in [1, 2, 4, 8, 16, 32, 64, 128]
                                     if bs <= max_batch_size]
            if not supported_batch_sizes:
                supported_batch_sizes = [1]

        self.max_batch_size = max_batch_size
        self.supported_batch_sizes = sorted(supported_batch_sizes)

        # batch_size → CUDAGraph 对象
        self._graphs: Dict[int, "torch.cuda.CUDAGraph"] = {}
        # batch_size → 输入buffer (预分配的固定shape tensor)
        self._input_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        # batch_size → 输出buffer
        self._output_buffers: Dict[int, torch.Tensor] = {}
        # 共享的 graph memory pool (所有graph共用，减少显存碎片)
        self._graph_pool = None

        # 统计
        self._replay_count = 0
        self._capture_count = 0

    def capture(self, model_forward_fn: Callable,
                batch_size: int,
                input_ids_shape: Tuple[int, ...],
                hidden_size: int = 4096,
                device: str = "cuda",
                dtype: torch.dtype = torch.float16,
                num_warmup: int = 3,
                **extra_buffer_shapes):
        """录制 CUDA Graph
        
        步骤:
        1. Warmup: 先执行 num_warmup 次 forward（让 CUDA runtime 预分配所有内存）
        2. 创建 CUDAGraph 对象
        3. 开始录制 (torch.cuda.graph context)
        4. 执行一次 forward (所有 kernel 被记录)
        5. 结束录制
        6. 保存 input/output buffer 引用
        
        Args:
            model_forward_fn: 模型 forward 函数, 签名: fn(input_ids, positions, **kwargs) -> hidden_states
            batch_size: 要录制的 batch_size
            input_ids_shape: input_ids tensor 的 shape, 通常 (batch_size,) for decode
            hidden_size: 模型隐藏层维度
            device: 设备
            dtype: 数据类型
            num_warmup: warmup 次数
            **extra_buffer_shapes: 额外输入 buffer 的 shape 定义
        """
        if not torch.cuda.is_available():
            return  # CPU 环境跳过

        # 1. 预分配 input/output buffers (固定shape，graph期间不能变)
        input_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        output_buffer = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)

        # 额外 buffers (如 block_tables, slot_mapping 等)
        extra_buffers = {}
        for name, shape in extra_buffer_shapes.items():
            extra_buffers[name] = torch.zeros(shape, dtype=torch.int32, device=device)

        # 2. Warmup — 确保所有 lazy 分配完成
        for _ in range(num_warmup):
            output_buffer[:] = model_forward_fn(input_ids, positions, **extra_buffers)

        # 3. 录制 Graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool):
            output_buffer[:] = model_forward_fn(input_ids, positions, **extra_buffers)

        # 使用第一个 graph 的 pool 给后续 graph 共享
        if self._graph_pool is None:
            self._graph_pool = graph.pool()

        # 4. 保存引用
        self._graphs[batch_size] = graph
        self._input_buffers[batch_size] = {
            "input_ids": input_ids,
            "positions": positions,
            **extra_buffers,
        }
        self._output_buffers[batch_size] = output_buffer
        self._capture_count += 1

        torch.cuda.synchronize()

    def replay(self, batch_size: int, input_ids: torch.Tensor,
               positions: Optional[torch.Tensor] = None,
               **kwargs) -> Optional[torch.Tensor]:
        """回放 CUDA Graph
        
        步骤:
        1. 找到 >= batch_size 的最小预录制 graph
        2. 将实际输入 copy 到 input buffer (in-place, 不分配新内存)
        3. graph.replay() — 一次性执行所有录制的 kernel
        4. 从 output buffer 中截取实际 batch_size 的结果
        
        Args:
            batch_size: 实际的 batch_size
            input_ids: 实际的 input token ids
            positions: 位置编码
            **kwargs: 额外的输入 (block_tables 等)
            
        Returns:
            模型输出 hidden_states, shape=(batch_size, hidden_size)
            如果无可用 graph 则返回 None (调用者应 fallback 到 eager mode)
        """
        padded_bs = self._get_padded_batch_size(batch_size)
        if padded_bs is None or padded_bs not in self._graphs:
            return None  # 无匹配的 graph，需 eager fallback

        graph = self._graphs[padded_bs]
        buffers = self._input_buffers[padded_bs]

        # In-place copy 输入到 buffer（必须 in-place，不能分配新 tensor）
        # 确保 device 和 dtype 匹配
        input_ids_src = input_ids[:batch_size]
        if input_ids_src.device != buffers["input_ids"].device or input_ids_src.dtype != buffers["input_ids"].dtype:
            input_ids_src = input_ids_src.to(device=buffers["input_ids"].device, dtype=buffers["input_ids"].dtype)
        buffers["input_ids"][:batch_size].copy_(input_ids_src)
        if positions is not None:
            positions_src = positions[:batch_size]
            if positions_src.device != buffers["positions"].device or positions_src.dtype != buffers["positions"].dtype:
                positions_src = positions_src.to(device=buffers["positions"].device, dtype=buffers["positions"].dtype)
            buffers["positions"][:batch_size].copy_(positions_src)

        # 拷贝额外输入
        for key, value in kwargs.items():
            if key in buffers and isinstance(value, torch.Tensor):
                # 确保 device 和 dtype 匹配
                value_src = value
                if value_src.device != buffers[key].device or value_src.dtype != buffers[key].dtype:
                    value_src = value_src.to(device=buffers[key].device, dtype=buffers[key].dtype)
                # 只拷贝有效部分
                actual_shape = value_src.shape
                buf_shape = buffers[key].shape
                slices = tuple(slice(0, min(a, b)) for a, b in zip(actual_shape, buf_shape))
                buffers[key][slices].copy_(value_src[slices])

        # 回放 Graph — 一次 API 调用重放所有 kernel
        graph.replay()
        self._replay_count += 1

        # 截取有效结果
        return self._output_buffers[padded_bs][:batch_size].clone()

    def _get_padded_batch_size(self, actual_batch_size: int) -> Optional[int]:
        """找到 >= actual_batch_size 的最小预录制 batch_size
        
        例: supported = [1,2,4,8,16,32], actual = 5 → 返回 8
        """
        for bs in self.supported_batch_sizes:
            if bs >= actual_batch_size:
                return bs
        return None  # 超出最大支持范围

    def is_captured(self, batch_size: int) -> bool:
        """检查指定 batch_size 是否已录制"""
        padded = self._get_padded_batch_size(batch_size)
        return padded is not None and padded in self._graphs

    def invalidate(self):
        """清除所有已录制的 Graph
        
        权重更新后必须调用:
        - Graph 中记录的是旧权重对应的 kernel 参数
        - 权重更新后 kernel 的参数（权重指针/值）已经变了
        - 必须重新录制才能使用新权重
        """
        self._graphs.clear()
        self._input_buffers.clear()
        self._output_buffers.clear()
        self._graph_pool = None

    def get_stats(self) -> dict:
        """统计信息"""
        return {
            "captured_sizes": list(self._graphs.keys()),
            "total_captures": self._capture_count,
            "total_replays": self._replay_count,
            "graph_pool_active": self._graph_pool is not None,
            "supported_batch_sizes": self.supported_batch_sizes,
        }
