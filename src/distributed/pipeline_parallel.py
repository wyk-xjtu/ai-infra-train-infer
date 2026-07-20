"""
流水线并行上下文 (Pipeline Parallel Context)

职责:
- 基于 ParallelContext 提供的 PP 进程组信息，管理流水线 stage 的层划分
- 提供 stage 判定 (is_first_stage / is_last_stage) 与本 rank 的层范围
- 转发 P2P 通信所需的 prev_rank / next_rank

设计说明:
- 本模块只负责 "PP 拓扑与层切分" 这一层职责，不涉及具体的 1F1B 调度与模型切分
  （后者由后续块的 scheduler / model 切分逻辑负责）。
- PP 进程组的创建在 ParallelContext.init_distributed(pp_size=...) 中完成，本类只读取。

参考: docs/pipeline_parallel_design.md (50-66, 195-241)
"""

from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def split_layers_to_stages(num_layers: int, pp_size: int) -> List[Tuple[int, int]]:
    """将 num_layers 均匀分配到 pp_size 个 stage。

    均匀分配策略：每个 stage 分 ``num_layers // pp_size`` 层，
    余数 ``num_layers % pp_size`` 层依次分给前几个 stage（前几个 stage 多 1 层）。

    Args:
        num_layers: Transformer 层总数
        pp_size: 流水线并行度（stage 数）

    Returns:
        长度为 pp_size 的列表，每个元素是该 stage 的 ``(start_layer_idx, end_layer_idx)``，
        采用左闭右开区间 ``[start, end)``。

    示例:
        split_layers_to_stages(32, 4) -> [(0,8),(8,16),(16,24),(24,32)]   # 4×8
        split_layers_to_stages(33, 4) -> [(0,9),(9,17),(17,25),(25,33)]   # [9,8,8,8]
        split_layers_to_stages(6, 4)  -> [(0,2),(2,4),(4,5),(5,6)]        # [2,2,1,1]
    """
    assert pp_size >= 1, f"pp_size must be >= 1, got {pp_size}"
    assert num_layers >= 0, f"num_layers must be >= 0, got {num_layers}"

    # [FIX-D] num_layers < pp_size 时部分 stage 将分到 0 层（空 stage），通常非预期。
    if num_layers < pp_size:
        logger.warning(
            "split_layers_to_stages: num_layers=%d < pp_size=%d，部分 stage 将为 0 层（空 stage），"
            "通常非预期；请确认 pp_size 设置是否合理。",
            num_layers, pp_size,
        )

    layers_per_stage = num_layers // pp_size
    remainder = num_layers % pp_size

    stages: List[Tuple[int, int]] = []
    start = 0
    for i in range(pp_size):
        # 前 remainder 个 stage 多分 1 层
        end = start + layers_per_stage + (1 if i < remainder else 0)
        stages.append((start, end))
        start = end
    return stages


class PipelineParallelContext:
    """Pipeline Parallel 上下文管理。

    职责:
    - 记录 stage 划分信息（层 → stage 的映射）
    - 提供 stage 判定与本 rank 的层范围
    - 转发 P2P 通信的对端 rank（prev_rank / next_rank）

    所有 PP 拓扑信息（pp_size / pp_rank / pp_group / prev_rank / next_rank）
    均来自传入的 ParallelContext，本类只做层切分与便捷查询。
    """

    def __init__(self, parallel_context, num_layers: int):
        """
        Args:
            parallel_context: 已初始化的 ParallelContext（含 PP 进程组信息）
            num_layers: 模型的 Transformer 层总数
        """
        self._pc = parallel_context
        self.num_layers = num_layers
        # 全部 stage 的层划分（列表，索引为 pp_rank）
        self.stages: List[Tuple[int, int]] = split_layers_to_stages(
            num_layers, self.pp_size
        )

    # ------------------------------------------------------------------
    # PP 拓扑信息（转发自 ParallelContext）
    # ------------------------------------------------------------------

    @property
    def pp_rank(self) -> int:
        """当前进程的 stage id"""
        return self._pc.pp_rank

    @property
    def pp_size(self) -> int:
        """流水线并行度（stage 数）"""
        return self._pc.pp_size

    @property
    def pp_group(self):
        """PP 通信组（pp_size<=1 时为 None）"""
        return self._pc.pp_group

    @property
    def prev_rank(self) -> Optional[int]:
        """前一个 stage 的全局 rank；第一个 stage 或无 PP 时为 None"""
        return self._pc.prev_rank

    @property
    def next_rank(self) -> Optional[int]:
        """后一个 stage 的全局 rank；最后一个 stage 或无 PP 时为 None"""
        return self._pc.next_rank

    # ------------------------------------------------------------------
    # Stage 判定与层范围
    # ------------------------------------------------------------------

    @property
    def is_first_stage(self) -> bool:
        """是否为第一个 stage（持有 embedding）"""
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        """是否为最后一个 stage（持有 final norm + lm_head）"""
        return self.pp_rank == self.pp_size - 1

    @property
    def stage_layer_range(self) -> Tuple[int, int]:
        """本 rank 负责的层范围 ``(start, end)``（左闭右开）"""
        return self.stages[self.pp_rank]

    @property
    def num_layers_this_stage(self) -> int:
        """本 rank 负责的层数"""
        start, end = self.stage_layer_range
        return end - start

    def __repr__(self) -> str:
        return (
            f"PipelineParallelContext(pp_rank={self.pp_rank}, pp_size={self.pp_size}, "
            f"num_layers={self.num_layers}, stage_layer_range={self.stage_layer_range}, "
            f"is_first={self.is_first_stage}, is_last={self.is_last_stage})"
        )
