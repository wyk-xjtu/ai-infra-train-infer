"""Rank-aware 训练产物写入器

根因：多进程(torchrun)下，各rank同时写同一文件导致数据损坏。
解决：全局文件仅rank0写入，per-rank文件各rank写自己的。

协议：
- 全局文件（config_snapshot, run_summary, loss_curve）: 仅 rank0 写
- Per-rank文件（metrics, memory_timeline）: 每个rank写自己的，文件名含rank
- Checkpoint: 根据ZeRO语义决定
"""
import os
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ArtifactWriter:
    """训练产物写入管理器

    使用方式:
        writer = ArtifactWriter(output_dir="outputs/exp1", rank=0, world_size=4)
        writer.write_main_json("run_summary.json", summary_dict)  # 仅rank0写入
        writer.write_rank_jsonl("metrics.jsonl", {"loss": 0.5})    # 写到 metrics_rank0.jsonl
    """

    def __init__(self, output_dir: str, rank: int = 0, world_size: int = 1):
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self._rank_files = {}  # 缓存打开的文件句柄
        os.makedirs(output_dir, exist_ok=True)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def write_main_json(self, filename: str, data: Any):
        """写入全局JSON文件（仅rank0执行）"""
        if not self.is_main:
            return
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def write_main_text(self, filename: str, text: str):
        """写入全局文本文件（仅rank0执行）"""
        if not self.is_main:
            return
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            f.write(text)

    def write_rank_jsonl(self, base_filename: str, row: dict):
        """写入per-rank的JSONL文件（每个rank写自己的）

        base_filename="metrics.jsonl" → 实际写入 "metrics_rank0.jsonl"
        每次运行首次打开时使用覆盖模式('w')，后续追加。
        """
        name, ext = os.path.splitext(base_filename)
        actual_filename = f"{name}_rank{self.rank}{ext}"
        path = os.path.join(self.output_dir, actual_filename)

        # 首次打开使用 'w' 模式（覆盖），避免跨运行数据混淆
        if actual_filename not in self._rank_files:
            self._rank_files[actual_filename] = open(path, 'w')

        f = self._rank_files[actual_filename]
        f.write(json.dumps(row, default=str) + '\n')
        f.flush()

    def write_rank_json(self, base_filename: str, data: Any):
        """写入per-rank的JSON文件"""
        name, ext = os.path.splitext(base_filename)
        actual_filename = f"{name}_rank{self.rank}{ext}"
        path = os.path.join(self.output_dir, actual_filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def close(self):
        """关闭所有打开的文件句柄"""
        for f in self._rank_files.values():
            f.close()
        self._rank_files.clear()

    def __del__(self):
        self.close()
