"""显存管理器 — OOM 预防与碎片监控"""
import torch
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class MemoryManager:
    """GPU 显存管理工具
    
    功能:
    1. 显存压力检查 — 超过阈值时 warning
    2. 碎片整理 — 调用 empty_cache 并监控效果
    3. OOM 缓冲区预留 — 防止突发 OOM
    """
    
    def __init__(self, device: torch.device, warning_threshold: float = 0.90,
                 critical_threshold: float = 0.95):
        self.device = device
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._buffer: Optional[torch.Tensor] = None
        
    def check_memory_pressure(self) -> Dict[str, float]:
        """检查当前显存使用率
        
        Returns:
            dict with keys: allocated_gb, reserved_gb, total_gb, utilization
        """
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "utilization": 0}
            
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        utilization = allocated / total
        
        stats = {
            "allocated_gb": allocated / 1e9,
            "reserved_gb": reserved / 1e9,
            "total_gb": total / 1e9,
            "utilization": utilization,
            "fragmentation_gb": (reserved - allocated) / 1e9,
        }
        
        if utilization > self.critical_threshold:
            logger.error(f"[MemoryManager] CRITICAL: GPU memory utilization {utilization:.1%} "
                        f"({stats['allocated_gb']:.1f}/{stats['total_gb']:.1f} GB)")
        elif utilization > self.warning_threshold:
            logger.warning(f"[MemoryManager] HIGH: GPU memory utilization {utilization:.1%} "
                          f"({stats['allocated_gb']:.1f}/{stats['total_gb']:.1f} GB)")
        
        return stats
    
    def defragment(self) -> float:
        """执行显存碎片整理，返回释放的显存量(GB)"""
        if not torch.cuda.is_available():
            return 0.0
            
        before = torch.cuda.memory_reserved(self.device)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved(self.device)
        freed_gb = (before - after) / 1e9
        
        if freed_gb > 0.1:
            logger.info(f"[MemoryManager] Defragmented: freed {freed_gb:.2f} GB")
        
        return freed_gb
    
    def reserve_buffer(self, size_mb: int = 512):
        """预留 OOM 缓冲区（可在 OOM 时释放以执行清理）"""
        if not torch.cuda.is_available() or self._buffer is not None:
            return
        try:
            self._buffer = torch.empty(size_mb * 1024 * 256, dtype=torch.float32,
                                       device=self.device)  # size_mb MB
            logger.info(f"[MemoryManager] Reserved {size_mb}MB OOM buffer")
        except torch.cuda.OutOfMemoryError:
            logger.warning("[MemoryManager] Cannot reserve OOM buffer (already tight)")
            self._buffer = None
    
    def release_buffer(self):
        """释放 OOM 缓冲区（在 OOM 紧急情况下调用）"""
        if self._buffer is not None:
            del self._buffer
            self._buffer = None
            torch.cuda.empty_cache()
            logger.info("[MemoryManager] Released OOM buffer")
    
    def get_stats(self) -> Dict[str, float]:
        """获取当前显存统计（无副作用）"""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
            "peak_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
            "total_gb": torch.cuda.get_device_properties(self.device).total_memory / 1e9,
        }
