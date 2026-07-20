"""
自研 Kernel 实现模块

包含 Triton 实现的高性能算子（教学/展示用途）。
实际生产环境使用对应的开源库（如 flash-attn）。
"""

try:
    from .flash_attention_triton import flash_attention_triton
except ImportError:
    flash_attention_triton = None

__all__ = ["flash_attention_triton"]
