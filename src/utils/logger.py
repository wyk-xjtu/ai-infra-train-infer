"""
结构化日志模块

提供统一的日志格式和配置:
- 格式: [时间] [级别] [模块] 消息
- 支持同时输出到console和文件
- 支持彩色终端输出
"""
import logging
import sys
from typing import Optional


_LOGGERS: dict = {}

# 日志格式
_LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ColorFormatter(logging.Formatter):
    """终端彩色日志格式化器"""

    COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "train_infer",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_color: bool = True,
) -> logging.Logger:
    """创建格式化的logger

    Args:
        name: logger名称
        level: 日志级别
        log_file: 可选的日志文件路径，设置后同时写入文件
        use_color: 终端输出是否使用彩色

    Returns:
        配置好的Logger实例
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    if use_color and sys.stdout.isatty():
        formatter = _ColorFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    else:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "train_infer") -> logging.Logger:
    """获取已创建的logger，若不存在则创建默认配置的logger

    Args:
        name: logger名称

    Returns:
        Logger实例
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    return setup_logger(name)
