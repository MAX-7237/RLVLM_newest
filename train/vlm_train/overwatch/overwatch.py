"""
overwatch.py

Utility class for creating a centralized/standardized logger (built on Rich) and accelerate handler.
"""
import logging
import logging.config
import os
from logging import LoggerAdapter
from typing import Union

# Overwatch Default Format String
RICH_FORMATTER, DATEFMT = "| >> %(message)s", "%m/%d [%H:%M:%S]"

# Set Logging Configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "simple-console",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOG_CONFIG)


# === rank_zero_only decorator ===
def rank_zero_only(fn):
    def wrapper(*args, **kwargs):
        # 优先使用 accelerate.PartialState
        try:
            from accelerate import PartialState
            if PartialState().is_main_process:
                return fn(*args, **kwargs)
            return None
        except Exception:
            pass

        # 退回到 torch.distributed
        try:
            import torch.distributed as dist
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                return fn(*args, **kwargs)
            return None
        except Exception:
            return fn(*args, **kwargs)
    return wrapper


# === Custom Contextual Logging Logic ===
class ContextAdapter(LoggerAdapter):
    CTX_PREFIXES = {
        0: "[*] "} | {idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]}

    def process(self, msg, kwargs):
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class DistributedOverwatch:
    def __init__(self, name: str) -> None:
        from accelerate import PartialState

        self.logger, self.distributed_state = ContextAdapter(
            logging.getLogger(name)), PartialState()

        # Logger delegation
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Only INFO on main rank, ERROR otherwise
        self.logger.setLevel(
            logging.INFO if self.distributed_state.is_main_process else logging.ERROR)

        # >>> add rank_zero_only here <<<
        self.rank_zero_only = rank_zero_only


class PureOverwatch:
    def __init__(self, name: str) -> None:
        self.logger = ContextAdapter(logging.getLogger(name))

        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        self.logger.setLevel(logging.INFO)

        # >>> add rank_zero_only here too <<<
        self.rank_zero_only = rank_zero_only


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    return DistributedOverwatch(name) if int(os.environ.get("WORLD_SIZE", -1)) > 1 else PureOverwatch(name)
