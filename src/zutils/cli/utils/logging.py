"""Logging utilities for the zutils CLI.

This module configures Loguru as the primary logger and bridges standard-library
``logging`` records into Loguru.
"""

from __future__ import annotations

import logging
import sys
from types import FrameType

from loguru import logger

DEFAULT_LOG_LEVEL = "INFO"


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    *,
    level: str = DEFAULT_LOG_LEVEL,
) -> None:
    """Configure Loguru and bridge stdlib logging to it."""

    logger.remove()
    logger.add(sys.stderr, level=level)

    # Bridge stdlib logging -> Loguru (for library modules using `logging`)
    logging.root.handlers = [_InterceptHandler()]
    logging.root.setLevel(logging.NOTSET)
