"""Logging utilities for the zutils CLI.

This module configures Loguru as the primary logger and bridges standard-library
``logging`` records into Loguru.
"""

from __future__ import annotations

import re
from datetime import datetime
import logging
import sys
from pathlib import Path
from types import FrameType

from loguru import logger


DEFAULT_STDERR_LOG_LEVEL = "INFO"
DEFAULT_FILE_LOG_LEVEL = "DEBUG"
DEFAULT_LOG_DIRECTORY = Path("logs")
DEFAULT_LOG_FILE_NAME = "{command}_{timestamp}.log"

# Redact auth_token (and similar) from log messages so tokens never appear in logs.
_REDACT_PATTERN = re.compile(
    r"(auth_token|token)=[^&\s\"']+",
    re.IGNORECASE,
)
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


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


def _redact_sensitive(record: dict) -> bool:
    """Filter that redacts token values in log messages. Mutates record and returns True."""
    record["message"] = _REDACT_PATTERN.sub(r"\1=***", record["message"])
    return True


def setup_logging(
    *,
    stderr_level: str = DEFAULT_STDERR_LOG_LEVEL,
    file_level: str = DEFAULT_FILE_LOG_LEVEL,
    log_file: Path | None = None,
) -> None:
    """Configure Loguru and bridge stdlib logging to it."""

    logger.remove()
    logger.add(sys.stderr, level=stderr_level, filter=_redact_sensitive)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=file_level, filter=_redact_sensitive)
        logger.info("Logging to {}", log_file)

    # Bridge stdlib logging -> Loguru (for library modules using `logging`)
    logging.root.handlers = [_InterceptHandler()]
    logging.root.setLevel(logging.NOTSET)


def create_log_file_path(command: str) -> Path:
    """Create a log file path for a given script name."""
    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    return DEFAULT_LOG_DIRECTORY / DEFAULT_LOG_FILE_NAME.format(
        command=command, timestamp=timestamp
    )
