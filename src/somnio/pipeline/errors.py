"""Errors raised by `somnio.pipeline`."""

from __future__ import annotations


class PipelineExecutionError(RuntimeError):
    """Base error for pipeline execution failures."""


class OutputConflictError(PipelineExecutionError):
    """Raised when runnable steps would write overlapping outputs concurrently."""


class DeadEndError(PipelineExecutionError):
    """Raised when no runnable steps exist but steps remain."""
