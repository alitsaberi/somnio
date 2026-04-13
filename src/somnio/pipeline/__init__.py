"""Pipeline execution for named TimeSeries bundles.

Public API intentionally stays small so we can later add CLI/YAML integration
without changing core semantics.
"""

from __future__ import annotations

from .config import Pipeline, Step, TransformSpec
from .engine import execute
from .errors import DeadEndError, OutputConflictError, PipelineExecutionError

__all__ = [
    "DeadEndError",
    "OutputConflictError",
    "Pipeline",
    "PipelineExecutionError",
    "Step",
    "TransformSpec",
    "execute",
]
