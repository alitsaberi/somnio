"""Pipeline execution for named TimeSeries bundles.

Public API intentionally stays small so we can later add CLI/YAML integration
without changing core semantics.
"""

from __future__ import annotations

from .types import Bundle, Pipeline, Step, Transform, TransformSpec
from .engine import execute
from .errors import DeadEndError, OutputConflictError, PipelineExecutionError

__all__ = [
    "Bundle",
    "DeadEndError",
    "OutputConflictError",
    "Pipeline",
    "PipelineExecutionError",
    "Step",
    "Transform",
    "TransformSpec",
    "execute",
]
