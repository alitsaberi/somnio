"""Configuration objects for `somnio.pipeline`.

Design goals:
- Keep configs trivially serializable (dataclasses of primitives).
- Allow "import-string" transform specs for process-based execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from somnio.data import TimeSeries

Bundle = dict[str, TimeSeries]
Backend = Literal["processes", "threads"]
Transform = Callable[[Bundle], Bundle]


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Import-string transform specification.

    Attributes:
        target: Import string in the form "pkg.module:callable".
        kwargs: Keyword arguments passed to the callable.
    """

    target: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Step:
    """A single pipeline step: Bundle -> Bundle.

    Attributes:
        name: Human-readable identifier (used in diagnostics).
        inputs: Names required to exist in the data store for this step to run.
        outputs: Names that this step will produce (reserved for conflict checks).
        transform: Either an import-string TransformSpec (recommended), or a
            direct callable (only safe for in-process execution).
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    transform: TransformSpec | Transform

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Step.name must be non-empty")
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError(f"Step.inputs must be unique for {self.name!r}")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError(f"Step.outputs must be unique for {self.name!r}")
        if not all(self.inputs):
            raise ValueError(f"Step.inputs must be non-empty strings for {self.name!r}")
        if not all(self.outputs):
            raise ValueError(
                f"Step.outputs must be non-empty strings for {self.name!r}"
            )


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Ordered list of steps executed with inferred dependencies.

    Step ordering is used for deterministic result commits when multiple steps
    finish concurrently.
    """

    steps: tuple[Step, ...]

    @classmethod
    def from_steps(cls, steps: list[Step]) -> "Pipeline":
        return cls(steps=tuple(steps))
