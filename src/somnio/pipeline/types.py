"""Runtime types for `somnio.pipeline`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from somnio.data import TimeSeries

Bundle = dict[str, TimeSeries]


@runtime_checkable
class Transform(Protocol):
    """Protocol for pipeline transforms.

    A transform maps a bundle of named `TimeSeries` to another bundle.
    Any callable matching the signature ``(bundle: Bundle, **kwargs) -> Bundle``
    satisfies this protocol structurally — no inheritance required.
    """

    def __call__(
        self, bundle: Bundle, /, **kwargs: Any
    ) -> Bundle: ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Specification for a single transform.

    `target` is either an import string (``"pkg.module:callable"``) that resolves to a `Transform` instance or a `Transform` instance itself.

    Attributes:
        target: Import string ``"pkg.module:callable"`` or a `Transform` instance.
        kwargs: Keyword arguments forwarded to the transform on every invocation.
    """

    target: str | Transform
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Step:
    """A single pipeline step: ``Bundle -> Bundle``.

    Attributes:
        name: Human-readable identifier (used in diagnostics).
        inputs: Names required to exist in the data store for this step to run.
        outputs: Names that this step will produce (reserved for conflict checks).
        transforms: One or more `TransformSpec` instances executed in order
            (bundle piping).  Use import-string targets for ``backend="processes"``;
            callable targets are safe for serial and threads backends.
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    transforms: tuple[TransformSpec, ...]

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
        if len(self.transforms) == 0:
            raise ValueError(f"Step.transforms must be non-empty for {self.name!r}")


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
