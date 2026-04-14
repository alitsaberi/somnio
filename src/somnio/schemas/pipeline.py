"""Pydantic schemas for pipeline serialization (YAML/JSON)."""

from __future__ import annotations

from typing import Any

from somnio.pipeline.types import Pipeline, Step, TransformSpec
from somnio.utils.imports import MissingOptionalDependency

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ModuleNotFoundError as e:
    if e.name != "pydantic":
        raise
    raise MissingOptionalDependency(
        "pydantic", extra="schemas", purpose="Pipeline schema validation"
    ) from e


class TransformSchema(BaseModel):
    """Serialized specification for a single transform.

    Attributes:
        target: Import string in the form ``"pkg.module:callable"``.
        kwargs: Keyword arguments forwarded to the callable on every invocation.
    """

    model_config = ConfigDict(extra="forbid")

    target: str = Field(
        ...,
        description='Import string in the form "pkg.module:callable".',
        pattern=r"^[\w.]+:\w+$",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the callable.",
    )

    def to_runtime(self) -> TransformSpec:
        """Convert to runtime `TransformSpec`."""
        return TransformSpec(target=self.target, kwargs=self.kwargs)


class StepSchema(BaseModel):
    """Serialized definition of a single pipeline step.

    Attributes:
        name: Human-readable identifier (used in diagnostics).
        inputs: Names required to exist in the data store for this step to run.
        outputs: Names that this step will produce (reserved for conflict checks).
        transforms: One or more transform specs executed in order (bundle piping).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Human-readable step identifier.")
    inputs: tuple[str, ...] = Field(
        default=(),
        description="Bundle keys this step consumes.",
    )
    outputs: tuple[str, ...] = Field(
        min_length=1,
        description="Bundle keys this step produces.",
    )
    transforms: tuple[TransformSchema, ...] = Field(
        min_length=1,
        description="Ordered list of transform specs applied to the input bundle.",
    )

    @field_validator("inputs", "outputs", mode="after")
    @classmethod
    def _unique_non_empty_strings(
        cls, v: tuple[str, ...], info: Any
    ) -> tuple[str, ...]:
        if not all(v):
            raise ValueError(f"{info.field_name} must contain non-empty strings")
        if len(v) != len(set(v)):
            raise ValueError(f"{info.field_name} must be unique")
        return v

    def to_runtime(self) -> Step:
        """Convert to runtime `Step`."""
        return Step(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            transforms=tuple(t.to_runtime() for t in self.transforms),
        )


class PipelineSchema(BaseModel):
    """Serialized pipeline definition.

    Validates a YAML/JSON dict into a `Pipeline` of `Step` dataclasses.
    All `TransformSpec` targets must be import strings (callable targets are
    not expressible in serialized form).

    Use `to_runtime()` to obtain the runtime `Pipeline` for execution.
    """

    model_config = ConfigDict(extra="forbid")

    steps: list[StepSchema] = Field(
        min_length=1,
        description="Ordered list of steps.",
    )

    def to_runtime(self) -> Pipeline:
        """Convert to runtime `Pipeline`."""
        return Pipeline(steps=tuple(s.to_runtime() for s in self.steps))
