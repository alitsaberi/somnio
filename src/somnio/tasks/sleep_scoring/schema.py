"""Sidecar metadata for sleep scoring models (ONNX)."""

from __future__ import annotations


from somnio.utils.imports import MissingOptionalDependency

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except ModuleNotFoundError as e:
    if e.name != "pydantic":
        raise
    raise MissingOptionalDependency(
        "pydantic", extra="schemas", purpose="Sleep scoring metadata validation"
    ) from e

from somnio.pipeline.types import TransformSpec
from somnio.schemas.pipeline import TransformSchema


class OnnxBindings(BaseModel):
    """Optional ONNX graph hints; omit and use session IO when unknown."""

    model_config = ConfigDict(extra="forbid")

    input_name: str | None = Field(
        default=None,
        description="Graph input tensor name; default: first model input.",
    )
    output_name: str | None = Field(
        default=None,
        description="Graph output tensor name; default: first model output.",
    )


class ModelMetadata(BaseModel):
    """Validated sidecar next to ``.onnx``.

    Example:
    ```yaml
        sample_rate_hz: 128
        n_periods_per_window: 35
        n_samples_per_period: 3840
        n_samples_per_prediction: 128
        n_channels: 1
        class_labels: [W, N1, N2, N3, REM]
        preprocessing: []
    ```
    """

    model_config = ConfigDict(extra="forbid")

    sample_rate_hz: float = Field(
        ...,
        gt=0,
        description="Sample rate (Hz) the model expects after preprocessing.",
    )
    n_periods_per_window: int = Field(
        ...,
        ge=1,
        description="ONNX input: periods along axis P (per batch item).",
    )
    n_samples_per_period: int = Field(
        ...,
        ge=1,
        description="ONNX input: samples along axis T per period.",
    )
    n_channels: int = Field(
        ...,
        ge=1,
        description="ONNX input: size of channel axis C (last axis in NPTC).",
    )
    n_samples_per_prediction: int = Field(
        ...,
        ge=1,
        description=(
            "Input samples aggregated into one prediction within a period. "
            "ONNX output time axis length is "
            "``n_samples_per_period // n_samples_per_prediction``."
        ),
    )
    class_labels: list[str] = Field(
        ...,
        min_length=1,
        description="Model output order: index ``i`` ↔ ``class_labels[i]``.",
    )
    preprocessing: list[TransformSchema] = Field(
        default_factory=list,
        description="Preprocessing transforms to apply to the input data.",
    )

    onnx: OnnxBindings = Field(default_factory=OnnxBindings)

    @property
    def n_predictions_per_period(self) -> int:
        """Predictions along the ONNX output time axis for one input period."""
        return self.n_samples_per_period // self.n_samples_per_prediction

    def preprocessing_transform_specs(self) -> tuple[TransformSpec, ...]:
        """Runtime `TransformSpec` tuple for `preprocessing` (import-string targets)."""
        return tuple(step.to_runtime() for step in self.preprocessing)

    @model_validator(mode="after")
    def _period_divisible_by_prediction_stride(self) -> ModelMetadata:
        n_sp = self.n_samples_per_period
        stride = self.n_samples_per_prediction
        if n_sp % stride != 0:
            raise ValueError(
                f"n_samples_per_period ({n_sp}) must be divisible by "
                f"n_samples_per_prediction ({stride})"
            )
        return self

    @field_validator("class_labels")
    @classmethod
    def _labels_unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("class_labels must be unique")
        return v
