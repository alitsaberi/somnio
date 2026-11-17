"""Sleep scoring model backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from somnio.tasks.sleep_scoring.backend import SleepScoringBackend
from somnio.tasks.sleep_scoring.models.metadata import (
    discover_metadata_path,
    load_model_metadata,
)

if TYPE_CHECKING:
    from somnio.tasks.sleep_scoring.models.onnx import OnnxSleepScoringModel
    from somnio.tasks.sleep_scoring.models.tensorflow import (
        TensorflowSleepScoringModel,
    )

__all__ = [
    "OnnxSleepScoringModel",
    "TensorflowSleepScoringModel",
    "SleepScoringBackend",
    "discover_metadata_path",
    "load_model_metadata",
]


def __getattr__(name: str):
    if name == "OnnxSleepScoringModel":
        from somnio.tasks.sleep_scoring.models.onnx import OnnxSleepScoringModel

        return OnnxSleepScoringModel
    if name == "TensorflowSleepScoringModel":
        from somnio.tasks.sleep_scoring.models.tensorflow import (
            TensorflowSleepScoringModel,
        )

        return TensorflowSleepScoringModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
