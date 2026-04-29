"""Sleep stage scoring pipelines (ONNX-first)."""

from somnio.tasks.sleep_scoring.backend import SleepScoringBackend
from somnio.tasks.sleep_scoring.score import (
    score_sleep_stages,
)
from somnio.tasks.sleep_scoring.schema import ModelMetadata
from somnio.tasks.sleep_scoring.windowing import (
    PeriodTimestampAlignment,
    WindowingResult,
    build_nptc_batches,
    build_nptc_batches_from_metadata,
)

__all__ = [
    "ModelMetadata",
    "SleepScoringBackend",
    "score_sleep_stages",
    "PeriodTimestampAlignment",
    "WindowingResult",
    "build_nptc_batches",
    "build_nptc_batches_from_metadata",
]
