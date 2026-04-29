"""Shared backend contract for sleep-scoring inference (ONNX, TensorFlow, …)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SleepScoringBackend(Protocol):
    """Contract for sleep-scoring inference."""

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Run forward pass on a batch of input data."""
        ...
