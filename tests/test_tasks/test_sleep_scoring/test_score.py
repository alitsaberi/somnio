"""Unit tests for sleep-scoring helpers (no real model weights)."""

import numpy as np
import pytest

from somnio.data import TimeSeries
from somnio.tasks.sleep_scoring.score import _as_bptk, score_sleep_stages
from somnio.tasks.sleep_scoring.schema import ModelMetadata


class _FakeBackend:
    def __init__(self, out: np.ndarray) -> None:
        self._out = out

    def predict(self, batch: np.ndarray) -> np.ndarray:
        assert batch.ndim == 4
        return self._out


def test_as_bptk_uses_actual_p():
    b, p, t_pred, k = 1, 7, 1, 5
    pred = np.zeros((b, p, k), dtype=np.float32)
    out = _as_bptk(
        pred,
        n_batch=b,
        n_periods_per_window=p,
        n_predictions_per_period=t_pred,
        n_classes=k,
    )
    assert out.shape == (b, p, t_pred, k)


def test_score_dynamic_oneshot():
    md = ModelMetadata(
        sample_rate_hz=128.0,
        n_periods_per_window=None,
        n_samples_per_period=4,
        n_samples_per_prediction=4,
        n_channels=1,
        class_labels=["W", "N1", "N2", "N3", "REM"],
    )
    n_samples = 8
    values = np.zeros((n_samples, 1), dtype=np.float64)
    step = int(1e9 / 128.0)
    timestamps = np.arange(n_samples, dtype=np.int64) * step
    ts = TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=("EEG",),
        units=("V",),
        sample_rate=128.0,
    )
    # one-shot: (1, 2 periods, 5 classes)
    fake = _FakeBackend(np.zeros((1, 2, 5), dtype=np.float32))
    out = score_sleep_stages(ts, backend=fake, metadata=md)
    assert out.values.shape == (2, 5)


def test_as_bptk_rejects_bad_shape():
    with pytest.raises(ValueError, match="Unexpected"):
        _as_bptk(
            np.zeros((1, 2, 3, 4, 5)),
            n_batch=1,
            n_periods_per_window=2,
            n_predictions_per_period=1,
            n_classes=5,
        )
