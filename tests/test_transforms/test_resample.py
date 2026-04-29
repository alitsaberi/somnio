"""Tests for somnio.transforms.resample."""

import numpy as np
import pytest

from somnio.data import TimeSeries
from somnio.transforms.resample import apply_resample


def _sine(sr: float, n: int) -> TimeSeries:
    t = np.arange(n, dtype=np.float64) / sr
    values = np.sin(2 * np.pi * 2.0 * t).reshape(n, 1)
    step = int(1e9 / sr)
    timestamps = np.arange(n, dtype=np.int64) * step
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=("EEG",),
        units=("V",),
        sample_rate=sr,
    )


def test_resample_halves_rate():
    ts = _sine(128.0, 256)
    out = apply_resample(ts, 64.0)
    assert out.sample_rate == 64.0
    assert out.n_samples == 128
    assert out.values.shape == (128, 1)


def test_resample_noop_near_equal():
    ts = _sine(128.0, 64)
    out = apply_resample(ts, 128.0)
    assert out is ts


def test_resample_requires_sample_rate():
    ts = TimeSeries(
        values=np.zeros((4, 1)),
        timestamps=np.arange(4, dtype=np.int64),
        channel_names=("a",),
        units=("V",),
        sample_rate=None,
    )
    with pytest.raises(ValueError, match="sample_rate"):
        apply_resample(ts, 128.0)
