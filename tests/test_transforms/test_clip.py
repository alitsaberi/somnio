"""Tests for somnio.transforms.clip."""

import numpy as np
import pytest

from somnio.data import TimeSeries
from somnio.transforms.clip import (
    apply_clip,
    apply_clip_iqr,
    apply_non_finite,
)


def _ts(values: np.ndarray) -> TimeSeries:
    n, c = values.shape
    return TimeSeries(
        values=values,
        timestamps=np.arange(n, dtype=np.int64),
        channel_names=tuple(f"c{i}" for i in range(c)),
        units=tuple("V" for _ in range(c)),
        sample_rate=100.0,
    )


def test_apply_clip_bounds():
    ts = _ts(np.array([[0.0], [10.0], [-5.0]], dtype=np.float64))
    out = apply_clip(ts, lower=-1.0, upper=3.0)
    assert np.allclose(out.values[:, 0], np.array([0.0, 3.0, -1.0]))


def test_apply_clip_iqr_clips_outlier():
    # Channel 0 has a large outlier; with iqr_factor small, it should clip.
    values = np.array([[0.0], [1.0], [2.0], [100.0]], dtype=np.float64)
    ts = _ts(values)
    out = apply_clip_iqr(ts, iqr_factor=1.0)
    assert out.values[-1, 0] < 100.0
    assert out.values[-1, 0] >= out.values[-2, 0]


def test_apply_non_finite_error():
    ts = _ts(np.array([[0.0], [np.nan]], dtype=np.float64))
    with pytest.raises(ValueError, match="non-finite"):
        apply_non_finite(ts, strategy="error")


def test_apply_non_finite_replace():
    ts = _ts(np.array([[0.0], [np.nan], [np.inf], [-np.inf]], dtype=np.float64))
    out = apply_non_finite(ts, strategy="replace", replace_with=7.0)
    assert np.allclose(out.values[:, 0], np.array([0.0, 7.0, 7.0, 7.0]))


def test_apply_non_finite_clip():
    ts = _ts(np.array([[0.0], [np.nan], [np.inf], [-np.inf]], dtype=np.float64))
    out = apply_non_finite(ts, strategy="clip", clip_lower=-1.0, clip_upper=2.0)
    assert np.allclose(out.values[:, 0], np.array([0.0, 0.5, 2.0, -1.0]))
