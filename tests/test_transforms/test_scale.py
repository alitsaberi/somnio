"""Tests for somnio.transforms.scale."""

import numpy as np

from somnio.data import TimeSeries
from somnio.transforms.scale import apply_scale


def _ts(values: np.ndarray) -> TimeSeries:
    n = values.shape[0]
    return TimeSeries(
        values=values,
        timestamps=np.arange(n, dtype=np.int64),
        channel_names=("c0", "c1"),
        units=("V", "V"),
        sample_rate=100.0,
    )


def test_apply_scale_zscore_center_and_scale():
    # Two channels with different offsets/scales.
    values = np.array(
        [
            [0.0, 10.0],
            [1.0, 12.0],
            [2.0, 14.0],
            [3.0, 16.0],
        ],
        dtype=np.float64,
    )
    ts = _ts(values)
    out = apply_scale(ts, method="zscore")

    # Mean ~0, std ~1 per channel (ddof=0).
    mu = np.mean(out.values, axis=0)
    sigma = np.std(out.values, axis=0, ddof=0)
    assert np.allclose(mu, 0.0, atol=1e-12)
    assert np.allclose(sigma, 1.0, atol=1e-12)

    # Metadata preserved.
    assert np.array_equal(out.timestamps, ts.timestamps)
    assert out.channel_names == ts.channel_names
    assert out.units == ts.units
    assert out.sample_rate == ts.sample_rate


def test_apply_scale_robust_median_and_iqr():
    # Construct a channel where median and IQR are easy to verify.
    # Note: IQR depends on the quantile definition; NumPy's default gives
    # Q25=0.75, Q75=2.25 for [0,1,2,3] → IQR=1.5.
    # Channel 1: [10, 10, 10, 10] median=10, IQR=0 -> eps handling.
    values = np.array(
        [
            [0.0, 10.0],
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
        ],
        dtype=np.float64,
    )
    ts = _ts(values)
    out = apply_scale(ts, method="robust", eps=1e-6)

    # Channel 0: scaled by (x - median) / IQR, using NumPy quantiles.
    median0 = float(np.median(values[:, 0]))
    iqr0 = float(np.quantile(values[:, 0], 0.75) - np.quantile(values[:, 0], 0.25))
    expected0 = (values[:, 0] - median0) / iqr0
    assert np.allclose(out.values[:, 0], expected0, atol=1e-12)

    # Channel 1: constant channel -> zeros after centering; scaling uses eps floor.
    assert np.allclose(out.values[:, 1], 0.0, atol=1e-12)


def test_apply_scale_explicit_center_scale_overrides_method():
    values = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    ts = _ts(values)
    out = apply_scale(ts, method="robust", center=[1.0, 2.0], scale=[2.0, 4.0])
    expected = (values - np.array([1.0, 2.0])) / np.array([2.0, 4.0])
    assert np.allclose(out.values, expected)
