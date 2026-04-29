"""Tests for sleep-scoring NPTC windowing (partial periods padded, not trimmed)."""

import numpy as np
import pytest

from somnio.data import TimeSeries
from somnio.tasks.sleep_scoring.schema import ModelMetadata
from somnio.tasks.sleep_scoring.windowing import (
    PeriodTimestampAlignment,
    build_nptc_batches,
    build_nptc_batches_from_metadata,
)


def _mono_ts(n_samples: int, *, sample_rate: float = 128.0) -> TimeSeries:
    values = np.arange(n_samples, dtype=np.float64).reshape(n_samples, 1)
    step = int(1e9 / sample_rate)
    timestamps = (np.arange(n_samples, dtype=np.int64) * step).copy()
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=("EEG",),
        units=("V",),
        sample_rate=sample_rate,
    )


def test_partial_last_period_padded_not_dropped():
    ts = _mono_ts(10)
    r = build_nptc_batches(
        ts,
        n_periods_per_window=3,
        n_samples_per_period=4,
        n_channels=1,
        sample_rate_hz=128.0,
        period_stride_samples=4,
        partial_period_padding="edge",
    )
    assert r.batches.shape == (1, 3, 4, 1)
    assert r.batches.dtype == np.float32
    assert len(r.period_start_sample) == 3
    np.testing.assert_array_equal(r.period_start_sample, [0, 4, 8])
    assert list(r.period_is_fully_observed) == [True, True, False]
    last_period = r.batches[0, 2, :, 0]
    np.testing.assert_array_equal(last_period[:2], [8.0, 9.0])
    np.testing.assert_array_equal(last_period[2:], [9.0, 9.0])


def test_no_partial_all_full_periods():
    ts = _mono_ts(8)
    r = build_nptc_batches(
        ts,
        n_periods_per_window=2,
        n_samples_per_period=4,
        n_channels=1,
        sample_rate_hz=128.0,
        period_stride_samples=4,
    )
    assert r.batches.shape == (1, 2, 4, 1)
    assert bool(np.all(r.period_is_fully_observed))
    assert bool(np.all(r.batch_slot_is_real_period))


def test_batch_tail_padding_masks_slots():
    ts = _mono_ts(16)
    r = build_nptc_batches(
        ts,
        n_periods_per_window=3,
        n_samples_per_period=4,
        n_channels=1,
        sample_rate_hz=128.0,
        period_stride_samples=4,
        batch_tail_padding="edge",
    )
    assert len(r.period_start_sample) == 4
    assert r.batches.shape == (2, 3, 4, 1)
    assert r.batch_slot_is_real_period.shape == (2, 3)
    assert list(r.batch_slot_is_real_period.flatten()) == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    np.testing.assert_array_equal(r.batches[1, 0], r.batches[1, 1])
    np.testing.assert_array_equal(r.batches[1, 0], r.batches[1, 2])


def test_sample_rate_mismatch_raises():
    ts = _mono_ts(8, sample_rate=100.0)
    with pytest.raises(ValueError, match="sample_rate"):
        build_nptc_batches(
            ts,
            n_periods_per_window=2,
            n_samples_per_period=4,
            n_channels=1,
            sample_rate_hz=128.0,
        )


def test_period_center_timestamp():
    ts = _mono_ts(8)
    r = build_nptc_batches(
        ts,
        n_periods_per_window=2,
        n_samples_per_period=4,
        n_channels=1,
        sample_rate_hz=128.0,
        period_stride_samples=4,
        timestamp_alignment=PeriodTimestampAlignment.PERIOD_CENTER,
    )
    expected0 = int(np.rint(np.mean(ts.timestamps[0:4].astype(np.float64))))
    assert r.period_timestamp_ns[0] == expected0


def test_build_from_metadata():
    md = ModelMetadata(
        sample_rate_hz=128.0,
        n_periods_per_window=2,
        n_samples_per_period=4,
        n_channels=1,
        class_labels=["W", "N1", "N2", "N3", "REM"],
    )
    ts = _mono_ts(8)
    r = build_nptc_batches_from_metadata(ts, md)
    assert r.batches.shape == (1, 2, 4, 1)
    assert r.sample_rate_hz == 128.0
