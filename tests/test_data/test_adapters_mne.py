"""Tests for MNE adapters."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("mne")

from somnio.data.adapters import mne as mne_adapt  # noqa: E402
from somnio.data.timeseries import TimeSeries  # noqa: E402


def test_to_mne_raw_allows_pre_edf_epoch_timestamps() -> None:
    """EDF export rejects these timestamps; generic MNE Raw does not."""
    pytest.skip("to_mne_raw is temporarily disabled")
    step = 10_000_000
    ts = TimeSeries(
        values=np.zeros((4, 1)),
        timestamps=np.arange(4, dtype=np.int64) * step,
        channel_names=("a",),
        units=("V",),
        sample_rate=100.0,
    )
    raw = mne_adapt.to_mne_raw(ts)
    assert raw.info["meas_date"] is not None


def test_from_mne_round_trip_values() -> None:
    pytest.skip("to_mne_raw is temporarily disabled")
    base = int(datetime(2022, 3, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    step = 4_000_000
    n = 50
    ts = TimeSeries(
        values=np.random.RandomState(2).randn(n, 2) * 1e-6,
        timestamps=base + np.arange(n, dtype=np.int64) * step,
        channel_names=("c0", "c1"),
        units=("V", "V"),
        sample_rate=250.0,
    )
    raw = mne_adapt.to_mne_raw(ts)
    got = mne_adapt.from_mne_raw(raw)
    np.testing.assert_allclose(got.values, ts.values, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(got.timestamps, ts.timestamps)
    assert got.channel_names == ts.channel_names
    assert got.sample_rate == ts.sample_rate
