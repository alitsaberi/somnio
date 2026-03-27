"""Tests for zutils.data.timeseries — Sample, TimeSeries, concat, collect_samples."""

from datetime import timedelta

import numpy as np
import pytest

from zutils.data import Sample, TimeSeries, collect_samples, concat


def make_ts(
    n_samples: int = 4,
    n_channels: int = 2,
    sample_rate: float | None = 256.0,
    t0: int = 0,
) -> TimeSeries:
    values = np.ones((n_samples, n_channels), dtype=np.float64)
    step = int(1e9 / sample_rate) if sample_rate else int(1e6)
    timestamps = np.array([t0 + i * step for i in range(n_samples)], dtype=np.int64)
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=("EEG_L", "EEG_R"),
        units=("V", "V"),
        sample_rate=sample_rate,
    )


def make_sample(timestamp: int = 0) -> Sample:
    return Sample(
        values=np.array([1.0, 2.0]),
        timestamp=timestamp,
        channel_names=("EEG_L", "EEG_R"),
        units=("V", "V"),
    )


class TestSample:
    def test_construction(self):
        s = make_sample(timestamp=1_000_000)
        assert s.values.dtype == np.float64
        assert s.values.shape == (2,)
        assert s.timestamp == 1_000_000

    def test_coerces_to_float64(self):
        s = Sample(
            values=np.array([1, 2], dtype=np.int32),
            timestamp=0,
            channel_names=("A", "B"),
            units=("V", "V"),
        )
        assert s.values.dtype == np.float64

    def test_rejects_2d_values(self):
        with pytest.raises(ValueError, match="1-D"):
            Sample(
                values=np.ones((2, 2)),
                timestamp=0,
                channel_names=("A", "B"),
                units=("V", "V"),
            )

    def test_channel_names_length_mismatch(self):
        with pytest.raises(ValueError, match="n_channels"):
            Sample(
                values=np.array([1.0, 2.0]),
                timestamp=0,
                channel_names=("A",),
                units=("V", "V"),
            )

    def test_units_length_mismatch(self):
        with pytest.raises(ValueError, match="n_channels"):
            Sample(
                values=np.array([1.0, 2.0]),
                timestamp=0,
                channel_names=("A", "B"),
                units=("V",),
            )

    def test_duplicate_channel_names_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            Sample(
                values=np.array([1.0, 2.0]),
                timestamp=0,
                channel_names=("A", "A"),
                units=("V", "V"),
            )


class TestTimeSeries:
    def test_construction(self):
        ts = make_ts()
        assert ts.values.dtype == np.float64
        assert ts.timestamps.dtype == np.int64
        assert ts.n_samples == 4
        assert ts.n_channels == 2
        assert ts.shape == (4, 2)

    def test_coerces_dtypes(self):
        ts = TimeSeries(
            values=np.ones((3, 2), dtype=np.int32),
            timestamps=np.array([0, 1, 2], dtype=np.float32),
            channel_names=["A", "B"],
            units=["V", "V"],
        )
        assert ts.values.dtype == np.float64
        assert ts.timestamps.dtype == np.int64

    def test_rejects_1d_values(self):
        with pytest.raises(ValueError, match="2-D"):
            TimeSeries(
                values=np.ones(4),
                timestamps=np.zeros(4, dtype=np.int64),
                channel_names=["A"],
                units=["V"],
            )

    def test_timestamps_shape_mismatch(self):
        with pytest.raises(ValueError, match="timestamps shape"):
            TimeSeries(
                values=np.ones((4, 2)),
                timestamps=np.zeros(3, dtype=np.int64),
                channel_names=["A", "B"],
                units=["V", "V"],
            )

    def test_timestamps_must_be_monotonic(self):
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            TimeSeries(
                values=np.ones((3, 2)),
                timestamps=np.array([2, 1, 3], dtype=np.int64),
                channel_names=["A", "B"],
                units=["V", "V"],
            )

        # Equal timestamps are allowed (non-decreasing, not strictly increasing)
        ts = TimeSeries(
            values=np.ones((3, 2)),
            timestamps=np.array([1, 1, 2], dtype=np.int64),
            channel_names=["A", "B"],
            units=["V", "V"],
        )
        assert ts.n_samples == 3

    def test_duplicate_channel_names_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            TimeSeries(
                values=np.ones((2, 2)),
                timestamps=np.zeros(2, dtype=np.int64),
                channel_names=["A", "A"],
                units=["V", "V"],
            )

    def test_negative_sample_rate_rejected(self):
        with pytest.raises(ValueError, match="sample_rate"):
            make_ts(sample_rate=-1.0)  # type: ignore[arg-type]

    def test_zero_sample_rate_rejected(self):
        with pytest.raises(ValueError, match="sample_rate"):
            TimeSeries(
                values=np.ones((2, 2)),
                timestamps=np.zeros(2, dtype=np.int64),
                channel_names=["A", "B"],
                units=["V", "V"],
                sample_rate=0.0,
            )

    def test_is_regular(self):
        assert make_ts(sample_rate=256.0).is_regular
        assert not make_ts(sample_rate=None).is_regular

    def test_duration(self):
        ts = make_ts(n_samples=257, sample_rate=256.0, t0=0)
        assert ts.duration > timedelta(0)

    def test_duration_single_sample(self):
        ts = make_ts(n_samples=1)
        assert ts.duration == timedelta(0)

    def test_channel_index_map(self):
        ts = make_ts()
        assert ts.channel_index_map == {"EEG_L": 0, "EEG_R": 1}

    def test_select_channels(self):
        ts = make_ts()
        sub = ts.select_channels(["EEG_R"])
        assert sub.channel_names == ("EEG_R",)
        assert sub.units == ("V",)
        assert sub.n_channels == 1
        assert sub.n_samples == ts.n_samples

    def test_select_channels_unknown_name(self):
        ts = make_ts()
        with pytest.raises(KeyError):
            ts.select_channels(["UNKNOWN"])

    def test_select_time_both_bounds(self):
        ts = make_ts(n_samples=8, sample_rate=256.0, t0=0)
        step = int(1e9 / 256.0)
        sub = ts.select_time(start=step, end=step * 3)
        assert sub.n_samples == 3
        assert sub.timestamps[0] >= step
        assert sub.timestamps[-1] <= step * 3

    def test_select_time_no_bounds(self):
        ts = make_ts()
        sub = ts.select_time()
        assert sub.n_samples == ts.n_samples

    def test_getitem_slice(self):
        ts = make_ts(n_samples=6)
        sub = ts[1:4]
        assert sub.n_samples == 3
        assert sub.channel_names == ts.channel_names
        assert sub.sample_rate == ts.sample_rate

    def test_getitem_int(self):
        ts = make_ts(n_samples=4)
        sub = ts[2]
        assert sub.n_samples == 1

    def test_getitem_negative_int(self):
        ts = make_ts(n_samples=4)
        sub = ts[-1]
        assert sub.n_samples == 1
        np.testing.assert_array_equal(sub.values[0], ts.values[-1])
        assert sub.timestamps[0] == ts.timestamps[-1]

    def test_getitem_int_out_of_range_raises(self):
        ts = make_ts(n_samples=4)
        with pytest.raises(IndexError):
            _ = ts[4]
        with pytest.raises(IndexError):
            _ = ts[-5]

    def test_getitem_preserves_units(self):
        ts = make_ts()
        sub = ts[0:2]
        assert sub.units == ts.units


class TestConcat:
    def test_basic(self):
        t0 = 0
        ts1 = make_ts(n_samples=4, t0=t0)
        step = int(1e9 / 256.0)
        ts2 = make_ts(n_samples=4, t0=t0 + 4 * step)
        result = concat([ts1, ts2])
        assert result.n_samples == 8
        assert result.channel_names == ts1.channel_names
        assert result.sample_rate == 256.0

    def test_sample_rate_mismatch_yields_none(self):
        t0 = 0
        ts1 = make_ts(sample_rate=256.0, t0=t0)
        # Ensure timestamps remain non-decreasing across the concatenation boundary.
        ts2 = make_ts(sample_rate=128.0, t0=int(ts1.timestamps[-1]) + 1)
        result = concat([ts1, ts2])
        assert result.sample_rate is None

    def test_channel_names_mismatch_raises(self):
        ts1 = make_ts()
        ts2 = TimeSeries(
            values=np.ones((4, 2)),
            timestamps=np.arange(4, dtype=np.int64),
            channel_names=("X", "Y"),
            units=("V", "V"),
        )
        with pytest.raises(ValueError, match="channel_names mismatch"):
            concat([ts1, ts2])

    def test_units_mismatch_raises(self):
        ts1 = make_ts()
        ts2 = TimeSeries(
            values=np.ones((4, 2)),
            timestamps=np.arange(4, dtype=np.int64),
            channel_names=("EEG_L", "EEG_R"),
            units=("m/s^2", "m/s^2"),
        )
        with pytest.raises(ValueError, match="units mismatch"):
            concat([ts1, ts2])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            concat([])

    def test_single_element(self):
        ts = make_ts()
        result = concat([ts])
        assert result.n_samples == ts.n_samples


class TestCollectSamples:
    def test_basic(self):
        samples = [make_sample(timestamp=i * 1_000_000) for i in range(5)]
        ts = collect_samples(samples)
        assert ts.n_samples == 5
        assert ts.n_channels == 2
        assert ts.sample_rate is None
        assert ts.timestamps[0] == 0
        assert ts.timestamps[-1] == 4_000_000

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            collect_samples([])

    def test_channel_names_mismatch_raises(self):
        s1 = make_sample()
        s2 = Sample(
            values=np.array([1.0, 2.0]),
            timestamp=1,
            channel_names=("X", "Y"),
            units=("V", "V"),
        )
        with pytest.raises(ValueError, match="channel_names mismatch"):
            collect_samples([s1, s2])

    def test_units_mismatch_raises(self):
        s1 = make_sample()
        s2 = Sample(
            values=np.array([1.0, 2.0]),
            timestamp=1,
            channel_names=("EEG_L", "EEG_R"),
            units=("m/s^2", "m/s^2"),
        )
        with pytest.raises(ValueError, match="units mismatch"):
            collect_samples([s1, s2])

    def test_values_stacked_correctly(self):
        s1 = Sample(
            values=np.array([1.0, 2.0]),
            timestamp=0,
            channel_names=("A", "B"),
            units=("V", "V"),
        )
        s2 = Sample(
            values=np.array([3.0, 4.0]),
            timestamp=1,
            channel_names=("A", "B"),
            units=("V", "V"),
        )
        ts = collect_samples([s1, s2])
        np.testing.assert_array_equal(ts.values[0], [1.0, 2.0])
        np.testing.assert_array_equal(ts.values[1], [3.0, 4.0])
