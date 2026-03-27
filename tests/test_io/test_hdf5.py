"""Tests for zutils.io.hdf5."""

import numpy as np

from zutils.data import Sample, TimeSeries
from zutils.io import FileFormat
from zutils.io.hdf5 import (
    HDF5Format,
    deserialize,
    read_hdf5,
    serialize,
    write_hdf5,
)


def make_ts(
    n_samples: int = 8,
    n_channels: int = 2,
    sample_rate: float | None = 256.0,
    t0: int = 0,
) -> TimeSeries:
    step = int(1e9 / sample_rate) if sample_rate else int(1e6)
    return TimeSeries(
        values=np.arange(n_samples * n_channels, dtype=np.float64).reshape(
            n_samples, n_channels
        ),
        timestamps=np.array([t0 + i * step for i in range(n_samples)], dtype=np.int64),
        channel_names=["EEG_L", "EEG_R"][:n_channels],
        units=["V", "V"][:n_channels],
        sample_rate=sample_rate,
    )


def make_sample(timestamp: int = 1_000_000) -> Sample:
    return Sample(
        values=np.array([0.001, -0.002]),
        timestamp=timestamp,
        channel_names=["EEG_L", "EEG_R"],
        units=["V", "V"],
    )


class TestSerialize:
    def test_timeseries_datasets(self):
        ts = make_ts()
        datasets, attrs = serialize(ts)
        assert set(datasets) == {"values", "timestamps"}
        np.testing.assert_array_equal(datasets["values"], ts.values)
        np.testing.assert_array_equal(datasets["timestamps"], ts.timestamps)
        assert datasets["values"].dtype == np.float64
        assert datasets["timestamps"].dtype == np.int64

    def test_timeseries_attributes(self):
        ts = make_ts(sample_rate=256.0)
        _, attrs = serialize(ts)
        assert attrs["channel_names"] == ["EEG_L", "EEG_R"]
        assert attrs["units"] == ["V", "V"]
        assert attrs["sample_rate"] == 256.0

    def test_timeseries_no_sample_rate_omits_key(self):
        ts = make_ts(sample_rate=None)
        _, attrs = serialize(ts)
        assert "sample_rate" not in attrs

    def test_sample_values_reshaped(self):
        s = make_sample()
        datasets, _ = serialize(s)
        assert datasets["values"].shape == (1, 2)
        assert datasets["timestamps"].shape == (1,)
        assert datasets["timestamps"][0] == s.timestamp

    def test_sample_no_sample_rate_key(self):
        s = make_sample()
        _, attrs = serialize(s)
        assert "sample_rate" not in attrs

    def test_channel_names_returned_as_list(self):
        ts = make_ts()
        _, attrs = serialize(ts)
        assert isinstance(attrs["channel_names"], list)
        assert isinstance(attrs["units"], list)


class TestDeserialize:
    def test_round_trip_with_sample_rate(self):
        ts = make_ts(sample_rate=256.0)
        datasets, attrs = serialize(ts)
        restored = deserialize(datasets, attrs)
        np.testing.assert_array_equal(restored.values, ts.values)
        np.testing.assert_array_equal(restored.timestamps, ts.timestamps)
        assert restored.channel_names == ts.channel_names
        assert restored.units == ts.units
        assert restored.sample_rate == ts.sample_rate

    def test_round_trip_without_sample_rate(self):
        ts = make_ts(sample_rate=None)
        datasets, attrs = serialize(ts)
        restored = deserialize(datasets, attrs)
        assert restored.sample_rate is None

    def test_values_dtype_float64(self):
        ts = make_ts()
        datasets, attrs = serialize(ts)
        assert deserialize(datasets, attrs).values.dtype == np.float64

    def test_timestamps_dtype_int64(self):
        ts = make_ts()
        datasets, attrs = serialize(ts)
        assert deserialize(datasets, attrs).timestamps.dtype == np.int64


class TestHDF5Format:
    def test_implements_file_format_protocol(self):
        assert isinstance(HDF5Format(), FileFormat)

    def test_default_compression_gzip(self, tmp_path):
        fmt = HDF5Format()
        path = tmp_path / "out.h5"
        fmt.write(path, {"eeg": make_ts()})
        import h5py

        with h5py.File(path, "r") as f:
            assert f["eeg"]["values"].compression == "gzip"

    def test_no_compression(self, tmp_path):
        fmt = HDF5Format(compression=None)
        path = tmp_path / "out.h5"
        fmt.write(path, {"eeg": make_ts()})
        import h5py

        with h5py.File(path, "r") as f:
            assert f["eeg"]["values"].compression is None

    def test_write_creates_file(self, tmp_path):
        path = tmp_path / "out.h5"
        HDF5Format().write(path, {"eeg": make_ts()})
        assert path.exists()

    def test_write_overwrites_existing(self, tmp_path):
        path = tmp_path / "out.h5"
        fmt = HDF5Format()
        fmt.write(path, {"a": make_ts()})
        fmt.write(path, {"b": make_ts()})
        result = fmt.read(path)
        assert "b" in result and "a" not in result

    def test_round_trip_single(self, tmp_path):
        ts = make_ts()
        path = tmp_path / "out.h5"
        fmt = HDF5Format()
        fmt.write(path, {"eeg": ts})
        result = fmt.read(path)
        assert set(result) == {"eeg"}
        np.testing.assert_array_equal(result["eeg"].values, ts.values)
        np.testing.assert_array_equal(result["eeg"].timestamps, ts.timestamps)
        assert result["eeg"].sample_rate == ts.sample_rate

    def test_round_trip_multiple(self, tmp_path):
        ts1 = make_ts(n_samples=4, t0=0)
        ts2 = make_ts(n_samples=6, t0=int(4e9), sample_rate=None)
        path = tmp_path / "out.h5"
        fmt = HDF5Format()
        fmt.write(path, {"eeg": ts1, "accel": ts2})
        result = fmt.read(path)
        assert set(result) == {"eeg", "accel"}
        assert result["accel"].sample_rate is None

    def test_round_trip_sample(self, tmp_path):
        s = make_sample(timestamp=999_000_000)
        path = tmp_path / "out.h5"
        fmt = HDF5Format()
        fmt.write(path, {"snap": s})
        result = fmt.read(path)
        restored = result["snap"]
        assert restored.n_samples == 1
        np.testing.assert_array_almost_equal(restored.values[0], s.values)
        assert restored.timestamps[0] == s.timestamp

    def test_empty_dict(self, tmp_path):
        path = tmp_path / "empty.h5"
        fmt = HDF5Format()
        fmt.write(path, {})
        assert fmt.read(path) == {}


class TestConvenienceFunctions:
    def test_write_read_hdf5_round_trip(self, tmp_path):
        ts = make_ts()
        path = tmp_path / "out.h5"
        write_hdf5(path, {"eeg": ts})
        result = read_hdf5(path)
        np.testing.assert_array_equal(result["eeg"].values, ts.values)
        assert result["eeg"].sample_rate == ts.sample_rate

    def test_write_hdf5_overwrites(self, tmp_path):
        path = tmp_path / "out.h5"
        write_hdf5(path, {"a": make_ts()})
        write_hdf5(path, {"b": make_ts()})
        assert "b" in read_hdf5(path)
        assert "a" not in read_hdf5(path)

    def test_values_preserved_exactly(self, tmp_path):
        ts = make_ts()
        path = tmp_path / "out.h5"
        write_hdf5(path, {"eeg": ts})
        np.testing.assert_array_equal(read_hdf5(path)["eeg"].values, ts.values)
