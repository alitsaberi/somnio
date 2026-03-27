"""Tests for zutils.io.hdf5."""

import numpy as np
import pytest

from zutils.data import Sample, TimeSeries
from zutils.io.hdf5 import (
    DatasetDoesNotExistError,
    GroupDoesNotExistError,
    HDF5Manager,
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
        restored = deserialize(datasets, attrs)
        assert restored.values.dtype == np.float64

    def test_timestamps_dtype_int64(self):
        ts = make_ts()
        datasets, attrs = serialize(ts)
        restored = deserialize(datasets, attrs)
        assert restored.timestamps.dtype == np.int64


class TestHDF5Manager:
    def test_context_manager_creates_file(self, tmp_path):
        path = tmp_path / "test.h5"
        with HDF5Manager(path) as mgr:
            assert mgr.file.id.valid
        assert path.exists()

    def test_file_closed_after_exit(self, tmp_path):
        path = tmp_path / "test.h5"
        with HDF5Manager(path) as mgr:
            f = mgr.file
        assert not f.id.valid

    def test_close_idempotent(self, tmp_path):
        path = tmp_path / "test.h5"
        mgr = HDF5Manager(path)
        mgr.close()
        mgr.close()  # should not raise

    def test_groups_empty_on_new_file(self, tmp_path):
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            assert mgr.groups == []

    def test_create_group(self, tmp_path):
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            mgr.create_group("eeg", channel_names=["A"])
            assert "eeg" in mgr.groups
            assert mgr.file["eeg"].attrs["channel_names"] == ["A"]

    def test_create_dataset_with_data(self, tmp_path):
        data = np.ones((4, 2), dtype=np.float64)
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            mgr.create_group("grp")
            ds = mgr.create_dataset("grp", "values", data=data)
            np.testing.assert_array_equal(ds[:], data)

    def test_create_dataset_missing_group_raises(self, tmp_path):
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            with pytest.raises(GroupDoesNotExistError):
                mgr.create_dataset("missing", "ds", data=np.ones(3))

    def test_create_dataset_no_data_no_shape_raises(self, tmp_path):
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            mgr.create_group("grp")
            with pytest.raises(ValueError, match="data or shape"):
                mgr.create_dataset("grp", "ds")

    def test_get_dataset(self, tmp_path):
        data = np.arange(6, dtype=np.float64)
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            mgr.create_group("grp")
            mgr.create_dataset("grp", "arr", data=data)
            ds = mgr.get_dataset("grp", "arr")
            np.testing.assert_array_equal(ds[:], data)

    def test_get_dataset_missing_group_raises(self, tmp_path):
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            with pytest.raises(GroupDoesNotExistError):
                mgr.get_dataset("no_group", "ds")

    def test_get_dataset_missing_dataset_raises(self, tmp_path):
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            mgr.create_group("grp")
            with pytest.raises(DatasetDoesNotExistError):
                mgr.get_dataset("grp", "missing")

    def test_append_1d(self, tmp_path):
        initial = np.array([1.0, 2.0, 3.0])
        extra = np.array([4.0, 5.0])
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            mgr.create_group("grp")
            mgr.create_dataset("grp", "arr", data=initial, max_shape=(None,))
            mgr.append("grp", "arr", extra)
            result = mgr.get_dataset("grp", "arr")[:]
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_append_2d(self, tmp_path):
        initial = np.ones((3, 2))
        extra = np.zeros((2, 2))
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            mgr.create_group("grp")
            mgr.create_dataset("grp", "arr", data=initial, max_shape=(None, 2))
            mgr.append("grp", "arr", extra)
            result = mgr.get_dataset("grp", "arr")[:]
        assert result.shape == (5, 2)

    def test_append_column_mismatch_raises(self, tmp_path):
        with HDF5Manager(tmp_path / "test.h5") as mgr:
            mgr.create_group("grp")
            mgr.create_dataset("grp", "arr", data=np.ones((2, 3)), max_shape=(None, 3))
            with pytest.raises(ValueError, match="Column count mismatch"):
                mgr.append("grp", "arr", np.ones((2, 2)))

    def test_compression_stored(self, tmp_path):
        with HDF5Manager(tmp_path / "test.h5", compression="gzip") as mgr:
            mgr.create_group("grp")
            mgr.create_dataset("grp", "ds", data=np.ones(100))
            ds = mgr.get_dataset("grp", "ds")
            assert ds.compression == "gzip"


class TestWriteReadHDF5:
    def test_round_trip_single_timeseries(self, tmp_path):
        ts = make_ts()
        path = tmp_path / "out.h5"
        write_hdf5(path, {"eeg": ts})
        result = read_hdf5(path)
        assert set(result) == {"eeg"}
        restored = result["eeg"]
        np.testing.assert_array_equal(restored.values, ts.values)
        np.testing.assert_array_equal(restored.timestamps, ts.timestamps)
        assert restored.channel_names == ts.channel_names
        assert restored.units == ts.units
        assert restored.sample_rate == ts.sample_rate

    def test_round_trip_multiple_groups(self, tmp_path):
        ts1 = make_ts(n_samples=4, t0=0)
        ts2 = make_ts(n_samples=6, t0=int(4e9), sample_rate=None)
        path = tmp_path / "out.h5"
        write_hdf5(path, {"eeg": ts1, "accel": ts2})
        result = read_hdf5(path)
        assert set(result) == {"eeg", "accel"}
        assert result["accel"].sample_rate is None

    def test_write_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "out.h5"
        ts_a = make_ts(n_samples=4)
        ts_b = make_ts(n_samples=8)
        write_hdf5(path, {"a": ts_a})
        write_hdf5(path, {"b": ts_b})
        result = read_hdf5(path)
        assert "b" in result
        assert "a" not in result

    def test_round_trip_sample(self, tmp_path):
        s = make_sample(timestamp=999_000_000)
        path = tmp_path / "out.h5"
        write_hdf5(path, {"snap": s})
        result = read_hdf5(path)
        restored = result["snap"]
        assert isinstance(restored, TimeSeries)
        assert restored.n_samples == 1
        np.testing.assert_array_almost_equal(restored.values[0], s.values)
        assert restored.timestamps[0] == s.timestamp

    def test_values_preserved_exactly(self, tmp_path):
        ts = make_ts()
        path = tmp_path / "out.h5"
        write_hdf5(path, {"eeg": ts})
        restored = read_hdf5(path)["eeg"]
        np.testing.assert_array_equal(restored.values, ts.values)

    def test_empty_dict_creates_empty_file(self, tmp_path):
        path = tmp_path / "empty.h5"
        write_hdf5(path, {})
        result = read_hdf5(path)
        assert result == {}
