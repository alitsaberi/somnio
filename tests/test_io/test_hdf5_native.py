"""Tests for native zutils HDF5 layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zutils.data.timeseries import TimeSeries

h5py = pytest.importorskip("h5py")

import zutils.io.hdf5 as zio_hdf5  # noqa: E402
from zutils.io.hdf5 import native  # noqa: E402


def _ts(n: int = 5, channels: int = 2, sample_rate: float | None = 100.0) -> TimeSeries:
    step = int(round(1e9 / sample_rate)) if sample_rate else 1_000_000_000
    ts = np.arange(n, dtype=np.int64) * step
    vals = np.arange(n * channels, dtype=np.float64).reshape(n, channels)
    names = tuple(f"ch{i}" for i in range(channels))
    units = tuple("V" for _ in range(channels))
    return TimeSeries(
        values=vals,
        timestamps=ts,
        channel_names=names,
        units=units,
        sample_rate=sample_rate,
    )


def test_native_roundtrip(tmp_path: Path) -> None:
    data = _ts()
    path = tmp_path / "t.h5"
    native.write(path, data, "g1")
    got = native.read(path, group="g1")
    np.testing.assert_array_equal(got.values, data.values)
    np.testing.assert_array_equal(got.timestamps, data.timestamps)
    assert got.channel_names == data.channel_names
    assert got.units == data.units
    assert got.sample_rate == data.sample_rate


def test_native_read_default_group(tmp_path: Path) -> None:
    data = _ts()
    path = tmp_path / "t.h5"
    native.write(path, data, "only")
    got = native.read(path)
    np.testing.assert_array_equal(got.values, data.values)


def test_native_read_all(tmp_path: Path) -> None:
    path = tmp_path / "m.h5"
    a, b = _ts(3, 2), _ts(4, 1, sample_rate=50.0)
    with h5py.File(path, "w") as f:
        for grp_name, ts in [("a", a), ("b", b)]:
            g = f.create_group(grp_name)
            g.create_dataset("data", data=ts.values)
            g.create_dataset("timestamp", data=ts.timestamps)
            g.attrs["channel_names"] = np.array(
                ts.channel_names, dtype=h5py.string_dtype(encoding="utf-8")
            )
            g.attrs["units"] = np.array(
                ts.units, dtype=h5py.string_dtype(encoding="utf-8")
            )
            if ts.sample_rate is not None:
                g.attrs["sample_rate"] = ts.sample_rate

    all_ts = native.read_all(path)
    assert set(all_ts.keys()) == {"a", "b"}
    np.testing.assert_array_equal(all_ts["a"].values, a.values)
    np.testing.assert_array_equal(all_ts["b"].values, b.values)


def test_native_append_second_group(tmp_path: Path) -> None:
    path = tmp_path / "a.h5"
    native.write(path, _ts(2), "g1")
    native.write(path, _ts(3, channels=3), "g2", append=True)
    g1, g2 = native.read(path, group="g1"), native.read(path, group="g2")
    assert g1.n_channels == 2
    assert g2.n_channels == 3


def test_native_serialize_deserialize_roundtrip() -> None:
    data = _ts()
    dset, attrs = native.serialize(data)
    back = native.deserialize(dset, attrs)
    np.testing.assert_array_equal(back.values, data.values)
    np.testing.assert_array_equal(back.timestamps, data.timestamps)
    assert back.channel_names == data.channel_names
    assert back.sample_rate == data.sample_rate


def test_native_writer_requires_group_name(tmp_path: Path) -> None:
    w = native.NativeHDF5Writer()
    with pytest.raises(TypeError, match="group_name"):
        w.write(tmp_path / "x.h5", _ts())


def test_native_read_ambiguous_group_raises(tmp_path: Path) -> None:
    path = tmp_path / "two.h5"
    native.write(path, _ts(2), "g1")
    native.write(path, _ts(3), "g2", append=True)
    with pytest.raises(ValueError, match="group must be specified"):
        native.read(path)


def test_native_read_invalid_group_raises(tmp_path: Path) -> None:
    path = tmp_path / "x.h5"
    native.write(path, _ts(), "ok")
    with h5py.File(path, "a") as f:
        f.create_group("empty")
    with pytest.raises(ValueError, match="not a native"):
        native.read(path, group="empty")


def test_native_write_raises_if_group_exists_on_append(tmp_path: Path) -> None:
    path = tmp_path / "a.h5"
    native.write(path, _ts(2), "g1")
    with pytest.raises(ValueError, match="already exists"):
        native.write(path, _ts(3), "g1", append=True)


def test_native_roundtrip_omits_sample_rate_attr_when_none(tmp_path: Path) -> None:
    data = _ts(sample_rate=None)
    path = tmp_path / "ns.h5"
    native.write(path, data, "g")
    got = native.read(path, group="g")
    assert got.sample_rate is None
    np.testing.assert_array_equal(got.values, data.values)


def test_native_reader_class_delegates(tmp_path: Path) -> None:
    path = tmp_path / "r.h5"
    data = _ts()
    native.write(path, data, "g")
    got = native.NativeHDF5Reader().read(path, group="g")
    np.testing.assert_array_equal(got.values, data.values)


def test_io_hdf5_subpackage_public_api() -> None:
    assert callable(zio_hdf5.read)
    assert callable(zio_hdf5.write)
    assert callable(zio_hdf5.read_all)
    assert callable(zio_hdf5.serialize)
    assert callable(zio_hdf5.deserialize)
    assert callable(zio_hdf5.usleep_read)
    assert callable(zio_hdf5.usleep_write)
    assert callable(zio_hdf5.align_timestamps_to_usleep_grid)
    assert zio_hdf5.NativeHDF5Reader is not None
    assert zio_hdf5.USleepHDF5Writer is not None
