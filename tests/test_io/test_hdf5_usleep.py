"""Tests for USleep-style HDF5 layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from somnio.data.timeseries import TimeSeries

pytest.importorskip("h5py")

import h5py  # noqa: E402
from somnio.io.hdf5 import usleep  # noqa: E402


def _regular_ts(
    n: int,
    channels: int,
    sample_rate: float,
    start_ns: int = 0,
) -> TimeSeries:
    step = int(round(1e9 / sample_rate))
    ts = start_ns + step * np.arange(n, dtype=np.int64)
    vals = np.linspace(0, 1, n * channels, dtype=np.float64).reshape(n, channels)
    names = tuple(f"EEG_{i}" for i in range(channels))
    units = tuple("V" for _ in range(channels))
    return TimeSeries(
        values=vals,
        timestamps=ts,
        channel_names=names,
        units=units,
        sample_rate=sample_rate,
    )


def test_usleep_roundtrip(tmp_path: Path) -> None:
    data = _regular_ts(100, 3, 128.0, start_ns=1_700_000_000_000_000_000)
    path = tmp_path / "u.h5"
    usleep.write(path, data)
    got = usleep.read(path)
    np.testing.assert_array_almost_equal(got.values, data.values)
    np.testing.assert_array_equal(got.timestamps, data.timestamps)
    assert got.channel_names == data.channel_names
    assert got.units == data.units
    assert got.sample_rate == pytest.approx(128.0)


def test_usleep_write_requires_sample_rate(tmp_path: Path) -> None:
    step = int(round(1e9 / 256.0))
    ts_arr = step * np.arange(10, dtype=np.int64)
    data = TimeSeries(
        values=np.zeros((10, 1), dtype=np.float64),
        timestamps=ts_arr,
        channel_names=("x",),
        units=("V",),
        sample_rate=None,
    )
    with pytest.raises(ValueError, match="sample_rate"):
        usleep.write(tmp_path / "no_sr.h5", data)


def test_usleep_rejects_irregular_timestamps(tmp_path: Path) -> None:
    data = TimeSeries(
        values=np.zeros((3, 1)),
        timestamps=np.array([0, 1, 3], dtype=np.int64),
        channel_names=("x",),
        units=("V",),
        sample_rate=1.0,
    )
    with pytest.raises(ValueError, match="grid"):
        usleep.write(tmp_path / "bad.h5", data)


def test_align_timestamps_to_usleep_grid_requires_sample_rate() -> None:
    data = TimeSeries(
        values=np.zeros((2, 1)),
        timestamps=np.array([0, 1_000_000_000], dtype=np.int64),
        channel_names=("x",),
        units=("V",),
        sample_rate=None,
    )
    with pytest.raises(ValueError, match="sample_rate"):
        usleep.align_timestamps_to_usleep_grid(data)


def test_usleep_channel_order_follows_channel_index_not_lexical(
    tmp_path: Path,
) -> None:
    """zmax-datasets uses channel_index for column order when names sort differently."""
    sr = 100.0
    n = 5
    step = int(round(1e9 / sr))
    ts = step * np.arange(n, dtype=np.int64)
    vals = np.column_stack(
        [
            np.arange(n, dtype=np.float64) * 10.0,
            np.arange(n, dtype=np.float64) * 100.0,
        ]
    )
    data = TimeSeries(
        values=vals,
        timestamps=ts,
        channel_names=("zebra", "apple"),
        units=("V", "V"),
        sample_rate=sr,
    )
    path = tmp_path / "order.h5"
    usleep.write(path, data)
    got = usleep.read(path)
    assert got.channel_names == ("zebra", "apple")
    np.testing.assert_array_equal(got.values[:, 0], vals[:, 0])
    np.testing.assert_array_equal(got.values[:, 1], vals[:, 1])


def test_usleep_read_without_channel_index_sorts_by_name(tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5"
    sr = 10.0
    n = 3
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = sr
        g = f.create_group("channels")
        g.create_dataset("b", data=np.ones(n, dtype=np.float64))
        g.create_dataset("a", data=np.zeros(n, dtype=np.float64))
    got = usleep.read(path)
    assert got.channel_names == ("a", "b")
    np.testing.assert_array_equal(got.values[:, 0], 0.0)
    np.testing.assert_array_equal(got.values[:, 1], 1.0)


def test_usleep_read_orders_by_channel_index_in_file(tmp_path: Path) -> None:
    """Explicit zmax-style file: lex order would be AAA, ZZZ; index says ZZZ first."""
    path = tmp_path / "idx.h5"
    sr = 100.0
    n = 4
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = sr
        f.attrs["start_timestamp_ns"] = np.int64(0)
        g = f.create_group("channels")
        z = g.create_dataset("ZZZ", data=np.full(n, 2.0, dtype=np.float64))
        z.attrs["channel_index"] = 0
        a = g.create_dataset("AAA", data=np.full(n, 9.0, dtype=np.float64))
        a.attrs["channel_index"] = 1
    got = usleep.read(path)
    assert got.channel_names == ("ZZZ", "AAA")
    np.testing.assert_array_equal(got.values[:, 0], 2.0)
    np.testing.assert_array_equal(got.values[:, 1], 9.0)


def test_usleep_read_ignores_per_dataset_sample_rate_attr(tmp_path: Path) -> None:
    """File-level rate wins; stray dataset attrs (e.g. from other exporters) ignored."""
    path = tmp_path / "ignore_ds_sr.h5"
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = 100.0
        g = f.create_group("channels")
        ds = g.create_dataset("x", data=np.zeros(5, dtype=np.float64))
        ds.attrs["sample_rate"] = 50.0
    got = usleep.read(path)
    assert got.sample_rate == pytest.approx(100.0)


def test_usleep_read_rejects_mixed_channel_index(tmp_path: Path) -> None:
    path = tmp_path / "mixed.h5"
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = 10.0
        g = f.create_group("channels")
        a = g.create_dataset("a", data=np.zeros(2, dtype=np.float64))
        a.attrs["channel_index"] = 0
        g.create_dataset("b", data=np.zeros(2, dtype=np.float64))
    with pytest.raises(ValueError, match="channel_index"):
        usleep.read(path)


def test_usleep_read_rejects_duplicate_channel_index(tmp_path: Path) -> None:
    path = tmp_path / "dup.h5"
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = 10.0
        g = f.create_group("channels")
        a = g.create_dataset("a", data=np.zeros(2, dtype=np.float64))
        a.attrs["channel_index"] = 0
        b = g.create_dataset("b", data=np.zeros(2, dtype=np.float64))
        b.attrs["channel_index"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        usleep.read(path)


def test_usleep_hdf5_protocol_wrappers(tmp_path: Path) -> None:
    data = _regular_ts(8, 2, 64.0)
    path = tmp_path / "wrap.h5"
    usleep.USleepHDF5Writer().write(path, data)
    got = usleep.USleepHDF5Reader().read(path)
    np.testing.assert_array_equal(got.values, data.values)
    assert got.sample_rate == pytest.approx(64.0)


def test_align_timestamps_then_write_with_jitter(tmp_path: Path) -> None:
    sr = 128.0
    step = int(round(1e9 / sr))
    ts = np.array([0, step, step * 2 + 1], dtype=np.int64)
    data = TimeSeries(
        values=np.linspace(0, 1, 9, dtype=np.float64).reshape(3, 3),
        timestamps=ts,
        channel_names=("a", "b", "c"),
        units=("V", "V", "V"),
        sample_rate=sr,
    )
    with pytest.raises(ValueError, match="grid"):
        usleep.write(tmp_path / "bad.h5", data)
    aligned = usleep.align_timestamps_to_usleep_grid(data)
    path = tmp_path / "ok.h5"
    usleep.write(path, aligned)
    got = usleep.read(path)
    np.testing.assert_array_equal(got.timestamps, aligned.timestamps)
    np.testing.assert_array_equal(got.values, aligned.values)
