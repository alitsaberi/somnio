"""USleep / zmax-datasets HDF5 export layout: ``/channels/{{name}}`` + file ``sample_rate``.

File-level ``sample_rate`` (Hz) is authoritative. Optional ``start_timestamp_ns``
(somnio extension for epoch round-trip). Per-channel 1-D datasets use attrs
``channel_index`` (write order, as in zmax-datasets) and optional ``unit``.
Per-dataset ``sample_rate`` is not written; if present on files from other tools,
:func:`read` ignores it and uses only the file attribute.

Timestamps are not stored per sample; on :func:`read` they are reconstructed as::

    t[i] = start_timestamp_ns + i * round(1e9 / sample_rate)

If every channel dataset has ``channel_index``, :func:`read` stacks columns in
that order (not lexicographic by name). Otherwise names are sorted for backward
compatibility.

:class:`~somnio.data.timeseries.TimeSeries` keeps per-sample timestamps as
authoritative; :func:`write` checks they match the grid above. Use
:func:`align_timestamps_to_usleep_grid` if sub-nanosecond jitter should be
snapped before writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.utils.imports import MissingOptionalDependency

try:
    import h5py
except ModuleNotFoundError as e:
    if e.name != "h5py":
        raise
    raise MissingOptionalDependency(
        "h5py", extra="hdf5", purpose="USleep HDF5 I/O"
    ) from e

CHANNELS_GROUP = "channels"
ATTR_SAMPLE_RATE = "sample_rate"
ATTR_START_TIMESTAMP = "start_timestamp_ns"
ATTR_CHANNEL_INDEX = "channel_index"


def align_timestamps_to_usleep_grid(data: TimeSeries) -> TimeSeries:
    """Return a copy with timestamps on the USleep integer nanosecond grid.

    Grid: ``t[i] = int(timestamps[0]) + i * round(1e9 / sample_rate)``.
    Values, channel names, units, and ``sample_rate`` are unchanged.

    Raises:
        ValueError: If ``sample_rate`` is None.
    """
    if data.sample_rate is None:
        raise ValueError(
            "align_timestamps_to_usleep_grid requires TimeSeries.sample_rate (Hz)"
        )
    if data.n_samples == 0:
        return TimeSeries(
            values=data.values.copy(),
            timestamps=data.timestamps.copy(),
            channel_names=list(data.channel_names),
            units=list(data.units),
            sample_rate=data.sample_rate,
        )
    sample_rate = float(data.sample_rate)
    start_ns = int(data.timestamps[0])
    step = int(round(1e9 / sample_rate))
    ts = start_ns + step * np.arange(data.n_samples, dtype=np.int64)
    return TimeSeries(
        values=data.values.copy(),
        timestamps=ts,
        channel_names=list(data.channel_names),
        units=list(data.units),
        sample_rate=data.sample_rate,
    )


def _dataset_channel_names(ch_root: h5py.Group) -> list[str]:
    """Resolve channel order: by ``channel_index`` if all sets have it, else by name sort."""
    names = [k for k in ch_root.keys() if isinstance(ch_root[k], h5py.Dataset)]
    if not names:
        raise ValueError(f"no channel datasets under /{CHANNELS_GROUP}")

    with_index: list[tuple[str, int]] = []
    without: list[str] = []
    for name in names:
        ds = ch_root[name]
        if ATTR_CHANNEL_INDEX in ds.attrs:
            with_index.append((str(name), int(ds.attrs[ATTR_CHANNEL_INDEX])))
        else:
            without.append(str(name))

    if with_index and without:
        raise ValueError(
            "All channel datasets must define "
            f"attrs['{ATTR_CHANNEL_INDEX}'], or none may define it; "
            f"mixed: indexed={[n for n, _ in with_index]}, missing_index={without}"
        )

    if with_index:
        with_index.sort(key=lambda x: x[1])
        idxs = [i for _, i in with_index]
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"duplicate {ATTR_CHANNEL_INDEX} values: {idxs}")
        return [n for n, _ in with_index]

    return sorted(names, key=str)


def read(path: Path | str) -> TimeSeries:
    """Read USleep-style HDF5: 1-D float datasets under ``/channels``.

    File attribute ``sample_rate`` (Hz) is required. The returned
    :class:`~somnio.data.timeseries.TimeSeries` always has ``sample_rate`` set.
    Optional ``start_timestamp_ns`` sets the first-sample Unix-epoch time in
    nanoseconds (default 0). Per-channel attrs: optional ``unit``; if every
    channel has ``channel_index``, column order follows those indices (zmax-datasets
    export). Only the file-level ``sample_rate`` is used; any per-dataset
    ``sample_rate`` attribute is ignored.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if CHANNELS_GROUP not in f:
            raise ValueError(f"expected /{CHANNELS_GROUP} group in {path}")
        ch_root = f[CHANNELS_GROUP]
        if not isinstance(ch_root, h5py.Group):
            raise ValueError(f"/{CHANNELS_GROUP} must be a group")

        names = _dataset_channel_names(ch_root)

        if ATTR_SAMPLE_RATE not in f.attrs:
            raise ValueError(f"file missing required attr '{ATTR_SAMPLE_RATE}'")
        sample_rate = float(f.attrs[ATTR_SAMPLE_RATE])
        if sample_rate <= 0:
            raise ValueError(f"{ATTR_SAMPLE_RATE} must be positive")

        series = [np.asarray(ch_root[name][...], dtype=np.float64) for name in names]
        lengths = {s.shape[0] for s in series}
        if len(lengths) != 1:
            raise ValueError(f"channel length mismatch: {lengths}")

        start_ns = int(f.attrs.get(ATTR_START_TIMESTAMP, 0))
        step = int(round(1e9 / sample_rate))
        n = series[0].shape[0]
        timestamps = start_ns + step * np.arange(n, dtype=np.int64)

        units_list: list[str] = []
        for name in names:
            ds = ch_root[name]
            u = ds.attrs.get("unit", b"")
            if isinstance(u, bytes):
                units_list.append(u.decode("utf-8"))
            else:
                units_list.append(str(u))

        values = np.column_stack(series)
        return TimeSeries(
            values=values,
            timestamps=timestamps,
            channel_names=tuple(str(n) for n in names),
            units=tuple(units_list),
            sample_rate=sample_rate,
        )


def write(path: Path | str, data: TimeSeries) -> None:
    """Write ``data`` in USleep-style layout (overwrites ``path``).

    Requires :attr:`~somnio.data.timeseries.TimeSeries.sample_rate` to be set
    (Hz). Timestamps must match the regular grid
    ``start_ns + i * round(1e9 / sample_rate)`` with ``start_ns`` equal to the
    first sample's time. ``start_timestamp_ns`` is stored from that first
    timestamp. Each channel dataset gets ``channel_index`` (column order).
    """
    path = Path(path)
    if data.sample_rate is None:
        raise ValueError(
            "USleep HDF5 write requires TimeSeries.sample_rate to be set (Hz)"
        )
    sample_rate = float(data.sample_rate)
    start_ns = int(data.timestamps[0]) if data.n_samples else 0

    if data.n_samples:
        step = int(round(1e9 / sample_rate))
        expected = start_ns + step * np.arange(data.n_samples, dtype=np.int64)
        if not np.array_equal(data.timestamps, expected):
            raise ValueError(
                "USleep HDF5 requires timestamps on the grid "
                "start_ns + i * round(1e9 / sample_rate). "
                "Call align_timestamps_to_usleep_grid() first if jitter is acceptable."
            )

    with h5py.File(path, "w") as f:
        f.attrs[ATTR_SAMPLE_RATE] = float(sample_rate)
        f.attrs[ATTR_START_TIMESTAMP] = start_ns

        g = f.create_group(CHANNELS_GROUP)
        for j, name in enumerate(data.channel_names):
            ds = g.create_dataset(
                name,
                data=data.values[:, j],
                compression="gzip",
                shuffle=True,
            )
            ds.attrs[ATTR_CHANNEL_INDEX] = j
            ds.attrs["unit"] = data.units[j]


class USleepHDF5Reader:
    """Stateless reader for the USleep HDF5 export layout."""

    def read(self, path: Path, **kwargs: Any) -> TimeSeries:
        return read(path)


class USleepHDF5Writer:
    """Stateless writer for the USleep HDF5 export layout."""

    def write(self, path: Path, data: TimeSeries, **kwargs: Any) -> None:
        write(path, data)
