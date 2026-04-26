"""Native somnio HDF5 layout: per-group ``data`` / ``timestamp`` datasets + attrs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.utils.imports import MissingOptionalDependency

try:
    import h5py
except ModuleNotFoundError as e:
    if e.name != "h5py":
        raise
    raise MissingOptionalDependency(
        "h5py", extra="hdf5", purpose="Native HDF5 I/O"
    ) from e


def _str_array_attrs(names: tuple[str, ...]) -> np.ndarray:
    return np.array(names, dtype=h5py.string_dtype(encoding="utf-8"))


def _attrs_to_str_tuple(arr: Any) -> tuple[str, ...]:
    if isinstance(arr, np.ndarray):
        return tuple(np.asarray(arr).astype(str).tolist())
    if isinstance(arr, bytes | str):
        return (str(arr),)
    return tuple(str(x) for x in arr)


def _read_group_sample(g: h5py.Group) -> TimeSeries:
    d = g["data"][...]
    t = g["timestamp"][...]
    ch = _attrs_to_str_tuple(g.attrs["channel_names"])
    units = _attrs_to_str_tuple(g.attrs["units"])
    sr = g.attrs.get("sample_rate", None)
    if sr is None or (isinstance(sr, np.floating) and np.isnan(sr)):
        sample_rate: float | None = None
    else:
        sample_rate = float(sr)
    return TimeSeries(
        values=np.asarray(d, dtype=np.float64),
        timestamps=np.asarray(t, dtype=np.int64),
        channel_names=ch,
        units=units,
        sample_rate=sample_rate,
    )


def _is_native_group(obj: h5py.Group) -> bool:
    return "data" in obj and "timestamp" in obj


def read(path: Path | str, *, group: str | None = None) -> TimeSeries:
    """Read one native-layout group from an HDF5 file.

    Args:
        path: HDF5 file path.
        group: HDF5 group name. If None, the file must contain exactly one
            native-layout top-level group (with ``data`` and ``timestamp``).

    Returns:
        The reconstructed :class:`~somnio.data.timeseries.TimeSeries`.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if group is not None:
            g = f[group]
            if not isinstance(g, h5py.Group) or not _is_native_group(g):
                raise ValueError(f"Group {group!r} is not a native somnio layout")
            return _read_group_sample(g)

        candidates = [
            k
            for k in f.keys()
            if isinstance(f[k], h5py.Group) and _is_native_group(f[k])
        ]
        if len(candidates) != 1:
            raise ValueError(
                "group must be specified when the file does not contain "
                f"exactly one native-layout group (found {len(candidates)}: {candidates})"
            )
        return _read_group_sample(f[candidates[0]])


def read_all(path: Path | str) -> dict[str, TimeSeries]:
    """Read every top-level native-layout group in the file.

    Groups without ``data``/``timestamp`` are skipped.
    """
    path = Path(path)
    out: dict[str, TimeSeries] = {}
    with h5py.File(path, "r") as f:
        for name, obj in f.items():
            if isinstance(obj, h5py.Group) and _is_native_group(obj):
                out[str(name)] = _read_group_sample(obj)
    return out


def write(
    path: Path | str,
    data: TimeSeries,
    group_name: str,
    *,
    append: bool = False,
) -> None:
    """Write a :class:`~somnio.data.timeseries.TimeSeries` as one native group.

    Args:
        path: HDF5 file path.
        data: Time series to store.
        group_name: Name of the HDF5 group (e.g. ``zmax_raw``).
        append: If True, open in append mode and add the group. If False,
            truncate/create the file and write only this group.
    """
    path = Path(path)
    mode = "a" if append else "w"
    with h5py.File(path, mode) as f:
        if group_name in f:
            raise ValueError(f"group {group_name!r} already exists")
        g = f.create_group(group_name)
        g.create_dataset("data", data=data.values, compression="gzip", shuffle=True)
        g.create_dataset("timestamp", data=data.timestamps, compression="gzip")
        g.attrs["channel_names"] = _str_array_attrs(data.channel_names)
        g.attrs["units"] = _str_array_attrs(tuple(u.symbol for u in data.units))
        if data.sample_rate is not None:
            g.attrs["sample_rate"] = float(data.sample_rate)


def serialize(ts: TimeSeries) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Encode a :class:`~somnio.data.timeseries.TimeSeries` for native HDF5 layout.

    Returns dataset arrays and group attributes suitable for incremental append
    (e.g. extend ``data``/``timestamp`` datasets in the same group).

    Returns:
        A pair ``(datasets, attrs)`` where ``datasets`` maps ``\"data\"`` and
        ``\"timestamp\"`` to arrays, and ``attrs`` holds ``channel_names``,
        ``units``, and optionally ``sample_rate``.
    """
    datasets = {"data": ts.values, "timestamp": ts.timestamps}
    attrs: dict[str, Any] = {
        "channel_names": ts.channel_names,
        "units": tuple(u.symbol for u in ts.units),
    }
    if ts.sample_rate is not None:
        attrs["sample_rate"] = float(ts.sample_rate)
    return datasets, attrs


def deserialize(
    datasets: Mapping[str, np.ndarray],
    attrs: Mapping[str, Any],
) -> TimeSeries:
    """Decode outputs of :func:`serialize` into a :class:`~somnio.data.timeseries.TimeSeries`."""
    try:
        values = datasets["data"]
        timestamps = datasets["timestamp"]
    except KeyError as exc:
        raise KeyError("datasets must contain 'data' and 'timestamp'") from exc
    ch = attrs["channel_names"]
    un = attrs["units"]
    if isinstance(ch, np.ndarray):
        ch = tuple(ch.astype(str).tolist())
    else:
        ch = tuple(ch)
    if isinstance(un, np.ndarray):
        un = tuple(un.astype(str).tolist())
    else:
        un = tuple(un)
    sr = attrs.get("sample_rate", None)
    if sr is not None:
        sr = float(sr)
    return TimeSeries(
        values=np.asarray(values, dtype=np.float64),
        timestamps=np.asarray(timestamps, dtype=np.int64),
        channel_names=ch,
        units=un,
        sample_rate=sr,
    )


class NativeHDF5Reader:
    """Stateless reader for the native somnio HDF5 layout."""

    def read(self, path: Path, **kwargs: Any) -> TimeSeries:
        """Read from ``path``; pass ``group`` to select a specific group name."""
        group = kwargs.get("group", None)
        return read(path, group=group)


class NativeHDF5Writer:
    """Stateless writer for the native somnio HDF5 layout."""

    def write(self, path: Path, data: TimeSeries, **kwargs: Any) -> None:
        """Write ``data``; pass ``group_name`` (required) and optional ``append``."""
        try:
            group_name = kwargs["group_name"]
        except KeyError as exc:
            raise TypeError("write() requires keyword argument 'group_name'") from exc
        append = bool(kwargs.get("append", False))
        write(path, data, group_name, append=append)
