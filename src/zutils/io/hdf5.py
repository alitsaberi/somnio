"""HDF5 file I/O for zutils data types.

This module owns all HDF5 serialization logic. Data types themselves carry no
format knowledge — conversion between ``Sample``/``TimeSeries`` and on-disk
HDF5 groups is handled here via ``serialize`` / ``deserialize``.

Two public entry points:

- **High-level** — ``write_hdf5`` / ``read_hdf5`` use the default
  ``HDF5Format`` layout.
- **Format instance** — instantiate ``HDF5Format`` directly to pass it
  wherever a ``FileFormat`` is expected, or subclass it to implement a
  different HDF5 layout (e.g. uSleep-style channels group).

Default storage layout per group::

    <group_name>/
        values      float64  (n_samples, n_channels)
        timestamps  int64    (n_samples,)

    <group_name>.attrs:
        channel_names   list[str]
        units           list[str]
        sample_rate     float | absent    absent means irregular (None)

Values are stored in SI base units without scaling.

Notes:
    ``HDF5Manager`` (real-time streaming / append-while-recording) is a
    slumber concern and lives in ``slumber.ext.units.storage``, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import h5py
import numpy as np

from zutils.data import Sample, TimeSeries


def serialize(
    obj: Union[Sample, TimeSeries],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert a ``Sample`` or ``TimeSeries`` to HDF5-ready dicts.

    Args:
        obj: Data object to serialize.

    Returns:
        A ``(datasets, attributes)`` pair:

        - ``datasets["values"]`` — float64 array, shape
          ``(n_samples, n_channels)`` (``(1, n_channels)`` for a ``Sample``).
        - ``datasets["timestamps"]`` — int64 array, shape ``(n_samples,)``
          (``(1,)`` for a ``Sample``).
        - ``attributes["channel_names"]`` — ``list[str]``.
        - ``attributes["units"]`` — ``list[str]``.
        - ``attributes["sample_rate"]`` — ``float`` Hz; absent when ``None``.
    """
    if isinstance(obj, Sample):
        datasets: dict[str, np.ndarray] = {
            "values": obj.values.reshape(1, -1),
            "timestamps": np.array([obj.timestamp], dtype=np.int64),
        }
        attrs: dict[str, Any] = {
            "channel_names": list(obj.channel_names),
            "units": list(obj.units),
        }
    else:
        datasets = {
            "values": obj.values,
            "timestamps": obj.timestamps,
        }
        attrs = {
            "channel_names": list(obj.channel_names),
            "units": list(obj.units),
        }
        if obj.sample_rate is not None:
            attrs["sample_rate"] = obj.sample_rate

    return datasets, attrs


def deserialize(datasets: dict[str, Any], attrs: dict[str, Any]) -> TimeSeries:
    """Reconstruct a ``TimeSeries`` from HDF5 group data.

    Args:
        datasets: Mapping with ``"values"`` and ``"timestamps"`` keys.
            Values may be ``h5py.Dataset`` objects or numpy arrays.
        attrs: Group-level attribute mapping with ``"channel_names"``,
            ``"units"``, and optionally ``"sample_rate"``.

    Returns:
        Reconstructed ``TimeSeries``.
    """
    return TimeSeries(
        values=np.asarray(datasets["values"], dtype=np.float64),
        timestamps=np.asarray(datasets["timestamps"], dtype=np.int64),
        channel_names=list(attrs["channel_names"]),
        units=list(attrs["units"]),
        sample_rate=float(attrs["sample_rate"]) if "sample_rate" in attrs else None,
    )


class HDF5Format:
    """Default zutils HDF5 layout: one top-level group per named signal.

    Implements :class:`~zutils.io.base.FileFormat`.  Subclass and override
    ``write`` / ``read`` to implement an alternative HDF5 layout (e.g.
    uSleep-style ``channels/`` group with a global ``sample_rate`` attribute).

    Args:
        compression: HDF5 compression filter for ``values`` and ``timestamps``
            datasets. ``None`` disables compression.
    """

    def __init__(self, compression: str | None = "gzip") -> None:
        self.compression = compression

    def write(
        self,
        path: Path,
        data: dict[str, Union[Sample, TimeSeries]],
    ) -> None:
        """Write named data objects to an HDF5 file.

        Overwrites any existing file at ``path``.

        Args:
            path: Destination HDF5 file path.
            data: Mapping of group name → ``Sample`` or ``TimeSeries``.
        """
        path = Path(path)
        if path.exists():
            path.unlink()

        with h5py.File(path, "w") as f:
            for group_name, obj in data.items():
                datasets, attrs = serialize(obj)
                group = f.create_group(group_name)
                group.attrs.update(attrs)
                for ds_name, array in datasets.items():
                    group.create_dataset(
                        ds_name, data=array, compression=self.compression
                    )

    def read(self, path: Path) -> dict[str, TimeSeries]:
        """Load all groups from an HDF5 file as ``TimeSeries`` objects.

        Args:
            path: Path to an HDF5 file written by :meth:`write`.

        Returns:
            Mapping of group name → ``TimeSeries``.
        """
        path = Path(path)
        result: dict[str, TimeSeries] = {}
        with h5py.File(path, "r") as f:
            for group_name in f:
                group = f[group_name]
                result[group_name] = deserialize(group, dict(group.attrs))
        return result


_default_format = HDF5Format()


def write_hdf5(
    path: Path,
    data: dict[str, Union[Sample, TimeSeries]],
) -> None:
    """Write named data objects to an HDF5 file using the default layout.

    Convenience wrapper around :meth:`HDF5Format.write`.
    Overwrites any existing file at ``path``.

    Args:
        path: Destination HDF5 file path.
        data: Mapping of group name → ``Sample`` or ``TimeSeries``.
    """
    _default_format.write(path, data)


def read_hdf5(path: Path) -> dict[str, TimeSeries]:
    """Load all groups from an HDF5 file using the default layout.

    Convenience wrapper around :meth:`HDF5Format.read`.

    Args:
        path: Path to an HDF5 file written by :func:`write_hdf5`.

    Returns:
        Mapping of group name → ``TimeSeries``.
    """
    return _default_format.read(path)
