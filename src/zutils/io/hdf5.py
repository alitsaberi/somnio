"""HDF5 file I/O for zutils data types.

This module owns all HDF5 serialization logic. Data types themselves carry no
format knowledge — conversion between ``Sample``/``TimeSeries`` and on-disk
HDF5 groups is handled here via ``serialize`` / ``deserialize``.

Storage layout per group::

    <group_name>/
        values      float64  (n_samples, n_channels)  or (1, n_channels) for Sample
        timestamps  int64    (n_samples,)              or (1,)             for Sample

    <group_name>.attrs:
        channel_names   list[str]
        units           list[str]
        sample_rate     float | absent                 absent means irregular (None)

Values are stored in SI base units exactly as they appear in the data type;
no scaling is applied on read or write.
"""

from __future__ import annotations

import time
import types
import warnings
from pathlib import Path
from typing import Any, Union

import h5py
import numpy as np

from zutils.data import Sample, TimeSeries


class GroupDoesNotExistError(ValueError):
    """Raised when a requested HDF5 group does not exist."""


class DatasetDoesNotExistError(ValueError):
    """Raised when a requested HDF5 dataset does not exist."""


class HDF5Manager:
    """Low-level HDF5 file manager with retry logic and context-manager support.

    Args:
        file_path: Path to the HDF5 file (created if absent, appended otherwise).
        compression: HDF5 compression filter applied to new datasets.
        max_retries: Number of open attempts before raising ``OSError``.
        retry_delay: Initial retry wait in seconds; doubled each attempt.
    """

    def __init__(
        self,
        file_path: Path,
        compression: str = "gzip",
        max_retries: int = 5,
        retry_delay: float = 0.1,
    ) -> None:
        self.file_path = Path(file_path)
        self.compression = compression
        self.file = self._open_with_retry(max_retries, retry_delay)

    def _open_with_retry(self, max_retries: int, retry_delay: float) -> h5py.File:
        """Open the HDF5 file with exponential-backoff retry on ``OSError``.

        Args:
            max_retries: Maximum open attempts.
            retry_delay: Initial sleep between attempts in seconds.

        Returns:
            Opened ``h5py.File`` in append mode.

        Raises:
            OSError: If the file cannot be opened after all attempts.
        """
        for attempt in range(max_retries):
            try:
                return h5py.File(self.file_path, "a")
            except OSError as exc:
                if attempt == max_retries - 1:
                    raise OSError(
                        f"Cannot open {self.file_path} after {max_retries} attempts."
                        f" Last error: {exc}"
                    ) from exc
                delay = retry_delay * (2**attempt)
                warnings.warn(
                    f"HDF5 open failed (attempt {attempt + 1}/{max_retries}),"
                    f" retrying in {delay:.2f}s: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                time.sleep(delay)

    def __enter__(self) -> "HDF5Manager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HDF5 file if still open."""
        if self.file.id.valid:
            self.file.close()

    @property
    def groups(self) -> list[str]:
        """Top-level group names in the file."""
        return list(self.file.keys())

    def create_group(self, group_name: str, **attributes: Any) -> h5py.Group:
        """Create a new top-level group with optional attributes.

        Args:
            group_name: Name for the new group.
            **attributes: Key/value metadata stored as HDF5 group attributes.

        Returns:
            The newly created ``h5py.Group``.
        """
        group = self.file.create_group(group_name)
        group.attrs.update(attributes)
        return group

    def create_dataset(
        self,
        group_name: str,
        dataset_name: str,
        data: np.ndarray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: Any = None,
        max_shape: tuple[int | None, ...] | None = None,
        **attributes: Any,
    ) -> h5py.Dataset:
        """Create a dataset inside an existing group.

        Either ``data`` or ``shape`` must be provided.  The dataset is
        created with ``maxshape`` so it can be extended later via
        :meth:`append`.

        Args:
            group_name: Containing group (must already exist).
            dataset_name: Name for the new dataset.
            data: Initial data array. Mutually exclusive with ``shape``.
            shape: Initial shape when no data is provided.
            dtype: NumPy dtype or compound dtype spec; inferred from data when absent.
            max_shape: Per-axis maximum extents. Defaults to all-``None`` (unlimited).
            **attributes: Metadata stored as dataset attributes.

        Returns:
            The created ``h5py.Dataset``.

        Raises:
            GroupDoesNotExistError: If ``group_name`` is not in the file.
            ValueError: If neither ``data`` nor ``shape`` is provided.
        """
        if group_name not in self.file:
            raise GroupDoesNotExistError(f"Group '{group_name}' does not exist.")
        if data is None and shape is None:
            raise ValueError("Either data or shape must be provided.")

        effective_shape = data.shape if data is not None else shape
        if max_shape is None:
            max_shape = tuple(None for _ in effective_shape)

        dataset = self.file[group_name].create_dataset(
            dataset_name,
            data=data,
            shape=shape,
            dtype=dtype,
            maxshape=max_shape,
            compression=self.compression,
        )
        dataset.attrs.update(attributes)
        return dataset

    def get_dataset(self, group_name: str, dataset_name: str) -> h5py.Dataset:
        """Return a dataset by group and dataset name.

        Args:
            group_name: Containing group name.
            dataset_name: Dataset name within the group.

        Returns:
            The ``h5py.Dataset`` object.

        Raises:
            GroupDoesNotExistError: If ``group_name`` is absent.
            DatasetDoesNotExistError: If ``dataset_name`` is absent in the group.
        """
        if group_name not in self.file:
            raise GroupDoesNotExistError(f"Group '{group_name}' does not exist.")
        if dataset_name not in self.file[group_name]:
            raise DatasetDoesNotExistError(
                f"Dataset '{dataset_name}' does not exist in group '{group_name}'."
            )
        return self.file[group_name][dataset_name]

    def append(self, group_name: str, dataset_name: str, data: np.ndarray) -> None:
        """Append rows to an existing resizable dataset.

        Args:
            group_name: Containing group name.
            dataset_name: Dataset to extend.
            data: New rows to append; must match the dataset's column count.

        Raises:
            GroupDoesNotExistError: If ``group_name`` is absent.
            DatasetDoesNotExistError: If ``dataset_name`` is absent.
            ValueError: If column counts differ for 2-D datasets.
        """
        dataset = self.get_dataset(group_name, dataset_name)

        if dataset.ndim == 2 and dataset.shape[1] != data.shape[-1]:
            raise ValueError(
                f"Column count mismatch: dataset has {dataset.shape[1]} columns,"
                f" data has {data.shape[-1]}."
            )

        new_len = dataset.shape[0] + len(data)
        dataset.resize(new_len, axis=0)
        dataset[-len(data) :] = data


def serialize(
    obj: Union[Sample, TimeSeries],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert a ``Sample`` or ``TimeSeries`` to HDF5-ready dicts.

    Args:
        obj: Data object to serialize.

    Returns:
        A ``(datasets, attributes)`` pair where:

        - ``datasets["values"]`` — float64 values array, shape
          ``(n_samples, n_channels)`` (or ``(1, n_channels)`` for a Sample).
        - ``datasets["timestamps"]`` — int64 timestamps array, shape
          ``(n_samples,)`` (or ``(1,)`` for a Sample).
        - ``attributes["channel_names"]`` — list of channel name strings.
        - ``attributes["units"]`` — list of SI unit strings.
        - ``attributes["sample_rate"]`` — float Hz, present only when not ``None``.
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
        datasets: Mapping with at least ``"values"`` and ``"timestamps"`` keys.
            Values may be ``h5py.Dataset`` objects or numpy arrays.
        attrs: Group-level attribute mapping with ``"channel_names"`` and
            ``"units"`` keys, and an optional ``"sample_rate"`` key.

    Returns:
        Reconstructed ``TimeSeries``.
    """
    values = np.asarray(datasets["values"], dtype=np.float64)
    timestamps = np.asarray(datasets["timestamps"], dtype=np.int64)
    channel_names = list(attrs["channel_names"])
    units = list(attrs["units"])
    sample_rate: float | None = (
        float(attrs["sample_rate"]) if "sample_rate" in attrs else None
    )
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=channel_names,
        units=units,
        sample_rate=sample_rate,
    )


def write_hdf5(path: Path, data: dict[str, Union[Sample, TimeSeries]]) -> None:
    """Write a mapping of named data objects to an HDF5 file.

    Each key becomes a top-level group.  An existing file is overwritten.

    Args:
        path: Destination HDF5 file path.
        data: Mapping of group name → ``Sample`` or ``TimeSeries``.
    """
    path = Path(path)
    if path.exists():
        path.unlink()

    with HDF5Manager(path) as mgr:
        for group_name, obj in data.items():
            datasets, attrs = serialize(obj)
            mgr.create_group(group_name, **attrs)
            for ds_name, array in datasets.items():
                mgr.create_dataset(group_name, ds_name, data=array)


def read_hdf5(path: Path) -> dict[str, TimeSeries]:
    """Load all groups from an HDF5 file as ``TimeSeries`` objects.

    Args:
        path: Path to an HDF5 file written by :func:`write_hdf5`.

    Returns:
        Mapping of group name → ``TimeSeries``.
    """
    path = Path(path)
    result: dict[str, TimeSeries] = {}
    with HDF5Manager(path) as mgr:
        for group_name in mgr.groups:
            group = mgr.file[group_name]
            result[group_name] = deserialize(
                datasets=group,
                attrs=dict(group.attrs),
            )
    return result
