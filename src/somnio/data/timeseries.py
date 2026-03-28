"""In-memory data types for sleep signal data.

Pure containers — no I/O, no logging, no serialization format knowledge.

Conventions (authoritative for all of somnio):
- Timestamps: int64 nanoseconds since Unix epoch (1970-01-01T00:00:00 UTC).
- Values dtype: float64.
- Physical units: SI base units tracked per-channel via ``units`` field.
  Use SI symbols: ``"V"`` (not ``"uV"``), ``"m/s^2"`` (not ``"g"``),
  ``"degC"`` for temperature.
- sample_rate: float in Hz when nominally regularly sampled, or None for
  irregular/unknown data. Timestamps remain authoritative; a specific file
  format may still require an exact spacing (see that format's writer).
- channel_names: unique, non-empty, underscore-separated (e.g. ``"EEG_L"``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property

import numpy as np


@dataclass
class Sample:
    """Single time-point, multi-channel measurement.

    Lightweight type for streaming (one sample per message).

    Attributes:
        values: Measurement values, shape ``(n_channels,)``, dtype float64.
        timestamp: Acquisition time in nanoseconds since Unix epoch.
        channel_names: Unique channel identifiers, length == n_channels.
        units: Physical unit per channel (SI symbols), length == n_channels.
            e.g. ``["V", "V", "m/s^2", "m/s^2", "m/s^2", "degC"]``.
    """

    values: np.ndarray
    timestamp: int
    channel_names: tuple[str, ...]
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        self.channel_names = tuple(self.channel_names)
        self.units = tuple(self.units)
        if self.values.ndim != 1:
            raise ValueError(
                f"values must be 1-D (n_channels,), got shape {self.values.shape}"
            )
        n = self.values.shape[0]
        if len(self.channel_names) != n:
            raise ValueError(
                f"len(channel_names)={len(self.channel_names)} != n_channels={n}"
            )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        if len(self.units) != n:
            raise ValueError(f"len(units)={len(self.units)} != n_channels={n}")


@dataclass
class TimeSeries:
    """Timestamp-first multi-channel timeseries.

    Core streaming/storage type. Supports irregular sampling natively;
    regular sampling is expressed via optional ``sample_rate`` metadata
    (nominal Hz when ``sample_rate`` is set). Timestamps are always per-sample
    and authoritative — they are not auto-corrected to match ``sample_rate``.

    Attributes:
        values: Measurement array, shape ``(n_samples, n_channels)``, dtype float64.
            All channels in SI base units (V, m/s^2, degC, ...).
        timestamps: Per-sample acquisition time, shape ``(n_samples,)``, dtype int64.
            Nanoseconds since Unix epoch. Monotonically non-decreasing.
        channel_names: Unique channel identifiers, length == n_channels.
        units: Physical unit per channel (SI symbols), length == n_channels.
        sample_rate: Nominal sample rate in Hz when data are intended to be
            regularly spaced, or None for irregular/unknown sampling.
    """

    values: np.ndarray
    timestamps: np.ndarray
    channel_names: tuple[str, ...]
    units: tuple[str, ...]
    sample_rate: float | None = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        self.timestamps = np.asarray(self.timestamps, dtype=np.int64)
        self.channel_names = tuple(self.channel_names)
        self.units = tuple(self.units)

        if self.values.ndim != 2:
            raise ValueError(
                f"values must be 2-D (n_samples, n_channels), got shape {self.values.shape}"
            )
        n_samples, n_channels = self.values.shape
        if self.timestamps.shape != (n_samples,):
            raise ValueError(
                f"timestamps shape {self.timestamps.shape} != (n_samples={n_samples},)"
            )
        if len(self.channel_names) != n_channels:
            raise ValueError(
                f"len(channel_names)={len(self.channel_names)} != n_channels={n_channels}"
            )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        if len(self.units) != n_channels:
            raise ValueError(f"len(units)={len(self.units)} != n_channels={n_channels}")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if n_samples > 1 and np.any(self.timestamps[1:] < self.timestamps[:-1]):
            raise ValueError("timestamps must be monotonically non-decreasing")

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.values.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.values.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of ``values``: ``(n_samples, n_channels)``."""
        return self.values.shape  # type: ignore[return-value]

    @property
    def duration(self) -> timedelta:
        """Wall-clock duration from first to last sample (nanosecond precision)."""
        if self.n_samples < 2:
            return timedelta(0)
        delta_ns = int(self.timestamps[-1]) - int(self.timestamps[0])
        return timedelta(microseconds=delta_ns / 1_000)

    @cached_property
    def channel_index_map(self) -> dict[str, int]:
        """Mapping from channel name to column index."""
        return {name: idx for idx, name in enumerate(self.channel_names)}

    @property
    def is_regular(self) -> bool:
        """True when ``sample_rate`` is set."""
        return self.sample_rate is not None

    def select_channels(self, names: list[str]) -> "TimeSeries":
        """Return a new TimeSeries restricted to the given channel names.

        Args:
            names: Ordered list of channel names to keep.

        Returns:
            New TimeSeries with only the requested channels.

        Raises:
            KeyError: If any name is not in ``channel_names``.
        """
        indices = [self.channel_index_map[n] for n in names]
        units = [self.units[i] for i in indices]
        return TimeSeries(
            values=self.values[:, indices],
            timestamps=self.timestamps.copy(),
            channel_names=list(names),
            units=units,
            sample_rate=self.sample_rate,
        )

    def select_time(
        self, start: int | None = None, end: int | None = None
    ) -> "TimeSeries":
        """Return a new TimeSeries within [start, end] nanosecond timestamps (inclusive).

        Args:
            start: Lower bound in ns since Unix epoch. None means no lower bound.
            end: Upper bound in ns since Unix epoch. None means no upper bound.

        Returns:
            New TimeSeries covering the requested time range.
        """
        mask = np.ones(self.n_samples, dtype=bool)
        if start is not None:
            mask &= self.timestamps >= start
        if end is not None:
            mask &= self.timestamps <= end
        return TimeSeries(
            values=self.values[mask],
            timestamps=self.timestamps[mask],
            channel_names=list(self.channel_names),
            units=list(self.units),
            sample_rate=self.sample_rate,
        )

    def __getitem__(self, key: int | slice) -> "TimeSeries":
        """Integer or slice indexing along the sample axis.

        Args:
            key: Integer index or slice.

        Returns:
            New TimeSeries for the selected samples.
        """
        if isinstance(key, int):
            n = self.n_samples
            if key < 0:
                key += n
            if key < 0 or key >= n:
                raise IndexError("TimeSeries index out of range")
            key = slice(key, key + 1)
        return TimeSeries(
            values=self.values[key],
            timestamps=self.timestamps[key],
            channel_names=list(self.channel_names),
            units=list(self.units),
            sample_rate=self.sample_rate,
        )


def concat(series: Sequence[TimeSeries]) -> TimeSeries:
    """Concatenate TimeSeries objects along the time axis.

    Args:
        series: Non-empty sequence of TimeSeries with matching channel_names and units.

    Returns:
        A new TimeSeries with samples from all inputs in order.
        ``sample_rate`` is propagated only if all inputs share the same value;
        otherwise it is set to None.

    Raises:
        ValueError: If ``series`` is empty or channel_names/units mismatch.
    """
    if len(series) == 0:
        raise ValueError("series must be non-empty")

    ref = series[0]
    for ts in series[1:]:
        if ts.channel_names != ref.channel_names:
            raise ValueError(
                f"channel_names mismatch: {ref.channel_names!r} vs {ts.channel_names!r}"
            )
        if ts.units != ref.units:
            raise ValueError(f"units mismatch: {ref.units!r} vs {ts.units!r}")

    rates = {ts.sample_rate for ts in series}
    sample_rate = rates.pop() if len(rates) == 1 else None

    return TimeSeries(
        values=np.concatenate([ts.values for ts in series], axis=0),
        timestamps=np.concatenate([ts.timestamps for ts in series]),
        channel_names=list(ref.channel_names),
        units=list(ref.units),
        sample_rate=sample_rate,
    )


def collect_samples(samples: Sequence[Sample]) -> TimeSeries:
    """Batch a sequence of Sample objects into a TimeSeries.

    Args:
        samples: Non-empty sequence of Sample objects with matching
            channel_names and units.

    Returns:
        TimeSeries with ``sample_rate=None`` (irregular by definition;
        caller may set it afterward if the source is known to be regular).

    Raises:
        ValueError: If ``samples`` is empty or channel_names/units mismatch.
    """
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")

    ref = samples[0]
    for s in samples[1:]:
        if s.channel_names != ref.channel_names:
            raise ValueError(
                f"channel_names mismatch: {ref.channel_names!r} vs {s.channel_names!r}"
            )
        if s.units != ref.units:
            raise ValueError(f"units mismatch: {ref.units!r} vs {s.units!r}")

    return TimeSeries(
        values=np.stack([s.values for s in samples]),
        timestamps=np.array([s.timestamp for s in samples], dtype=np.int64),
        channel_names=list(ref.channel_names),
        units=list(ref.units),
        sample_rate=None,
    )
