"""NextSense dual-ear CSV export reader."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.data.units import UNKNOWN, UV, Unit

_START_DATA = re.compile(r"^#=+\s*Start Data\s*=+\s*$", re.IGNORECASE)
_META_LINE = re.compile(r"^#\s*([^:]+):\s*(.*)\s*$")
_HEADER_COLS = re.compile(r"^#\s*(SAMPLE_NUMBER,.+)\s*$", re.IGNORECASE)

# Columns that are not signal channels.
_META_COLUMNS = frozenset(
    {
        "SAMPLE_NUMBER",
        "SAMPLING_TIMESTAMP",
        "RECEPTION_TIME",
    }
)

# Default units: NextSense EEG exports are microvolt-scale; ACC/GYRO unknown.
_EEG_HINTS = ("ELW", "ELC", "ERW", "ERC", "EEG")
_ACC_HINTS = ("ACC_", "ACC")
_GYRO_HINTS = ("GYRO_", "GYRO")


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("-", "_")


def _unit_for_channel(name: str) -> Unit:
    upper = name.upper()
    if any(hint in upper for hint in _EEG_HINTS):
        return UV
    if any(upper.startswith(hint) or hint in upper for hint in _ACC_HINTS):
        return UNKNOWN
    if any(upper.startswith(hint) or hint in upper for hint in _GYRO_HINTS):
        return UNKNOWN
    return UNKNOWN


def _parse_header(path: Path) -> tuple[dict[str, str], list[str], int]:
    """Return metadata, column names, and the 0-based data start line index."""
    metadata: dict[str, str] = {}
    columns: list[str] | None = None
    data_start: int | None = None

    with path.open(newline="", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            if _START_DATA.match(line):
                data_start = index + 1
                break
            meta = _META_LINE.match(line)
            if meta is not None:
                metadata[meta.group(1).strip()] = meta.group(2).strip()
                continue
            cols = _HEADER_COLS.match(line)
            if cols is not None:
                columns = [c.strip() for c in cols.group(1).split(",")]
                continue

    if columns is None:
        raise ValueError(f"NextSense CSV missing column header in {path}")
    if data_start is None:
        raise ValueError(f"NextSense CSV missing Start Data marker in {path}")
    return metadata, columns, data_start


def _signal_columns(columns: Sequence[str]) -> list[str]:
    return [
        c for c in columns if c.upper() not in _META_COLUMNS and not c.startswith("#")
    ]


def _resolve_channel_indices(
    columns: Sequence[str],
    channels: Sequence[str] | None,
) -> tuple[list[str], list[int]]:
    signal_cols = _signal_columns(columns)
    if channels is None:
        names = [_normalize_channel_name(c) for c in signal_cols]
        indices = [columns.index(c) for c in signal_cols]
        return names, indices

    by_norm = {_normalize_channel_name(c): (c, columns.index(c)) for c in signal_cols}
    names: list[str] = []
    indices: list[int] = []
    missing: list[str] = []
    for requested in channels:
        key = _normalize_channel_name(requested)
        if key not in by_norm:
            missing.append(requested)
            continue
        original, index = by_norm[key]
        names.append(_normalize_channel_name(original))
        indices.append(index)
    if missing:
        raise ValueError(
            f"Requested channel(s) not found: {missing}. "
            f"Available: {[_normalize_channel_name(c) for c in signal_cols]}"
        )
    return names, indices


def _load_array(
    path: Path,
    *,
    data_start: int,
    columns: Sequence[str],
    channel_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load values and sampling timestamps (ms) from the data section."""
    usecols = list(channel_indices)
    ts_index: int | None = None
    for index, name in enumerate(columns):
        if name.upper() == "SAMPLING_TIMESTAMP":
            ts_index = index
            break
    if ts_index is not None:
        usecols = [*channel_indices, ts_index]

    raw = np.loadtxt(
        path,
        delimiter=",",
        skiprows=data_start,
        usecols=usecols,
        dtype=np.float64,
        ndmin=2,
    )
    if raw.size == 0:
        values = np.zeros((0, len(channel_indices)), dtype=np.float64)
        timestamps_ms = np.zeros(0, dtype=np.float64)
        return values, timestamps_ms

    if ts_index is None:
        values = raw if raw.ndim == 2 else raw.reshape(-1, 1)
        timestamps_ms = np.arange(values.shape[0], dtype=np.float64)
        return values, timestamps_ms

    values = raw[:, : len(channel_indices)]
    timestamps_ms = raw[:, -1]
    return values, timestamps_ms


def _timestamps_ns(
    timestamps_ms: np.ndarray,
    *,
    sample_rate: float,
    n_samples: int,
) -> np.ndarray:
    if n_samples == 0:
        return np.zeros(0, dtype=np.int64)
    # Prefer device sampling timestamps when they look like absolute ms.
    if timestamps_ms.size == n_samples and np.any(timestamps_ms > 0):
        # Treat as milliseconds; if values are already large epoch-like, still ms→ns.
        return np.round(timestamps_ms * 1e6).astype(np.int64)
    step = int(round(1e9 / sample_rate))
    return np.arange(n_samples, dtype=np.int64) * step


def read(
    path: Path | str | Sequence[Path | str],
    *,
    channels: Sequence[str] | None = None,
) -> TimeSeries:
    """Load a NextSense CSV export (single file or left+right pair) into a TimeSeries.

    Args:
        path: One CSV path, or a sequence of two paths (left/right). When two
            files are given they must share the same sample grid; non-overlapping
            signal columns are concatenated. Duplicate channel names from the
            second file get a ``_R`` suffix when needed.
        channels: Optional subset of signal channel names to load.

    Returns:
        Regularly sampled :class:`~somnio.data.timeseries.TimeSeries` with
        EEG channels tagged as microvolts (``uV``).
    """
    if isinstance(path, (str, Path)):
        paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]
    if not paths:
        raise ValueError("path must contain at least one NextSense CSV")
    for file_path in paths:
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

    series_list: list[TimeSeries] = []
    for file_path in paths:
        series_list.append(_read_one(file_path, channels=channels))

    if len(series_list) == 1:
        return series_list[0]
    return _merge_pair(series_list[0], series_list[1])


def _read_one(
    path: Path,
    *,
    channels: Sequence[str] | None,
) -> TimeSeries:
    metadata, columns, data_start = _parse_header(path)
    names, indices = _resolve_channel_indices(columns, channels)

    rate_raw = metadata.get("eegStreamingRate") or metadata.get("eegstreamingrate")
    if rate_raw is None:
        raise ValueError(f"NextSense CSV missing eegStreamingRate in {path}")
    sample_rate = float(rate_raw)
    if sample_rate <= 0:
        raise ValueError(f"Invalid eegStreamingRate={sample_rate} in {path}")

    values, timestamps_ms = _load_array(
        path,
        data_start=data_start,
        columns=columns,
        channel_indices=indices,
    )
    timestamps = _timestamps_ns(
        timestamps_ms,
        sample_rate=sample_rate,
        n_samples=values.shape[0],
    )
    units = tuple(_unit_for_channel(name) for name in names)
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=tuple(names),
        units=units,
        sample_rate=sample_rate,
    )


def _merge_pair(left: TimeSeries, right: TimeSeries) -> TimeSeries:
    if left.sample_rate != right.sample_rate:
        raise ValueError(
            f"NextSense pair sample_rate mismatch: {left.sample_rate} vs {right.sample_rate}"
        )
    n = min(left.n_samples, right.n_samples)
    if n == 0:
        raise ValueError("NextSense pair has no overlapping samples")

    left_vals = left.values[:n].copy()
    right_vals = right.values[:n]
    timestamps = left.timestamps[:n]

    names = list(left.channel_names)
    units = list(left.units)
    extra_cols: list[np.ndarray] = []
    used = set(names)

    for index, name in enumerate(right.channel_names):
        col = right_vals[:, index]
        if name in used:
            left_index = names.index(name)
            left_empty = np.allclose(left_vals[:, left_index], 0.0)
            right_empty = np.allclose(col, 0.0)
            if left_empty and not right_empty:
                left_vals[:, left_index] = col
            elif not left_empty and not right_empty:
                suffix = f"{name}_R"
                while suffix in used:
                    suffix = f"{suffix}_R"
                names.append(suffix)
                units.append(right.units[index])
                extra_cols.append(col)
                used.add(suffix)
            continue
        names.append(name)
        units.append(right.units[index])
        extra_cols.append(col)
        used.add(name)

    values = left_vals if not extra_cols else np.column_stack([left_vals, *extra_cols])
    return TimeSeries(
        values=values,
        timestamps=timestamps.copy(),
        channel_names=tuple(names),
        units=tuple(units),
        sample_rate=left.sample_rate,
    )


class NextSenseCSVReader:
    """Stateless reader for NextSense CSV exports."""

    def read(
        self, path: Path | str | Sequence[Path | str], **kwargs: Any
    ) -> TimeSeries:
        """Read NextSense CSV; supports ``channels`` kwarg."""
        return read(path, **kwargs)


def read_metadata(path: Path | str) -> Mapping[str, str]:
    """Return NextSense metadata key/value pairs without loading signal data."""
    metadata, _columns, _start = _parse_header(Path(path))
    return metadata


def list_channels(path: Path | str) -> tuple[str, ...]:
    """Return normalized signal channel names from a NextSense CSV header."""
    _metadata, columns, _start = _parse_header(Path(path))
    return tuple(_normalize_channel_name(c) for c in _signal_columns(columns))
