"""Internal helpers for ``somnio.io.edf``: export dependency checks and EDF header rules."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


from somnio.data.adapters.mne import to_mne_raw
from somnio.data.timeseries import TimeSeries
from somnio.utils.imports import MissingOptionalDependency


def require_edf_compatible_timestamps(data: TimeSeries) -> None:
    """Raise if the first sample time cannot be stored in an EDF header (edfio).

    EDF allows recording dates only between 1985 and 2084.
    """
    start_ns = int(data.timestamps[0])
    meas_date = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    if not (1985 <= meas_date.year <= 2084):
        raise ValueError(
            "The first sample's timestamp maps to a calendar year outside 1985–2084, "
            "which EDF headers cannot represent (see edfio). Shift "
            "TimeSeries.timestamps so the recording start falls in that range, or "
            "use another format (e.g. HDF5)."
        )


def write_edf(
    path: Path,
    data: TimeSeries,
    *,
    overwrite: bool = False,
    verbose: str | bool | None = None,
) -> None:
    """
    Write a TimeSeries to an EDF file using MNE's exporter (``edfio`` backend).

    Requires ``data.sample_rate``. Measurement date is taken from the first sample
    timestamp.
    """
    try:
        import edfio  # noqa: F401
        from mne.export import export_raw
    except ModuleNotFoundError as e:
        if e.name not in ("edfio", "mne"):
            raise
        raise MissingOptionalDependency(
            e.name, extra="edf", purpose="EDF export"
        ) from e

    require_edf_compatible_timestamps(data)
    raw = to_mne_raw(data)
    export_raw(path, raw, overwrite=overwrite, verbose=verbose)
