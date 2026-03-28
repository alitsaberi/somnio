"""Internal helpers for ``somnio.io.edf``: export dependency checks and EDF header rules."""

from __future__ import annotations

from datetime import datetime, timezone

from somnio.data.timeseries import TimeSeries

from somnio.data.adapters.mne import import_mne


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


def ensure_export_edf_deps() -> None:
    """Raise if EDF export dependencies are missing."""
    import_mne()
    try:
        import edfio  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Exporting EDF requires edfio. Install with: pip install somnio[edf]"
        ) from exc
