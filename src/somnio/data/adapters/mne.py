"""MNE ↔ :class:`~somnio.data.timeseries.TimeSeries`."""

from __future__ import annotations

from datetime import timezone
from typing import Final

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.data.units import DEG_C, UNKNOWN, V, Unit
from somnio.utils.imports import MissingOptionalDependency

try:
    import mne
    from mne.io.constants import FIFF
except ModuleNotFoundError as e:
    if e.name != "mne":
        raise
    raise MissingOptionalDependency("mne", extra="edf", purpose="MNE adapter") from e


_FIFF_TO_UNIT: Final[dict[int, Unit]] = {
    int(FIFF.FIFF_UNIT_V): V,
    int(FIFF.FIFF_UNIT_CEL): DEG_C,
    # TODO: Expand somnio.data.units to support more FIFF units, then map them
    #       here (e.g., seconds, tesla, meters, joules). For now, fall back.
}


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def _units_from_info(info: object) -> tuple[Unit, ...]:
    chs = info["chs"]
    out: list[Unit] = []
    for ch in chs:
        u = int(ch["unit"])
        out.append(_FIFF_TO_UNIT.get(u, UNKNOWN))
    return tuple(out)


def from_mne_raw(raw: mne.io.BaseRaw) -> TimeSeries:
    """Convert a preloaded MNE :class:`~mne.io.BaseRaw` to :class:`~somnio.data.timeseries.TimeSeries`.

    Values are MNE's SI representation (e.g. Volts for EEG). Timestamps use
    ``meas_date`` when present; otherwise nanoseconds are relative with the first
    sample at 0 ns.
    """
    raw = raw.copy()
    if not raw.preload:
        raw.load_data()

    data = raw.get_data()
    values = np.asarray(data.T, dtype=np.float64)
    n_samples = values.shape[0]
    times_s = np.asarray(raw.times, dtype=np.float64)
    if times_s.shape != (n_samples,):
        raise ValueError(
            f"MNE raw.times shape {times_s.shape} != ({n_samples},) from get_data()"
        )

    meas_date = raw.info.get("meas_date")
    if meas_date is not None:
        if meas_date.tzinfo is None:
            meas_date = meas_date.replace(tzinfo=timezone.utc)
        start_ns = int(meas_date.timestamp() * 1e9)
    else:
        start_ns = 0

    abs_times_s = float(raw.first_time) + times_s
    timestamps = start_ns + np.round(abs_times_s * 1e9).astype(np.int64)

    ch_names = tuple(_normalize_channel_name(ch["ch_name"]) for ch in raw.info["chs"])
    units = _units_from_info(raw.info)
    sfreq = float(raw.info["sfreq"])
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=ch_names,
        units=units,
        sample_rate=sfreq,
    )


def to_mne_raw(data: TimeSeries) -> mne.io.BaseRaw:
    """Build an MNE :class:`~mne.io.RawArray` from ``data``."""
    # TODO: [EDF] Implement this.
    raise NotImplementedError(
        "EDF export relies on correct MNE channel typing and unit metadata, "
        "which is pending."
    )
