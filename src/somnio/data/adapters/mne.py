"""MNE ↔ :class:`~somnio.data.timeseries.TimeSeries`."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.utils.imports import MissingOptionalDependency

try:
    import mne
except ModuleNotFoundError as e:
    if e.name != "mne":
        raise
    raise MissingOptionalDependency("mne", extra="edf", purpose="MNE adapter") from e


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def _units_from_info(info: object) -> tuple[str, ...]:
    FIFF = mne.io.constants.FIFF

    mapping = {
        int(FIFF.FIFF_UNIT_V): "V",
        int(FIFF.FIFF_UNIT_T): "T",
        int(FIFF.FIFF_UNIT_M): "m",
        int(FIFF.FIFF_UNIT_J): "J",
        int(FIFF.FIFF_UNIT_SEC): "s",
    }
    chs = info["chs"]
    out: list[str] = []
    for ch in chs:
        u = int(ch["unit"])
        out.append(mapping.get(u, "V"))
    return tuple(out)


def _ch_types_for_units(units: tuple[str, ...]) -> list[str]:
    types: list[str] = []
    for u in units:
        uu = u.strip()
        if uu == "V":
            types.append("eeg")
        elif uu in {"m/s^2", "m/s²"}:
            types.append("misc")
        elif uu == "degC":
            types.append("misc")
        else:
            types.append("misc")
    return types


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
    """Build an MNE :class:`~mne.io.RawArray` from ``data``.

    Requires ``data.sample_rate``. Sets ``meas_date`` from the first sample's
    Unix timestamp (any year MNE accepts).
    """
    if data.sample_rate is None:
        raise ValueError("TimeSeries.sample_rate must be set for MNE Raw")
    if data.n_samples == 0:
        raise ValueError("Cannot build MNE Raw from an empty TimeSeries")

    sfreq = float(data.sample_rate)
    ch_types = _ch_types_for_units(data.units)
    info = mne.create_info(
        list(data.channel_names),
        sfreq,
        ch_types=ch_types,
    )
    arr = np.asarray(data.values.T, dtype=np.float64)
    raw = mne.io.RawArray(arr, info)
    start_ns = int(data.timestamps[0])
    meas_date = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    raw.set_meas_date(meas_date)
    return raw
