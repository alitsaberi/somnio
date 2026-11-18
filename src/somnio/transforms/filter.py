"""Signal filtering transforms for TimeSeries data.

Requires the ``mne`` extra::

    pip install 'somnio[mne]'
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.utils.imports import MissingOptionalDependency

try:
    from mne.filter import filter_data
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "mne", extra="mne", purpose="Signal filtering transforms"
    ) from exc

logger = logging.getLogger(__name__)

FilterMethod = Literal["fir", "iir"]
FilterType = Literal["highpass", "lowpass", "bandpass", "bandstop"]


def _classify_filter_type(
    low_cutoff: float | None,
    high_cutoff: float | None,
) -> FilterType:
    """Infer filter type from cutoff combination.

    * ``low_cutoff`` only → **highpass**
    * ``high_cutoff`` only → **lowpass**
    * ``low_cutoff < high_cutoff`` → **bandpass**
    * ``low_cutoff > high_cutoff`` → **bandstop** (stop between
      ``high_cutoff`` and ``low_cutoff``)
    """
    if low_cutoff is None and high_cutoff is None:
        raise ValueError("At least one of low_cutoff or high_cutoff must be provided.")

    if low_cutoff is not None and high_cutoff is not None:
        if low_cutoff == high_cutoff:
            raise ValueError(
                f"low_cutoff and high_cutoff must differ; both are {low_cutoff} Hz."
            )
        if low_cutoff < high_cutoff:
            return "bandpass"
        return "bandstop"

    if low_cutoff is not None:
        return "highpass"

    return "lowpass"


def _validate_cutoffs(
    sample_rate: float,
    low_cutoff: float | None,
    high_cutoff: float | None,
) -> None:
    """Ensure cutoffs are positive and strictly below Nyquist."""
    nyquist = sample_rate / 2.0
    for name, value in (("low_cutoff", low_cutoff), ("high_cutoff", high_cutoff)):
        if value is None:
            continue
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be a positive finite frequency, got {value}."
            )
        if value >= nyquist:
            raise ValueError(
                f"{name}={value} Hz must be strictly below Nyquist "
                f"({nyquist} Hz at sample_rate={sample_rate} Hz)."
            )


def _auto_l_trans_bandwidth(l_freq: float) -> float:
    """MNE auto low-side transition bandwidth (Hz)."""
    return float(min(max(0.25 * l_freq, 2.0), l_freq))


def _auto_h_trans_bandwidth(h_freq: float, nyquist: float) -> float:
    """MNE auto high-side transition bandwidth (Hz)."""
    return float(min(max(0.25 * h_freq, 2.0), nyquist - h_freq))


def _resolve_trans_bandwidth(
    value: float | str,
    *,
    name: str,
    auto_hz: float,
) -> float:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(
                f"{name} must be 'auto' or a positive float, got {value!r}."
            )
        return auto_hz
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite frequency, got {value}.")
    return float(value)


def _mne_trans_bandwidths(
    sample_rate: float,
    low_cutoff: float | None,
    high_cutoff: float | None,
    l_trans_bandwidth: float | str,
    h_trans_bandwidth: float | str,
) -> tuple[float | str, float | str]:
    """Resolve transition bandwidths for :func:`mne.filter.filter_data`.

    Band-stop with MNE's ``'auto'`` transition widths can fail FIR design for
    common stop bands, so band-stop always gets concrete Hz values.
    """
    filter_type = _classify_filter_type(low_cutoff, high_cutoff)
    nyquist = sample_rate / 2.0

    if filter_type != "bandstop":
        return l_trans_bandwidth, h_trans_bandwidth

    assert low_cutoff is not None and high_cutoff is not None
    # Ascending stop edges are (high_cutoff, low_cutoff) when API has low > high.
    # MNE FIR band-stop design is fragile with unequal auto transition widths
    # (can request a shorter filter than the wider edge needs). Use a common
    # width = min(auto_l, auto_h) unless the caller passed explicit values.
    auto_l = _auto_l_trans_bandwidth(high_cutoff)
    auto_h = _auto_h_trans_bandwidth(low_cutoff, nyquist)
    common_auto = min(auto_l, auto_h)
    if l_trans_bandwidth == "auto" and h_trans_bandwidth == "auto":
        return common_auto, common_auto
    l_trans = _resolve_trans_bandwidth(
        l_trans_bandwidth,
        name="l_trans_bandwidth",
        auto_hz=common_auto,
    )
    h_trans = _resolve_trans_bandwidth(
        h_trans_bandwidth,
        name="h_trans_bandwidth",
        auto_hz=common_auto,
    )
    return l_trans, h_trans


def _require_sample_rate(ts: TimeSeries) -> float:
    if ts.sample_rate is None:
        raise ValueError(
            "TimeSeries.sample_rate must not be None; "
            "filtering requires a known sample rate."
        )
    return float(ts.sample_rate)


def _filtered_timeseries(ts: TimeSeries, filtered_values: np.ndarray) -> TimeSeries:
    return TimeSeries(
        values=np.asarray(filtered_values, dtype=np.float64),
        timestamps=ts.timestamps.copy(),
        channel_names=list(ts.channel_names),
        units=list(ts.units),
        sample_rate=ts.sample_rate,
    )


def apply_filter(
    ts: TimeSeries,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    *,
    method: FilterMethod = "fir",
    numtaps: int | None = None,
    l_trans_bandwidth: float | str = "auto",
    h_trans_bandwidth: float | str = "auto",
    order: int = 4,
) -> TimeSeries:
    """Apply a zero-phase FIR or IIR filter to all channels via MNE.

    Cutoff convention matches :func:`mne.filter.filter_data`:

    * Both ``None`` — the original TimeSeries is returned unchanged.
    * ``low_cutoff`` only — **highpass** filter.
    * ``high_cutoff`` only — **lowpass** filter.
    * ``low_cutoff < high_cutoff`` — **bandpass** filter.
    * ``low_cutoff > high_cutoff`` — **bandstop** filter (stop band between
      ``high_cutoff`` and ``low_cutoff``).

    **FIR** (``method="fir"``, default): MNE overlap-add FIR with Hamming
    ``firwin`` design and zero-phase delay compensation (same defaults as
    offline WU-Sleep / zmax-datasets prep).

    **IIR** (``method="iir"``): MNE forward-backward Butterworth
    (``iir_params`` with ``ftype="butter"``), default ``order=4``.

    Args:
        ts: Input time-series. ``ts.sample_rate`` must not be ``None``.
        low_cutoff: Lower cutoff frequency in Hz, or ``None``.
        high_cutoff: Upper cutoff frequency in Hz, or ``None``.
        method: ``"fir"`` or ``"iir"``.
        numtaps: Explicit FIR length in samples (passed to MNE as
            ``filter_length``). ``None`` uses MNE ``"auto"``. Ignored for IIR.
        l_trans_bandwidth: Low-side FIR transition width in Hz, or ``"auto"``.
        h_trans_bandwidth: High-side FIR transition width in Hz, or ``"auto"``.
        order: IIR (Butterworth) filter order (default 4). Ignored for FIR.

    Returns:
        New :class:`~somnio.data.timeseries.TimeSeries` with filtered values,
        or the original ``ts`` if both cutoffs are ``None``.

    Raises:
        ValueError: If ``ts.sample_rate`` is ``None``, cutoffs are invalid /
            equal / at or above Nyquist, or ``method`` is unknown.
    """
    if low_cutoff is None and high_cutoff is None:
        return ts

    if method not in ("fir", "iir"):
        raise ValueError(f"method must be 'fir' or 'iir', got {method!r}.")

    sample_rate = _require_sample_rate(ts)
    _validate_cutoffs(sample_rate, low_cutoff, high_cutoff)
    filter_type = _classify_filter_type(low_cutoff, high_cutoff)

    l_trans, h_trans = _mne_trans_bandwidths(
        sample_rate,
        low_cutoff,
        high_cutoff,
        l_trans_bandwidth,
        h_trans_bandwidth,
    )

    # MNE expects shape (n_channels, n_times)
    data = np.asarray(ts.values, dtype=np.float64).T
    filter_kwargs: dict = {
        "l_freq": low_cutoff,
        "h_freq": high_cutoff,
        "method": method,
        "l_trans_bandwidth": l_trans,
        "h_trans_bandwidth": h_trans,
        "copy": True,
        "verbose": False,
    }

    if method == "fir":
        if numtaps is not None:
            if numtaps < 3:
                raise ValueError(f"numtaps must be >= 3, got {numtaps}.")
            filter_kwargs["filter_length"] = int(numtaps)
        filtered = filter_data(data, sample_rate, **filter_kwargs)
        logger.info(
            "apply_filter(fir/mne): %s [low=%s Hz, high=%s Hz] "
            "filter_length=%s trans=(%s, %s) on %d channel(s)",
            filter_type,
            low_cutoff,
            high_cutoff,
            filter_kwargs.get("filter_length", "auto"),
            l_trans,
            h_trans,
            ts.n_channels,
        )
    else:
        if numtaps is not None:
            logger.debug("apply_filter: numtaps=%s ignored for method='iir'", numtaps)
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}.")
        filter_kwargs["iir_params"] = {"order": int(order), "ftype": "butter"}
        filtered = filter_data(data, sample_rate, **filter_kwargs)
        logger.info(
            "apply_filter(iir/mne): %s [low=%s Hz, high=%s Hz] order=%d on %d channel(s)",
            filter_type,
            low_cutoff,
            high_cutoff,
            order,
            ts.n_channels,
        )

    return _filtered_timeseries(ts, filtered.T)


def apply_fir_filter(
    ts: TimeSeries,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    *,
    numtaps: int | None = None,
    l_trans_bandwidth: float | str = "auto",
    h_trans_bandwidth: float | str = "auto",
) -> TimeSeries:
    """Apply a zero-phase FIR filter (alias for :func:`apply_filter`).

    Kept for backward compatibility. New code should call
    :func:`apply_filter` directly.
    """
    return apply_filter(
        ts,
        low_cutoff,
        high_cutoff,
        method="fir",
        numtaps=numtaps,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
    )


def apply_butterworth_filter(
    ts: TimeSeries,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    *,
    order: int = 4,
) -> TimeSeries:
    """Apply a zero-phase Butterworth IIR filter (alias for :func:`apply_filter`).

    Default ``order=4``. Prefer this (or ``method="iir"``) for cutoffs ≲ 1 Hz
    when the recording may be short relative to a long FIR.
    """
    return apply_filter(
        ts,
        low_cutoff,
        high_cutoff,
        method="iir",
        order=order,
    )
