"""Signal filtering transforms for TimeSeries data.

Requires the ``signal`` extra (SciPy)::

    pip install 'somnio[signal]'
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.utils.imports import MissingOptionalDependency

try:
    from scipy.signal import butter, filtfilt, firwin, sosfiltfilt
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "scipy", extra="signal", purpose="Signal filtering transforms"
    ) from exc

logger = logging.getLogger(__name__)

# Hamming + firwin: filter length ≈ 3.3 / (trans_bandwidth / fs) samples (MNE).
_HAMMING_FIRWIN_FACTOR = 3.3

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
    """MNE-like auto low-side transition bandwidth (Hz)."""
    # min(max(0.25 * l_freq, 2.0), l_freq) keeps trans ≤ cutoff for very low
    # l_freq (e.g. 0.3 Hz → 0.3 Hz) so the stopband remains reachable.
    return float(min(max(0.25 * l_freq, 2.0), l_freq))


def _auto_h_trans_bandwidth(h_freq: float, nyquist: float) -> float:
    """MNE-like auto high-side transition bandwidth (Hz)."""
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


def _transition_bandwidths(
    sample_rate: float,
    low_cutoff: float | None,
    high_cutoff: float | None,
    filter_type: FilterType,
    l_trans_bandwidth: float | str,
    h_trans_bandwidth: float | str,
) -> tuple[float | None, float | None, float]:
    """Return ``(l_trans, h_trans, shortest_trans)`` in Hz for FIR design."""
    nyquist = sample_rate / 2.0
    l_trans: float | None = None
    h_trans: float | None = None

    # Auto rules key off the relevant edge frequency. For bandstop our API
    # uses low_cutoff > high_cutoff; the ascending stop edges are
    # (high_cutoff, low_cutoff).
    if filter_type == "highpass":
        assert low_cutoff is not None
        l_trans = _resolve_trans_bandwidth(
            l_trans_bandwidth,
            name="l_trans_bandwidth",
            auto_hz=_auto_l_trans_bandwidth(low_cutoff),
        )
    elif filter_type == "lowpass":
        assert high_cutoff is not None
        h_trans = _resolve_trans_bandwidth(
            h_trans_bandwidth,
            name="h_trans_bandwidth",
            auto_hz=_auto_h_trans_bandwidth(high_cutoff, nyquist),
        )
    elif filter_type == "bandpass":
        assert low_cutoff is not None and high_cutoff is not None
        l_trans = _resolve_trans_bandwidth(
            l_trans_bandwidth,
            name="l_trans_bandwidth",
            auto_hz=_auto_l_trans_bandwidth(low_cutoff),
        )
        h_trans = _resolve_trans_bandwidth(
            h_trans_bandwidth,
            name="h_trans_bandwidth",
            auto_hz=_auto_h_trans_bandwidth(high_cutoff, nyquist),
        )
    else:
        assert low_cutoff is not None and high_cutoff is not None
        l_trans = _resolve_trans_bandwidth(
            l_trans_bandwidth,
            name="l_trans_bandwidth",
            auto_hz=_auto_l_trans_bandwidth(high_cutoff),
        )
        h_trans = _resolve_trans_bandwidth(
            h_trans_bandwidth,
            name="h_trans_bandwidth",
            auto_hz=_auto_h_trans_bandwidth(low_cutoff, nyquist),
        )

    widths = [w for w in (l_trans, h_trans) if w is not None]
    if not widths:
        raise ValueError("Internal error: no transition bandwidths resolved.")
    return l_trans, h_trans, min(widths)


def _max_fir_numtaps(n_samples: int) -> int:
    """Largest odd FIR length that satisfies ``filtfilt`` padding constraints.

    ``scipy.signal.filtfilt`` requires ``n_samples >= 3 * (numtaps - 1) + 1``
    for the default padlen.
    """
    # n >= 3*(N-1)+1  ⇒  N <= (n-1)//3 + 1
    max_n = (n_samples - 1) // 3 + 1
    if max_n % 2 == 0:
        max_n -= 1
    return max(max_n, 3)


def _fir_numtaps_from_trans(
    sample_rate: float,
    n_samples: int,
    trans_bandwidth: float,
    numtaps: int | None,
) -> int:
    """Choose odd FIR length from transition bandwidth (or an explicit override)."""
    max_n = _max_fir_numtaps(n_samples)
    if max_n < 3:
        raise ValueError(
            f"Signal is too short for FIR filtering ({n_samples} samples); "
            "need enough samples for filtfilt padding with at least 3 taps."
        )

    if numtaps is not None:
        if numtaps < 3:
            raise ValueError(f"numtaps must be >= 3, got {numtaps}.")
        n = int(numtaps) | 1
        if n > max_n:
            raise ValueError(
                f"Requested numtaps={numtaps} (using odd length {n}) exceeds the "
                f"maximum {max_n} feasible for a signal of length {n_samples} "
                f"(filtfilt requires n_samples >= 3*(numtaps-1)+1)."
            )
        return n

    # numtaps ≈ factor * fs / trans_bandwidth (Hamming + firwin).
    needed = int(np.ceil(_HAMMING_FIRWIN_FACTOR * sample_rate / trans_bandwidth))
    needed |= 1  # odd length
    needed = max(needed, 3)

    if needed > max_n:
        min_samples = 3 * (needed - 1) + 1
        min_seconds = min_samples / sample_rate
        raise ValueError(
            f"Signal is too short for the requested FIR filter: need about "
            f"{needed} taps (≈{_HAMMING_FIRWIN_FACTOR:.1f}*fs/trans_bandwidth "
            f"with trans_bandwidth={trans_bandwidth:.4g} Hz), but only {max_n} "
            f"taps fit in {n_samples} samples at {sample_rate:g} Hz. "
            f"Provide at least ~{min_samples} samples (≈{min_seconds:.1f} s), "
            f"widen the transition bandwidth, pass an explicit shorter numtaps, "
            f'or use method="iir".'
        )
    return needed


def _design_fir_coeffs(
    sample_rate: float,
    n_samples: int,
    low_cutoff: float | None,
    high_cutoff: float | None,
    *,
    numtaps: int | None = None,
    l_trans_bandwidth: float | str = "auto",
    h_trans_bandwidth: float | str = "auto",
) -> tuple[FilterType, np.ndarray, int, float, float | None, float | None]:
    """Design FIR coefficients; return type, coeffs, numtaps, and bandwidths."""
    filter_type = _classify_filter_type(low_cutoff, high_cutoff)
    l_trans, h_trans, shortest = _transition_bandwidths(
        sample_rate,
        low_cutoff,
        high_cutoff,
        filter_type,
        l_trans_bandwidth,
        h_trans_bandwidth,
    )
    n_taps = _fir_numtaps_from_trans(sample_rate, n_samples, shortest, numtaps)
    nyq = sample_rate / 2.0

    if filter_type == "bandpass":
        assert low_cutoff is not None and high_cutoff is not None
        coeffs = firwin(n_taps, [low_cutoff / nyq, high_cutoff / nyq], pass_zero=False)
    elif filter_type == "bandstop":
        assert low_cutoff is not None and high_cutoff is not None
        coeffs = firwin(n_taps, [high_cutoff / nyq, low_cutoff / nyq], pass_zero=True)
    elif filter_type == "highpass":
        assert low_cutoff is not None
        coeffs = firwin(n_taps, low_cutoff / nyq, pass_zero=False)
    else:
        assert high_cutoff is not None
        coeffs = firwin(n_taps, high_cutoff / nyq, pass_zero=True)

    return filter_type, coeffs, n_taps, shortest, l_trans, h_trans


def _design_butterworth_sos(
    sample_rate: float,
    low_cutoff: float | None,
    high_cutoff: float | None,
    order: int,
) -> tuple[FilterType, np.ndarray]:
    """Design a Butterworth SOS matrix for zero-phase ``sosfiltfilt``."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}.")

    filter_type = _classify_filter_type(low_cutoff, high_cutoff)
    nyq = sample_rate / 2.0

    if filter_type == "bandpass":
        assert low_cutoff is not None and high_cutoff is not None
        wn = [low_cutoff / nyq, high_cutoff / nyq]
        btype = "bandpass"
    elif filter_type == "bandstop":
        assert low_cutoff is not None and high_cutoff is not None
        wn = [high_cutoff / nyq, low_cutoff / nyq]
        btype = "bandstop"
    elif filter_type == "highpass":
        assert low_cutoff is not None
        wn = low_cutoff / nyq
        btype = "highpass"
    else:
        assert high_cutoff is not None
        wn = high_cutoff / nyq
        btype = "lowpass"

    sos = butter(order, wn, btype=btype, output="sos")
    return filter_type, sos


def _require_sample_rate(ts: TimeSeries) -> float:
    if ts.sample_rate is None:
        raise ValueError(
            "TimeSeries.sample_rate must not be None; "
            "filtering requires a known sample rate."
        )
    return float(ts.sample_rate)


def _filtered_timeseries(ts: TimeSeries, filtered_values: np.ndarray) -> TimeSeries:
    return TimeSeries(
        values=np.asarray(filtered_values),
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
    """Apply a zero-phase FIR or IIR filter to all channels.

    The filter type is determined by which cutoffs are provided:

    * Both ``None`` — the original TimeSeries is returned unchanged.
    * ``low_cutoff`` only — **highpass** filter.
    * ``high_cutoff`` only — **lowpass** filter.
    * ``low_cutoff < high_cutoff`` — **bandpass** filter.
    * ``low_cutoff > high_cutoff`` — **bandstop** filter (stop band between
      ``high_cutoff`` and ``low_cutoff``).

    **FIR** (``method="fir"``, default): length is derived from transition
    bandwidth (MNE-like), not a fixed ~1 s window. Auto transition widths::

        l_trans = min(max(0.25 * l_freq, 2.0), l_freq)
        h_trans = min(max(0.25 * h_freq, 2.0), nyquist - h_freq)

    and ``numtaps ≈ 3.3 * sample_rate / min(l_trans, h_trans)`` (odd, Hamming /
    ``firwin``). For cutoffs ≲ 1 Hz this yields multi-second filters; if the
    signal is too short to fit that length under ``filtfilt`` padding rules, a
    :class:`ValueError` is raised. Prefer ``method="iir"`` when recordings are
    short or cutoffs are very low.

    **IIR** (``method="iir"``): zero-phase Butterworth via SciPy ``butter`` +
    ``sosfiltfilt``, default ``order=4``.

    Args:
        ts: Input time-series. ``ts.sample_rate`` must not be ``None``.
        low_cutoff: Lower cutoff frequency in Hz, or ``None``.
        high_cutoff: Upper cutoff frequency in Hz, or ``None``.
        method: ``"fir"`` or ``"iir"``.
        numtaps: Explicit odd FIR length; ``None`` selects from transition
            bandwidth. Ignored for IIR.
        l_trans_bandwidth: Low-side FIR transition width in Hz, or ``"auto"``.
        h_trans_bandwidth: High-side FIR transition width in Hz, or ``"auto"``.
        order: IIR (Butterworth) filter order (default 4). Ignored for FIR.

    Returns:
        New :class:`~somnio.data.timeseries.TimeSeries` with filtered values,
        or the original ``ts`` if both cutoffs are ``None``.

    Raises:
        ValueError: If ``ts.sample_rate`` is ``None``, cutoffs are invalid /
            equal / at or above Nyquist, ``method`` is unknown, or the signal
            is too short for the designed FIR length.
    """
    if low_cutoff is None and high_cutoff is None:
        return ts

    sample_rate = _require_sample_rate(ts)
    _validate_cutoffs(sample_rate, low_cutoff, high_cutoff)

    if method == "fir":
        filter_type, coeffs, n_taps, shortest, l_trans, h_trans = _design_fir_coeffs(
            sample_rate,
            ts.n_samples,
            low_cutoff,
            high_cutoff,
            numtaps=numtaps,
            l_trans_bandwidth=l_trans_bandwidth,
            h_trans_bandwidth=h_trans_bandwidth,
        )
        filtered_values = filtfilt(coeffs, 1.0, ts.values, axis=0)
        logger.info(
            "apply_filter(fir): %s [low=%s Hz, high=%s Hz] numtaps=%d "
            "trans_bandwidth=%.4g Hz (l=%s, h=%s) on %d channel(s)",
            filter_type,
            low_cutoff,
            high_cutoff,
            n_taps,
            shortest,
            l_trans,
            h_trans,
            ts.n_channels,
        )
    elif method == "iir":
        if numtaps is not None:
            logger.debug("apply_filter: numtaps=%s ignored for method='iir'", numtaps)
        filter_type, sos = _design_butterworth_sos(
            sample_rate, low_cutoff, high_cutoff, order
        )
        filtered_values = sosfiltfilt(sos, ts.values, axis=0)
        logger.info(
            "apply_filter(iir): %s [low=%s Hz, high=%s Hz] order=%d on %d channel(s)",
            filter_type,
            low_cutoff,
            high_cutoff,
            order,
            ts.n_channels,
        )
    else:
        raise ValueError(f"method must be 'fir' or 'iir', got {method!r}.")

    return _filtered_timeseries(ts, filtered_values)


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
    :func:`apply_filter` directly. FIR length is chosen from transition
    bandwidth; see :func:`apply_filter` for details and the note about low
    cutoffs ≲ 1 Hz.
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
    when the recording may be too short for a long FIR.
    """
    return apply_filter(
        ts,
        low_cutoff,
        high_cutoff,
        method="iir",
        order=order,
    )
