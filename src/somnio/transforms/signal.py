"""Signal processing transforms for TimeSeries data.

Requires the ``signal`` extra (SciPy)::

    pip install 'somnio[signal]'
"""

from __future__ import annotations

import logging

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.pipeline.types import Bundle
from somnio.utils.imports import MissingOptionalDependency

try:
    from scipy.signal import filtfilt, firwin
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "scipy", extra="signal", purpose="Signal processing transforms"
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fir_numtaps(sample_rate: float, n_samples: int) -> int:
    """Return an odd numtaps feasible for the given signal length.

    Targets ``numtaps ≈ sample_rate`` (≈ 1-second filter) and reduces in steps
    of 2 until ``scipy.signal.filtfilt``'s padding requirement
    ``n_samples >= 3 * (numtaps - 1) + 1`` is satisfied.
    """
    candidate = int(sample_rate) | 1  # nearest odd integer >= sample_rate
    while candidate > 3 and n_samples < 3 * (candidate - 1) + 1:
        candidate -= 2
    return max(candidate, 3)


def _design_fir_coeffs(
    sample_rate: float,
    n_samples: int,
    low_cutoff: float | None,
    high_cutoff: float | None,
) -> tuple[str, np.ndarray]:
    """Design FIR filter coefficients from the given cutoff combination.

    Filter type is inferred from which cutoffs are provided:

    * ``low_cutoff`` only → **highpass**
    * ``high_cutoff`` only → **lowpass**
    * ``low_cutoff < high_cutoff`` → **bandpass**
    * ``low_cutoff > high_cutoff`` → **bandstop** (stop between
      ``high_cutoff`` and ``low_cutoff``)

    Args:
        sample_rate: Nominal sample rate in Hz.
        n_samples: Signal length (used to cap ``numtaps`` for short signals).
        low_cutoff: Lower edge in Hz, or ``None``.
        high_cutoff: Upper edge in Hz, or ``None``.

    Returns:
        Tuple of ``(filter_type, coeffs)`` where ``filter_type`` is one of
        ``"highpass"``, ``"lowpass"``, ``"bandpass"``, or ``"bandstop"``, and
        ``coeffs`` is a 1-D FIR coefficient array for ``scipy.signal.filtfilt``.

    Raises:
        ValueError: If both cutoffs are ``None``, or if they are equal.
    """
    if low_cutoff is None and high_cutoff is None:
        raise ValueError("At least one of low_cutoff or high_cutoff must be provided.")

    nyq = sample_rate / 2.0
    numtaps = _fir_numtaps(sample_rate, n_samples)

    if low_cutoff is not None and high_cutoff is not None:
        if low_cutoff == high_cutoff:
            raise ValueError(
                f"low_cutoff and high_cutoff must differ; both are {low_cutoff} Hz."
            )
        if low_cutoff < high_cutoff:
            return "bandpass", firwin(
                numtaps, [low_cutoff / nyq, high_cutoff / nyq], pass_zero=False
            )
        else:
            # firwin expects cutoffs in ascending order
            return "bandstop", firwin(
                numtaps, [high_cutoff / nyq, low_cutoff / nyq], pass_zero=True
            )

    if low_cutoff is not None:
        return "highpass", firwin(numtaps, low_cutoff / nyq, pass_zero=False)

    return "lowpass", firwin(numtaps, high_cutoff / nyq, pass_zero=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_fir_filter(
    ts: TimeSeries,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
) -> TimeSeries:
    """Apply a zero-phase FIR filter to all channels of a TimeSeries at once.

    The filter type is determined by which cutoffs are provided:

    * Both ``None`` — the original TimeSeries is returned unchanged.
    * ``low_cutoff`` only — **highpass** filter.
    * ``high_cutoff`` only — **lowpass** filter.
    * ``low_cutoff < high_cutoff`` — **bandpass** filter.
    * ``low_cutoff > high_cutoff`` — **bandstop** filter (stop band between
      ``high_cutoff`` and ``low_cutoff``).

    All channels are filtered simultaneously (``filtfilt`` applied to the 2-D
    values array along the time axis).  The returned TimeSeries has the same
    shape, timestamps, channel names, units, and sample rate as the input;
    only the values differ.

    Args:
        ts: Input time-series. ``ts.sample_rate`` must not be ``None``.
        low_cutoff: Lower cutoff frequency in Hz, or ``None``.
        high_cutoff: Upper cutoff frequency in Hz, or ``None``.

    Returns:
        New :class:`~somnio.data.timeseries.TimeSeries` with filtered values,
        or the original ``ts`` if both cutoffs are ``None``.

    Raises:
        ValueError: If ``ts.sample_rate`` is ``None``, or if both cutoffs are
            equal.
    """
    if low_cutoff is None and high_cutoff is None:
        return ts

    if ts.sample_rate is None:
        raise ValueError(
            "TimeSeries.sample_rate must not be None; "
            "FIR filtering requires a known sample rate."
        )

    filter_type, coeffs = _design_fir_coeffs(
        ts.sample_rate, ts.n_samples, low_cutoff, high_cutoff
    )
    filtered_values = filtfilt(coeffs, 1.0, ts.values, axis=0)

    logger.debug(
        "apply_fir_filter: %s [low=%s Hz, high=%s Hz] applied to %d channel(s)",
        filter_type,
        low_cutoff,
        high_cutoff,
        ts.n_channels,
    )

    return TimeSeries(
        values=np.asarray(filtered_values),
        timestamps=ts.timestamps.copy(),
        channel_names=list(ts.channel_names),
        units=list(ts.units),
        sample_rate=ts.sample_rate,
    )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def fir_filter(
    bundle: Bundle,
    /,
    *,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
) -> Bundle:
    """Bundle → Bundle transform that applies :func:`apply_fir_filter` to every TimeSeries.

    Every value in the bundle must be a
    :class:`~somnio.data.timeseries.TimeSeries`; mixed bundles containing
    ``list[Event]`` or ``Epochs`` will raise a ``TypeError``.

    Args:
        bundle: Input bundle. All values must be TimeSeries instances.
        low_cutoff: Lower cutoff frequency in Hz, or ``None``.
        high_cutoff: Upper cutoff frequency in Hz, or ``None``.

    Returns:
        New bundle where every TimeSeries has been filtered.

    Raises:
        TypeError: If any bundle value is not a TimeSeries.
    """
    result: Bundle = {}
    for key, value in bundle.items():
        if not isinstance(value, TimeSeries):
            raise TypeError(
                f"Bundle key {key!r} is {type(value).__name__!r}, expected TimeSeries."
            )
        result[key] = apply_fir_filter(value, low_cutoff, high_cutoff)
    return result
