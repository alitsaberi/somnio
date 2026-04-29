"""Primitive left/right eye-movement event detection.

Requires the ``signal`` extra (SciPy)::

    pip install 'somnio[signal]'
"""

from __future__ import annotations

import logging

import numpy as np

from somnio.data.annotations import Event
from somnio.data.timeseries import TimeSeries
from somnio.transforms.filter import apply_fir_filter
from somnio.data.units import UV, convert_values
from somnio.utils.imports import MissingOptionalDependency
from somnio.tasks.eye_movement.defaults import (
    HIGH_CUTOFF_HZ,
    LOW_CUTOFF_HZ,
    MAX_AMPLITUDE_RATIO,
    MAX_EVENT_DURATION_S,
    MAX_EVENT_GAP_S,
    MAX_EVENT_SKEWNESS,
    MAX_PEAK_AMPLITUDE_UV,
    MIN_AMPLITUDE_RATIO,
    MIN_CORRELATION,
    MIN_EVENT_DURATION_S,
    MIN_EVENT_SKEWNESS,
    MIN_PEAK_AMPLITUDE_UV,
    MIN_PEAK_GAP_S,
    RELATIVE_BASELINE,
    RELATIVE_PEAK_PROMINENCE,
)
from somnio.tasks.eye_movement.event import (
    EVENT_TYPE,
    LEFT_LABEL,
    RIGHT_LABEL,
    filter_by_pattern,
    merge_events,
)

try:
    from scipy.signal import find_peaks
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "scipy", extra="signal", purpose="Eye-movement event detection"
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core peak-based event detector (package-internal)
# ---------------------------------------------------------------------------


def _detect_events(
    product_signal: np.ndarray,
    diff_signal: np.ndarray,
    timestamps: np.ndarray,
    sample_rate: float,
    *,
    min_peak_amplitude_uv: float,
    max_peak_amplitude_uv: float,
    min_peak_gap_s: float,
    relative_peak_prominence: float,
    min_event_duration_s: float,
    max_event_duration_s: float,
    min_event_skewness: float,
    max_event_skewness: float,
    relative_baseline: float,
) -> list[Event]:
    """Detect primitive L/R eye-movement events from pre-computed signals.

    Args:
        product_signal: 1-D array — negative product of filtered left and right
            channels.  Peaks here correspond to conjugate eye movements.
        diff_signal: 1-D array — difference (left − right) of filtered channels.
            Sign at each peak determines movement direction.
        timestamps: Unix nanosecond timestamps aligned with ``product_signal``,
            shape ``(n_samples,)``, dtype int64.
        sample_rate: Nominal sample rate in Hz.
        min_peak_amplitude_uv: Minimum channel amplitude at the peak in microvolts (µV).
            The product height lower bound is this value squared.
        max_peak_amplitude_uv: Maximum channel amplitude in microvolts (µV).
            The product height upper bound is this value squared.
        min_peak_gap_s: Minimum time between consecutive peaks in seconds.
        relative_peak_prominence: Minimum peak prominence expressed as a
            fraction of ``min_peak_amplitude_uv ** 2``.
        min_event_duration_s: Minimum event duration in seconds.
        max_event_duration_s: Maximum event duration in seconds.
        min_event_skewness: Lower bound on skewness ``(peak_pos / width) − 0.5``.
            A value of 0 means the peak is centred; negative means early;
            positive means late.
        max_event_skewness: Upper bound on skewness.
        relative_baseline: Fraction of ``min_peak_amplitude_uv ** 2`` used as the
            amplitude threshold when walking left/right from a peak to find the
            event boundaries.

    Returns:
        Ordered list of :class:`~somnio.data.annotations.Event` objects.
        Each event has:

        * ``type``: ``"eye_movement"``.
        * ``label``: movement direction (``"L"`` or ``"R"``).
        * ``onset``: absolute Unix timestamp in **nanoseconds** (taken from ``timestamps``).
        * ``duration``: event duration in **nanoseconds**.
        * ``extras``: additional measurements:

          * ``"peak_amplitude_uv2"`` — product-signal value at the peak (µV²).
          * ``"skewness"`` — normalised peak offset within the event.
    """
    product_height = (min_peak_amplitude_uv**2, max_peak_amplitude_uv**2)
    peaks, _ = find_peaks(
        product_signal,
        height=product_height,
        distance=max(1, int(min_peak_gap_s * sample_rate)),
        prominence=relative_peak_prominence * product_height[0],
    )
    logger.debug("Found %d candidate product peaks", len(peaks))

    baseline_threshold = relative_baseline * product_height[0]
    n = len(product_signal)
    events: list[Event] = []

    for peak_idx in peaks:
        # Walk leftward to find event start (first sample above threshold).
        start_idx = int(peak_idx)
        while start_idx > 0 and product_signal[start_idx] > baseline_threshold:
            start_idx -= 1

        # Walk rightward to find event end (last sample above threshold).
        end_idx = int(peak_idx)
        while end_idx < n - 1 and product_signal[end_idx] > baseline_threshold:
            end_idx += 1

        event_width = end_idx - start_idx
        if event_width == 0:
            continue

        skewness = ((int(peak_idx) - start_idx) / event_width) - 0.5
        duration_s = event_width / sample_rate

        if not (min_event_skewness <= skewness <= max_event_skewness):
            logger.debug(
                "Rejecting peak at t=%.3fs — skewness %.3f not in [%.3f, %.3f]",
                (int(timestamps[peak_idx]) - int(timestamps[0])) / 1e9,
                skewness,
                min_event_skewness,
                max_event_skewness,
            )
            continue

        if not (min_event_duration_s <= duration_s <= max_event_duration_s):
            logger.debug(
                "Rejecting peak at t=%.3fs — duration %.3fs not in [%.3f, %.3f]",
                (int(timestamps[peak_idx]) - int(timestamps[0])) / 1e9,
                duration_s,
                min_event_duration_s,
                max_event_duration_s,
            )
            continue

        direction = LEFT_LABEL if diff_signal[int(peak_idx)] > 0 else RIGHT_LABEL
        onset = int(timestamps[start_idx])
        # Duration derived from timestamps to handle irregular sampling correctly.
        duration = int(timestamps[end_idx]) - int(timestamps[start_idx])

        events.append(
            Event(
                onset=onset,
                duration=duration,
                type=EVENT_TYPE,
                label=direction,
                extras={
                    "peak_amplitude_uv2": float(product_signal[peak_idx]),
                    "skewness": float(skewness),
                },
            )
        )

    logger.info("Detected %d primitive eye-movement events", len(events))
    return events


def _filter_by_signal(
    sequences: list[Event],
    left_vals: np.ndarray,
    right_vals: np.ndarray,
    timestamps: np.ndarray,
    *,
    min_correlation: float,
    min_amplitude_ratio: float,
    max_amplitude_ratio: float,
) -> list[Event]:
    """Reject sequences with poor anti-correlation or outlying amplitude ratio.

    Uses the already-converted µV arrays and timestamps from the calling
    function to avoid re-computing them for every sequence.
    """
    valid: list[Event] = []

    for seq in sequences:
        label = str(seq.label)
        seq_end = seq.onset + seq.duration
        mask = (timestamps >= seq.onset) & (timestamps <= seq_end)

        if mask.sum() < 2:
            logger.info(
                "Sequence %r at onset=%.3fs rejected — fewer than 2 samples in window",
                label,
                seq.onset / 1e9,
            )
            continue

        lw = left_vals[mask]
        rw = right_vals[mask]

        correlation = float(np.corrcoef(lw, rw)[0, 1])
        if not np.isfinite(correlation):
            logger.info(
                "Sequence %r at onset=%.3fs rejected — non-finite correlation %r",
                label,
                seq.onset / 1e9,
                correlation,
            )
            continue
        if correlation > -min_correlation:
            logger.info(
                "Sequence %r at onset=%.3fs rejected — correlation %.3f > -%.3f",
                label,
                seq.onset / 1e9,
                correlation,
                min_correlation,
            )
            continue

        amplitude_ratio = float(np.std(lw)) / (float(np.std(rw)) + np.finfo(float).eps)
        if not (min_amplitude_ratio <= amplitude_ratio <= max_amplitude_ratio):
            logger.info(
                "Sequence %r at onset=%.3fs rejected — amplitude_ratio %.3f not in [%.3f, %.3f]",
                label,
                seq.onset / 1e9,
                amplitude_ratio,
                min_amplitude_ratio,
                max_amplitude_ratio,
            )
            continue

        valid.append(
            Event(
                onset=seq.onset,
                duration=seq.duration,
                type=seq.type,
                label=seq.label,
                extras={
                    **seq.extras,
                    "correlation": correlation,
                    "amplitude_ratio": amplitude_ratio,
                },
            )
        )

    return valid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_lr_eye_movements(
    ts: TimeSeries,
    left: str,
    right: str,
    *,
    preprocess: bool = True,
    # primitive-event params
    min_peak_amplitude_uv: float = MIN_PEAK_AMPLITUDE_UV,
    max_peak_amplitude_uv: float = MAX_PEAK_AMPLITUDE_UV,
    min_peak_gap_s: float = MIN_PEAK_GAP_S,
    relative_peak_prominence: float = RELATIVE_PEAK_PROMINENCE,
    min_event_duration_s: float = MIN_EVENT_DURATION_S,
    max_event_duration_s: float = MAX_EVENT_DURATION_S,
    min_event_skewness: float = MIN_EVENT_SKEWNESS,
    max_event_skewness: float = MAX_EVENT_SKEWNESS,
    relative_baseline: float = RELATIVE_BASELINE,
    low_cutoff_hz: float = LOW_CUTOFF_HZ,
    high_cutoff_hz: float = HIGH_CUTOFF_HZ,
    # sequence params
    merge: bool = True,
    max_event_gap_s: float = MAX_EVENT_GAP_S,
    accepted_pattern: str | None = None,
    # signal-filtering params
    min_correlation: float = MIN_CORRELATION,
    min_amplitude_ratio: float = MIN_AMPLITUDE_RATIO,
    max_amplitude_ratio: float = MAX_AMPLITUDE_RATIO,
) -> tuple[list[Event], list[Event]]:
    """Detect L/R eye-movement sequences from a two-channel EOG TimeSeries.

    Full pipeline in one call:

    1. Optionally bandpass-filter the two EOG channels.
    2. Detect primitive L/R events via peak detection on the negative-product
       signal (see :func:`detect_primitive_events`).
    3. Merge temporally close primitives into sequences
       (see :func:`~somnio.tasks.eye_movement.sequences.build_sequences`).
    4. Discard sequences whose label does not match *accepted_pattern*.
    5. Discard sequences with insufficient anti-correlation or an outlying
       amplitude ratio between the left and right channels.

    Args:
        ts: Input time-series containing ``left`` and ``right`` EOG channels.
            ``ts.sample_rate`` must be set.
        left: Channel name for the left EOG electrode.
        right: Channel name for the right EOG electrode.
        preprocess: When ``True`` (default) apply a bandpass FIR filter before
            detection.  Set to ``False`` if the signal is already filtered.
        min_peak_amplitude_uv: Minimum EOG amplitude at a peak in **µV**.
        max_peak_amplitude_uv: Maximum EOG amplitude at a peak in **µV**.
        min_peak_gap_s: Minimum gap between consecutive peaks in **seconds**.
        relative_peak_prominence: Peak prominence threshold relative to
            ``min_peak_amplitude_uv ** 2``.
        min_event_duration_s: Minimum primitive event duration in **seconds**.
        max_event_duration_s: Maximum primitive event duration in **seconds**.
        min_event_skewness: Lower bound on normalised peak offset within the
            event window (0 = centred).
        max_event_skewness: Upper bound on normalised peak offset.
        relative_baseline: Boundary-walk threshold as a fraction of
            ``min_peak_amplitude_uv ** 2``.
        low_cutoff_hz: FIR low passband edge in **Hz** (``preprocess=True`` only).
        high_cutoff_hz: FIR high passband edge in **Hz** (``preprocess=True`` only).
        merge: When ``True`` (default) merge temporally close primitive events
            into sequences.
        max_event_gap_s: Maximum gap between primitive events (in **seconds**)
            to still merge them into one sequence.  Default: ``0.2``.
        accepted_pattern: Regex that must fully match the sequence label.
            Default is ``None`` (no pattern filtering).
        min_correlation: Minimum magnitude of the *negative* Pearson
            correlation between left and right channels over the sequence
            window.  Sequences where ``corr > −min_correlation`` are rejected.
            Default: ``0.6``.
        min_amplitude_ratio: Lower bound on ``std(left) / std(right)``.
        max_amplitude_ratio: Upper bound on ``std(left) / std(right)``.

    Returns:
        ``(sequences, primitives)`` where

        * ``sequences`` — validated sequence events.  Each has ``label`` set to
          the direction string (e.g. ``"LRLR"``), ``extras["n_events"]``,
          ``extras["correlation"]``, and ``extras["amplitude_ratio"]``.
        * ``primitives`` — all primitive L/R events detected before grouping
          (before any sequence-level filtering).

    Raises:
        ValueError: If ``ts.channel_names`` is not exactly ``{left, right}``.
        ValueError: If ``ts.sample_rate`` is ``None``.
        MissingOptionalDependency: If ``scipy`` is not installed.
    """
    if ts.sample_rate is None:
        raise ValueError(
            "TimeSeries.sample_rate must not be None; "
            "eye-movement detection requires a known sample rate."
        )

    if set(ts.channel_names) != {left, right}:
        raise ValueError(
            f"TimeSeries must contain exactly the two channels {[left, right]!r}; "
            f"got {list(ts.channel_names)!r}."
        )

    eog = ts.select_channels([left, right])
    if preprocess:
        eog = apply_fir_filter(eog, low_cutoff_hz, high_cutoff_hz)

    left_vals = convert_values(eog.values[:, 0], eog.units[0], UV)
    right_vals = convert_values(eog.values[:, 1], eog.units[1], UV)
    product_signal = -(left_vals * right_vals)
    diff_signal = left_vals - right_vals

    primitives = _detect_events(
        product_signal,
        diff_signal,
        eog.timestamps,
        eog.sample_rate,  # type: ignore[arg-type]
        min_peak_amplitude_uv=min_peak_amplitude_uv,
        max_peak_amplitude_uv=max_peak_amplitude_uv,
        min_peak_gap_s=min_peak_gap_s,
        relative_peak_prominence=relative_peak_prominence,
        min_event_duration_s=min_event_duration_s,
        max_event_duration_s=max_event_duration_s,
        min_event_skewness=min_event_skewness,
        max_event_skewness=max_event_skewness,
        relative_baseline=relative_baseline,
    )

    sequences = (
        merge_events(primitives, max_event_gap_s=max_event_gap_s)
        if merge
        else primitives
    )

    if accepted_pattern:
        sequences = filter_by_pattern(sequences, accepted_pattern)

    sequences = _filter_by_signal(
        sequences,
        left_vals,
        right_vals,
        eog.timestamps,
        min_correlation=min_correlation,
        min_amplitude_ratio=min_amplitude_ratio,
        max_amplitude_ratio=max_amplitude_ratio,
    )

    logger.info(
        "detect_lr_eye_movements: %d primitives → %d valid sequences",
        len(primitives),
        len(sequences),
    )
    return sequences, primitives
