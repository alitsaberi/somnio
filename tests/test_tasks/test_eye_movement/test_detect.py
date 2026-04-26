"""Tests for somnio.tasks.eye_movement.detect.

Requires the ``signal`` extra (SciPy).  All tests are skipped when SciPy is
not installed so the default test suite stays light.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module when scipy is not available.
pytest.importorskip("scipy")

from somnio.data.annotations import Event
from somnio.data.timeseries import TimeSeries
from somnio.tasks.eye_movement.defaults import (
    MAX_EVENT_DURATION_S,
    MAX_EVENT_SKEWNESS,
    MAX_PEAK_AMPLITUDE_UV,
    MIN_EVENT_DURATION_S,
    MIN_EVENT_SKEWNESS,
    MIN_PEAK_AMPLITUDE_UV,
    MIN_PEAK_GAP_S,
    RELATIVE_BASELINE,
    RELATIVE_PEAK_PROMINENCE,
)
from somnio.tasks.eye_movement.detect import _detect_events, detect_lr_eye_movements
from somnio.tasks.eye_movement.event import EVENT_TYPE, LEFT_LABEL, RIGHT_LABEL

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 100.0  # Hz
_T0_NS = 1_700_000_000_000_000_000  # arbitrary anchor
_STEP_NS = int(1e9 / _SAMPLE_RATE)


def _make_timestamps(n: int) -> np.ndarray:
    return (_T0_NS + np.arange(n, dtype=np.int64) * _STEP_NS).astype(np.int64)


def _gaussian(n: int, center: int, sigma: float, amplitude: float) -> np.ndarray:
    """1-D Gaussian centered at *center* sample, std-dev *sigma* samples."""
    t = np.arange(n, dtype=float)
    return amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _product_and_diff_for_event(
    n: int,
    center: int,
    direction: str,
    amplitude_uv: float = 200.0,
    sigma: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct (product_signal, diff_signal) with a single eye-movement peak.

    For ``direction="L"``: left > 0, right < 0 → diff > 0 at peak.
    For ``direction="R"``: left < 0, right > 0 → diff < 0 at peak.
    """
    sign = 1 if direction == LEFT_LABEL else -1
    left_uv = sign * _gaussian(n, center, sigma, amplitude_uv)
    right_uv = -sign * _gaussian(n, center, sigma, amplitude_uv)
    product_signal = -(left_uv * right_uv)  # positive peak
    diff_signal = left_uv - right_uv
    return product_signal, diff_signal


def _make_eog_ts(
    events: list[tuple[float, str]],
    *,
    sample_rate: float = _SAMPLE_RATE,
    duration_s: float = 10.0,
    amplitude_uv: float = 200.0,
    sigma_s: float = 0.1,
) -> TimeSeries:
    """Build a synthetic two-channel EOG ``TimeSeries`` with known eye movements.

    Each entry in *events* is ``(center_time_s, direction)`` where direction is
    ``"L"`` or ``"R"``.  Values are stored in Volts (as required by somnio).
    """
    n = int(sample_rate * duration_s)
    step_ns = int(1e9 / sample_rate)
    timestamps = (_T0_NS + np.arange(n, dtype=np.int64) * step_ns).astype(np.int64)
    t = np.arange(n) / sample_rate

    left_uv = np.zeros(n)
    right_uv = np.zeros(n)

    for center_s, direction in events:
        gauss = amplitude_uv * np.exp(-0.5 * ((t - center_s) / sigma_s) ** 2)
        if direction == LEFT_LABEL:
            left_uv += gauss
            right_uv -= gauss
        else:
            left_uv -= gauss
            right_uv += gauss

    # Store in Volts; detect_lr_eye_movements converts internally to µV.
    values = np.column_stack([left_uv * 1e-6, right_uv * 1e-6])
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=("EOG_L", "EOG_R"),
        units=("V", "V"),
        sample_rate=sample_rate,
    )


# Default kwargs for _detect_events (match module defaults).
_DETECT_KWARGS: dict = dict(
    min_peak_amplitude_uv=MIN_PEAK_AMPLITUDE_UV,
    max_peak_amplitude_uv=MAX_PEAK_AMPLITUDE_UV,
    min_peak_gap_s=MIN_PEAK_GAP_S,
    relative_peak_prominence=RELATIVE_PEAK_PROMINENCE,
    min_event_duration_s=MIN_EVENT_DURATION_S,
    max_event_duration_s=MAX_EVENT_DURATION_S,
    min_event_skewness=MIN_EVENT_SKEWNESS,
    max_event_skewness=MAX_EVENT_SKEWNESS,
    relative_baseline=RELATIVE_BASELINE,
)


# ---------------------------------------------------------------------------
# Tests for _detect_events
# ---------------------------------------------------------------------------


class TestDetectEvents:
    """Unit tests for the private peak-based event detector."""

    def _run(
        self,
        n: int,
        center: int,
        direction: str,
        amplitude_uv: float = 200.0,
        sigma: float = 10.0,
        **overrides,
    ) -> list[Event]:
        product, diff = _product_and_diff_for_event(
            n, center, direction, amplitude_uv=amplitude_uv, sigma=sigma
        )
        timestamps = _make_timestamps(n)
        kwargs = {**_DETECT_KWARGS, **overrides}
        return _detect_events(product, diff, timestamps, _SAMPLE_RATE, **kwargs)

    def test_single_left_event_detected(self):
        events = self._run(n=1000, center=200, direction="L")
        assert len(events) == 1
        e = events[0]
        assert e.type == EVENT_TYPE
        assert e.label == LEFT_LABEL

    def test_single_right_event_detected(self):
        events = self._run(n=1000, center=200, direction="R")
        assert len(events) == 1
        assert events[0].label == RIGHT_LABEL

    def test_event_onset_is_unix_ns(self):
        events = self._run(n=1000, center=200, direction="L")
        # onset must be >= the first timestamp (absolute time, not relative)
        assert events[0].onset >= _T0_NS

    def test_event_duration_is_positive_ns(self):
        events = self._run(n=1000, center=200, direction="L")
        assert events[0].duration > 0

    def test_event_has_extras(self):
        events = self._run(n=1000, center=200, direction="L")
        e = events[0]
        assert "peak_amplitude_uv2" in e.extras
        assert "skewness" in e.extras
        assert e.extras["peak_amplitude_uv2"] > 0

    def test_two_peaks_two_events(self):
        n = 1000
        p1, d1 = _product_and_diff_for_event(n, center=200, direction="L")
        p2, d2 = _product_and_diff_for_event(n, center=700, direction="R")
        product = p1 + p2
        diff = d1 + d2
        timestamps = _make_timestamps(n)
        events = _detect_events(
            product, diff, timestamps, _SAMPLE_RATE, **_DETECT_KWARGS
        )
        assert len(events) == 2
        assert events[0].label == LEFT_LABEL
        assert events[1].label == RIGHT_LABEL

    def test_amplitude_below_minimum_not_detected(self):
        # amplitude = 60 µV → product peak = 3600 < MIN^2 = 6400
        events = self._run(n=1000, center=200, direction="L", amplitude_uv=60.0)
        assert len(events) == 0

    def test_amplitude_above_maximum_not_detected(self):
        # amplitude = 600 µV → product peak = 360000 > MAX^2 = 302500
        events = self._run(n=1000, center=200, direction="L", amplitude_uv=600.0)
        assert len(events) == 0

    def test_duration_too_long_rejected(self):
        # Large σ → wide event exceeding MAX_EVENT_DURATION_S (1.2 s)
        # σ=80 samples @ 100 Hz → event width ≈ 2*2.9*80 = 464 samples = 4.64 s
        events = self._run(n=2000, center=500, direction="L", sigma=80.0)
        assert len(events) == 0

    def test_empty_signal_returns_empty(self):
        product = np.zeros(100)
        diff = np.zeros(100)
        timestamps = _make_timestamps(100)
        events = _detect_events(
            product, diff, timestamps, _SAMPLE_RATE, **_DETECT_KWARGS
        )
        assert events == []

    def test_skewness_of_symmetric_peak_near_zero(self):
        events = self._run(n=1000, center=200, direction="L", sigma=10.0)
        assert len(events) == 1
        assert abs(events[0].extras["skewness"]) <= 0.3


# ---------------------------------------------------------------------------
# Tests for detect_lr_eye_movements
# ---------------------------------------------------------------------------


class TestDetectLrEyeMovements:
    """Integration tests for the public detection API."""

    def test_single_left_movement(self):
        ts = _make_eog_ts([(2.0, "L")])
        sequences, primitives = detect_lr_eye_movements(
            ts, left="EOG_L", right="EOG_R", preprocess=False
        )
        assert len(primitives) >= 1
        assert primitives[0].label == LEFT_LABEL

    def test_single_right_movement(self):
        ts = _make_eog_ts([(2.0, "R")])
        sequences, primitives = detect_lr_eye_movements(
            ts, left="EOG_L", right="EOG_R", preprocess=False
        )
        assert len(primitives) >= 1
        assert primitives[0].label == RIGHT_LABEL

    def test_returns_tuple_of_lists(self):
        ts = _make_eog_ts([(2.0, "L")])
        result = detect_lr_eye_movements(
            ts, left="EOG_L", right="EOG_R", preprocess=False
        )
        assert isinstance(result, tuple) and len(result) == 2
        sequences, primitives = result
        assert isinstance(sequences, list)
        assert isinstance(primitives, list)

    def test_events_have_absolute_ns_timestamps(self):
        ts = _make_eog_ts([(2.0, "L")])
        _, primitives = detect_lr_eye_movements(
            ts, left="EOG_L", right="EOG_R", preprocess=False
        )
        assert len(primitives) >= 1
        e = primitives[0]
        # onset must be within the input TimeSeries' time range
        assert ts.timestamps[0] <= e.onset <= ts.timestamps[-1]
        assert e.duration > 0

    def test_two_alternating_movements_produce_sequence(self):
        # L at 2 s and R at 5 s — 3 s apart, well within a single recording.
        # With merge=True and a large gap, we expect each primitive to be its
        # own sequence (default max_event_gap=0.2 s).
        ts = _make_eog_ts([(2.0, "L"), (5.0, "R")])
        sequences, primitives = detect_lr_eye_movements(
            ts,
            left="EOG_L",
            right="EOG_R",
            preprocess=False,
            accepted_pattern=None,
        )
        assert len(primitives) == 2
        labels = [p.label for p in primitives]
        assert LEFT_LABEL in labels and RIGHT_LABEL in labels

    def test_merge_close_events_into_sequence(self):
        # Place L at 2 s and R at 2.6 s.  Use a large max_event_gap so they merge.
        ts = _make_eog_ts([(2.0, "L"), (2.6, "R")])
        sequences, primitives = detect_lr_eye_movements(
            ts,
            left="EOG_L",
            right="EOG_R",
            preprocess=False,
            merge=True,
            max_event_gap_s=1.0,
            accepted_pattern=None,
        )
        assert len(primitives) == 2
        # Merged into a single sequence
        assert len(sequences) == 1
        assert sequences[0].label == "LR"

    def test_raises_when_sample_rate_is_none(self):
        n = 100
        ts = TimeSeries(
            values=np.zeros((n, 2)),
            timestamps=np.arange(n, dtype=np.int64),
            channel_names=("EOG_L", "EOG_R"),
            units=("V", "V"),
            sample_rate=None,
        )
        with pytest.raises(ValueError, match="sample_rate"):
            detect_lr_eye_movements(ts, left="EOG_L", right="EOG_R")

    def test_raises_when_wrong_channel_names(self):
        ts = _make_eog_ts([(2.0, "L")])
        with pytest.raises(ValueError, match="channels"):
            detect_lr_eye_movements(ts, left="WRONG_L", right="EOG_R", preprocess=False)

    def test_raises_when_extra_channels_present(self):
        # TimeSeries with 3 channels — not exactly {left, right}
        n = 100
        ts = TimeSeries(
            values=np.zeros((n, 3)),
            timestamps=_make_timestamps(n),
            channel_names=("EOG_L", "EOG_R", "EEG"),
            units=("V", "V", "V"),
            sample_rate=_SAMPLE_RATE,
        )
        with pytest.raises(ValueError, match="channels"):
            detect_lr_eye_movements(ts, left="EOG_L", right="EOG_R", preprocess=False)

    def test_merge_false_returns_unmerged_primitives(self):
        ts = _make_eog_ts([(2.0, "L"), (5.0, "R")])
        sequences, primitives = detect_lr_eye_movements(
            ts,
            left="EOG_L",
            right="EOG_R",
            preprocess=False,
            merge=False,
            accepted_pattern=None,
        )
        # With merge=False, sequences IS the primitives list (no grouping).
        assert len(sequences) == len(primitives)

    def test_pattern_filter_removes_single_primitives(self):
        # accepted_pattern requires at least 2 alternating pairs
        ts = _make_eog_ts([(2.0, "L")])
        sequences, primitives = detect_lr_eye_movements(
            ts,
            left="EOG_L",
            right="EOG_R",
            preprocess=False,
            accepted_pattern=r"(LR|RL){2,}",
        )
        # A lone "L" primitive cannot match — no valid sequences
        assert len(sequences) == 0
        assert len(primitives) >= 1
