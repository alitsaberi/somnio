"""Tests for somnio.tasks.eye_movement.event.

Covers direction labeling, event validation, temporal merging, and pattern
filtering.  No SciPy dependency is required.
"""

from __future__ import annotations

import pytest

from somnio.data.annotations import Event
from somnio.tasks.eye_movement.event import (
    EVENT_TYPE,
    LEFT_LABEL,
    RIGHT_LABEL,
    filter_by_pattern,
    is_valid_eye_movement_event,
    merge_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0_NS = 1_700_000_000_000_000_000  # arbitrary Unix epoch anchor (ns)
_S = int(1e9)  # 1 second in nanoseconds


def _make_event(
    label: str,
    onset_s: float = 0.0,
    duration_s: float = 0.5,
) -> Event:
    return Event(
        onset=_T0_NS + int(onset_s * _S),
        duration=int(duration_s * _S),
        type=EVENT_TYPE,
        label=label,
    )


# ---------------------------------------------------------------------------
# is_valid_eye_movement_event
# ---------------------------------------------------------------------------


class TestIsValidEyeMovementEvent:
    def test_single_left(self):
        assert is_valid_eye_movement_event(_make_event(LEFT_LABEL))

    def test_single_right(self):
        assert is_valid_eye_movement_event(_make_event(RIGHT_LABEL))

    def test_sequence_label(self):
        assert is_valid_eye_movement_event(_make_event("LRLR"))

    def test_wrong_type_rejected(self):
        e = Event(onset=_T0_NS, duration=_S, type="other", label="L")
        assert not is_valid_eye_movement_event(e)

    def test_empty_label_rejected(self):
        e = Event(onset=_T0_NS, duration=_S, type=EVENT_TYPE, label="")
        assert not is_valid_eye_movement_event(e)

    def test_none_label_rejected(self):
        e = Event(onset=_T0_NS, duration=_S, type=EVENT_TYPE, label=None)
        assert not is_valid_eye_movement_event(e)

    def test_invalid_character_rejected(self):
        e = Event(onset=_T0_NS, duration=_S, type=EVENT_TYPE, label="LXR")
        assert not is_valid_eye_movement_event(e)

    def test_int_label_rejected(self):
        e = Event(onset=_T0_NS, duration=_S, type=EVENT_TYPE, label=1)
        assert not is_valid_eye_movement_event(e)


# ---------------------------------------------------------------------------
# merge_events
# ---------------------------------------------------------------------------


class TestMergeEvents:
    def test_empty_returns_empty(self):
        assert merge_events([], max_event_gap_s=0.2) == []

    def test_single_event_passthrough(self):
        e = _make_event("L", onset_s=0.0, duration_s=0.4)
        result = merge_events([e], max_event_gap_s=0.2)
        assert len(result) == 1
        assert result[0].label == "L"
        assert result[0].onset == e.onset
        assert result[0].duration == e.duration

    def test_two_close_events_merged(self):
        # Gap between end of first (0.5 s) and onset of second (0.6 s) = 0.1 s < 0.2 s
        e1 = _make_event("L", onset_s=0.0, duration_s=0.5)
        e2 = _make_event("R", onset_s=0.6, duration_s=0.5)
        result = merge_events([e1, e2], max_event_gap_s=0.2)
        assert len(result) == 1
        merged = result[0]
        assert merged.label == "LR"
        assert merged.onset == e1.onset
        assert merged.duration == e2.onset + e2.duration - e1.onset

    def test_two_distant_events_kept_separate(self):
        # Gap = 1.0 s > 0.2 s
        e1 = _make_event("L", onset_s=0.0, duration_s=0.5)
        e2 = _make_event("R", onset_s=1.5, duration_s=0.5)
        result = merge_events([e1, e2], max_event_gap_s=0.2)
        assert len(result) == 2
        assert result[0].label == "L"
        assert result[1].label == "R"

    def test_three_events_two_groups(self):
        # Events 0 and 1 close; event 2 far away
        e0 = _make_event("L", onset_s=0.0, duration_s=0.4)
        e1 = _make_event("R", onset_s=0.5, duration_s=0.4)
        e2 = _make_event("L", onset_s=5.0, duration_s=0.4)
        result = merge_events([e0, e1, e2], max_event_gap_s=0.2)
        assert len(result) == 2
        assert result[0].label == "LR"
        assert result[1].label == "L"

    def test_label_concatenation_order(self):
        events = [
            _make_event("L", onset_s=0.0, duration_s=0.3),
            _make_event("R", onset_s=0.35, duration_s=0.3),
            _make_event("L", onset_s=0.7, duration_s=0.3),
            _make_event("R", onset_s=1.05, duration_s=0.3),
        ]
        result = merge_events(events, max_event_gap_s=0.1)
        assert len(result) == 1
        assert result[0].label == "LRLR"

    def test_invalid_event_raises(self):
        bad = Event(onset=_T0_NS, duration=_S, type="wrong", label="L")
        with pytest.raises(AssertionError):
            merge_events([bad], max_event_gap_s=0.2)

    def test_gap_exactly_at_threshold_merges(self):
        # Gap exactly at max_event_gap_s should still merge (<=)
        gap_ns = int(0.2 * _S)
        e1 = _make_event("L", onset_s=0.0, duration_s=0.5)
        e2 = Event(
            onset=e1.onset + e1.duration + gap_ns,
            duration=int(0.5 * _S),
            type=EVENT_TYPE,
            label="R",
        )
        result = merge_events([e1, e2], max_event_gap_s=0.2)
        assert len(result) == 1
        assert result[0].label == "LR"


# ---------------------------------------------------------------------------
# filter_by_pattern
# ---------------------------------------------------------------------------


class TestFilterByPattern:
    def test_empty_list_returns_empty(self):
        assert filter_by_pattern([], r"(LR|RL){2,}") == []

    def test_matching_pattern_kept(self):
        events = [_make_event("LRLR"), _make_event("LRLRLR", onset_s=2.0)]
        result = filter_by_pattern(events, r"(LR|RL){2,}")
        assert len(result) == 2

    def test_non_matching_label_rejected(self):
        events = [
            _make_event("L"),
            _make_event("LR", onset_s=1.0),
            _make_event("LRLR", onset_s=2.0),
        ]
        result = filter_by_pattern(events, r"(LR|RL){2,}")
        assert len(result) == 1
        assert result[0].label == "LRLR"

    def test_single_char_pattern(self):
        events = [_make_event("L"), _make_event("R", onset_s=1.0)]
        left_only = filter_by_pattern(events, "L")
        assert len(left_only) == 1
        assert left_only[0].label == "L"

    def test_none_label_treated_as_empty_string(self):
        e = Event(onset=_T0_NS, duration=_S, type=EVENT_TYPE, label=None)
        result = filter_by_pattern([e], r"(LR|RL){2,}")
        assert result == []
