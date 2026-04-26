"""Tests for somnio.data.annotations — Event, Epochs, conversions.

Note: The annotation types use integer nanoseconds (ns) for onset/duration.
"""

import numpy as np
import pytest

from somnio.data import Event, Epochs, epochs_to_events, events_to_epochs


class TestEvent:
    def test_construction_coerces_types(self):
        with pytest.raises(TypeError, match="onset must be an integer"):
            Event(onset=1.5, duration=2, type="X")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="duration must be an integer"):
            Event(onset=1, duration=2.2, type="X")  # type: ignore[arg-type]

        e = Event(onset=1, duration=2, type=123, label=456, extras={"confidence": 0.95})  # type: ignore[arg-type]
        assert e.onset == 1
        assert e.duration == 2
        assert e.type == "123"
        assert e.label == 456
        assert e.extras == {"confidence": 0.95}

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError, match="duration must be >= 0"):
            Event(onset=0, duration=-1, type="X")

    def test_rejects_negative_onset(self):
        with pytest.raises(ValueError, match="onset must be >= 0"):
            Event(onset=-1, duration=0, type="X")

    def test_rejects_non_string_extras_keys(self):
        with pytest.raises(ValueError, match="extras keys must be strings"):
            Event(onset=0, duration=0, type="X", extras={1: "nope"})  # type: ignore[arg-type]


class TestEpochs:
    def test_construction_coerces_dtype_strings(self):
        epochs = Epochs(
            labels=["N2", "W"],
            period_length=int(30e9),
            onset=int(10e9),
        )
        assert epochs.labels.shape == (2,)
        assert epochs.labels.dtype == object
        assert epochs.onset == int(10e9)
        assert epochs.period_length == int(30e9)

    def test_construction_coerces_dtype_ints(self):
        epochs = Epochs(labels=[0, 1], period_length=int(30e9))
        assert epochs.labels.dtype.kind in {"i", "u"}
        np.testing.assert_array_equal(epochs.labels, np.array([0, 1], dtype=np.int64))

    def test_rejects_non_1d_labels(self):
        with pytest.raises(ValueError, match="labels must be 1-D"):
            Epochs(labels=[[0, 1]], period_length=int(30e9))

    def test_rejects_non_positive_period_length(self):
        with pytest.raises(ValueError, match="period_length must be > 0"):
            Epochs(labels=[0], period_length=0)

    def test_rejects_non_finite_onset(self):
        with pytest.raises(TypeError):
            Epochs(labels=[0], period_length=int(30e9), onset=float("nan"))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            Epochs(labels=[0], period_length=int(30e9), onset=float("inf"))  # type: ignore[arg-type]


class TestConversions:
    def test_epochs_to_events(self):
        epochs = Epochs(
            labels=np.array(["N2", "W"], dtype=object),
            period_length=int(30e9),
            onset=int(10e9),
        )
        events = epochs_to_events(epochs)
        assert len(events) == 2
        assert events[0].onset == int(10e9)
        assert events[0].duration == int(30e9)
        assert events[0].type == "epoch"
        assert events[0].label == "N2"
        assert events[1].onset == int(40e9)
        assert events[1].label == "W"

    def test_events_to_epochs_roundtrip_int_descriptions(self):
        events = [
            Event(onset=0, duration=int(30e9), type="epoch", label=0),
            Event(onset=int(30e9), duration=int(30e9), type="epoch", label=1),
        ]
        epochs = events_to_epochs(events, period_length=int(30e9))
        assert epochs.period_length == int(30e9)
        assert epochs.onset == 0
        assert epochs.labels.dtype.kind in {"i", "u"}
        np.testing.assert_array_equal(epochs.labels, np.array([0, 1], dtype=np.int64))

        events2 = epochs_to_events(epochs)
        assert [e.label for e in events2] == [0, 1]

    def test_events_to_epochs_requires_contiguity(self):
        events = [
            Event(onset=0, duration=int(30e9), type="epoch", label="0"),
            # Non-contiguous onset (should be 30s)
            Event(onset=int(40e9), duration=int(30e9), type="epoch", label="1"),
        ]
        with pytest.raises(ValueError, match="not contiguous fixed-period epochs"):
            events_to_epochs(events, period_length=int(30e9))

    def test_events_to_epochs_rejects_non_finite_period_length(self):
        events = [Event(onset=0, duration=int(30e9), type="epoch", label="0")]
        with pytest.raises((TypeError, ValueError, OverflowError)):
            events_to_epochs(events, period_length=float("nan"))  # type: ignore[arg-type]
        with pytest.raises((TypeError, ValueError, OverflowError)):
            events_to_epochs(events, period_length=float("inf"))  # type: ignore[arg-type]
