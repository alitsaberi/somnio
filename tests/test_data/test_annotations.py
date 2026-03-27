"""Tests for zutils.data.annotations — Event, Epochs, conversions."""

import numpy as np
import pytest

from zutils.data import Event, Epochs, epochs_to_events, events_to_epochs


class TestEvent:
    def test_construction_coerces_types(self):
        e = Event(
            onset="1.5",  # type: ignore[arg-type]
            duration=2,
            description=123,  # type: ignore[arg-type]
            extras={"confidence": 0.95},
        )
        assert isinstance(e.onset, float)
        assert isinstance(e.duration, float)
        assert e.description == "123"
        assert e.extras == {"confidence": 0.95}

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError, match="duration must be >= 0"):
            Event(onset=0.0, duration=-1.0, description="X")

    def test_rejects_negative_onset(self):
        with pytest.raises(ValueError, match="onset must be >= 0"):
            Event(onset=-0.1, duration=0.0, description="X")

    def test_rejects_non_string_extras_keys(self):
        with pytest.raises(ValueError, match="extras keys must be strings"):
            Event(onset=0.0, duration=0.0, description="X", extras={1: "nope"})


class TestEpochs:
    def test_construction_coerces_dtype_strings(self):
        epochs = Epochs(labels=["N2", "W"], period_length=30.0, onset=10.0)
        assert epochs.labels.shape == (2,)
        assert epochs.labels.dtype == object
        assert epochs.onset == 10.0
        assert epochs.period_length == 30.0

    def test_construction_coerces_dtype_ints(self):
        epochs = Epochs(labels=[0, 1], period_length=30.0)
        assert epochs.labels.dtype.kind in {"i", "u"}
        np.testing.assert_array_equal(epochs.labels, np.array([0, 1], dtype=np.int64))

    def test_rejects_non_1d_labels(self):
        with pytest.raises(ValueError, match="labels must be 1-D"):
            Epochs(labels=[[0, 1]], period_length=30.0)

    def test_rejects_non_positive_period_length(self):
        with pytest.raises(ValueError, match="period_length must be > 0"):
            Epochs(labels=[0], period_length=0.0)

    def test_rejects_non_finite_onset(self):
        with pytest.raises(ValueError, match="onset must be finite"):
            Epochs(labels=[0], period_length=30.0, onset=float("nan"))
        with pytest.raises(ValueError, match="onset must be finite"):
            Epochs(labels=[0], period_length=30.0, onset=float("inf"))


class TestConversions:
    def test_epochs_to_events(self):
        epochs = Epochs(
            labels=np.array(["N2", "W"], dtype=object), period_length=30.0, onset=10.0
        )
        events = epochs_to_events(epochs)
        assert len(events) == 2
        assert events[0].onset == pytest.approx(10.0)
        assert events[0].duration == pytest.approx(30.0)
        assert events[0].description == "N2"
        assert events[1].onset == pytest.approx(40.0)
        assert events[1].description == "W"

    def test_events_to_epochs_roundtrip_int_descriptions(self):
        events = [
            Event(onset=0.0, duration=30.0, description="0"),
            Event(onset=30.0, duration=30.0, description="1"),
        ]
        epochs = events_to_epochs(events, period_length=30.0)
        assert epochs.period_length == pytest.approx(30.0)
        assert epochs.onset == pytest.approx(0.0)
        assert epochs.labels.dtype.kind in {"i", "u"}
        np.testing.assert_array_equal(epochs.labels, np.array([0, 1], dtype=np.int64))

        events2 = epochs_to_events(epochs)
        assert [e.description for e in events2] == ["0", "1"]

    def test_events_to_epochs_requires_contiguity(self):
        events = [
            Event(onset=0.0, duration=30.0, description="0"),
            # Non-contiguous onset (should be 30.0)
            Event(onset=40.0, duration=30.0, description="1"),
        ]
        with pytest.raises(ValueError, match="not contiguous fixed-period epochs"):
            events_to_epochs(events, period_length=30.0)

    def test_events_to_epochs_rejects_non_finite_period_length(self):
        events = [Event(onset=0.0, duration=30.0, description="0")]
        with pytest.raises(ValueError, match="period_length must be finite"):
            events_to_epochs(events, period_length=float("nan"))
        with pytest.raises(ValueError, match="period_length must be finite"):
            events_to_epochs(events, period_length=float("inf"))
