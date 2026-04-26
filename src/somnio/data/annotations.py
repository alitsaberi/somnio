"""Lightweight annotation types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _require_int_ns(name: str, value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(
        f"{name} must be an integer nanosecond value, got {type(value).__name__}"
    )


@dataclass
class Event:
    """A single annotated occurrence with arbitrary properties.

    Attributes:
        onset: Start time in int nanoseconds since Unix epoch.
        duration: Duration in int nanoseconds (0 for instantaneous events).
        type: Event semantic type/family (e.g., "sleep_stage", "arousal", "eye_movement").
        label: Optional event label/value (e.g., "N2", "L", "R", 2).
        extras: Arbitrary per-event metadata (e.g., {"confidence": 0.95}).
    """

    onset: int
    duration: int
    type: str
    label: str | int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.onset = _require_int_ns("onset", self.onset)
        self.duration = _require_int_ns("duration", self.duration)
        self.type = str(self.type)
        if self.onset < 0:
            raise ValueError(f"onset must be >= 0, got {self.onset}")
        if self.duration < 0:
            raise ValueError(f"duration must be >= 0, got {self.duration}")
        if not self.type:
            raise ValueError("type must be non-empty")

        if not isinstance(self.extras, dict):
            raise ValueError("extras must be a dict")
        bad_key = next((k for k in self.extras.keys() if not isinstance(k, str)), None)
        if bad_key is not None:
            raise ValueError("extras keys must be strings")


@dataclass
class Epochs:
    """Fixed-period epoch annotations (e.g., 30-second sleep stages).

    Convenience type for the common pattern of one label per fixed-length
    window.

    Attributes:
        labels: Label per epoch, shape ``(n_epochs,)``. Typically string or int.
        period_length: Duration of each epoch in int nanoseconds (e.g., 30e9).
        onset: Start time of the first epoch in int nanoseconds since Unix epoch.
    """

    labels: np.ndarray
    period_length: int
    onset: int = 0

    def __post_init__(self) -> None:
        self.period_length = _require_int_ns("period_length", self.period_length)
        self.onset = _require_int_ns("onset", self.onset)
        if self.period_length <= 0:
            raise ValueError(f"period_length must be > 0, got {self.period_length!r}")
        if self.onset < 0:
            raise ValueError(f"onset must be >= 0, got {self.onset!r}")

        labels = np.asarray(self.labels)
        if labels.ndim != 1:
            raise ValueError(
                f"labels must be 1-D (n_epochs,), got shape {labels.shape}"
            )

        # Normalize label dtype: allow {int, string, object(strings)}.
        if labels.dtype.kind in {"i", "u"}:
            labels = labels.astype(np.int64)
        elif labels.dtype.kind in {"U", "S"}:
            labels = labels.astype(object)
        elif labels.dtype.kind == "O":
            # Keep as object; callers can store strings/ints.
            labels = labels.astype(object)
        else:
            raise ValueError(
                "labels must be an integer dtype, a string dtype, or dtype=object"
            )

        self.labels = labels


def epochs_to_events(epochs: Epochs) -> list[Event]:
    """Expand each epoch label to an :class:`Event`.

    Each epoch becomes an ``Event`` with:
    - ``onset = epochs.onset + i * epochs.period_length``
    - ``duration = epochs.period_length``
    - ``type = "epoch"``
    - ``label = epochs.labels[i]`` (int-like labels stay int; otherwise str)
    """

    events: list[Event] = []
    step = epochs.period_length
    for i, label in enumerate(epochs.labels):
        if isinstance(label, (np.integer,)):
            event_label: int | str = int(label)
        else:
            event_label = str(label)
        events.append(
            Event(
                onset=epochs.onset + i * step,
                duration=step,
                type="epoch",
                label=event_label,
            )
        )
    return events


def events_to_epochs(events: list[Event], period_length: int) -> Epochs:
    """Collapse contiguous fixed-duration events into an :class:`Epochs`.

    Args:
        events: Events with fixed duration equal to ``period_length``.
            They must be contiguous with step size ``period_length`` when sorted
            by onset.
        period_length: Fixed epoch duration in int nanoseconds.
    """

    if len(events) == 0:
        raise ValueError("events must be non-empty")

    step = int(period_length)
    if step <= 0:
        raise ValueError(f"period_length must be > 0, got {period_length!r}")

    events_sorted = sorted(events, key=lambda e: e.onset)
    first_onset = events_sorted[0].onset

    # Validate fixed duration + contiguity.
    for i, event in enumerate(events_sorted):
        expected_onset = first_onset + i * step
        if event.duration != step:
            raise ValueError(
                f"event duration must equal period_length={step}, got {event.duration}"
            )
        if event.onset != expected_onset:
            raise ValueError(
                "events are not contiguous fixed-period epochs when sorted by onset"
            )

    if any(e.label is None for e in events_sorted):
        raise ValueError("all events must have a non-None label to build Epochs")

    labels_list = [e.label for e in events_sorted]
    if all(isinstance(x, (int, np.integer)) for x in labels_list):
        labels = np.asarray([int(x) for x in labels_list], dtype=np.int64)
    else:
        labels = np.asarray([str(x) for x in labels_list], dtype=object)

    return Epochs(labels=labels, period_length=step, onset=first_onset)
