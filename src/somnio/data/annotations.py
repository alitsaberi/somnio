"""Lightweight annotation types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Event:
    """A single annotated occurrence with arbitrary properties.

    Attributes:
        onset: Start time in seconds from recording start.
        duration: Duration in seconds (0.0 for instantaneous events).
        description: Event type/label (e.g., "N2", "arousal", "stimulus").
        extras: Arbitrary per-event metadata (e.g., {"confidence": 0.95}).
    """

    onset: float
    duration: float
    description: str
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.onset = float(self.onset)
        self.duration = float(self.duration)
        self.description = str(self.description)
        if not np.isfinite(self.onset):
            raise ValueError(f"onset must be finite, got {self.onset!r}")
        if not np.isfinite(self.duration):
            raise ValueError(f"duration must be finite, got {self.duration!r}")
        if self.onset < 0:
            raise ValueError(f"onset must be >= 0, got {self.onset}")
        if self.duration < 0:
            raise ValueError(f"duration must be >= 0, got {self.duration}")

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
        period_length: Duration of each epoch in seconds (e.g., 30.0).
        onset: Start time of the first epoch in seconds from recording start.
    """

    labels: np.ndarray
    period_length: float
    onset: float = 0.0

    def __post_init__(self) -> None:
        self.period_length = float(self.period_length)
        self.onset = float(self.onset)
        if not np.isfinite(self.onset):
            raise ValueError(f"onset must be finite, got {self.onset!r}")
        if not np.isfinite(self.period_length):
            raise ValueError(
                f"period_length must be finite, got {self.period_length!r}"
            )
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
    - ``description = str(epochs.labels[i])``
    """

    events: list[Event] = []
    step = epochs.period_length
    for i, label in enumerate(epochs.labels):
        events.append(
            Event(
                onset=epochs.onset + i * step,
                duration=step,
                description=str(label),
            )
        )
    return events


def _parse_int_label(description: str) -> int | None:
    """Return integer value when `description` is an int-like string."""

    s = description.strip()
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-"):
        rest = s[1:]
        if rest.isdigit():
            return -int(rest)
        return None
    if s.isdigit():
        return int(s)
    return None


def events_to_epochs(events: list[Event], period_length: float) -> Epochs:
    """Collapse contiguous fixed-duration events into an :class:`Epochs`.

    Args:
        events: Events with fixed duration equal to ``period_length``.
            They must be contiguous with step size ``period_length`` when sorted
            by onset.
        period_length: Fixed epoch duration in seconds.
    """

    if len(events) == 0:
        raise ValueError("events must be non-empty")

    step = float(period_length)
    if not np.isfinite(step):
        raise ValueError(f"period_length must be finite, got {period_length!r}")
    if step <= 0:
        raise ValueError(f"period_length must be > 0, got {period_length!r}")

    events_sorted = sorted(events, key=lambda e: e.onset)
    first_onset = events_sorted[0].onset

    # Validate fixed duration + contiguity.
    atol = 1e-9
    for i, event in enumerate(events_sorted):
        expected_onset = first_onset + i * step
        if not np.isclose(event.duration, step, rtol=0.0, atol=atol):
            raise ValueError(
                f"event duration must equal period_length={step}, got {event.duration}"
            )
        if not np.isclose(event.onset, expected_onset, rtol=0.0, atol=atol):
            raise ValueError(
                "events are not contiguous fixed-period epochs when sorted by onset"
            )

    # Infer dtype: if all labels are int-like, return int epochs; else object.
    parsed_ints: list[int] = []
    all_ints = True
    for event in events_sorted:
        parsed = _parse_int_label(event.description)
        if parsed is None:
            all_ints = False
            break
        parsed_ints.append(parsed)

    if all_ints:
        labels = np.asarray(parsed_ints, dtype=np.int64)
    else:
        labels = np.asarray([e.description for e in events_sorted], dtype=object)

    return Epochs(labels=labels, period_length=step, onset=first_onset)
