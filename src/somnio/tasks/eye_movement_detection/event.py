"""Eye-movement event validation, merging, and pattern filtering."""

from __future__ import annotations

import logging
import re

from somnio.data.annotations import Event

logger = logging.getLogger(__name__)

EVENT_TYPE: str = "eye_movement"
LEFT_LABEL: str = "L"
RIGHT_LABEL: str = "R"

_VALID_LABELS: frozenset[str] = frozenset({LEFT_LABEL, RIGHT_LABEL})


def is_valid_eye_movement_event(event: Event) -> bool:
    """Return ``True`` if *event* is a well-formed eye-movement event.

    The check passes when all of the following hold:

    * ``event.type == EVENT_TYPE`` (``"eye_movement"``)
    * ``event.label`` is a non-empty :class:`str`
    * every character of ``event.label`` is ``LEFT_LABEL`` (``"L"``) or
      ``RIGHT_LABEL`` (``"R"``)

    This covers both primitive events (label ``"L"`` or ``"R"``) and merged
    sequence events (e.g. label ``"LRLR"``).
    """
    if event.type != EVENT_TYPE:
        return False
    label = event.label
    if not isinstance(label, str) or not label:
        return False
    return all(c in _VALID_LABELS for c in label)


def merge_events(
    events: list[Event],
    /,
    max_event_gap_s: float,
) -> list[Event]:
    """Merge eye-movement events by temporal proximity.

    Consecutive events whose gap — ``onset(next) − end(prev)`` — is at most
    ``max_event_gap_s`` seconds are grouped and merged into a single sequence
    event.  The merged event's ``label`` is the concatenated direction string
    of its constituent events (e.g. ``"LRLR"``).

    Args:
        events: Ordered list of valid eye-movement events.
        max_event_gap_s: Maximum gap between the end of one event and the
            onset of the next (in **seconds**) for them to be merged into the
            same sequence.

    Returns:
        List of sequence :class:`~somnio.data.annotations.Event` objects.
        Each sequence spans from the onset of its first constituent event to
        the end of its last, and carries:

        * ``type``: ``"eye_movement"``
        * ``label``: direction string, e.g. ``"LRLR"``.

    Raises:
        AssertionError: If any event fails :func:`is_valid_eye_movement_event`.
    """
    if not events:
        return []

    max_event_gap_ns = int(max_event_gap_s * 1e9)

    _assert_valid(events[0])

    sequences: list[Event] = []
    current_group: list[Event] = [events[0]]

    for event in events[1:]:
        _assert_valid(event)

        prev = current_group[-1]
        gap_ns = event.onset - (prev.onset + prev.duration)

        if gap_ns <= max_event_gap_ns:
            current_group.append(event)
        else:
            sequences.append(_merge_group(current_group))
            current_group = [event]

    sequences.append(_merge_group(current_group))

    logger.info(
        "merge_events: merged %d events into %d sequences",
        len(events),
        len(sequences),
    )
    return sequences


def filter_by_pattern(
    events: list[Event],
    accepted_pattern: str,
) -> list[Event]:
    """Keep only events whose label fully matches *accepted_pattern*.

    Args:
        events: Eye-movement events.
        accepted_pattern: Regex pattern to match against the event ``label``.

    Returns:
        Filtered list of eye-movement events.
    """
    compiled = re.compile(accepted_pattern)
    valid: list[Event] = []

    for event in events:
        label = str(event.label) if event.label is not None else ""
        if compiled.fullmatch(label):
            valid.append(event)
        else:
            logger.info(
                "Event %r at onset=%.3fs rejected — pattern %r not matched",
                label,
                event.onset / 1e9,
                accepted_pattern,
            )

    return valid


def _assert_valid(event: Event) -> None:
    assert is_valid_eye_movement_event(event), f"Invalid eye-movement event: {event}"


def _merge_group(events: list[Event]) -> Event:
    """Merge a contiguous group of eye-movement events into one event."""
    label = "".join(str(e.label) for e in events)
    onset = events[0].onset
    duration = (events[-1].onset + events[-1].duration) - onset
    return Event(
        onset=onset,
        duration=duration,
        type=EVENT_TYPE,
        label=label,
    )
