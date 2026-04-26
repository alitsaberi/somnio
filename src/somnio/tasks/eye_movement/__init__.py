"""Eye-movement detection task."""

from somnio.tasks.eye_movement.event import (
    EVENT_TYPE,
    LEFT_LABEL,
    RIGHT_LABEL,
    is_valid_eye_movement_event,
    filter_by_pattern,
    merge_events,
)
from somnio.tasks.eye_movement.detect import detect_lr_eye_movements

__all__ = [
    "EVENT_TYPE",
    "LEFT_LABEL",
    "RIGHT_LABEL",
    "is_valid_eye_movement_event",
    "detect_lr_eye_movements",
    "merge_events",
    "filter_by_pattern",
]
