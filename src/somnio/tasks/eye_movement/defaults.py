"""Default parameters for eye-movement detection."""

from __future__ import annotations

# Preprocessing
LOW_CUTOFF_HZ: float = 0.2
HIGH_CUTOFF_HZ: float = 10.0

# Primitive-event filtering
MIN_PEAK_AMPLITUDE_UV: float = 80.0
MAX_PEAK_AMPLITUDE_UV: float = 550.0
MIN_PEAK_GAP_S: float = 0.5
RELATIVE_PEAK_PROMINENCE: float = 0.7
MIN_EVENT_DURATION_S: float = 0.0
MAX_EVENT_DURATION_S: float = 1.2
MIN_EVENT_SKEWNESS: float = -0.3
MAX_EVENT_SKEWNESS: float = 0.3
RELATIVE_BASELINE: float = 0.05

# Sequence building
MAX_EVENT_GAP_S: float = 0.2

# Signal-quality filters
MIN_CORRELATION: float = 0.6
MIN_AMPLITUDE_RATIO: float = 0.5
MAX_AMPLITUDE_RATIO: float = 2.0
