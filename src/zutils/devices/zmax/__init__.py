"""ZMax headband TCP live-mode client (Hypnodyne / ZMax USB server)."""

from zutils.devices.zmax.client import ConnectionClosedError, ZMax, ensure_connected
from zutils.devices.zmax.constants import (
    DEFAULTS,
    EXPECTED_DATA_LENGTH,
    LED_MAX_INTENSITY,
    LED_MIN_INTENSITY,
    MIN_DATA_LENGTH,
    SAMPLE_RATE,
    STIMULATION_MAX_DURATION,
    STIMULATION_MAX_REPETITIONS,
    STIMULATION_MIN_DURATION,
    STIMULATION_MIN_REPETITIONS,
)
from zutils.devices.zmax.enums import DataType, DongleStatus, LEDColor

__all__ = [
    "ConnectionClosedError",
    "DataType",
    "DEFAULTS",
    "DongleStatus",
    "EXPECTED_DATA_LENGTH",
    "LEDColor",
    "LED_MAX_INTENSITY",
    "LED_MIN_INTENSITY",
    "MIN_DATA_LENGTH",
    "SAMPLE_RATE",
    "STIMULATION_MAX_DURATION",
    "STIMULATION_MAX_REPETITIONS",
    "STIMULATION_MIN_DURATION",
    "STIMULATION_MIN_REPETITIONS",
    "ZMax",
    "ensure_connected",
]
