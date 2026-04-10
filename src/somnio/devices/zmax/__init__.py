"""ZMax headband TCP live-mode client (Hypnodyne / ZMax USB server)."""

from somnio.devices.zmax.client import ConnectionClosedError, ZMax
from somnio.devices.zmax.constants import (
    DEFAULT_IP,
    DEFAULT_PORT,
    EXPECTED_DATA_LENGTH,
    LED_MAX_INTENSITY,
    LED_MIN_INTENSITY,
    MIN_DATA_LENGTH,
    SAMPLE_RATE,
    STIMULATION_MAX_DURATION,
    STIMULATION_MAX_DURATION_S,
    STIMULATION_MAX_REPETITIONS,
    STIMULATION_MIN_DURATION,
    STIMULATION_MIN_DURATION_S,
    STIMULATION_MIN_REPETITIONS,
    STIMULATION_TIME_UNIT_S,
)
from somnio.devices.zmax.enums import DataType, DongleStatus, LEDColor

__all__ = [
    "ConnectionClosedError",
    "DataType",
    "DEFAULT_IP",
    "DEFAULT_PORT",
    "DongleStatus",
    "EXPECTED_DATA_LENGTH",
    "LEDColor",
    "LED_MAX_INTENSITY",
    "LED_MIN_INTENSITY",
    "MIN_DATA_LENGTH",
    "SAMPLE_RATE",
    "STIMULATION_MAX_DURATION",
    "STIMULATION_MAX_DURATION_S",
    "STIMULATION_MAX_REPETITIONS",
    "STIMULATION_MIN_DURATION",
    "STIMULATION_MIN_REPETITIONS",
    "STIMULATION_MIN_DURATION_S",
    "STIMULATION_TIME_UNIT_S",
    "ZMax",
]
