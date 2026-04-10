"""Binary parsing, sensor scaling, and stimulation encoding for ZMax live-mode."""

import math
import logging
from .constants import (
    LED_MAX_INTENSITY,
    LED_MIN_INTENSITY,
    STIMULATION_MAX_DURATION,
    STIMULATION_MAX_DURATION_S,
    STIMULATION_MAX_REPETITIONS,
    STIMULATION_MIN_DURATION,
    STIMULATION_MIN_DURATION_S,
    STIMULATION_MIN_REPETITIONS,
    STIMULATION_PWM_MAX,
    STIMULATION_TIME_UNIT_S,
)

logger = logging.getLogger(__name__)


def seconds_to_stimulation_units(seconds: float, *, name: str) -> int:
    """Convert a duration in seconds to wire-protocol units (1 unit = 100 ms)."""
    if seconds <= 0:
        raise ValueError(f"{name} must be positive")

    units = int(round(seconds / STIMULATION_TIME_UNIT_S))
    if not (STIMULATION_MIN_DURATION <= units <= STIMULATION_MAX_DURATION):
        raise ValueError(
            f"{name} must be between {STIMULATION_MIN_DURATION_S:g} s and {STIMULATION_MAX_DURATION_S:g} s "
            f"(device resolution {STIMULATION_TIME_UNIT_S * 1000:.0f} ms)"
        )

    quantized_seconds = units * STIMULATION_TIME_UNIT_S
    if not math.isclose(quantized_seconds, seconds, rel_tol=0.0, abs_tol=1e-9):
        logger.warning(
            "%s quantized to %d units (%.1f ms)",
            name,
            units,
            quantized_seconds * 1000,
        )

    return units


def stimulation_led_intensity_to_pwm(led_intensity_percent: int) -> int:
    """Map LED intensity (1–100 %) to the wire-protocol PWM value (0–254)."""
    if not (LED_MIN_INTENSITY <= led_intensity_percent <= LED_MAX_INTENSITY):
        raise ValueError(
            f"LED intensity must be between {LED_MIN_INTENSITY} and {LED_MAX_INTENSITY}"
        )
    return int(led_intensity_percent / 100 * STIMULATION_PWM_MAX)


def validate_stimulation_repetitions(repetitions: int) -> None:
    """Raise ``ValueError`` if *repetitions* is outside the device range."""
    if not (STIMULATION_MIN_REPETITIONS <= repetitions <= STIMULATION_MAX_REPETITIONS):
        raise ValueError(
            f"Repetitions must be between {STIMULATION_MIN_REPETITIONS}"
            f" and {STIMULATION_MAX_REPETITIONS}"
        )


def get_byte_at(buffer: str, index: int) -> int:
    """Read one byte from a space-separated hex buffer.

    The buffer is encoded as hex pairs separated by spaces:
    ``"00 1A FF ..."`` where each pair is one byte and ``index`` is
    the zero-based byte position.

    Args:
        buffer: Space-separated hex string (2 hex chars + 1 space per byte,
            no trailing space).
        index: Zero-based byte index.

    Returns:
        Integer value of the byte at *index*.
    """
    hex_str = buffer[index * 3 : index * 3 + 2]
    return int(hex_str, 16)


def get_word_at(buffer: str, index: int) -> int:
    """Read a big-endian 16-bit word from a space-separated hex buffer.

    Combines the byte at *index* (high byte) and *index + 1* (low byte)
    into a single 16-bit integer.

    Args:
        buffer: Space-separated hex string.
        index: Zero-based byte index of the high byte.

    Returns:
        16-bit integer value.
    """
    return get_byte_at(buffer, index) * 256 + get_byte_at(buffer, index + 1)


def scale_eeg(value: int) -> float:
    """Convert raw 16-bit EEG word to Volts (SI)."""
    uv_range = 3952.0
    microvolts = (value - 32768) * uv_range / 65536.0
    return microvolts * 1e-6


_G_STANDARD = 9.80665


def scale_accelerometer(value: int) -> float:
    """Convert raw 16-bit accelerometer word to m/s² (SI)."""
    g = value * 4 / 4096 - 2
    return g * _G_STANDARD


def scale_battery(value: int) -> float:
    """Convert raw 16-bit battery ADC word to Volts (SI)."""
    return value / 1024 * 6.6


def scale_body_temperature(value: int) -> float:
    """Convert raw 16-bit thermistor ADC word to °C."""
    return 15 + (value / 1024 * 3.3 - 1.0446) / 0.0565537333333333


def dec2hex(decimal: int, pad: int = 2) -> str:
    """Format an integer as an uppercase hex string with zero-padding.

    Args:
        decimal: Non-negative integer to convert.
        pad: Minimum number of hex digits (zero-padded on the left).

    Returns:
        Uppercase hex string of at least *pad* characters.
    """
    return format(decimal, f"0{pad}x").upper()
