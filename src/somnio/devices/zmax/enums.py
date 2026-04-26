"""ZMax data-type enumerations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from somnio.data.units import DEG_C, M_S2, UNITLESS, UNKNOWN, Unit, V

from .protocol import (
    get_word_at,
    scale_accelerometer,
    scale_battery,
    scale_body_temperature,
    scale_eeg,
)


@dataclass(frozen=True)
class DataTypeConfig:
    """Per-signal metadata for decoding one live packet.

    Attributes:
        file_name: EDF channel stem / live-mode label (e.g. ``"EEG R"``).
        buffer_position: Index of the high byte of the 16-bit word in the
            hex data buffer (see :func:`~protocol.get_word_at`).
        unit: Physical unit for the channel (SI metadata).
        scale_function: Optional callable that maps the raw 16-bit word to a
            physical value in SI units.  ``None`` means the raw integer is
            returned as-is.
    """

    file_name: str
    buffer_position: int
    unit: Unit
    scale_function: Callable[[int], float] | None = None

    def get_value(self, buffer: str) -> int | float:
        """Extract and optionally scale the value for this channel.

        Args:
            buffer: Space-separated hex string from a ZMax live-mode packet.

        Returns:
            Raw 16-bit integer if no scale function, otherwise the scaled
            float value in SI units.
        """
        value = get_word_at(buffer, self.buffer_position)
        if self.scale_function:
            return self.scale_function(value)
        return value


class DongleStatus(Enum):
    """USB dongle presence as reported by the server."""

    UNKNOWN = "unknown"
    INSERTED = "inserted"
    REMOVED = "removed"


class LEDColor(Enum):
    """RGB PWM triple used in ``LIVEMODE_SENDBYTES`` flash commands.

    Each member is a ``(R, G, B)`` tuple of PWM duty values (0–2 scale).
    """

    RED = (2, 0, 0)
    YELLOW = (2, 2, 0)
    GREEN = (0, 2, 0)
    CYAN = (0, 2, 2)
    BLUE = (0, 0, 2)
    PURPLE = (2, 0, 2)
    WHITE = (2, 2, 2)
    OFF = (0, 0, 0)


class DataType(Enum):
    """Canonical live-stream channels (order matches default multi-EDF layout)."""

    EEG_RIGHT = DataTypeConfig("EEG R", 1, V, scale_eeg)
    EEG_LEFT = DataTypeConfig("EEG L", 3, V, scale_eeg)
    ACCELEROMETER_X = DataTypeConfig("dX", 5, M_S2, scale_accelerometer)
    ACCELEROMETER_Y = DataTypeConfig("dY", 7, M_S2, scale_accelerometer)
    ACCELEROMETER_Z = DataTypeConfig("dZ", 9, M_S2, scale_accelerometer)
    BODY_TEMP = DataTypeConfig("BODY TEMP", 36, DEG_C, scale_body_temperature)
    BATTERY = DataTypeConfig("BATT", 23, V, scale_battery)
    NOISE = DataTypeConfig("NOISE", 19, UNITLESS)
    LIGHT = DataTypeConfig("LIGHT", 21, UNITLESS)
    NASAL_LEFT = DataTypeConfig("NASAL L", 11, UNKNOWN)
    NASAL_RIGHT = DataTypeConfig("NASAL R", 13, UNKNOWN)
    OXIMETER_INFRARED_AC = DataTypeConfig("OXY_IR_AC", 27, UNKNOWN)
    OXIMETER_RED_AC = DataTypeConfig("OXY_R_AC", 25, UNKNOWN)
    OXIMETER_DARK_AC = DataTypeConfig("OXY_DARK_AC", 34, UNKNOWN)
    OXIMETER_INFRARED_DC = DataTypeConfig("OXY_IR_DC", 17, UNKNOWN)
    OXIMETER_RED_DC = DataTypeConfig("OXY_R_DC", 15, UNKNOWN)
    OXIMETER_DARK_DC = DataTypeConfig("OXY_DARK_DC", 32, UNKNOWN)

    def __str__(self) -> str:
        return self.name

    @property
    def category(self) -> str:
        """First component of the member name before the first underscore.

        Examples: ``"EEG"``, ``"ACCELEROMETER"``, ``"OXIMETER"``.
        """
        return self.name.split("_")[0]

    @property
    def file_name(self) -> str:
        """EDF channel stem / live-mode label (delegates to ``value.file_name``)."""
        return self.value.file_name

    @classmethod
    def get_by_category(cls, category: str) -> list[DataType]:
        """Return all members whose :attr:`category` equals *category*.

        Args:
            category: Category prefix, e.g. ``"EEG"`` or ``"ACCELEROMETER"``.

        Returns:
            List of matching :class:`DataType` members (may be empty).
        """
        return [data_type for data_type in cls if data_type.category == category]
