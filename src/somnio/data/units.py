"""Unit definitions and parsing utilities.

Somnio stores signal arrays as plain NumPy arrays and tracks physical units as
lightweight metadata on :class:`~somnio.data.timeseries.TimeSeries` and
:class:`~somnio.data.timeseries.Sample`.

This module provides a small, structured unit type to avoid ad-hoc unit strings
throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, TypeAlias

import numpy as np


class Dimension(str, Enum):
    """Physical dimensions used in somnio unit metadata."""

    VOLTAGE = "voltage"
    ACCELERATION = "acceleration"
    TEMPERATURE = "temperature"
    DIMENSIONLESS = "dimensionless"


@dataclass(frozen=True, slots=True)
class Unit:
    """Physical unit metadata.

    Attributes:
        symbol: Canonical short symbol (e.g. ``"V"``).
        dimension: Physical dimension.
        scale_to_si: Multiplicative scale factor to convert a quantity in this
            unit to the canonical SI unit for the given dimension.
            For example, microvolts ``uV`` has ``scale_to_si=1e-6`` relative to
            volts ``V``.
    """

    symbol: str
    dimension: Dimension
    scale_to_si: float = 1.0

    def __str__(self) -> str:  # pragma: no cover
        return self.symbol


UnitLike: TypeAlias = Unit | str


# Canonical SI units used by somnio.
V: Final[Unit] = Unit("V", Dimension.VOLTAGE, 1.0)
M_S2: Final[Unit] = Unit("m/s^2", Dimension.ACCELERATION, 1.0)
DEG_C: Final[Unit] = Unit("degC", Dimension.TEMPERATURE, 1.0)

# Unitless unit (used for internal metadata).
UNITLESS: Final[Unit] = Unit("1", Dimension.DIMENSIONLESS, 1.0)

# Placeholder for unknown / missing unit metadata.
UNKNOWN: Final[Unit] = Unit("unknown", Dimension.DIMENSIONLESS, 1.0)

# Non-SI convenience units (not recommended for internal storage).
UV: Final[Unit] = Unit("uV", Dimension.VOLTAGE, 1e-6)
MV: Final[Unit] = Unit("mV", Dimension.VOLTAGE, 1e-3)
G: Final[Unit] = Unit("g", Dimension.ACCELERATION, 9.80665)

UNITS: Final[tuple[Unit, ...]] = (
    V,
    M_S2,
    DEG_C,
    UNITLESS,
    UNKNOWN,
    UV,
    MV,
    G,
)


_BY_SYMBOL: Final[dict[str, Unit]] = {u.symbol: u for u in UNITS}


def parse_unit(unit: UnitLike) -> Unit:
    """Normalize a unit input to a :class:`Unit`.

    Args:
        unit: A :class:`Unit` instance or a known unit symbol string.

    Returns:
        Normalized :class:`Unit` instance.

    Raises:
        ValueError: If `unit` is an unknown symbol.
        TypeError: If `unit` is neither `Unit` nor `str`.
    """

    if isinstance(unit, Unit):
        return unit
    if isinstance(unit, str):
        u = _BY_SYMBOL.get(unit)
        if u is None:
            raise ValueError(
                f"Unknown unit symbol {unit!r}. Known symbols: {sorted(_BY_SYMBOL)}"
            )
        return u
    raise TypeError(f"unit must be Unit or str, got {type(unit).__name__}")


def parse_unit_or(unit: UnitLike | None, *, default: UnitLike = UNKNOWN) -> Unit:
    """Parse a unit, falling back when missing/unknown."""
    if unit is None:
        return parse_unit(default)

    if isinstance(unit, str) and unit.strip() == "":
        return parse_unit(default)

    try:
        return parse_unit(unit)  # type: ignore[arg-type]
    except ValueError:
        return parse_unit(default)


def is_si_unit(unit: UnitLike) -> bool:
    """Return True when the unit is already in canonical SI form."""

    u = parse_unit(unit)
    return u.scale_to_si == 1.0


def convert_values(
    values: np.ndarray, from_unit: UnitLike, to_unit: UnitLike
) -> np.ndarray:
    """Convert numeric values between compatible units.

    Conversion is purely multiplicative using each unit's ``scale_to_si``:

    ``values_to = values_from * (from.scale_to_si / to.scale_to_si)``

    Args:
        values: Numeric array to convert.
        from_unit: Source unit (Unit or symbol string).
        to_unit: Target unit (Unit or symbol string).

    Returns:
        Converted NumPy array (float64).

    Raises:
        ValueError: If units have different dimensions.
    """

    src = parse_unit(from_unit)
    dst = parse_unit(to_unit)
    if src.dimension is not dst.dimension:
        raise ValueError(
            f"Cannot convert {src.symbol!r} ({src.dimension.value}) to "
            f"{dst.symbol!r} ({dst.dimension.value}): incompatible dimensions."
        )
    x = np.asarray(values, dtype=np.float64)
    return x * (src.scale_to_si / dst.scale_to_si)
