"""Base protocol for zutils file format handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from zutils.data import TimeSeries


@runtime_checkable
class FileFormat(Protocol):
    """Protocol for reading and writing zutils data to a specific file format.

    Each concrete format (e.g. ``HDF5Format``, ``EDFFormat``) implements this
    protocol.  Within a single format, multiple layout strategies are possible
    by providing different implementations (e.g. ``HDF5Format`` vs a uSleep
    HDF5 variant).

    Example:
        ```python
        def save(fmt: FileFormat, path: Path, data: dict[str, TimeSeries]) -> None:
            fmt.write(path, data)
        ```
    """

    def write(self, path: Path, data: dict[str, TimeSeries]) -> None:
        """Write named ``TimeSeries`` objects to ``path``.

        Args:
            path: Destination file path.
            data: Mapping of name → ``TimeSeries`` to persist.
        """
        ...

    def read(self, path: Path) -> dict[str, TimeSeries]:
        """Read named ``TimeSeries`` objects from ``path``.

        Args:
            path: Source file path.

        Returns:
            Mapping of name → ``TimeSeries``.
        """
        ...
