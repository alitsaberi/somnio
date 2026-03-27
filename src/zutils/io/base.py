"""I/O protocols for signal and annotation readers and writers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from zutils.data.annotations import Epochs, Event
from zutils.data.timeseries import TimeSeries


class SignalReader(Protocol):
    """Reads signal data from disk in a specific format+layout."""

    def read(self, path: Path, **kwargs) -> TimeSeries: ...


class SignalWriter(Protocol):
    """Writes signal data to disk in a specific format+layout."""

    def write(self, path: Path, data: TimeSeries, **kwargs) -> None: ...


class AnnotationReader(Protocol):
    """Reads annotations from disk in a specific format+layout."""

    def read(self, path: Path, **kwargs) -> list[Event] | Epochs: ...


class AnnotationWriter(Protocol):
    """Writes annotations to disk in a specific format+layout."""

    def write(self, path: Path, data: list[Event] | Epochs, **kwargs) -> None: ...
