"""Multiplexed single-file EDF/BDF layout via MNE."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from somnio.data.adapters.mne import from_mne_raw
from somnio.data.timeseries import TimeSeries

from somnio.io.edf.utils import write_edf
from somnio.utils.imports import MissingOptionalDependency

try:
    import mne
except ModuleNotFoundError as e:
    if e.name != "mne":
        raise
    raise MissingOptionalDependency(
        "mne", extra="edf", purpose="Standard EDF I/O"
    ) from e


def _is_bdf(path: Path) -> bool:
    return path.suffix.lower() == ".bdf"


def _normalize_requested_channels(
    requested: Sequence[str],
    available: list[str],
) -> list[str]:
    """Match requested names to MNE channel names (spaces ↔ underscores)."""
    by_norm = {name.replace(" ", "_"): name for name in available}
    missing: list[str] = []
    picked: list[str] = []
    for name in requested:
        key = name.replace(" ", "_")
        if key in by_norm:
            picked.append(by_norm[key])
        elif name in available:
            picked.append(name)
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Requested channel(s) not found: {missing}. Available: {available}"
        )
    return picked


def read(
    path: Path | str,
    *,
    preload: bool = True,
    verbose: str | bool | None = "ERROR",
    units: dict[str, str] | str | None = None,
    channels: Sequence[str] | None = None,
) -> TimeSeries:
    """Load one multiplexed EDF or BDF file into a :class:`~somnio.data.timeseries.TimeSeries`.

    Uses :func:`mne.io.read_raw_bdf` for ``.bdf`` and :func:`mne.io.read_raw_edf`
    otherwise. Channel names are normalized (spaces → underscores).
    Physical scaling follows MNE (data returned in SI, e.g. Volts for EEG).

    Args:
        path: Path to ``.edf`` / ``.bdf`` / ``.gdf`` supported by MNE.
        preload: Passed through to MNE when ``channels`` is ``None``. When
            ``channels`` is set, the file is opened without preload, channels
            are picked, then data are loaded.
        verbose: MNE verbosity.
        units: Optional channel units hint forwarded to MNE when the file lacks
            unit metadata.
        channels: Optional subset of channel names to load (saves RAM). Names
            may use spaces or underscores.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    reader = mne.io.read_raw_bdf if _is_bdf(path) else mne.io.read_raw_edf
    kwargs: dict[str, Any] = {
        "verbose": verbose,
    }
    if units is not None:
        kwargs["units"] = units

    if channels is not None:
        raw = reader(path, preload=False, **kwargs)
        picked = _normalize_requested_channels(list(channels), list(raw.ch_names))
        raw.pick(picked)
        raw.load_data()
    else:
        kwargs["preload"] = preload
        raw = reader(path, **kwargs)

    return from_mne_raw(raw)


def write(
    path: Path | str,
    data: TimeSeries,
    *,
    overwrite: bool = False,
    verbose: str | bool | None = None,
) -> None:
    """Write ``data`` to a single EDF file.

    Requires ``data.sample_rate``. Measurement date is taken from the first sample
    timestamp.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True)")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_edf(path, data, overwrite=overwrite, verbose=verbose)


class StandardEDFReader:
    """Stateless reader for a single multiplexed EDF/BDF file."""

    def read(self, path: Path, **kwargs: Any) -> TimeSeries:
        """Read EDF/BDF at ``path``; supports ``preload``, ``verbose``, ``units``, ``channels``."""
        return read(path, **kwargs)


class StandardEDFWriter:
    """Stateless writer for a single multiplexed EDF file."""

    def write(self, path: Path, data: TimeSeries, **kwargs: Any) -> None:
        """Write ``data``; supports ``overwrite`` and ``verbose`` kwargs."""
        overwrite = bool(kwargs.get("overwrite", False))
        verbose = kwargs.get("verbose", None)
        write(path, data, overwrite=overwrite, verbose=verbose)
