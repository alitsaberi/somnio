"""Multiplexed single-file EDF layout via MNE (optional ``edf`` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zutils.data.adapters.mne import from_mne_raw, import_mne, to_mne_raw
from zutils.data.timeseries import TimeSeries

from zutils.io.edf.utils import (
    ensure_export_edf_deps,
    require_edf_compatible_timestamps,
)


def read(
    path: Path | str,
    *,
    preload: bool = True,
    verbose: str | bool | None = "ERROR",
    units: dict[str, str] | str | None = None,
) -> TimeSeries:
    """Load one multiplexed EDF (+/BDF) file into a :class:`~zutils.data.timeseries.TimeSeries`.

    Uses :func:`mne.io.read_raw_edf`. Channel names are normalized (spaces → underscores).
    Physical scaling follows MNE (data returned in SI, e.g. Volts for EEG).

    Args:
        path: Path to ``.edf`` / ``.bdf`` / ``.gdf`` supported by MNE.
        preload: Passed through to MNE.
        verbose: MNE verbosity.
        units: Optional channel units hint forwarded to
            :func:`mne.io.read_raw_edf` when the file lacks unit metadata.
    """
    import_mne()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    from mne.io import read_raw_edf

    kwargs: dict[str, Any] = {
        "preload": preload,
        "verbose": verbose,
    }
    if units is not None:
        kwargs["units"] = units

    raw = read_raw_edf(path, **kwargs)
    return from_mne_raw(raw)


def write(
    path: Path | str,
    data: TimeSeries,
    *,
    overwrite: bool = False,
    verbose: str | bool | None = None,
) -> None:
    """Write ``data`` to a single EDF file using MNE's exporter (``edfio`` backend).

    Requires ``data.sample_rate``. Measurement date is taken from the first sample
    timestamp.
    """
    ensure_export_edf_deps()
    require_edf_compatible_timestamps(data)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True)")
    path.parent.mkdir(parents=True, exist_ok=True)

    mne = import_mne()
    raw = to_mne_raw(data)
    mne.export.export_raw(path, raw, overwrite=overwrite, verbose=verbose)


class StandardEDFReader:
    """Stateless reader for a single multiplexed EDF file."""

    def read(self, path: Path, **kwargs: Any) -> TimeSeries:
        """Read EDF at ``path``; supports ``preload``, ``verbose``, ``units`` kwargs."""
        return read(path, **kwargs)


class StandardEDFWriter:
    """Stateless writer for a single multiplexed EDF file."""

    def write(self, path: Path, data: TimeSeries, **kwargs: Any) -> None:
        """Write ``data``; supports ``overwrite`` and ``verbose`` kwargs."""
        overwrite = bool(kwargs.get("overwrite", False))
        verbose = kwargs.get("verbose", None)
        write(path, data, overwrite=overwrite, verbose=verbose)
